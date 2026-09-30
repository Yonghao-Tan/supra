#ifndef SUPRA_TESTCASE_MEMORY_ACTIONS_HPP
#define SUPRA_TESTCASE_MEMORY_ACTIONS_HPP

#include <json.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

struct TestcaseExpectedLayout {
    std::uint64_t base, bytes, element, stride;
    explicit TestcaseExpectedLayout(const nlohmann::json& region) :
        base(region.at("address")), bytes(region.at("bytes")),
        element(region.value("element_bytes", bytes)), stride(region.value("stride_bytes", element)) {
        if (!bytes || base + bytes < base || !element || bytes % element || stride < element ||
                (bytes / element - 1) > (UINT64_MAX - base - element) / stride)
            throw std::runtime_error("invalid strided expected address range");
    }
    std::uint64_t address(std::uint64_t byte) const {
        if (byte >= bytes) throw std::runtime_error("expected byte index outside region");
        return base + (byte / element) * stride + byte % element;
    }
};

// Compare a value while its DDR lifetime is still valid, before a later
// execution overwrites it. This function only reads actual DDR bytes.
template<class Read>
nlohmann::json testcase_compare_expected(const nlohmann::json& cfg,
        const std::filesystem::path& input_directory,
        const std::filesystem::path& output, int after_execution, Read read_byte) {
    nlohmann::json comparisons = nlohmann::json::array();
    for (const auto& region : cfg.at("expected")) {
        if (region.value("execution_index", -1) != after_execution) continue;
        const auto name = region.at("name").template get<std::string>();
        if (name.empty() || std::filesystem::path(name).filename() != name || name == "." || name == "..")
            throw std::runtime_error("expected name must be a file basename");
        const TestcaseExpectedLayout layout(region);
        const auto count = layout.bytes;
        std::ifstream expected((input_directory / region.at("path").template get<std::string>()), std::ios::binary);
        if (!expected || std::filesystem::file_size((input_directory / region.at("path").template get<std::string>())) != count)
            throw std::runtime_error("expected file size mismatch: " + name);
        std::ofstream actual(output / (name + ".actual.bin"), std::ios::binary);
        if (!actual) throw std::runtime_error("cannot open actual output");
        std::uint64_t differences = 0;
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto actual_address = layout.address(i);
            const int value = read_byte(actual_address);
            const int reference = expected.get();
            if (value < 0 || reference < 0) throw std::runtime_error("DDR/expected read failed");
            actual.put(static_cast<char>(value));
            if (value != reference) {
                if (differences < 8) std::cerr << "TESTCASE_MISMATCH " << name
                    << " address=0x" << std::hex << actual_address
                    << " actual=0x" << value << " expected=0x" << reference << std::dec << '\n';
                ++differences;
            }
        }
        actual.close();
        if (!actual) throw std::runtime_error("actual output write failed");
        comparisons.push_back({{"name", name}, {"bytes", count}, {"mismatches", differences}});
        if (after_execution >= 0) comparisons.back()["execution_index"] = after_execution;
    }
    return comparisons;
}

template<class Write>
std::uint64_t testcase_initialize_segments(const nlohmann::json& cfg,
                                    const std::filesystem::path& directory, Write write) {
    std::uint64_t initialized = 0;
    if (!cfg.contains("initial_segments")) return initialized;
    for (const auto& segment : cfg.at("initial_segments")) {
        const auto address = segment.at("address").get<std::uint64_t>();
        const auto count = segment.at("bytes").get<std::uint64_t>();
        const auto path = directory / segment.at("path").get<std::string>();
        std::ifstream input(path, std::ios::binary);
        if (!count || address + count < address || !input || std::filesystem::file_size(path) != count)
            throw std::runtime_error("invalid initial DDR segment range/file");
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto value = input.get();
            if (value < 0 || write(address + i, std::uint8_t(value)))
                throw std::runtime_error("initial DDR segment load failed");
        }
        initialized += count;
    }
    return initialized;
}

// These host copies occur only between drained executions, outside accelerator timing.
template<class Read, class Write>
std::uint64_t testcase_copy_memory(const nlohmann::json& execution, Read read, Write write) {
    std::uint64_t copied = 0;
    if (!execution.contains("copies")) return copied;
    for (const auto& copy : execution.at("copies")) {
        const auto source = copy.at("source_address").get<std::uint64_t>();
        const auto destination = copy.at("destination_address").get<std::uint64_t>();
        const auto count = copy.at("bytes").get<std::uint64_t>();
        if (!count || source + count < source || destination + count < destination)
            throw std::runtime_error("invalid inter-execution DDR copy range");
        std::vector<std::uint8_t> actual(count);
        for (std::uint64_t i = 0; i < count; ++i) {
            const int value = read(source + i);
            if (value < 0) throw std::runtime_error("inter-execution DDR source read failed");
            actual[i] = value;
        }
        for (std::uint64_t i = 0; i < count; ++i)
            if (write(destination + i, actual[i]))
                throw std::runtime_error("inter-execution DDR copy requires drained memory");
        copied += count;
    }
    return copied;
}

#endif
