#include "testcase_memory_actions.hpp"
#include <array>
#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) throw std::runtime_error("supply an external self-test directory");
    const std::filesystem::path root = argv[1];
    std::filesystem::create_directories(root);
    std::array<std::uint8_t, 16> memory{};
    for (unsigned i = 0; i < memory.size(); ++i) memory[i] = i;
    auto read = [&](std::uint64_t address) { return address < memory.size() ? int(memory[address]) : -1; };
    auto write = [&](std::uint64_t address, std::uint8_t value) {
        if (address >= memory.size()) return 1;
        memory[address] = value; return 0;
    };
    const nlohmann::json execution = {{"copies", {{{"source_address", 0}, {"destination_address", 2}, {"bytes", 8}}}}};
    assert(testcase_copy_memory(execution, read, write) == 8);
    for (unsigned i = 0; i < 8; ++i) assert(memory[i + 2] == i);
    assert(testcase_copy_memory(nlohmann::json::object(), read, write) == 0);
    bool failed = false;
    try { testcase_copy_memory(execution, read, [](auto, auto) { return 1; }); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    failed = false;
    try { testcase_copy_memory(execution, [](auto) { return -1; }, write); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    {
        std::ofstream input(root / "segment.bin", std::ios::binary);
        input.put(0x80); input.put(0xff);
    }
    nlohmann::json cfg = {{"initial_segments", {{{"address", 12}, {"bytes", 2}, {"path", "segment.bin"}}}}};
    assert(testcase_initialize_segments(cfg, root, write) == 2);
    assert(memory[12] == 0x80 && memory[13] == 0xff);
    const nlohmann::json expected_cfg = {{"expected", {
        {{"name", "before"}, {"address", 12}, {"bytes", 2},
         {"path", "segment.bin"}, {"execution_index", 0}},
        {{"name", "after"}, {"address", 12}, {"bytes", 2},
         {"path", "segment.bin"}}
    }}};
    const auto before = testcase_compare_expected(expected_cfg, root, root, 0, read);
    assert(before.size() == 1 && before[0]["name"] == "before" &&
           before[0]["bytes"] == 2 && before[0]["mismatches"] == 0);
    assert(testcase_compare_expected(expected_cfg, root, root, 1, read).empty());
    memory[12] = 0;
    const auto after = testcase_compare_expected(expected_cfg, root, root, -1, read);
    assert(after.size() == 1 && after[0]["name"] == "after" &&
           after[0]["mismatches"] == 1);
    // A later write must neither erase the earlier result nor overwrite its raw file.
    std::ifstream captured(root / "before.actual.bin", std::ios::binary);
    assert(captured.get() == 0x80 && captured.get() == 0xff);
    assert(before[0]["mismatches"] == 0);
    cfg["initial_segments"][0]["bytes"] = 3;
    failed = false;
    try { testcase_initialize_segments(cfg, root, write); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    TestcaseExpectedLayout contiguous({{"address", 8}, {"bytes", 20}});
    assert(contiguous.address(19) == 27);
    TestcaseExpectedLayout strided({{"address", 8}, {"bytes", 20}, {"element_bytes", 10}, {"stride_bytes", 16}});
    assert(strided.address(9) == 17 && strided.address(10) == 24 && strided.address(19) == 33);
    for (const auto& bad : std::vector<nlohmann::json>{
            {{"address", 0}, {"bytes", 20}, {"element_bytes", 0}},
            {{"address", 0}, {"bytes", 20}, {"element_bytes", 3}},
            {{"address", 0}, {"bytes", 20}, {"element_bytes", 10}, {"stride_bytes", 9}},
            {{"address", UINT64_MAX - 20}, {"bytes", 20}, {"element_bytes", 10}, {"stride_bytes", 16}}}) {
        failed = false;
        try { TestcaseExpectedLayout invalid(bad); }
        catch (const std::runtime_error&) { failed = true; }
        assert(failed);
    }
    std::cout << "PASS testcase actual memory copies, segments, strided expected fields and range/read/write errors\n";
}
