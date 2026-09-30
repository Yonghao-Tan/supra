#ifndef SUPRA_HEAD_CHECKPOINT_CHECKER_HPP
#define SUPRA_HEAD_CHECKPOINT_CHECKER_HPP

#include <json.hpp>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <set>
#include <string>
#include <vector>

class HeadCheckpointChecker {
    std::vector<HeadCheckpointChecker> checks_;
    std::vector<std::string> check_directories_;
    struct Tensor {
        std::string name;
        unsigned width = 0;
        std::vector<std::uint8_t> expected, actual, seen;
        std::uint64_t events = 0, mismatches = 0, duplicates = 0;
    };
    std::vector<Tensor> tensors_;
    std::size_t tokens_ = 0, vocabulary_ = 0;
    std::size_t execution_index_ = 0;
    bool enabled_ = false, active_ = true;

    template<class Words>
    static std::uint32_t bits(const Words& words, unsigned offset, unsigned width) {
        const auto value = std::uint64_t(words[offset / 32]) >> (offset % 32);
        return std::uint32_t(value & ((std::uint64_t{1} << width) - 1));
    }

    void put(unsigned tensor, std::size_t index, std::uint32_t raw) {
        auto& value = tensors_.at(tensor);
        if (index >= value.seen.size())
            throw std::runtime_error("head checkpoint index outside tensor: " + value.name);
        ++value.events;
        if (value.seen[index]) ++value.duplicates;
        value.seen[index] = 1;
        std::uint32_t reference = 0;
        for (unsigned byte = 0; byte < value.width; ++byte) {
            reference |= std::uint32_t(value.expected[index * value.width + byte]) << (8 * byte);
            value.actual[index * value.width + byte] = raw >> (8 * byte);
        }
        if (reference != raw) {
            if (value.mismatches < 8)
                std::cerr << "HEAD_RAW_MISMATCH " << value.name << " index=" << index
                    << " actual=0x" << std::hex << raw << " expected=0x" << reference << std::dec << '\n';
            ++value.mismatches;
        }
    }

    void matrix(unsigned tensor, std::size_t token, std::size_t column, std::uint32_t raw) {
        const auto columns = tensor < 2 ? 4096 : vocabulary_;
        if (token >= tokens_ || column >= columns)
            throw std::runtime_error("head checkpoint token/column outside tensor: " + tensors_[tensor].name);
        put(tensor, token * columns + column, raw);
    }

public:
    HeadCheckpointChecker(const nlohmann::json& cfg, const std::filesystem::path& case_directory) {
        if (!cfg.contains("head_checkpoints")) return;
        const auto& head = cfg.at("head_checkpoints");
        if (head.is_array()) {
            if (head.empty()) throw std::runtime_error("empty head checkpoint list");
            std::set<std::string> names;
            for (const auto& entry : head) {
                if (!entry.is_object() || !entry.contains("execution_index"))
                    throw std::runtime_error("head checkpoint list requires explicit execution indices");
                const auto name = std::string("head_execution_") + entry.at("execution_index").dump();
                if (!names.insert(name).second) throw std::runtime_error("duplicate head checkpoint scope");
                auto local = cfg;
                local["head_checkpoints"] = entry;
                checks_.emplace_back(local, case_directory);
                check_directories_.push_back(name);
            }
            return;
        }

        tokens_ = head.at("tokens").get<std::size_t>();
        vocabulary_ = head.at("vocab").get<std::size_t>();
        if (!tokens_ || tokens_ > 96 || !vocabulary_ || vocabulary_ > 126464 || vocabulary_ % 8)
            throw std::runtime_error("invalid head checkpoint tokens/vocabulary");
        if (head.contains("execution_index")) {
            const auto& index = head.at("execution_index");
            if (!index.is_number_integer() || index.get<std::int64_t>() < 0 ||
                    index.get<std::uint64_t>() >= cfg.at("executions").size())
                throw std::runtime_error("head checkpoint execution_index outside launches");
            execution_index_ = index.get<std::size_t>();
            active_ = false;
        } else if (cfg.at("executions").size() != 1) {
            throw std::runtime_error("multiple launches require an explicit head checkpoint execution_index");
        }
        const std::array<std::string, 10> names{{"norm_output", "activation_codes", "activation_scale", "accumulator", "output",
            "candidate_top1", "candidate_logit", "candidate_confidence", "candidate_probability", "candidate_action"}};
        const std::array<std::string, 10> dtypes{{"<u2", "|i1", "<u2", "<i4", "<u2", "<u4", "<u2", "<u2", "<u2", "<u2"}};
        const std::array<unsigned, 10> widths{{2, 1, 2, 4, 2, 4, 2, 2, 2, 2}};
        const std::array<std::size_t, 10> counts{{tokens_ * 4096, tokens_ * 4096, tokens_, tokens_ * vocabulary_, tokens_ * vocabulary_,
            tokens_, tokens_, tokens_, tokens_, tokens_}};
        auto expected = head.at("expected");
        if (!expected.is_array() || expected.size() != 5)
            throw std::runtime_error("head requires expected norm_output, activation_codes, activation_scale, accumulator and output tensors");
        if (head.contains("candidates")) {
            const auto& candidates = head.at("candidates");
            if (!candidates.is_array() || candidates.size() != 5)
                throw std::runtime_error("candidate checks require all five expected fields");
            for (const auto& candidate : candidates) expected.push_back(candidate);
        }
        tensors_.resize(expected.size());
        for (unsigned i = 0; i < tensors_.size(); ++i) {
            const nlohmann::json* entry = nullptr;
            for (const auto& item : expected) if (item.at("name") == names[i]) {
                if (entry) throw std::runtime_error("duplicate head expected tensor: " + names[i]);
                entry = &item;
            }
            if (!entry || entry->at("dtype") != dtypes[i] || entry->at("bytes") != counts[i] * widths[i])
                throw std::runtime_error("missing or malformed head expected tensor: " + names[i]);
            const auto shape = i == 2 || i >= 5 ? std::vector<std::size_t>{tokens_} :
                std::vector<std::size_t>{tokens_, i < 2 ? 4096 : vocabulary_};
            if (entry->at("shape").get<std::vector<std::size_t>>() != shape)
                throw std::runtime_error("head expected shape differs: " + names[i]);
            const auto path = case_directory / entry->at("path").get<std::string>();
            std::ifstream input(path, std::ios::binary);
            if (!input || std::filesystem::file_size(path) != counts[i] * widths[i])
                throw std::runtime_error("head expected file size differs: " + names[i]);
            auto& value = tensors_[i];
            value.name = names[i]; value.width = widths[i];
            value.expected.resize(counts[i] * widths[i]);
            input.read(reinterpret_cast<char*>(value.expected.data()), value.expected.size());
            if (!input) throw std::runtime_error("cannot read head expected: " + names[i]);
            value.actual.resize(value.expected.size()); value.seen.resize(counts[i]);
        }
        enabled_ = true;
    }

    void begin_execution(std::size_t index) {
        for (auto& check : checks_) check.begin_execution(index);
        active_ = index == execution_index_;
    }

    // Call after clk=0 eval and before the rising edge that accepts these events.
    template<class Dut>
    void observe(const Dut& dut) {
        for (auto& check : checks_) check.observe(dut);
        if (!enabled_ || !active_ || dut.rst) return;
        // The shared RMSNorm also emits samples during Transformer execution.
        if (dut.post_active && dut.rms_trace_sample_valid)
            for (unsigned lane = 0; lane < 64; ++lane)
                if (bits(dut.rms_trace_sample_byte_enable, lane * 2, 2) == 3)
                    matrix(0, dut.post_norm_group_base + lane / 8,
                        dut.rms_trace_sample_element + lane % 8, bits(dut.rms_trace_sample_data, lane * 16, 16));
        if (dut.post_quant_observe_valid)
            for (unsigned lane = 0; lane < 64; ++lane)
                if ((dut.post_quant_observe_mask >> lane) & 1)
                    matrix(1, dut.post_quant_observe_row + lane / 8,
                        dut.post_quant_observe_element + lane % 8, bits(dut.post_quant_observe_values, lane * 8, 8));
        if (dut.post_scale_observe_valid)
            for (unsigned lane = 0; lane < 8; ++lane)
                if ((dut.post_scale_observe_mask >> lane) & 1)
                    put(2, dut.post_scale_observe_row + lane, bits(dut.post_scale_observe_values, lane * 16, 16));
        if (dut.post_int32_observe_valid)
            for (unsigned lane = 0; lane < 64; ++lane)
                if ((dut.post_int32_observe_mask >> lane) & 1)
                    matrix(3, dut.post_int32_observe_row + lane / 8,
                        dut.post_int32_observe_vocab + lane % 8, bits(dut.post_int32_observe_values, lane * 32, 32));
        if (dut.post_logit_observe_valid)
            for (unsigned lane = 0; lane < 64; ++lane)
                if ((dut.post_logit_observe_mask >> lane) & 1)
                    matrix(4, dut.post_logit_observe_row + lane / 8,
                        dut.post_logit_observe_vocab + lane % 8, bits(dut.post_logit_observe_values, lane * 16, 16));
        if (tensors_.size() == 10 && dut.post_candidate_observe_valid) {
            const auto token = dut.post_candidate_observe_row;
            put(5, token, dut.post_candidate_observe_top1);
            put(6, token, dut.post_candidate_observe_logit);
            put(7, token, dut.post_candidate_observe_confidence);
            put(8, token, dut.post_candidate_observe_probability);
            put(9, token, dut.post_candidate_observe_action);
        }
    }

    nlohmann::json finish(const std::filesystem::path& output) const {
        if (!checks_.empty()) {
            nlohmann::json results = nlohmann::json::array();
            std::uint64_t errors = 0;
            for (std::size_t i = 0; i < checks_.size(); ++i) {
                const auto path = output / check_directories_[i];
                std::filesystem::create_directories(path);
                auto result = checks_[i].finish(path);
                result["directory"] = check_directories_[i];
                errors += result.at("mismatches").get<std::uint64_t>();
                results.push_back(result);
            }
            return {{"status", errors ? "FAIL" : "PASS"}, {"mismatches", errors}, {"checks", results}};
        }

        if (!enabled_) return {{"status", "NOT_REQUESTED"}, {"mismatches", 0}};
        nlohmann::json results = nlohmann::json::array();
        std::uint64_t errors = 0;
        for (const auto& value : tensors_) {
            std::uint64_t missing = 0;
            for (auto seen : value.seen) missing += !seen;
            errors += missing + value.duplicates + value.mismatches;
            std::ofstream actual(output / ("head." + value.name + ".actual.bin"), std::ios::binary);
            actual.write(reinterpret_cast<const char*>(value.actual.data()), value.actual.size());
            actual.close();
            if (!actual) throw std::runtime_error("head actual output write failed: " + value.name);
            results.push_back({{"name", value.name}, {"elements", value.seen.size()},
                {"events", value.events}, {"bytes", value.actual.size()}, {"mismatches", value.mismatches},
                {"missing", missing}, {"duplicates", value.duplicates}});
        }
        return {{"status", errors ? "FAIL" : "PASS"}, {"mismatches", errors}, {"tensors", results}};
    }
};

#endif
