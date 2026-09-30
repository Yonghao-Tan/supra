#ifndef SUPRA_ATTENTION_CHECKPOINT_CHECKER_HPP
#define SUPRA_ATTENTION_CHECKPOINT_CHECKER_HPP

#include <json.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <set>
#include <vector>

class AttentionCheckpointChecker {
    std::vector<AttentionCheckpointChecker> checks_;
    std::vector<std::string> check_directories_;
    struct Tensor {
        std::string name;
        unsigned width;
        std::vector<std::uint8_t> expected, actual, seen;
        std::uint64_t mismatches = 0, duplicates = 0, events = 0;
    };
    std::array<Tensor, 10> tensors_;
    std::vector<std::vector<std::size_t>> token_map_;
    std::vector<std::size_t> token_positions_;
    std::size_t layer_ = 0, tokens_ = 0, sequence_ = 0, heads_ = 0;
    std::size_t target_execution_ = 0;
    std::size_t preserve_event_index_ = 0, residual_event_index_ = 0;
    bool enabled_ = false, query_enabled_ = false, intermediate_enabled_ = false;
    bool preserve_enabled_ = false, residual_enabled_ = false;
    bool refill_enabled_ = false, active_ = false;
    bool context_enabled_ = false;

    template<class Words>
    static std::uint32_t raw(const Words& words, unsigned offset, unsigned width) {
        return (words[offset / 32] >> (offset % 32)) & ((1u << width) - 1);
    }

    void put(unsigned tensor, unsigned layer, unsigned round, unsigned head,
             unsigned physical, unsigned key, std::uint32_t value) {
        if (layer != layer_ || round >= token_map_.size() || head >= heads_ ||
                physical >= token_map_[round].size())
            throw std::runtime_error("Attention checkpoint event outside configured input: tensor=" +
                std::to_string(tensor) + " layer=" + std::to_string(layer) + " round=" +
                std::to_string(round) + " head=" + std::to_string(head) + " physical=" +
                std::to_string(physical));
        const auto token = token_map_[round][physical];
        std::size_t index = 0;
        if (tensor == 0 || tensor == 4 || tensor == 5) {
            if (key >= sequence_) throw std::runtime_error("Attention probability key outside sequence");
            index = (head * tokens_ + token) * sequence_ + key;
        } else if (tensor == 1 || tensor == 3) {
            if (key != 0) throw std::runtime_error("Attention scale key must be zero");
            index = head * tokens_ + token;
        } else if (tensor == 2 || tensor == 9) {
            if (key >= 128) throw std::runtime_error("Query code element outside head dimension");
            index = (head * tokens_ + token) * 128 + key;
        } else {
            throw std::runtime_error("unknown Attention checkpoint tensor");
        }
        put_raw(tensor, index, value);
    }

    void put_raw(unsigned tensor, std::size_t index, std::uint32_t value) {
        auto& target = tensors_[tensor];
        if (index >= target.seen.size())
            throw std::runtime_error(target.name + " event index exceeds configured input");
        ++target.events;
        if (target.seen[index]) ++target.duplicates;
        target.seen[index] = 1;
        std::uint32_t reference = 0;
        for (unsigned byte = 0; byte < target.width; ++byte) {
            reference |= std::uint32_t(target.expected[index * target.width + byte]) << (8 * byte);
            target.actual[index * target.width + byte] = value >> (8 * byte);
        }
        if (reference != value) {
            if (target.mismatches < 8)
                std::cerr << "ATTENTION_RAW_MISMATCH " << target.name << " index=" << index
                          << " actual=0x" << std::hex << value << " expected=0x" << reference << std::dec << '\n';
            ++target.mismatches;
        }
    }

    void put_hidden_stream(unsigned tensor, std::size_t& event_index, std::uint16_t value) {
        put_raw(tensor, event_index++, value);
    }

    void put_hidden_at(unsigned tensor, std::size_t index, std::uint16_t value) {
        put_raw(tensor, index, value);
    }

public:
    AttentionCheckpointChecker(const nlohmann::json& cfg, const std::filesystem::path& directory) {
        if (!cfg.contains("attention_checkpoints")) return;
        const auto& settings = cfg.at("attention_checkpoints");
        if (settings.is_array()) {
            if (settings.empty()) throw std::runtime_error("empty attention checkpoint list");
            std::set<std::string> names;
            for (const auto& entry : settings) {
                if (!entry.is_object() || !entry.contains("execution_index"))
                    throw std::runtime_error("attention checkpoint list requires explicit execution indices");
                const auto name = std::string("attention_execution_") + entry.at("execution_index").dump() + "_layer_" + entry.at("layer").dump();
                if (!names.insert(name).second) throw std::runtime_error("duplicate attention checkpoint scope");
                auto local = cfg;
                local["attention_checkpoints"] = entry;
                checks_.emplace_back(local, directory);
                check_directories_.push_back(name);
            }
            return;
        }

        if (cfg.at("executions").size() != 1 && !settings.contains("execution_index"))
            throw std::runtime_error("Attention checkpoints require an explicit launch index");
        target_execution_ = settings.value("execution_index", std::size_t{0});
        if (target_execution_ >= cfg.at("executions").size())
            throw std::runtime_error("Attention checkpoint launch index outside configured launches");
        layer_ = settings.at("layer").get<std::size_t>();
        tokens_ = settings.at("tokens").get<std::size_t>();
        sequence_ = settings.at("sequence").get<std::size_t>();
        heads_ = settings.at("heads").get<std::size_t>();
        if (layer_ >= 32 || !tokens_ || tokens_ > 2048 || !sequence_ || sequence_ > 2048 || heads_ != 32)
            throw std::runtime_error("Attention checkpoint dimensions outside hardware capacity");
        token_map_ = settings.at("physical_to_reference_token").get<decltype(token_map_)>();
        if (token_map_.empty() || token_map_.size() > 64)
            throw std::runtime_error("Attention checkpoint round count outside metadata capacity");
        std::vector<bool> seen(tokens_);
        for (const auto& round : token_map_) {
            if (round.empty() || round.size() > 48)
                throw std::runtime_error("Attention checkpoint token count outside round capacity");
            for (auto token : round) {
                if (token >= tokens_ || seen[token])
                    throw std::runtime_error("Attention checkpoint token map is not a permutation");
                seen[token] = true;
            }
        }
        for (bool token : seen) if (!token)
            throw std::runtime_error("Attention checkpoint token map omits input tokens");
        const auto& expected = settings.at("expected");
        if (!expected.is_array())
            throw std::runtime_error("Attention checkpoints require Probability and optional Query/intermediate tensors");
        const nlohmann::json* preserve_entry = nullptr;
        const nlohmann::json* refill_entry = nullptr;
        const nlohmann::json* residual_entry = nullptr;
        const nlohmann::json* context_entry = nullptr;
        for (const auto& item : expected) {
            const auto name = item.at("name").get<std::string>();
            auto** entry = name == "preserved_hidden" ? &preserve_entry :
                name == "preserved_hidden_refill" ? &refill_entry :
                name == "attention_residual" ? &residual_entry :
                name == "context" ? &context_entry : nullptr;
            if (entry) {
                if (*entry) throw std::runtime_error("duplicate hidden stream tensor");
                *entry = &item;
            }
        }
        preserve_enabled_ = preserve_entry != nullptr;
        refill_enabled_ = refill_entry != nullptr;
        residual_enabled_ = residual_entry != nullptr;
        context_enabled_ = context_entry != nullptr;
        const auto attention_tensor_count = expected.size() - unsigned(preserve_enabled_) -
            unsigned(refill_enabled_) - unsigned(residual_enabled_) - unsigned(context_enabled_);
        if (attention_tensor_count != 2 && attention_tensor_count != 4 &&
                attention_tensor_count != 6)
            throw std::runtime_error("Attention checkpoints require Probability and optional Query/intermediate tensors");
        query_enabled_ = attention_tensor_count >= 4;
        intermediate_enabled_ = attention_tensor_count == 6;
        if (query_enabled_) {
            token_positions_ = settings.at("token_positions").get<std::vector<std::size_t>>();
            if (token_positions_.size() != tokens_ ||
                    std::any_of(token_positions_.begin(), token_positions_.end(),
                        [&](std::size_t position) { return position >= sequence_; }))
                throw std::runtime_error("Attention Query logical positions differ from configured tokens");
        }
        for (unsigned i = 0; i < (intermediate_enabled_ ? 6u : query_enabled_ ? 4u : 2u); ++i) {
            auto& target = tensors_[i];
            target.name = std::array<const char*, 6>{"probability_codes", "probability_scale",
                "query_codes", "query_scale", "scores", "probabilities"}[i];
            target.width = i == 0 || i == 2 ? 1 : 2;
            const auto elements = i == 0 || i >= 4 ? sequence_ : i == 2 ? 128u : 1u;
            const std::vector<std::size_t> shape{heads_, tokens_, elements};
            const auto count = heads_ * tokens_ * elements;
            const nlohmann::json* entry = nullptr;
            for (const auto& item : expected) if (item.at("name") == target.name) {
                if (entry) throw std::runtime_error("duplicate Attention expected tensor");
                entry = &item;
            }
            if (!entry || entry->at("dtype") != (target.width == 2 ? "<u2" : "|i1") ||
                    entry->at("shape").get<std::vector<std::size_t>>() != shape ||
                    entry->at("bytes") != count * target.width)
                throw std::runtime_error("Attention expected tensor shape/dtype/size differs");
            const auto path = directory / entry->at("path").get<std::string>();
            std::ifstream stream(path, std::ios::binary);
            if (!stream || std::filesystem::file_size(path) != count * target.width)
                throw std::runtime_error("Attention expected file size differs");
            target.expected.resize(count * target.width);
            stream.read(reinterpret_cast<char*>(target.expected.data()), target.expected.size());
            if (!stream) throw std::runtime_error("Attention expected read failed");
            target.actual.resize(target.expected.size());
            target.seen.resize(count);
        }
        const auto configure_hidden_stream = [&](unsigned tensor, const char* name,
                                                  const nlohmann::json* entry,
                                                  bool token_major) {
            if (!entry) return;
            auto& target = tensors_[tensor];
            target.name = name;
            target.width = 2;
            const std::vector<std::size_t> shape{tokens_, 4096};
            const auto count = tokens_ * 4096;
            if (entry->at("dtype") != "<u2" ||
                    entry->at("shape").get<std::vector<std::size_t>>() != shape ||
                    entry->at("bytes") != count * 2)
                throw std::runtime_error(std::string(name) + " shape/dtype/size differs");
            const auto path = directory / entry->at("path").get<std::string>();
            std::ifstream stream(path, std::ios::binary);
            if (!stream || std::filesystem::file_size(path) != count * 2)
                throw std::runtime_error(std::string(name) + " expected file size differs");
            std::vector<std::uint8_t> logical(count * 2);
            stream.read(reinterpret_cast<char*>(logical.data()), logical.size());
            if (!stream) throw std::runtime_error("preserved hidden expected read failed");
            auto stream_map = token_map_;
            if (token_major && settings.contains("preserved_hidden_token_order")) {
                const auto order = settings.at("preserved_hidden_token_order").get<std::vector<std::size_t>>();
                auto sorted = order;
                std::sort(sorted.begin(), sorted.end());
                if (sorted.size() != tokens_)
                    throw std::runtime_error("preserved hidden token order has wrong length");
                for (std::size_t i = 0; i < tokens_; ++i)
                    if (sorted[i] != i) throw std::runtime_error("preserved hidden token order is not a permutation");
                stream_map = {order};
            }
            if (!token_major && settings.contains("attention_residual_token_order")) {
                stream_map = settings.at("attention_residual_token_order").get<decltype(token_map_)>();
                if (stream_map.size() != token_map_.size())
                    throw std::runtime_error("Attention residual round count differs");
                for (std::size_t r = 0; r < stream_map.size(); ++r)
                    if (stream_map[r].size() > token_map_[r].size() ||
                            !std::equal(stream_map[r].begin(), stream_map[r].end(), token_map_[r].begin()))
                        throw std::runtime_error("Attention residual must select each round's output prefix");
            }
            target.expected.reserve(logical.size());
            for (const auto& round : stream_map)
                for (unsigned outer = 0;
                     outer < (token_major ? round.size() : 4096 / 8); ++outer)
                    for (unsigned inner = 0;
                         inner < (token_major ? 4096 / 8 : round.size()); ++inner)
                        for (unsigned lane = 0; lane < 8; ++lane) {
                            const auto token = round[token_major ? outer : inner];
                            const auto stripe = token_major ? inner : outer;
                            const auto source = (token * 4096 + stripe * 8 + lane) * 2;
                            target.expected.push_back(logical[source]);
                            target.expected.push_back(logical[source + 1]);
                        }
            target.actual.resize(target.expected.size());
            target.seen.resize(target.expected.size() / target.width);
        };
        configure_hidden_stream(6, "preserved_hidden", preserve_entry, true);
        configure_hidden_stream(7, "attention_residual", residual_entry, false);
        if (refill_entry) {
            if (refill_entry->at("dtype") != "<u2" ||
                    refill_entry->at("shape") != nlohmann::json::array({tokens_, 4096}) ||
                    refill_entry->at("bytes") != tokens_ * 4096 * 2)
                throw std::runtime_error("Hidden refill checkpoint requires BF16 [tokens,4096]");
            auto& target = tensors_[8];
            target.name = "preserved_hidden_refill";
            target.width = 2;
            const auto path =
                directory / refill_entry->at("path").get<std::string>();
            std::ifstream stream(path, std::ios::binary);
            target.expected.resize(tokens_ * 4096 * 2);
            stream.read(reinterpret_cast<char*>(target.expected.data()),
                        target.expected.size());
            if (!stream || stream.peek() != std::char_traits<char>::eof())
                throw std::runtime_error(
                    "preserved hidden refill expected read failed");
            target.actual.resize(target.expected.size());
            target.seen.resize(tokens_ * 4096);
        }
        if (context_enabled_) {
            if (context_entry->at("dtype") != "<u2" ||
                    context_entry->at("shape") != nlohmann::json::array({heads_, tokens_, 128}) ||
                    context_entry->at("bytes") != heads_ * tokens_ * 128 * 2)
                throw std::runtime_error("Context checkpoint requires full BF16 [heads,tokens,128]");
            auto& target = tensors_[9];
            target.name = "context";
            target.width = 2;
            const auto path = directory / context_entry->at("path").get<std::string>();
            target.expected.resize(heads_ * tokens_ * 128 * 2);
            std::ifstream stream(path, std::ios::binary);
            if (!stream || std::filesystem::file_size(path) != target.expected.size())
                throw std::runtime_error("Context checkpoint file size differs");
            stream.read(reinterpret_cast<char*>(target.expected.data()), target.expected.size());
            if (!stream) throw std::runtime_error("Context checkpoint read failed");
            target.actual.resize(target.expected.size());
            target.seen.resize(heads_ * tokens_ * 128);
        }
        enabled_ = true;
        active_ = cfg.at("executions").size() == 1;
    }

    void begin_execution(std::size_t launch) {
        for (auto& check : checks_) check.begin_execution(launch);
        active_ = enabled_ && launch == target_execution_;
    }

    // Observe accepted writes before the rising clock edge; never drive the DUT.
    template<class Dut>
    void observe(const Dut& dut) {
        for (auto& check : checks_) check.observe(dut);
        if (!active_ || dut.rst) return;
        // Preservation can run before the command layer advances from the
        // previous execution. Its layer comes from the execution controller.
        if (preserve_enabled_ && dut.hidden_ddr_observe_valid &&
                dut.hidden_ddr_observe_preserve &&
                dut.hidden_ddr_observe_preserve_layer == layer_) {
            if (dut.hidden_ddr_observe_byte_enable != 0xffffU)
                throw std::runtime_error("preserved hidden DDR byte enable differs");
            for (unsigned lane = 0; lane < 8; ++lane)
                put_hidden_stream(6, preserve_event_index_,
                    raw(dut.hidden_ddr_observe_data, lane * 16, 16));
        }
        // Refill follows QKV and residual follows Attention in the same layer;
        // arithmetic checkpoints likewise belong to the accepted command.
        if (dut.observed_layer != layer_) return;
        if (context_enabled_ && dut.context_bf16_observe_valid) {
            const auto word = dut.context_bf16_observe_address;
            for (unsigned lane = 0; lane < 8; ++lane)
                if (((dut.context_bf16_observe_byte_enable >> (lane * 2)) & 3u) == 3u)
                    put(9, dut.observed_layer, dut.debug_row_token_batch_index,
                        dut.attention_score_observe_head, word / 16,
                        (word % 16) * 8 + lane,
                        raw(dut.context_bf16_observe_data, lane * 16, 16));
        }
        if (residual_enabled_ && dut.hidden_ddr_observe_valid &&
                dut.hidden_ddr_observe_ffn) {
            if (dut.hidden_ddr_observe_byte_enable != 0xffffU)
                throw std::runtime_error("Attention residual DDR byte enable differs");
            for (unsigned lane = 0; lane < 8; ++lane)
                put_hidden_stream(7, residual_event_index_,
                    raw(dut.hidden_ddr_observe_data, lane * 16, 16));
        }
        if (refill_enabled_ && dut.hidden_local_write_observe_valid &&
                dut.hidden_local_write_observe_preserved_refill) {
            const auto round = dut.debug_row_token_batch_index;
            const auto physical = dut.hidden_local_write_observe_physical_row;
            const auto word = dut.hidden_local_write_observe_channel_word;
            if (round >= token_map_.size() || physical >= token_map_[round].size() ||
                    word >= 4096 / 8)
                throw std::runtime_error("preserved hidden refill metadata outside configured input");
            if (dut.hidden_local_write_observe_byte_enable != 0xffffU)
                throw std::runtime_error("preserved hidden refill byte enable differs");
            const auto token = token_map_[round][physical];
            for (unsigned lane = 0; lane < 8; ++lane)
                put_hidden_at(8, token * 4096 + word * 8 + lane,
                    raw(dut.hidden_local_write_observe_data, lane * 16, 16));
        }
        if (query_enabled_ && dut.qkv_quantized_observe_valid &&
                dut.qkv_quantized_observe_qkv_select == 0) {
            const auto round = dut.debug_row_token_batch_index;
            const auto physical = dut.qkv_quantized_observe_row;
            if (round >= token_map_.size() || physical >= token_map_[round].size() ||
                    dut.qkv_quantized_observe_head >= heads_ ||
                    dut.qkv_quantized_observe_half > 1 ||
                    dut.qkv_quantized_observe_lane_mask != ~std::uint64_t{0} ||
                    dut.qkv_quantized_observe_token_position !=
                        token_positions_.at(token_map_[round][physical]))
                throw std::runtime_error("Query checkpoint metadata differs from configured token map");
            if (dut.qkv_quantized_observe_half == 0)
                put(3, dut.observed_layer, round, dut.qkv_quantized_observe_head,
                    physical, 0, dut.qkv_quantized_observe_scale);
            for (unsigned lane = 0; lane < 64; ++lane)
                put(2, dut.observed_layer, round, dut.qkv_quantized_observe_head,
                    physical, dut.qkv_quantized_observe_half * 64 + lane,
                    raw(dut.qkv_quantized_observe_values, lane * 8, 8));
        }
        if (intermediate_enabled_ && dut.attention_score_observe_valid)
            for (unsigned lane = 0; lane < 64; ++lane)
                if (raw(dut.attention_score_observe_byte_enable, lane * 2, 2) == 3)
                    put(4, dut.observed_layer, dut.debug_row_token_batch_index,
                        dut.attention_score_observe_head,
                        dut.attention_score_observe_query_base + lane / 8,
                        dut.attention_score_observe_key_base + lane % 8,
                        raw(dut.attention_score_observe_data, lane * 16, 16));
        if (intermediate_enabled_ && dut.softmax_bf16_observe_valid)
            for (unsigned lane = 0; lane < 64; ++lane)
                if (raw(dut.softmax_bf16_observe_byte_enable, lane * 2, 2) == 3)
                    put(5, dut.observed_layer, dut.debug_row_token_batch_index,
                        dut.softmax_bf16_observe_head,
                        dut.softmax_bf16_observe_row + lane / 8,
                        dut.softmax_bf16_observe_key + lane % 8,
                        raw(dut.softmax_bf16_observe_data, lane * 16, 16));
        if (dut.softmax_quantized_observe_valid)
            for (unsigned lane = 0; lane < 64; ++lane)
                if ((dut.softmax_quantized_observe_byte_enable >> lane) & 1)
                    put(0, dut.observed_layer, dut.debug_row_token_batch_index,
                        dut.softmax_quantized_observe_head, dut.softmax_quantized_observe_row + lane / 8,
                        dut.softmax_quantized_observe_key + lane % 8,
                        raw(dut.softmax_quantized_observe_data, lane * 8, 8));
        if (dut.softmax_scale_observe_valid)
            for (unsigned lane = 0; lane < 8; ++lane)
                if ((dut.softmax_scale_observe_row_mask >> lane) & 1)
                    put(1, dut.observed_layer, dut.debug_row_token_batch_index,
                        dut.softmax_scale_observe_head, dut.softmax_scale_observe_row + lane, 0,
                        raw(dut.softmax_scale_observe_value, lane * 16, 16));
    }

    nlohmann::json replay_complete(const std::filesystem::path& directory,
                                  const nlohmann::json& previous) {
        if (!checks_.empty()) {
            if (!previous.contains("checks") || previous.at("checks").size() != checks_.size())
                throw std::runtime_error("complete Attention checkpoint list required for replay");
            nlohmann::json results = nlohmann::json::array();
            std::uint64_t errors = 0;
            for (std::size_t i = 0; i < checks_.size(); ++i) {
                if (previous.at("checks").at(i).at("directory") != check_directories_[i])
                    throw std::runtime_error("Attention replay scope differs");
                auto result = checks_[i].replay_complete(directory / check_directories_[i], previous.at("checks").at(i));
                result["directory"] = check_directories_[i];
                errors += result.at("mismatches").get<std::uint64_t>();
                results.push_back(result);
            }
            return {{"status", errors ? "FAIL" : "PASS"}, {"mismatches", errors}, {"checks", results}};
        }
        if (!enabled_ || previous.at("tensors").empty())
            throw std::runtime_error("complete Attention trace required for replay");
        nlohmann::json results = nlohmann::json::array();
        std::uint64_t errors = 0;
        for (unsigned index = 0; index < tensors_.size(); ++index) {
            auto& target = tensors_[index];
            if (target.expected.empty()) continue;
            auto found = std::find_if(previous.at("tensors").begin(), previous.at("tensors").end(),
                [&](const auto& item) { return item.at("name") == target.name; });
            if (found == previous.at("tensors").end() ||
                    found->at("duplicates") != 0 || found->at("events") != target.seen.size())
                throw std::runtime_error("incomplete accepted-event trace for " + target.name);
            const auto path = directory / ("attention." + target.name + ".actual.bin");
            const auto recorded_bytes = found->at("bytes").get<std::size_t>();
            if (found->at("missing") != 0 || recorded_bytes != target.expected.size() ||
                    std::filesystem::file_size(path) != recorded_bytes)
                throw std::runtime_error("wrong full raw payload size for " + target.name);
            std::ifstream stream(path, std::ios::binary);
            for (std::size_t element = 0; element < target.seen.size(); ++element) {
                std::uint32_t value = 0;
                for (unsigned byte = 0; byte < target.width; ++byte) {
                    const int read = stream.get();
                    if (read < 0) throw std::runtime_error("incomplete raw trace for " + target.name);
                    value |= std::uint32_t(read) << (8 * byte);
                }
                put_raw(index, element, value);
            }
            errors += target.mismatches;
            results.push_back({{"name", target.name}, {"events", target.events}, {"missing", 0},
                {"duplicates", target.duplicates}, {"mismatches", target.mismatches}, {"bytes", target.actual.size()}});
        }
        return {{"status", errors ? "FAIL" : "PASS_REPLAYED"}, {"mismatches", errors}, {"tensors", results}};
    }

    nlohmann::json finish(const std::filesystem::path& directory) const {
        if (!checks_.empty()) {
            nlohmann::json results = nlohmann::json::array();
            std::uint64_t errors = 0;
            for (std::size_t i = 0; i < checks_.size(); ++i) {
                const auto path = directory / check_directories_[i];
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
        for (unsigned index = 0;
             index < (intermediate_enabled_ ? 6u : query_enabled_ ? 4u : 2u);
             ++index) {
            const auto& tensor = tensors_[index];
            std::uint64_t missing = 0;
            for (auto seen : tensor.seen) missing += !seen;
            errors += missing + tensor.mismatches + tensor.duplicates;
            std::ofstream stream(directory / ("attention." + tensor.name + ".actual.bin"), std::ios::binary);
            stream.write(reinterpret_cast<const char*>(tensor.actual.data()), tensor.actual.size());
            stream.close();
            if (!stream) throw std::runtime_error("Attention actual write failed");
            results.push_back({{"name", tensor.name}, {"events", tensor.events}, {"missing", missing},
                {"duplicates", tensor.duplicates}, {"mismatches", tensor.mismatches}, {"bytes", tensor.actual.size()}});
        }
        if (preserve_enabled_) {
            const auto& tensor = tensors_[6];
            std::uint64_t missing = 0;
            for (auto seen : tensor.seen) missing += !seen;
            errors += missing + tensor.mismatches + tensor.duplicates;
            std::ofstream stream(directory / "attention.preserved_hidden.actual.bin", std::ios::binary);
            stream.write(reinterpret_cast<const char*>(tensor.actual.data()), tensor.actual.size());
            stream.close();
            if (!stream) throw std::runtime_error("preserved hidden actual write failed");
            results.push_back({{"name", tensor.name}, {"events", tensor.events}, {"missing", missing},
                {"duplicates", tensor.duplicates}, {"mismatches", tensor.mismatches}, {"bytes", tensor.actual.size()}});
        }
        if (residual_enabled_) {
            const auto& tensor = tensors_[7];
            std::uint64_t missing = 0;
            for (auto seen : tensor.seen) missing += !seen;
            errors += missing + tensor.mismatches + tensor.duplicates;
            std::ofstream stream(directory / "attention.attention_residual.actual.bin", std::ios::binary);
            stream.write(reinterpret_cast<const char*>(tensor.actual.data()), tensor.actual.size());
            stream.close();
            if (!stream) throw std::runtime_error("Attention residual actual write failed");
            results.push_back({{"name", tensor.name}, {"events", tensor.events}, {"missing", missing},
                {"duplicates", tensor.duplicates}, {"mismatches", tensor.mismatches}, {"bytes", tensor.actual.size()}});
        }
        if (refill_enabled_) {
            const auto& tensor = tensors_[8];
            std::uint64_t missing = 0;
            for (auto seen : tensor.seen) missing += !seen;
            errors += missing + tensor.mismatches + tensor.duplicates;
            std::ofstream stream(directory / "attention.preserved_hidden_refill.actual.bin",
                                 std::ios::binary);
            stream.write(reinterpret_cast<const char*>(tensor.actual.data()),
                         tensor.actual.size());
            stream.close();
            if (!stream) throw std::runtime_error("preserved hidden refill actual write failed");
            results.push_back({{"name", tensor.name}, {"events", tensor.events}, {"missing", missing},
                {"duplicates", tensor.duplicates}, {"mismatches", tensor.mismatches}, {"bytes", tensor.actual.size()}});
        }
        if (context_enabled_) {
            const auto& tensor = tensors_[9];
            std::uint64_t missing = 0;
            for (auto seen : tensor.seen) missing += !seen;
            errors += missing + tensor.mismatches + tensor.duplicates;
            std::ofstream stream(directory / "attention.context.actual.bin", std::ios::binary);
            stream.write(reinterpret_cast<const char*>(tensor.actual.data()), tensor.actual.size());
            stream.close();
            if (!stream) throw std::runtime_error("Context actual write failed");
            results.push_back({{"name", tensor.name}, {"events", tensor.events}, {"missing", missing},
                {"duplicates", tensor.duplicates}, {"mismatches", tensor.mismatches}, {"bytes", tensor.actual.size()}});
        }
        return {{"status", errors ? "FAIL" : "PASS"}, {"mismatches", errors}, {"tensors", results}};
    }
};
#endif
