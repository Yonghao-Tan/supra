#ifndef SUPRA_TESTCASE_HANDOFF_ACTIONS_HPP
#define SUPRA_TESTCASE_HANDOFF_ACTIONS_HPP

#include <json.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>
#include "tb/generated/execution_config_packer.hpp"
#include "tb/generated/attention_dependency_job_packer.hpp"
extern "C" {
#include "cmodel/forward_postprocess_model.h"
}

namespace testcase_handoff {
inline void set(std::vector<std::uint8_t>& bytes, unsigned offset, unsigned width, std::uint64_t value) {
    for (unsigned b = 0; b < width; ++b) bytes.at(offset + b) = value >> (b * 8);
}

inline std::vector<std::uint8_t> metadata(const std::vector<forward_postprocess_next_token_descriptor>& tokens,
                                         unsigned sequence, unsigned capture) {
    std::vector<std::uint8_t> result;
    for (unsigned first = 0, round = 0; first < tokens.size(); ++round) {
        unsigned count = 0, slots = 0;
        while (count < 48 && first + count < tokens.size()) {
            const auto cost = tokens[first + count].activation_bits == 8 ? 2u : 1u;
            if (slots + cost > 64) break;
            slots += cost; ++count;
        }
        std::array<forward_postprocess_physical_row, 48> physical{};
        std::array<std::uint8_t, 48> inverse{};
        std::uint8_t groups = 0, semantic = 0;
        if (!count || forward_postprocess_pack_next_tokens(tokens.data() + first, count, sequence,
                physical.data(), inverse.data(), &groups, &semantic))
            throw std::runtime_error("C11 rejected actual handoff token packing");
        const auto base = result.size(), bytes = forward_postprocess_token_metadata_token_batch_bytes(count);
        result.resize(base + bytes);
        result[base] = count; result[base + 1] = groups; result[base + 2] = semantic;
        set(result, base + 4, 2, round); set(result, base + 6, 2, first);
        set(result, base + 8, 2, bytes); set(result, base + 10, 2, 16);
        set(result, base + 12, 2, 32 + count * 16); set(result, base + 14, 2, ((count + 15) / 16) * 16);
        set(result, base + 16, 4, 1); set(result, base + 20, 4, capture);
        for (unsigned token = 0; token < count; ++token) {
            const auto& item = physical[token]; const auto offset = base + 32 + token * 16;
            result[base + 3] |= item.source == FORWARD_POSTPROCESS_EMBEDDING_TOKEN;
            set(result, offset, 4, item.source_index); set(result, offset + 4, 2, item.token_position);
            set(result, offset + 6, 2, item.kv_index);
            const std::array<std::uint8_t, 8> tail{{item.token_ordinal, item.source, item.activation_bits,
                item.query_group, item.cache_group, item.compute_group, item.pe_slot, item.phase_mask}};
            std::copy(tail.begin(), tail.end(), result.begin() + offset + 8);
        }
        std::copy_n(inverse.begin(), count, result.begin() + base + 32 + count * 16);
        first += count;
    }
    return result;
}

struct Row {
    std::uint64_t address;
    unsigned position, ordinal, bits, round, physical;
};

template<class Read>
std::vector<Row> read_metadata(Read read, std::uint64_t base, unsigned capacity, unsigned& bytes) {
    std::vector<Row> result;
    bytes = 0;
    for (unsigned round = 0; bytes < capacity; ++round) {
        const auto count = read(base + bytes, 1);
        if (!count) break;
        if (count > 48 || bytes + 32 > capacity) throw std::runtime_error("invalid actual metadata header");
        const unsigned span = read(base + bytes + 8, 2), ordinal = read(base + bytes + 6, 2);
        if (span != forward_postprocess_token_metadata_token_batch_bytes(count) || bytes + span > capacity ||
                read(base + bytes + 4, 2) != round || ordinal != result.size() ||
                read(base + bytes + 12, 2) != 32 + count * 16 ||
                read(base + bytes + 14, 2) != ((count + 15) / 16) * 16)
            throw std::runtime_error("actual metadata span/ordinal differs");
        std::array<bool, 48> ordinals{};
        for (unsigned physical = 0; physical < count; ++physical) {
            const auto address = base + bytes + 32 + physical * 16;
            const unsigned local = read(address + 8, 1), bits = read(address + 10, 1);
            if (local >= count || ordinals[local] || (bits != 4 && bits != 8) ||
                    read(base + bytes + 32 + count * 16 + local, 1) != physical)
                throw std::runtime_error("actual metadata inverse/precision invalid");
            ordinals[local] = true;
            result.push_back({address, unsigned(read(address + 4, 2)), ordinal + local, bits, round, physical});
        }
        bytes += span;
    }
    if (result.empty() || result.size() > 2048) throw std::runtime_error("actual metadata tokens outside supported range");
    return result;
}
}

// Host metadata packing runs only after DDR drains; it never reads expected data.
template<class ReadByte, class WriteByte>
nlohmann::json testcase_handoff_actions(const nlohmann::json& execution, ReadByte read_byte, WriteByte write_byte) {
    nlohmann::json records = nlohmann::json::array();
    if (!execution.contains("handoff_actions")) return records;
    const auto read = [&](std::uint64_t address, unsigned width) {
        std::uint64_t value = 0;
        for (unsigned b = 0; b < width; ++b) {
            const auto raw = read_byte(address + b);
            if (raw < 0) throw std::runtime_error("handoff actual DDR read failed");
            value |= std::uint64_t(raw) << (b * 8);
        }
        return value;
    };
    const auto write = [&](std::uint64_t address, unsigned width, std::uint64_t value) {
        for (unsigned b = 0; b < width; ++b)
            if (write_byte(address + b, std::uint8_t(value >> (b * 8))))
                throw std::runtime_error("handoff host write requires drained DDR");
    };
    const auto publish = [&](std::uint64_t address, const auto& data) {
        for (unsigned b = 0; b < data.size(); ++b) write(address + b, 1, data[b]);
    };
    for (const auto& action : execution.at("handoff_actions")) {
        const auto kind = action.at("kind").template get<std::string>();
        const auto config = action.at("config_address").template get<std::uint64_t>();
        auto base = action.at("metadata_address").template get<std::uint64_t>();
        auto capacity = action.at("metadata_capacity").template get<unsigned>();
        if (!capacity || capacity > 65536) throw std::runtime_error("invalid handoff metadata capacity");
        if (action.contains("history_address")) {
            const auto history = action.at("history_address").template get<std::uint64_t>();
            const auto table = action.at("token_table_address").template get<std::uint64_t>();
            const auto states = action.at("source_state_address").template get<std::uint64_t>();
            const auto event = action.at("source_event_address").template get<std::uint64_t>();
            const unsigned capture = action.at("capture"), sequence = action.at("sequence");
            const unsigned previous = action.at("source_current_begin"), count = action.at("source_state_count");
            if ((count != 32 && count != 64) || previous + count > sequence ||
                    read(history, 2) != previous || read(history + 2, 2) != sequence ||
                    read(history + 4, 4) + 1 != capture || read(event, 4) != capture)
                throw std::runtime_error("packed history block/sequence/capture differs from completed forward");
            unsigned bytes = 0;
            const auto completed = testcase_handoff::read_metadata(read,
                action.at("source_metadata_address").template get<std::uint64_t>(),
                action.at("source_metadata_capacity").template get<unsigned>(), bytes);
            for (const auto& row : completed) {
                if (row.position >= sequence) throw std::runtime_error("completed KV position exceeds history");
                const auto header = row.address - 32 - row.physical * 16;
                if (!(read(header + 24, 6) & (UINT64_C(1) << row.physical))) write(history + 16 + row.position, 1, 0);
            }
            for (unsigned row = 0; row < count; ++row) {
                const auto state = states + row * 32;
                const unsigned position = read(state, 2), token = read(state + 8, 4);
                if (position != previous + row || read(state + 20, 4) != capture || token >= 126464)
                    throw std::runtime_error("completed packed state position/capture/token differs");
                const auto address = table + position * 8;
                write(address, 8, (read(address, 8) & ((UINT64_C(1) << 35)-1)) | (std::uint64_t(token) << 35));
                write(history + 16 + position, 1, read(state + 19, 1));
            }
            const auto transitions = read(history + 8, 4) | read(event + 8, 4) |
                read(event + 12, 4) | read(event + 16, 4) | read(event + 48, 4);
            write(history + 8, 4, transitions);
            write(history + 4, 4, capture);
        }
        if (kind == "full_sequence_from_state") {
            const unsigned sequence = action.at("sequence"), capture = action.at("capture");
            const unsigned previous = action.at("source_current_begin"), current = action.at("current_begin");
            const auto states = action.at("source_state_address").template get<std::uint64_t>();
            const auto table = action.at("table_address").template get<std::uint64_t>();
            const bool reset = action.value("reset_current_state", false);
            if (!sequence || sequence > 2048 || previous + 32 > sequence || current + 32 > sequence ||
                    current != previous + (reset ? 32u : 0u) ||
                    read(config + EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET, 2) != sequence ||
                    read(config + EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET, 2) ||
                    read(config + EXECUTION_CONFIG_JOINT_CONFIGURATION_OFFSET_OFFSET, 2))
                throw std::runtime_error("baseline full-sequence configuration differs");
            std::vector<forward_postprocess_next_token_descriptor> tokens(sequence);
            for (unsigned position = 0; position < sequence; ++position) {
                auto& token = tokens[position];
                token.source_index = read(table + position * 4, 4);
                if (position >= previous && position < previous + 32) {
                    const auto state = states + (position - previous) * 32;
                    if (read(state, 2) != position || read(state + 20, 4) != capture ||
                            read(state + 7, 1) != 8 ||
                            (read(state + 5, 1) != FORWARD_POSTPROCESS_MASKED &&
                             read(state + 5, 1) != FORWARD_POSTPROCESS_LOCKED) ||
                            (reset && read(state + 5, 1) != FORWARD_POSTPROCESS_LOCKED))
                        throw std::runtime_error("baseline actual state position/capture/precision differs");
                    token.source_index = read(state + 8, 4);
                }
                if (token.source_index >= 126464) throw std::runtime_error("baseline token exceeds vocabulary");
                token.token_position = token.kv_index = position;
                token.source = FORWARD_POSTPROCESS_EMBEDDING_TOKEN; token.activation_bits = 8;
            }
            if (reset && action.contains("state_address")) {
                const auto next = action.at("state_address").template get<std::uint64_t>();
                if (action.at("state_count").template get<unsigned>() != 32)
                    throw std::runtime_error("baseline new block needs 32 reset state entries");
                for (unsigned row = 0; row < 32; ++row) {
                    const auto state = next + row * 32;
                    if (read(state, 2) != current + row || read(state + 5, 1) != FORWARD_POSTPROCESS_MASKED ||
                            read(state + 6, 1) || read(state + 12, 4) != UINT32_MAX || read(state + 16, 2) ||
                            read(state + 8, 4) != tokens[current + row].source_index)
                        throw std::runtime_error("baseline new block state is not the initial masked state");
                }
            }
            const auto packed = testcase_handoff::metadata(tokens, sequence, capture);
            if (packed.size() > capacity) throw std::runtime_error("baseline metadata exceeds capacity");
            for (unsigned position = 0; position < sequence; ++position)
                write(table + position * 4, 4, tokens[position].source_index);
            for (unsigned b = 0; b < capacity; ++b) write(base + b, 1, b < packed.size() ? packed[b] : 0);
            records.push_back({{"kind", kind}, {"tokens", sequence}, {"updated_block_begin", previous}});
        } else if (kind == "packed_boundary_from_state") {
            const unsigned sequence = action.at("sequence"), capture = action.at("capture");
            const unsigned current = action.at("current_begin"), previous = action.at("source_current_begin");
            const unsigned state_count = action.at("source_state_count"), maturity = action.at("maturity_age");
            const auto history = action.at("history_address").template get<std::uint64_t>();
            const auto table = action.at("token_table_address").template get<std::uint64_t>();
            const auto selector = action.at("table_address").template get<std::uint64_t>();
            const auto states = action.at("source_state_address").template get<std::uint64_t>();
            const auto policy = action.at("precision_policy").template get<std::string>();
            const bool all_a8 = action.at("block_initialization_all_a8");
            if (!sequence || sequence > 2048 || current != previous + 32 || current + 32 > sequence || !maturity ||
                    (policy != "original" && policy != "masked_only" && policy != "mature_only" && policy != "all_a8"))
                throw std::runtime_error("packed boundary sequence/block/precision configuration differs");
            for (unsigned row = 0; row < 32; ++row)
                if (read(states + row * 32 + 5, 1) != FORWARD_POSTPROCESS_LOCKED)
                    throw std::runtime_error("packed boundary predecessor is not completed");
            const auto transition_mask = read(history + 8, 4);
            std::vector<forward_postprocess_next_token_descriptor> tokens(sequence);
            for (unsigned position = 0; position < sequence; ++position) {
                auto& token = tokens[position];
                token.token_position = token.kv_index = position;
                token.source = FORWARD_POSTPROCESS_EMBEDDING_TOKEN;
                token.source_index = read(table + position * 8, 8) >> 35;
                if (token.source_index >= 126464) throw std::runtime_error("packed boundary token exceeds vocabulary");
                token.activation_bits = all_a8 ? 8u : action.at("context_bits").template get<unsigned>();
                if (position >= current && position < current + 32 && !all_a8) {
                    const auto state = states + (position - previous) * 32;
                    const unsigned value = state_count == 64 ? read(state + 5, 1) : FORWARD_POSTPROCESS_MASKED;
                    const auto age = state_count == 64 ? std::int16_t(read(state + 16, 2)) : -1;
                    const bool mature = value == FORWARD_POSTPROCESS_LOCKED && age >= int(maturity);
                    const bool masked = value == FORWARD_POSTPROCESS_MASKED;
                    const bool a4 = policy == "original" ? masked || mature :
                        policy == "masked_only" ? masked : policy == "mature_only" && mature;
                    token.activation_bits = a4 ? 4 : 8;
                }
                const auto address = selector + position * 8;
                auto record = read(address, 8);
                record &= ~((UINT64_C(1) << 10) | (UINT64_C(1) << 34) | 255u);
                record |= std::uint64_t(position < current || position >= current + 32 ? read(history + 16 + position, 1) != 0 : false) << 10;
                // History-based selection sets deep-row precision separately.
                if (action.at("historical_only").template get<bool>()) record |= read(address, 8) & (UINT64_C(1) << 34);
                else record |= std::uint64_t(token.activation_bits == 8) << 34;
                write(address, 8, record);
            }
            if (action.contains("next_state_address")) {
                const auto next = action.at("next_state_address").template get<std::uint64_t>();
                const unsigned count = action.at("next_state_count");
                if (count != 32 && count != 64) throw std::runtime_error("boundary next state count differs");
                for (unsigned row = 0; row < count; ++row) {
                    const auto destination = next + row * 32;
                    if (row < 32 && state_count == 64) {
                        for (unsigned b = 0; b < 32; ++b) write(destination + b, 1, read(states + (32 + row) * 32 + b, 1));
                        write(destination + 3, 1, 0); write(destination + 24, 8, 0);
                    } else {
                        for (unsigned b = 0; b < 32; ++b) write(destination + b, 1, 0);
                        write(destination, 2, current + row); write(destination + 2, 1, row % 32);
                        write(destination + 3, 1, row / 32);
                        write(destination + 4, 1, action.at("block_index").template get<unsigned>() + row / 32);
                        write(destination + 8, 4, tokens.at(current + row).source_index);
                        write(destination + 12, 4, UINT32_MAX); write(destination + 16, 2, 65535);
                        write(destination + 7, 1, 4);
                    }
                    write(destination + 20, 4, capture);
                }
            }
            const auto packed = testcase_handoff::metadata(tokens, sequence, capture);
            if (packed.size() > capacity) throw std::runtime_error("packed boundary metadata exceeds capacity");
            for (unsigned b = 0; b < capacity; ++b) write(base + b, 1, b < packed.size() ? packed[b] : 0);
            if (!action.at("historical_only").template get<bool>()) {
                // Probability columns are dense over enabled eight-key groups.
                std::vector<unsigned> groups;
                std::array<unsigned, 256> masks{};
                for (unsigned p = 0; p < sequence; ++p)
                    if (transition_mask ? (p >= previous && p < previous+32 && (transition_mask & (UINT64_C(1) << (p-previous)))) :
                            (p < current || p >= current+32)) masks[p/8] |= 1u << (p%8);
                for (unsigned g = 0; g < masks.size(); ++g) if (masks[g]) groups.push_back(g);
                const auto probability = action.at("probability_address").template get<std::uint64_t>();
                const auto limit = action.at("probability_limit").template get<std::uint64_t>();
                const auto jobs = action.at("jobs_address").template get<std::uint64_t>();
                const auto probability_config = action.at("probability_config_address").template get<std::uint64_t>();
                unsigned bytes = 0;
                const auto layout = testcase_handoff::read_metadata(read, action.at("layout_address").template get<std::uint64_t>(),
                    action.at("layout_capacity").template get<unsigned>(), bytes);
                const unsigned batch_stride = groups.size()*128, head_stride = batch_stride*6, round_stride = head_stride*32;
                if (probability + std::uint64_t(layout.back().round+1)*round_stride > limit)
                    throw std::runtime_error("live boundary probability columns exceed prepared capacity");
                std::array<bool, 256> initialized{};
                unsigned count = 0;
                for (const auto& row : layout) {
                    const bool inside = row.position >= current && row.position < current+32;
                    if (transition_mask ? inside : !inside) continue;
                    for (unsigned gi = 0; gi < groups.size(); ++gi) {
                        const unsigned group = groups[gi]; attention_dependency_job job{};
                        job.source_base = probability + row.round*round_stride + (row.physical/8)*batch_stride + gi*128 + (row.physical%8)*16;
                        job.source_limit = limit; job.head_stride = head_stride; job.lane_mask = masks[group];
                        job.output_base = selector + (transition_mask ? row.position*8 : group*64);
                        job.output_limit = selector + ((sequence+7)/8)*64;
                        job.operation = transition_mask ? (gi ? 3 : 7) : (initialized[group] ? 2 : 6);
                        initialized[group] = true;
                        if (count >= action.at("jobs_capacity").template get<unsigned>()) throw std::runtime_error("live boundary jobs exceed prepared capacity");
                        publish(jobs + count++*64, pack_attention_dependency_job(job));
                    }
                }
                write(probability_config + 16, 4, batch_stride); write(probability_config + 20, 4, head_stride);
                write(probability_config + 24, 4, round_stride);
                for (unsigned word = 0; word < 4; ++word) {
                    std::uint64_t enabled = 0;
                    for (unsigned lane = 0; lane < 64; ++lane) if (masks[word*64+lane]) enabled |= UINT64_C(1) << lane;
                    write(probability_config + 32 + word*8, 8, enabled);
                }
                write(action.at("refresh_address").template get<std::uint64_t>() + 164, 4, count);
            }
            write(history, 2, current); write(history + 8, 4, 0);
            records.push_back({{"kind", kind}, {"tokens", sequence}, {"transition_mask", transition_mask}});
        } else if (kind == "state_to_cross_block_inputs") {
            const auto states = action.at("state_address").template get<std::uint64_t>();
            const auto sequence = action.at("sequence").template get<unsigned>();
            const auto current = action.at("current_begin").template get<unsigned>();
            const auto capture = action.at("capture").template get<unsigned>();
            if (current < 32 || sequence != current + 32 || sequence > 2048)
                throw std::runtime_error("boundary requires predecessor/current 32-token blocks");
            std::map<unsigned, std::uint64_t> state_by_position;
            for (unsigned i = 0; i < 64; ++i) {
                const auto address = states + i * 32; const unsigned position = read(address, 2);
                if (position < current - 32 || position >= sequence || !state_by_position.emplace(position, address).second ||
                        read(address + 20, 4) != capture)
                    throw std::runtime_error("boundary actual state identity differs");
                if (position < current && read(address + 5, 1) != FORWARD_POSTPROCESS_LOCKED)
                    throw std::runtime_error("boundary predecessor block is not completed");
            }
            std::vector<forward_postprocess_next_token_descriptor> tokens(sequence);
            std::vector<unsigned> changed;
            const auto table = action.at("table_address").template get<std::uint64_t>();
            for (unsigned position = 0; position < sequence; ++position) {
                auto& token = tokens[position]; token.token_position = token.kv_index = position;
                token.source = FORWARD_POSTPROCESS_EMBEDDING_TOKEN; token.source_index = position % 64; token.activation_bits = 8;
                bool mandatory = false;
                if (position >= current - 32) {
                    const auto address = state_by_position.at(position);
                    token.source_index = read(address + 8, 4);
                    if (position >= current) token.activation_bits = read(address + 7, 1);
                    mandatory = position < current && read(address + 19, 1);
                    if (position < current && (read(address + 25, 1) & 1)) changed.push_back(position);
                }
                const auto record = (std::uint64_t(position >= current) << 8) | (std::uint64_t(position < current) << 9) |
                    (std::uint64_t(mandatory) << 10) | (std::uint64_t(position) << 12) |
                    (std::uint64_t(token.activation_bits == 8) << 34) | (std::uint64_t(position) << 35);
                write(table + position * 8, 8, record);
            }
            if (changed.empty()) throw std::runtime_error("boundary has no actual changed predecessor token");
            const auto packed = testcase_handoff::metadata(tokens, sequence, capture);
            if (packed.size() > capacity) throw std::runtime_error("boundary metadata exceeds capacity");
            for (unsigned b = 0; b < capacity; ++b) write(base + b, 1, b < packed.size() ? packed[b] : 0);
            unsigned bytes = 0;
            const auto actual = testcase_handoff::read_metadata(read, base, capacity, bytes);
            const auto probability = action.at("probability_address").template get<std::uint64_t>();
            const auto jobs = action.at("jobs_address").template get<std::uint64_t>();
            const unsigned groups = (sequence + 7) / 8, batch_stride = groups * 128;
            const unsigned head_stride = batch_stride * 6, round_stride = head_stride * 32;
            const auto probability_limit = probability + std::uint64_t(actual.back().round + 1) * round_stride;
            if (probability_limit > jobs) throw std::runtime_error("actual scout probability output overlaps relation jobs");
            unsigned count = 0;
            for (const auto& token : actual) if (token.position < current) {
                bool first = true;
                for (unsigned group = 0; group < groups; ++group) {
                    attention_dependency_job job{};
                    for (unsigned lane = 0; lane < 8; ++lane)
                        if (std::find(changed.begin(), changed.end(), group * 8 + lane) != changed.end()) job.lane_mask |= 1u << lane;
                    if (!job.lane_mask) continue;
                    job.source_base = probability + token.round * round_stride + (token.physical / 8) * batch_stride + group * 128 + (token.physical % 8) * 16;
                    job.source_limit = probability_limit; job.head_stride = head_stride;
                    job.output_base = table + token.position * 8; job.output_limit = table + ((sequence + 7) / 8) * 64;
                    job.operation = first ? 7 : 3; first = false;
                    if (count >= action.at("jobs_capacity").template get<unsigned>()) throw std::runtime_error("boundary relation jobs exceed capacity");
                    publish(jobs + count++ * 64, pack_attention_dependency_job(job));
                }
            }
            const auto refresh = action.at("refresh_address").template get<std::uint64_t>();
            write(refresh + 164, 4, count);
            write(refresh + 176 + 8, 8, probability_limit);
            write(config + EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET, 8, base + bytes);
            records.push_back({{"kind", kind}, {"tokens", sequence}, {"changed", changed.size()}, {"relation_jobs", count}});
        } else if (kind != "execution_from_metadata") {
            throw std::runtime_error("unsupported handoff action: " + kind);
        }
        if (action.contains("layout_address")) {
            // The prepared layout fixes DDR row placement, not selection. Require
            // the actual selection to fit it and copy all token sources from RTL.
            // A different selection fails here, before any consumer is launched.
            unsigned actual_bytes = 0, layout_bytes = 0;
            const auto actual = testcase_handoff::read_metadata(read, base, capacity, actual_bytes);
            const auto layout = action.at("layout_address").template get<std::uint64_t>();
            const auto layout_capacity = action.at("layout_capacity").template get<unsigned>();
            const auto placed = testcase_handoff::read_metadata(read, layout, layout_capacity, layout_bytes);
            std::map<unsigned, testcase_handoff::Row> selected;
            std::map<unsigned, bool> writes;
            for (const auto& row : actual) {
                if (!selected.emplace(row.position, row).second)
                    throw std::runtime_error("actual selection contains duplicate positions");
                const auto header = row.address - 32 - row.physical * 16;
                writes[row.position] = !(read(header + 24, 6) & (UINT64_C(1) << row.physical));
            }
            if (placed.size() != actual.size()) throw std::runtime_error("actual selection count differs from prepared layout");
            const unsigned current = action.at("current_begin"), end = action.at("current_end");
            const unsigned future_mask = action.contains("future_result_address") ?
                unsigned(read(action.at("future_result_address").template get<std::uint64_t>(), 4)) : 0;
            std::map<unsigned, testcase_handoff::Row> positions;
            const bool separate_l31 = read(config + EXECUTION_CONFIG_FLAGS_OFFSET, 4) & (1u << 13);
            std::vector<std::vector<testcase_handoff::Row>> layouts{placed};
            if (separate_l31) {
                const auto directory = read(config + EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET, 8);
                const auto l31_base = read(directory + 16, 8), l31_limit = read(directory + 24, 8);
                unsigned l31_bytes = 0;
                layouts.push_back(testcase_handoff::read_metadata(read, l31_base, l31_limit - l31_base, l31_bytes));
                if (layouts.back().size() != placed.size()) throw std::runtime_error("L31 selection count differs from input layout");
            }
            std::map<unsigned, unsigned> input_ordinals;
            for (const auto& row : placed) input_ordinals.emplace(row.position, row.ordinal);
            for (unsigned layout_index = 0; layout_index < layouts.size(); ++layout_index) {
              positions.clear();
              for (const auto& row : layouts[layout_index]) {
                const auto found = selected.find(row.position);
                if (found == selected.end() || found->second.bits != row.bits || !positions.emplace(row.position, row).second)
                    throw std::runtime_error("actual selection position/precision differs from prepared layout at position " + std::to_string(row.position));
                const auto& source = found->second;
                const auto header = row.address - 32 - row.physical * 16;
                if (read(header + 3, 1) & 2) {
                    const bool output = row.physical < read(header + 30, 1);
                    const bool required = (current <= row.position && row.position < end) ||
                        (end <= row.position && row.position < end + 32 && (future_mask & (1u << (row.position - end))));
                    if (output != required) throw std::runtime_error("L31 output layout differs from actual consumer set");
                }
                write(row.address, 4, layout_index ? input_ordinals.at(row.position) : read(source.address, 4));
                write(row.address + 6, 2, read(source.address + 6, 2));
                write(row.address + 9, 1, layout_index ? FORWARD_POSTPROCESS_RESIDENT_HIDDEN : read(source.address + 9, 1));
                write(header + 3, 1, (read(header + 3, 1) & ~1u) | (layout_index ? 0u : unsigned(read(source.address + 9, 1) == FORWARD_POSTPROCESS_EMBEDDING_TOKEN)));
                auto suppressed = read(header + 24, 6);
                const auto mask = UINT64_C(1) << row.physical;
                suppressed = writes.at(row.position) ? suppressed & ~mask : suppressed | mask;
                write(header + 24, 6, suppressed);
              }
            }
            if (action.contains("state_address")) {
                const auto states = action.at("state_address").template get<std::uint64_t>();
                const unsigned state_count = action.at("state_count");
                if (state_count != 32 && state_count != 64) throw std::runtime_error("regular state range requires one or two blocks");
                std::map<unsigned, unsigned> prediction_states;
                for (unsigned i = 0; i < state_count; ++i) {
                    const auto address = states + i * 32;
                    const unsigned position = read(address, 2);
                    if (position != current + i) throw std::runtime_error("carried state position differs from current/future blocks");
                    const auto selected_row = positions.find(position);
                    if (selected_row != positions.end()) {
                        write(address + 7, 1, selected_row->second.bits);
                        if (writes.at(position)) {
                            write(address + 18, 1, 1); write(address + 19, 1, 0);
                        }
                    }
                    // Current proposal history is local to this forward. Future
                    // history persists for retry/admission across forwards.
                    if (i < 32) write(address + 28, 3, 0);
                    const auto current_state = read(address + 5, 1);
                    const bool current_prediction = action.value("closeout_kind", 0u) == 1u ?
                        current_state == FORWARD_POSTPROCESS_TENTATIVE : current_state != FORWARD_POSTPROCESS_LOCKED;
                    if ((i < 32 && current_prediction) ||
                            (i >= 32 && (future_mask & (1u << (i - 32)))))
                        prediction_states.emplace(position, i);
                }
                const unsigned prediction_count = read(config + EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET, 2);
                const auto predictions = read(config + EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET, 8);
                if (prediction_count != prediction_states.size()) throw std::runtime_error("actual prediction count differs from prepared head");
                for (unsigned i = 0; i < prediction_count; ++i) {
                    const auto descriptor = predictions + i * 32;
                    const unsigned position = read(descriptor, 2);
                    const auto selected_state = prediction_states.find(position);
                    if (selected_state == prediction_states.end() || !positions.count(position))
                        throw std::runtime_error("prepared head consumer absent from actual prediction set");
                    const auto state = states + selected_state->second * 32;
                    const auto& row = positions.at(position);
                    write(descriptor + 6, 1, row.bits);
                    write(descriptor + 7, 1, read(state + 5, 1));
                    write(descriptor + 8, 4, read(state + 8, 4));
                    write(descriptor + 12, 4, read(state + 8, 4));
                    write(descriptor + 20, 2, row.ordinal);
                    write(descriptor + 24, 4, selected_state->second);
                    write(descriptor + 28, 4, selected_state->second);
                    prediction_states.erase(selected_state);
                }
            }
            base = layout;
            capacity = layout_capacity;
            records.push_back({{"kind", "actual_metadata_to_layout"}, {"tokens", actual.size()}, {"bytes", layout_bytes}});
        }
        unsigned bytes = 0;
        const auto tokens = testcase_handoff::read_metadata(read, base, capacity, bytes);
        write(config + EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET, 2, tokens.size());
        if (!(read(config + EXECUTION_CONFIG_FLAGS_OFFSET, 4) & (1u << 13))) {
            write(config + EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET, 8, base);
            write(config + EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET, 8, base + bytes);
        }
        if (action.contains("source_index_offset")) {
            const auto offset = action.at("source_index_offset").template get<unsigned>();
            for (const auto& token : tokens) {
                if (read(token.address + 9, 1) != FORWARD_POSTPROCESS_RESIDENT_HIDDEN)
                    throw std::runtime_error("source-index relocation requires resident hidden tokens");
                const auto source = read(token.address, 4);
                if (source + offset > 2047) throw std::runtime_error("relocated hidden source token exceeds capacity");
                write(token.address, 4, source + offset);
            }
        }
        if (action.contains("output_address")) {
            const auto output = action.at("output_address").template get<std::uint64_t>();
            const auto output_tokens = action.value("output_capacity_tokens", unsigned(tokens.size()));
            if (output_tokens < tokens.size() || output_tokens > 2048) throw std::runtime_error("invalid hidden arena token capacity");
            write(config + EXECUTION_CONFIG_OUTPUT_HIDDEN_BASE_OFFSET, 8, output);
            write(config + EXECUTION_CONFIG_OUTPUT_HIDDEN_LIMIT_OFFSET, 8, output + std::uint64_t(output_tokens) * 8192);
        }
        if (action.contains("commit_table_address")) {
            const auto table = action.at("commit_table_address").template get<std::uint64_t>();
            const auto sequence = action.at("sequence").template get<unsigned>();
            for (unsigned p = 0; p < sequence; ++p) write(table + p * 8, 8, 0);
            for (const auto& token : tokens) {
                if (token.position >= sequence) throw std::runtime_error("deep position outside sequence");
                write(table + token.position * 8, 8, 256);
            }
        }
        records.push_back({{"kind", "execution_from_metadata"}, {"tokens", tokens.size()}, {"bytes", bytes}, {"rounds", tokens.back().round + 1}});
    }
    return records;
}

#endif
