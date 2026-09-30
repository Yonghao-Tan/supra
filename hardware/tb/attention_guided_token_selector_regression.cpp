#include "Vattention_guided_token_selector_tb.h"
#include "verilated.h"
#include "json.hpp"
#include "generated/token_refresh_config_packer.hpp"
#include "generated/attention_dependency_job_packer.hpp"
#include "generated/attention_probability_config_packer.hpp"
#include "generated/uaps_config_packer.hpp"
#include "generated/uaps_attempt_config_packer.hpp"
#include "generated/uaps_result_packer.hpp"
extern "C" {
#include "../cmodel/token_refresh_model.h"
}
#include "generated/refresh_score_config_packer.hpp"
#include "generated/in_block_refresh_budget_packer.hpp"

#include <cstdint>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include "generated/cross_block_shortlist_config_packer.hpp"
#include "generated/token_state_entry_packer.hpp"

namespace {
using nlohmann::json;
constexpr std::uint64_t kCacheBases[] = {0x1000000, 0x2000000, 0x3000000, 0x4000000, 0x5000000, 0x6000000};
constexpr unsigned kCacheSizes[] = {8388608, 8388608, 131072};
std::uint8_t cache_pattern(std::uint64_t address) {
    return std::uint8_t((address * 0x9e3779b9ull) ^ (address >> 7) ^ (address >> 17));
}
struct Row {
    unsigned score = 0, eligible_order = 0, required_order = 0;
    unsigned source_index = 0;
    bool a8 = false;
    bool forecast_a8 = false;
    bool prediction_token_mask = false;
    bool kv_write_disable = false;
    bool current = false, eligible = false, mandatory = false, required = false;
};
struct TestCase {
    std::string name;
    std::vector<Row> rows;
    unsigned target = 0, quota = 0;
    std::vector<unsigned> expected;
    std::vector<std::uint8_t> jobs, probabilities;
    std::vector<unsigned> expected_scores;
    std::vector<std::uint8_t> joint_config, joint_base, joint_future, joint_result;
    std::map<std::uint64_t, std::vector<std::uint8_t>> extra_memory;
    std::vector<std::uint8_t> pending_expected;
    std::vector<std::uint8_t> budget_expected;
    std::vector<unsigned> shortlist_eligible;
    std::vector<unsigned> shortlist_mandatory;
    std::vector<unsigned> shortlist_orders;
    unsigned dependency_layers = 0;
    bool execution_extension = false;
    bool new_pending_sequence = false;
    bool advance_dependency_block = false;
    bool capture_p8_only = false;
    bool dependency_p8 = true;
    std::uint64_t relation_job_address = 0x30000;
    std::vector<std::uint8_t> dependency_expected;
    bool allocate_precision = false, all_a8 = false;
    unsigned context_a8_token_count = 0;
    std::map<unsigned, unsigned> expected_bits;
    std::map<unsigned, unsigned> expected_sources;
    std::vector<unsigned> expected_kv_disabled;
    std::vector<std::uint8_t> expected_table;
    bool state_table_update = false;
    std::vector<unsigned> loaded_kv_positions;
    bool live_consumed = false;
    bool paired_pending = false;
    bool closeout = false;
    std::vector<std::uint8_t> cross_pending_expected;
    bool initial_selected_consumed = false;
    unsigned consume_current_begin = 32;
    bool publish_result = false;
    bool qkvo_group = false;
    unsigned metadata_capacity = 0x4000;
    bool deep_precision = false;
    unsigned deep_a8_limit = 0;
    std::vector<std::uint8_t> attempts_expected;
};
std::uint64_t encode(const Row& row) {
    return row.score | std::uint64_t(row.current) << 8 | std::uint64_t(row.eligible) << 9 |
        std::uint64_t(row.mandatory) << 10 | std::uint64_t(row.required) << 11 |
        std::uint64_t(row.eligible_order) << 12 | std::uint64_t(row.required_order) << 23 |
        std::uint64_t(row.a8) << 34 | std::uint64_t(row.source_index) << 35 | std::uint64_t(row.forecast_a8) << 52 |
        std::uint64_t(row.prediction_token_mask) << 53 | std::uint64_t(row.kv_write_disable) << 54;
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
std::vector<TestCase> load(const char* path) {
    std::ifstream input(path);
    require(input.good(), "missing boundary test_case");
    json document;
    input >> document;
    require(document.at("schema") == "supra-atse-cross-block-token-selection/v1", "boundary schema mismatch");
    std::vector<TestCase> result;
    for (const auto& record : document.at("records")) {
        TestCase test_case;
        test_case.name = record.at("name");
        const auto& source = record.at("inputs");
        const auto scores = source.at("score_q8").at("raw").get<std::vector<unsigned>>();
        test_case.rows.resize(scores.size());
        for (unsigned position = 0; position < scores.size(); ++position) test_case.rows[position].score = scores[position];
        for (const auto name : {"current_positions", "eligible_positions", "mandatory_candidate_positions", "required_candidate_positions"}) {
            unsigned order = 0;
            for (auto position : source.at(name).at("raw").get<std::vector<unsigned>>()) {
                auto& row = test_case.rows.at(position);
                const std::string field(name);
                if (field == "current_positions") row.current = true;
                if (field == "eligible_positions") { row.eligible = true; row.eligible_order = order; }
                if (field == "mandatory_candidate_positions") row.mandatory = true;
                if (field == "required_candidate_positions") { row.required = true; row.required_order = order; }
                ++order;
            }
        }
        test_case.target = source.at("target_token_count");
        test_case.quota = source.at("required_candidate_count");
        test_case.expected = record.at("expected").at("deep_positions").at("raw").get<std::vector<unsigned>>();
        for (unsigned position = 0; position < scores.size(); ++position) {
            test_case.rows[position].a8 = source.contains("deep_activation_bits") ?
                source.at("deep_activation_bits") == 8 : !test_case.rows[position].current;
            test_case.rows[position].source_index = source.contains("deep_activation_bits") ? position : (position*17)%scores.size();
        }
        if (source.contains("initial_activation_bits")) {
            const auto& bits = source.at("initial_activation_bits").at("raw");
            require(bits.size() == scores.size(), "boundary initial precision length differs");
            for (unsigned p = 0; p < scores.size(); ++p) {
                const unsigned value = bits.at(p);
                require(value == 4 || value == 8, "boundary initial precision is not A4/A8");
                test_case.rows[p].a8 = value == 8;
            }
            const int limit = source.at("deep_a8_limit");
            test_case.deep_precision = limit >= 0;
            test_case.deep_a8_limit = limit >= 0 ? limit : 0;
            const auto& expected = record.at("expected").at("deep_bits").at("raw");
            require(expected.size() == test_case.expected.size(), "boundary expected precision length differs");
            for (unsigned i = 0; i < test_case.expected.size(); ++i)
                test_case.expected_bits[test_case.expected[i]] = expected.at(i);
        }
        if (source.contains("previous_pending")) {
            test_case.shortlist_eligible = source.at("eligible_positions").at("raw").get<std::vector<unsigned>>();
            for (auto& row : test_case.rows) row.eligible = false;
            cross_block_shortlist_config config{};
            config.pending_base = 0x70000; config.pending_limit = config.pending_base+scores.size()*16;
            config.shortlist_token_count = source.at("shortlist_token_count");
            config.flags = source.value("shortlist_flags", 0);
            config.relative_score_floor_bf16 = source.value("relative_score_floor_bf16", 0);
            config.protected_begin = source.value("protected_begin", 0);
            config.protected_end = source.value("protected_end", 0);
            if (record.at("expected").contains("shortlist_mandatory_positions"))
                test_case.shortlist_mandatory = record.at("expected").at("shortlist_mandatory_positions").at("raw").get<std::vector<unsigned>>();
            if (record.at("expected").contains("shortlist_orders"))
                test_case.shortlist_orders = record.at("expected").at("shortlist_orders").at("raw").get<std::vector<unsigned>>();
            const auto descriptor = pack_cross_block_shortlist_config(config);
            test_case.extra_memory[0x8600] = std::vector<std::uint8_t>(descriptor.begin(), descriptor.end());
            auto& pending = test_case.extra_memory[0x70000]; pending.resize(scores.size()*16, 0x5a);
            for (unsigned position = 0; position < scores.size(); ++position) {
                const unsigned raw = source.at("previous_pending").at("raw").at(position);
                pending[position*16] = raw; pending[position*16+1] = raw >> 8;
                if (source.contains("previous_future_pending")) {
                    const unsigned future = source.at("previous_future_pending").at("raw").at(position);
                    pending[position*16+6] = future; pending[position*16+7] = future >> 8;
                }
            }
        }
        result.push_back(test_case);
    }
    return result;
}

class Regression {
  public:
    void run_observed_boundary(const char* path) {
        for (const auto& test_case : load(path)) execute(test_case, 0, true, true);
        std::cout << "PASS observed_boundary reference=" << path << " selection/precision/metadata/cache_commit\n";
    }
    void run_precision(const char* path) {
        const auto cases = precision_test_cases(path);
        require(!cases.empty(), "empty context precision reference");
        for (const auto& test_case : cases) execute(test_case, 0, false, true);
        std::cout << "PASS context_precision reference=" << path << " records=" << cases.size() << '\n';
    }
    void run_observed_regular(const char* path) {
        for (const auto& test_case : regular_test_cases(path, true))
            execute(test_case, 0, false, true);
        std::cout << "PASS observed_feature1 reference=" << path
                  << " dependency/pending/budget/selection/precision/metadata threads=" << dut_.threads() << '\n';
    }
    void run(const std::vector<TestCase>& test_cases) {
        if (const auto* group=std::getenv("SUPRA_ATSE_CASE_GROUP"); group && std::string(group)=="grouped-metadata") {
            for (const unsigned count : {97u,289u,432u}) {
                for (const unsigned a8 : {0u,40u,count}) {
                    TestCase item;
                    item.name="grouped_metadata_"+std::to_string(count)+"_a8_"+std::to_string(a8);
                    item.rows.resize(count); item.target=count; item.publish_result=true; item.qkvo_group=true;
                    for (unsigned p=0;p<count;++p) {
                        item.rows[p].current=true; item.rows[p].a8=p<a8;
                        item.rows[p].source_index=p; item.expected.push_back(p);
                    }
                    execute(item,0,false,true);
                    item.qkvo_group=false;
                    execute(item,0,false,true);
                }
            }
            std::cout << "PASS grouped metadata capacity, tail, configuration restore\n";
            return;
        }
        if (const auto* group=std::getenv("SUPRA_ATSE_CASE_GROUP"); group && std::string(group)=="metadata-result") {
            TestCase result{"published_selection_result",std::vector<Row>(16),3,0,{0,1,2}};
            result.rows[0].current=true;
            for (unsigned p=1;p<16;++p) { result.rows[p].eligible=true; result.rows[p].eligible_order=p; }
            result.publish_result=true;
            execute(result,0,true);
            execute(result,25,true);
            execute(result,0,true);
            result.metadata_capacity=112;
            execute(result,0,true);
            result.metadata_capacity=96;
            execute(result,27,true);
            result.metadata_capacity=0x4000;
            result.publish_result=false;
            execute(result,0,true);
            std::cout << "PASS selection with/without result publication, write-error/drain and recovery\n";
            return;
        }
        if (const auto* group=std::getenv("SUPRA_ATSE_CASE_GROUP")) {
            require(std::string(group)=="live-state" || std::string(group)=="pending" || std::string(group)=="initial-pending" || std::string(group)=="paired-pending", "unknown ATSE case group");
            if (std::string(group)=="live-state") run_joint_state_cases();
            else if (std::string(group)=="initial-pending") run_initial_pending_cases();
            else if (std::string(group)=="paired-pending") run_paired_pending_cases();
            else run_live_pending_cases();
            std::cout << "PASS attention_refresh group=" << group << " runtime_threads=" << dut_.threads() << "\n";
            return;
        }
        dut_.rst = 1;
        edge();
        dut_.rst = 0;
        for (unsigned index = 0; index < test_cases.size(); ++index)
            execute(test_cases[index], 0, true, index != 0);
        TestCase block_initialization{"block_initialization352_a4_metadata_commit", std::vector<Row>(512), 352, 0, {}};
        for (unsigned position = 0; position < block_initialization.rows.size(); ++position) {
            auto& row = block_initialization.rows[position];
            row.current = position < 32;
            row.eligible = position >= 32;
            row.eligible_order = position;
            row.source_index = (position*17)%512;
            if (position < block_initialization.target) block_initialization.expected.push_back(position);
        }
        // Equal scores retain eligibility order after the mandatory current block.
        execute(block_initialization, 0, true);
        block_initialization.name = "block_initialization352_cache_commit_only";
        execute(block_initialization, 0, true, false, nullptr, false, false);
        block_initialization.name = "block_initialization432_a4_metadata_commit";
        block_initialization.target = 432; block_initialization.expected.clear();
        for (unsigned i = 0; i < 432; ++i) block_initialization.expected.push_back(i);
        execute(block_initialization, 0, true);
        block_initialization.name = "block_initialization431_a8_metadata_commit";
        block_initialization.target = 431; block_initialization.expected.pop_back();
        for (auto& row : block_initialization.rows) row.a8 = true;
        execute(block_initialization, 0, true);
        TestCase maximum{"position2047_and_required_overlap", std::vector<Row>(2048), 5, 2, {2, 8, 12, 2046, 2047}};
        maximum.rows[2047].current = true;
        for (unsigned position = 0; position < 2047; ++position) {
            auto& row = maximum.rows[position];
            row.eligible = true;
            row.eligible_order = 2046-position;
            row.required_order = position;
            row.required = position == 2 || position == 8 || position == 12;
            row.mandatory = position == 2;
            row.score = position == 12 ? 254 : 0;
        }
        // Required quota includes mandatory 2; remaining winner is 12.
        // Equal-score optional candidates use descending original list order.
        maximum.expected = {2, 12, 2045, 2046, 2047};
        execute(maximum, 0);
        TestCase all_current{"no_optional", std::vector<Row>(1), 1, 0, {0}};
        all_current.rows[0].current = true;
        execute(all_current, 0);
        auto capture_only = all_current;
        capture_only.name = "p8_capture_without_relation_jobs";
        capture_only.capture_p8_only = true;
        execute(capture_only, 0, false, true, nullptr, true);
        TestCase layered{"p8_layers_replace_max_retain", std::vector<Row>(17), 1, 0, {4}};
        layered.rows[4].current = true;
        layered.dependency_layers = 2;
        layered.probabilities.resize(512);
        const std::array<unsigned, 8> codes{0, 32, 64, 96, 1, 2, 3, 127};
        for (unsigned head = 0; head < 32; ++head) {
            std::copy(codes.begin(), codes.end(), layered.probabilities.begin()+head*16);
            layered.probabilities[head*16+8] = 0x80;
            layered.probabilities[head*16+9] = 0x3b;
        }
        layered.extra_memory[0x70000].resize(48);
        for (unsigned byte = 1; byte < 48; byte += 2) layered.extra_memory[0x70000][byte] = 0x3f;
        attention_dependency_job layer_job{};
        layer_job.source_base = 0x40000; layer_job.source_limit = 0x40200;
        layer_job.head_stride = 16; layer_job.lane_mask = 0x55; layer_job.operation = 9;
        layer_job.output_base = 0x70010; layer_job.output_limit = 0x70030;
        const auto encoded_layer_job = pack_attention_dependency_job(layer_job);
        layered.jobs.assign(encoded_layer_job.begin(), encoded_layer_job.end());
        execute(layered, 0, false, true);
        layered.name = "bf16_layers_replace_max_retain";
        layered.dependency_p8 = false;
        layer_job.operation = 1;
        const auto bf16_layer_job = pack_attention_dependency_job(layer_job);
        layered.jobs.assign(bf16_layer_job.begin(), bf16_layer_job.end());
        const std::array<unsigned, 8> bf16_values{0, 0x3e00, 0x3e80, 0x3ec0, 0x3b80, 0x3c00, 0x3c40, 0x3efe};
        for (unsigned head = 0; head < 32; ++head)
            for (unsigned lane = 0; lane < 8; ++lane) {
                layered.probabilities[head*16+lane*2] = bf16_values[lane];
                layered.probabilities[head*16+lane*2+1] = bf16_values[lane] >> 8;
            }
        execute(layered, 0, false, true);
        layered.name = "bf16_l31_job_layout_and_entry_state";
        layered.execution_extension = true;
        execute(layered, 0, false, true);
        execute(scout_test_case(), 0, false, true);
        const auto shortlist_cases = load("cases/control/atse_previous_shortlist.json");
        for (const auto& test_case : shortlist_cases)
            execute(test_case, 0, false, true);
        const auto floor_case = std::find_if(shortlist_cases.begin(), shortlist_cases.end(), [](const TestCase& item) {
            return item.name == "block_initialization_future_floor";
        });
        require(floor_case != shortlist_cases.end(), "missing generator block initialization floor input");
        execute(*floor_case, 13, false, true); // Accepted multiply must drain before abort ack.
        auto excessive_protection = *floor_case;
        excessive_protection.name = "block_initialization_nonzero_floor_mandatory_exceeds_cap";
        excessive_protection.extra_memory.at(0x8600)[24] = 310 & 255;
        excessive_protection.extra_memory.at(0x8600)[25] = 310 >> 8;
        execute(excessive_protection, 23, false, true);
        auto invalid_future = *floor_case;
        invalid_future.name = "block_initialization_future_requires_dependency_only";
        invalid_future.extra_memory.at(0x8600)[18] = 5;
        invalid_future.extra_memory.at(0x8600)[20] = 0;
        invalid_future.extra_memory.at(0x8600)[21] = 0;
        execute(invalid_future, 24, false, true);
        // Same DUT, no reset: ordinary required-order semantics must survive
        // the preceding DDR block initialization configuration changes.
        execute(test_cases.front(), 0, false, true);
        auto shortlist_pending = shortlist_cases.front();
        shortlist_pending.name = "shortlist_preserves_source_a_kv_disable";
        for (unsigned position : shortlist_pending.expected) {
            if (shortlist_pending.rows[position].current) continue;
            shortlist_pending.rows[position].kv_write_disable = true;
            shortlist_pending.expected_kv_disabled.push_back(position);
        }
        execute(shortlist_pending, 0, false, true);
        execute(shortlist_cases.front(), 14, false, true);
        execute(shortlist_cases.front(), 15, false, true);
        execute(shortlist_cases.front(), 16, false, true);
        const auto joint_cases = joint_test_cases();
        for (const auto& test_case : joint_cases) {
            execute(test_case, 0, false, true);
            if (!test_case.attempts_expected.empty()) {
                auto replay = test_case;
                replay.name += "_same_capture_retry";
                replay.extra_memory[0xc000] = last_attempt_state_;
                execute(replay, 0, false, true);
            }
        }
        const auto unlimited = std::find_if(joint_cases.begin(), joint_cases.end(), [](const TestCase& item) {
            return item.name == "joint_future_admission_limit_-1";
        });
        const auto limited = std::find_if(joint_cases.begin(), joint_cases.end(), [](const TestCase& item) {
            return item.name == "joint_future_admission_limit_2";
        });
        require(unlimited != joint_cases.end() && limited != joint_cases.end(), "missing generator attempt test cases");
        auto next_block = *unlimited;
        next_block.name = "future_admission_new_block_actual_previous_counts";
        next_block.joint_config[6] = 96; next_block.joint_config[12] = 64;
        next_block.extra_memory[0x9040] = limited->extra_memory.at(0x9040);
        next_block.extra_memory[0x9040][20] = 0x80;
        next_block.extra_memory[0x9040][21] = 0x3e;
        next_block.extra_memory[0xc000] = last_attempt_state_;
        for (unsigned row = 32; row < 38; ++row) { next_block.rows[row+32] = next_block.rows[row]; next_block.rows[row] = Row{}; }
        for (auto& position : next_block.expected) if (position >= 32) position += 32;
        for (unsigned entry = 0; entry < next_block.joint_base.size(); entry += 16)
            if (next_block.joint_base[entry] >= 32) next_block.joint_base[entry] += 32;
        next_block.attempts_expected.assign(144, 0);
        auto put32 = [](auto& bytes, unsigned offset, std::uint32_t value) {
            for (unsigned byte = 0; byte < 4; ++byte) bytes.at(offset+byte) = value>>(byte*8);
        };
        put32(next_block.attempts_expected, 0, 0x31425441);
        put32(next_block.attempts_expected, 4, 64); put32(next_block.attempts_expected, 8, 99);
        const unsigned added = unsigned(next_block.joint_result[4]) | unsigned(next_block.joint_result[5])<<8 |
            unsigned(next_block.joint_result[6])<<16 | unsigned(next_block.joint_result[7])<<24;
        put32(next_block.attempts_expected, 12, added);
        for (unsigned row = 0; row < 32; ++row) put32(next_block.attempts_expected, 16+4*row, (added>>row)&1);
        execute(next_block, 0, false, true);
        auto alias = *limited;
        alias.name = "attempt_state_alias_descriptor";
        put32(alias.extra_memory[0x9040], 0, 0x8000);
        put32(alias.extra_memory[0x9040], 8, 0x8090);
        execute(alias, 20, false, true);
        for (unsigned malformed = 0; malformed < 3; ++malformed) {
            auto invalid = *limited;
            invalid.name = "joint_invalid_retry_target_"+std::to_string(malformed);
            auto& extension = invalid.extra_memory[0x9040];
            if (malformed == 0) { extension[20] = 0x81; extension[21] = 0x3f; }
            if (malformed == 1) extension[22] = 65;
            if (malformed == 2) { put32(extension, 0, 0); extension[20] = 0x80; extension[21] = 0x3e; }
            execute(invalid, 20, false, true);
        }
        execute(*limited, 22, false, true);
        execute(*limited, 0, false, true);
        for (const auto& test_case : deep_test_cases()) execute(test_case, 0, true);
        execute(joint_cases.front(), 11, false, true);
        execute(joint_cases.front(), 12, false, true);
        execute(joint_cases.front(), 13, false, true);
        run_live_pending_cases();
        const auto pending_cases = pending_test_cases();
        std::vector<std::uint8_t> persisted_pending;
        for (auto pending : pending_cases) {
            if (pending.new_pending_sequence) persisted_pending.clear();
            auto& input = pending.extra_memory.at(0x50000);
            if (!persisted_pending.empty()) {
                require(input.size() == persisted_pending.size(), "pending chain changed table size");
                for (unsigned row = 0; row < input.size()/16; ++row) {
                    for (unsigned byte = 0; byte < 10; ++byte)
                        require(input[row*16+byte] == persisted_pending[row*16+byte],
                            "persisted RTL pending differs from next generator input");
                    // The next forward supplies consumed/predicted flags; all
                    // stored numeric fields come from the previous RTL write.
                    persisted_pending[row*16+10] = input[row*16+10];
                }
                input = persisted_pending;
            }
            execute(pending, 0, false, true, &persisted_pending);
        }
        std::cout << "PASS pending_state_chain forwards=" << pending_cases.size()
                  << " persisted_vectors=5 new_attention_and_changes=test_case\n";
        execute(pending_cases.front(), 13, false, true);
        for (const auto* name : {"atse_attention_dependencies.json", "atse_in_block_boundaries.json"})
            for (const auto& regular : regular_test_cases(name)) execute(regular, 0, false, true);
        const auto state_changes = published_state_changes(regular_test_cases("atse_attention_dependencies.json").front());
        execute(state_changes, 0, false, true);
        execute(state_changes, 17, false, true);
        execute(state_changes, 18, false, true);
        const auto state_table = state_table_test_case();
        execute(state_table, 17, false, true);
        execute(state_table, 19, false, true);
        execute(state_table, 21, false, true);
        std::vector<std::uint8_t> selected_pending;
        execute(state_table, 0, false, true, &selected_pending);
        auto consumed = state_table;
        consumed.name = "state_table_consumes_previous_base";
        consumed.extra_memory.at(0x50000) = selected_pending;
        consumed.pending_expected[1] = 0;
        execute(consumed, 0, false, true);
        auto state_joint = state_table;
        state_joint.name = "state_pending_to_joint_no_future_table";
        state_joint.target = 4; state_joint.expected.push_back(33);
        state_joint.expected_bits[33] = 8; state_joint.expected_sources[33] = 11;
        uaps_config state_joint_config{};
        state_joint_config.magic = 0x314a4449; state_joint_config.version = 1; state_joint_config.bytes = 64;
        state_joint_config.future_token_count = 1; state_joint_config.next_block_start = 33;
        state_joint_config.max_next_tokens = 1; state_joint_config.priority_control = 0xc1;
        state_joint_config.result_base = 0x28000; state_joint_config.result_limit = 0x28010;
        const auto state_joint_bytes = pack_uaps_config(state_joint_config);
        state_joint.joint_config = {state_joint_bytes.begin(), state_joint_bytes.end()};
        uaps_result state_joint_result{};
        state_joint_result.future_prediction_mask = 1; state_joint_result.added_future_token_mask = 1;
        state_joint_result.future_prediction_count = 1; state_joint_result.added_future_token_count = 1;
        state_joint_result.base_activation_slots = 5; state_joint_result.joint_activation_slots = 7;
        state_joint_result.next_pass_activation_slots = 8;
        const auto state_joint_expected = pack_uaps_result(state_joint_result);
        state_joint.joint_result = {state_joint_expected.begin(), state_joint_expected.end()};
        auto missing_state = state_joint;
        missing_state.name = "joint_future_missing_published_state";
        state_joint_config.next_block_start = 3;
        const auto missing_config = pack_uaps_config(state_joint_config);
        missing_state.joint_config = {missing_config.begin(), missing_config.end()};
        execute(missing_state, 20, false, true);
        execute(state_joint, 0, false, true);
        // A pending future remains in the state snapshot but is not a due
        // future prediction. Its previous refresh-required flag cannot force it.
        auto pending_a = state_joint;
        pending_a.name = "source_a_pending_excluded_from_joint";
        pending_a.extra_memory.at(0x40000)[31] = 1;
        pending_a.extra_memory.at(0x40000)[19] = 1;
        pending_a.expected_table[33*8+6] |= 0x40;
        pending_a.expected = {1,2,5};
        uaps_result pending_a_result{};
        pending_a_result.base_activation_slots = pending_a_result.joint_activation_slots = 5;
        pending_a_result.next_pass_activation_slots = 6;
        const auto pending_a_bytes = pack_uaps_result(pending_a_result);
        pending_a.joint_result = {pending_a_bytes.begin(), pending_a_bytes.end()};
        execute(pending_a, 0, false, true);

        // Feature1 may still use a pending future as context; preserve its query
        // and suppress K/V writes after the packer's physical A8/A4 permutation.
        auto pending_context = state_table;
        pending_context.name = "source_a_context_preserves_query_disables_kv";
        pending_context.extra_memory.at(0x40000)[31] = 1;
        pending_context.extra_memory.at(0x40000)[19] = 1;
        pending_context.expected_table[33*8+6] |= 0x40;
        pending_context.extra_memory.at(0x50000)[32*16] = 0x80;
        pending_context.extra_memory.at(0x50000)[32*16+1] = 0x3f;
        pending_context.pending_expected[32*16] = 0x80;
        pending_context.pending_expected[32*16+1] = 0x3f;
        pending_context.pending_expected[32*16+10] = 1;
        pending_context.extra_memory.at(0x50000)[4*16+10] = 0;
        pending_context.pending_expected[4*16+10] = 0;
        pending_context.expected = {1,2,33};
        pending_context.expected_bits[33] = 8;
        pending_context.expected_sources[33] = 11;
        pending_context.expected_kv_disabled = {33};
        pending_context.joint_config = state_joint.joint_config;
        pending_context.joint_result = pending_a.joint_result;
        execute(pending_context, 0, false, true);

        auto target_full = state_joint;
        target_full.name = "published_current_predictions_exhaust_target";
        // Current tokens 1/2 are unresolved; token 5 is context. A MASKED future
        // row must not be added when target=2, despite room in execution rows.
        target_full.extra_memory.at(0x40000)[5] = 0;
        target_full.extra_memory.at(0x40000)[8] = 63;
        auto future_table_row = encode(target_full.rows[33]);
        future_table_row |= (std::uint64_t(1) << 9) | (std::uint64_t(1) << 34);
        future_table_row = (future_table_row & ~(std::uint64_t(0x1ffff) << 35)) | (std::uint64_t(63) << 35);
        for (unsigned byte = 0; byte < 8; ++byte) target_full.expected_table[33*8+byte] = future_table_row >> (byte*8);
        target_full.joint_config[6] = 96;
        uaps_attempt_config target_extension{};
        target_extension.prediction_target = 2;
        const auto target_suffix = pack_uaps_attempt_config(target_extension);
        target_full.extra_memory[0x9040] = {target_suffix.begin(), target_suffix.end()};
        target_full.expected = {1,2,5};
        state_joint_result.future_prediction_mask = state_joint_result.added_future_token_mask = 0;
        state_joint_result.future_prediction_count = state_joint_result.added_future_token_count = 0;
        state_joint_result.joint_activation_slots = 5; state_joint_result.next_pass_activation_slots = 6;
        const auto no_future = pack_uaps_result(state_joint_result);
        target_full.joint_result = {no_future.begin(), no_future.end()};
        execute(target_full, 0, false, true);
        run_joint_state_cases();
        const auto precision_cases = precision_test_cases();
        for (const auto& test_case : precision_cases) execute(test_case, 0, false, true);
        std::cout << "PASS context_precision actual_generator_records=" << precision_cases.size()
                  << " quota=10/12 input_pending=test_case selected_rows=test_case final_metadata_bits=raw\n";
        TestCase invalid = all_current;
        invalid.name = "invalid_start";
        invalid.target = 2;
        execute(invalid, 1);
        invalid = all_current;
        invalid.name = "mandatory_outside_eligible";
        invalid.rows[0].mandatory = true;
        execute(invalid, 2);
        execute(test_cases.front(), 3);
        execute(test_cases.front(), 4);
        execute(test_cases.front(), 5);
        execute(test_cases.front(), 8, true);
        execute(test_cases.front(), 10, true);
        execute(all_current, 0);
        dut_.final();
        std::cout << "PASS attention_refresh boundary_cases=" << test_cases.size() + 3
                  << " invalid_start/row/read_error/abort_restart/revoked_read=checked"
                  << " rows=1..2048 output_position2047=checked no_reset_relaunch=checked"
                  << " cycles=" << cycles_ << " threads=" << dut_.threads() << '\n';
    }
  private:
    TestCase state_table_test_case() {
        TestCase test_case;
        test_case.name = "state_table_sparse_odd_region";
        test_case.rows.resize(35); test_case.target = 3; test_case.expected = {1,2,5};
        test_case.state_table_update = true; test_case.allocate_precision = true; test_case.context_a8_token_count = 1;
        for (unsigned position = 0; position < test_case.rows.size(); ++position) {
            auto& row = test_case.rows[position]; row.source_index = position;
            row.score = position; row.eligible_order = position; row.required_order = position;
        }
        refresh_score_config config{};
        config.magic = 0x314e5041; config.version = 1; config.bytes = 80;
        config.token_count = 33; config.keys = 33; config.block_end = 33; config.relation_row_shift = 7; config.flags = 6;
        config.relation_base = 0x30000; config.relation_limit = 0x30000+33*128;
        config.change_base = 0x40000; config.change_limit = 0x40000+5*32;
        config.pending_base = 0x50000; config.pending_limit = 0x50000+33*16;
        config.regular_budget_base = 0x60000; config.regular_budget_limit = 0x60010;
        const auto packed = pack_refresh_score_config(config);
        test_case.extra_memory[0x8500] = {packed.begin(), packed.end()};
        test_case.extra_memory[0x30000].resize(33*128);
        auto& pending = test_case.extra_memory[0x50000]; pending.resize(33*16);
        pending[1] = 0x3f;
        pending[3*16+1] = 0x3e;
        for (const unsigned position : {5,32,33}) pending[(position-1)*16+10] = 2;
        test_case.pending_expected.resize(33*16);
        test_case.pending_expected[1] = 0x3f;
        test_case.pending_expected[3*16+1] = 0x3e;
        for (const unsigned position : {1,2,5}) test_case.pending_expected[(position-1)*16+10] = 3;
        auto expected_rows = test_case.rows;
        const std::array<unsigned, 5> positions{33,2,32,1,2047}, states{1,1,2,0,2}, bits{8,8,4,4,8}, tokens{11,7,9,63,126463};
        auto& state_bytes = test_case.extra_memory[0x40000];
        for (unsigned index = 0; index < positions.size(); ++index) {
            token_state_entry state{};
            state.token_position = positions[index]; state.state = states[index]; state.activation_bits = bits[index];
            state.token_id = tokens[index]; state.capture_index = 99; state.refresh_required = positions[index] == 32;
            const auto bytes = pack_token_state_entry(state);
            state_bytes.insert(state_bytes.end(), bytes.begin(), bytes.end());
            if (positions[index] >= 35) continue;
            auto& row = expected_rows[positions[index]];
            row.source_index = tokens[index]; row.a8 = bits[index] == 8;
            row.current = positions[index] < 33; row.eligible = !row.current;
            row.mandatory = state.refresh_required;
            row.prediction_token_mask = row.current && states[index] != 2;
            row.forecast_a8 = row.current && states[index] == 0;
        }
        for (const auto& row : expected_rows)
            for (unsigned byte = 0; byte < 8; ++byte) test_case.expected_table.push_back(encode(row)>>(byte*8));
        test_case.expected_bits = {{1,4},{2,8},{5,8}};
        test_case.expected_sources = {{1,63},{2,7},{5,5}};
        in_block_refresh_budget budget{};
        budget.target_token_count_x2 = 6; budget.region_start = 1;
        const auto before = pack_in_block_refresh_budget(budget);
        test_case.extra_memory[0x60000] = {before.begin(), before.end()};
        budget.regular_steps = 1; budget.cumulative_active_token_count = 3;
        const auto after = pack_in_block_refresh_budget(budget);
        test_case.budget_expected = {after.begin(), after.end()};
        return test_case;
    }
    void run_joint_state_cases() {
        const auto cases=joint_state_test_cases();
        for (const auto& test_case:cases) execute(test_case,0,false,true);
        for (unsigned fault=0;fault<3;++fault) {
            auto invalid=cases.front();invalid.name="live_state_invalid_"+std::to_string(fault);
            if (fault==0) invalid.extra_memory.at(0x9040)[26]=0;
            if (fault==1) invalid.extra_memory.at(0x9040)[29]=2;
            // Missing current state must fail instead of using the prior case's
            // cached state. Out-of-region records remain legal inputs.
            if (fault==2) invalid.extra_memory.at(0x40000)[0]=0;
            execute(invalid,20,false,true);
        }
        execute(cases.front(),0,false,true);
    }
    std::vector<TestCase> joint_state_test_cases() {
        std::ifstream stream("cases/control/uaps_live_state.json");
        json document; stream >> document;
        require(document.at("schema") == "supra-joint-state-selection/v1", "live-state schema");
        std::vector<TestCase> result;
        for (const auto& record : document.at("records")) {
            TestCase test_case;
            test_case.name = "live_state_"+record.at("name").get<std::string>();
            test_case.rows.resize(96); test_case.target = 48;
            test_case.state_table_update = true; test_case.allocate_precision = true;
            test_case.context_a8_token_count = record.at("context_a8");
            const auto base = record.at("base_positions").get<std::vector<unsigned>>();
            const auto bits = record.at("base_bits").get<std::vector<unsigned>>();
            const auto& expected = record.at("expected");
            test_case.expected = base;
            const auto added = expected.at("added").get<std::vector<unsigned>>();
            for (unsigned i=0; i<base.size(); ++i) test_case.expected_bits[base[i]] = bits[i];
            for (unsigned i=0; i<added.size(); ++i) {
                test_case.expected.push_back(64+added[i]);
                test_case.expected_bits[64+added[i]] = expected.at("added_bits").at(i);
            }
            std::sort(test_case.expected.begin(),test_case.expected.end());
            refresh_score_config config{};
            config.magic=0x314e5041; config.version=1; config.bytes=80;
            config.token_count=64; config.keys=64; config.block_end=64; config.relation_row_shift=7; config.flags=6;
            config.relation_base=0x30000; config.relation_limit=0x32000;
            config.change_base=0x40000; config.change_limit=0x40800;
            config.pending_base=0x50000; config.pending_limit=0x50400;
            config.regular_budget_base=0x60000; config.regular_budget_limit=0x60010;
            const auto packed=pack_refresh_score_config(config);
            test_case.extra_memory[0x8500]={packed.begin(),packed.end()};
            test_case.extra_memory[0x30000].resize(8192);
            auto& pending=test_case.extra_memory[0x50000]; pending.resize(1024);
            for (unsigned i=0;i<64;++i) {
                const unsigned score=record.at("dependency_bf16").at(i).get<unsigned>();
                pending[i*16]=score; pending[i*16+1]=score>>8;
            }
            test_case.pending_expected=pending;
            for (const auto position : base) test_case.pending_expected[(position-32)*16+10]=position<35 ? 3:1;
            for (unsigned pos=0;pos<96;++pos) {
                auto& row=test_case.rows[pos]; row.source_index=pos;
                auto updated=row;
                if (pos>=32) {
                    const unsigned status=record.at(pos<64 ? "current_states":"future_states").at(pos%32);
                    const bool pending_a=pos==66;
                    token_state_entry entry{};
                    entry.token_position=pos; entry.state=status; entry.activation_bits=status==1 ? 8:4;
                    entry.token_id=status==0 ? 63:pos; entry.capture_index=99; entry.source_a_pending=pending_a;
                    const auto state=pack_token_state_entry(entry);
                    auto& states=test_case.extra_memory[0x40000];states.insert(states.end(),state.begin(),state.end());
                    updated.source_index=entry.token_id; updated.a8=entry.activation_bits==8;
                    updated.current=pos<64; updated.eligible=pos>=64;
                    updated.prediction_token_mask=pos<64 && status!=2;
                    updated.forecast_a8=pos<64 && status==0; updated.kv_write_disable=pending_a;
                    test_case.expected_sources[pos]=entry.token_id;
                }
                for (unsigned b=0;b<8;++b) test_case.expected_table.push_back(encode(updated)>>(8*b));
            }
            test_case.expected_kv_disabled={66};
            in_block_refresh_budget budget{};budget.target_token_count_x2=12;budget.region_start=32;
            const auto before=pack_in_block_refresh_budget(budget);test_case.extra_memory[0x60000]={before.begin(),before.end()};
            budget.regular_steps=1;budget.cumulative_active_token_count=6;
            const auto after=pack_in_block_refresh_budget(budget);test_case.budget_expected={after.begin(),after.end()};
            uaps_config joint{};joint.magic=0x314a4449;joint.version=1;joint.bytes=96;
            joint.future_token_count=32;joint.next_block_start=64;joint.max_next_tokens=8;joint.priority_control=0xe0;
            joint.result_base=0x28000;joint.result_limit=0x28010;
            const auto descriptor=pack_uaps_config(joint);test_case.joint_config={descriptor.begin(),descriptor.end()};
            uaps_attempt_config suffix{};
            suffix.source_b_flags=record.at("source_b_flags");
            suffix.block_step_index=record.at("block_step_index");
            suffix.max_current_unresolved=record.at("max_current");suffix.max_handoff_tokens=record.at("max_handoff");
            suffix.admission_budget=record.at("admission_budget");suffix.state_control=record.at("allow_deferred").get<bool>() ? 3:1;
            suffix.min_reuse_score_bf16=record.at("min_reuse_score_bf16");
            const auto extra=pack_uaps_attempt_config(suffix);test_case.extra_memory[0x9040]={extra.begin(),extra.end()};
            uaps_result output{};
            for (const unsigned pos:expected.at("progress").get<std::vector<unsigned>>()) output.future_prediction_mask|=1u<<pos;
            for (const auto pos:added) output.added_future_token_mask|=1u<<pos;
            output.future_prediction_count=expected.at("progress").size();output.added_future_token_count=added.size();
            const auto forecast=expected.at("forecast_base_bits").get<std::vector<unsigned>>();
            for (unsigned i=0;i<base.size();++i) {
                output.base_activation_slots+=bits[i]/4;
                const bool reused=base[i]>=64 && ((output.future_prediction_mask>>(base[i]-64))&1);
                output.next_pass_activation_slots+=reused ? 2:forecast[i]/4;
            }
            output.joint_activation_slots=output.base_activation_slots;
            for (unsigned i=0;i<added.size();++i) {
                output.joint_activation_slots+=expected.at("added_bits").at(i).get<unsigned>()/4;
                output.next_pass_activation_slots+=2;
            }
            const auto packed_result=pack_uaps_result(output);test_case.joint_result={packed_result.begin(),packed_result.end()};
            result.push_back(std::move(test_case));
        }
        return result;
    }
    std::vector<TestCase> precision_test_cases(const char* path = "cases/control/psme_context_precision.json") {
        std::ifstream stream(path);
        require(stream.good(), "missing context precision reference");
        json document; stream >> document;
        require(document.at("schema") == "supra-psme-context-precision/v1", "wrong context precision schema");
        std::vector<TestCase> result;
        for (const auto& entry : document.at("records")) {
            TestCase test_case;
            test_case.name = "context_precision_"+std::to_string(entry.at("parameters").at("fixed_context_a8_token_count").get<unsigned>())+"_"+entry.at("capture_index").dump();
            test_case.expected = entry.at("input_positions").at("raw").get<std::vector<unsigned>>();
            const unsigned begin = test_case.expected.front()/32*32;
            const unsigned rows = (test_case.expected.back()-begin+32)/32*32;
            require(rows <= 96, "packed context test_case exceeds regular region");
            test_case.rows.resize(begin+rows); test_case.target = test_case.expected.size();
            test_case.allocate_precision = true;
            test_case.context_a8_token_count = entry.at("parameters").at("fixed_context_a8_token_count");
            test_case.all_a8 = entry.at("tail_transfer_active");
            refresh_score_config config{};
            config.magic = 0x314e5041; config.version = 1; config.bytes = 80;
            config.token_count = rows; config.keys = rows; config.block_end = rows;
            config.relation_row_shift = rows > 64 ? 8 : rows > 32 ? 7 : 6;
            config.relation_base = 0x30000; config.relation_limit = config.relation_base+(rows<<config.relation_row_shift);
            config.change_base = 0x40000; config.change_limit = config.change_base+rows*4;
            config.pending_base = 0x50000; config.pending_limit = config.pending_base+rows*16;
            config.regular_budget_base = 0x60000; config.regular_budget_limit = 0x60010;
            const auto packed = pack_refresh_score_config(config);
            test_case.extra_memory[0x8500] = {packed.begin(), packed.end()};
            test_case.extra_memory[0x30000].resize(config.relation_limit-config.relation_base);
            test_case.extra_memory[0x40000].resize(rows*4);
            auto& pending = test_case.extra_memory[0x50000]; pending.resize(rows*16);
            for (unsigned index = 0; index < test_case.expected.size(); ++index) {
                const auto position = test_case.expected[index];
                auto& row = test_case.rows[position]; row.source_index = position;
                row.a8 = entry.at("input_activation_bits").at("raw").at(index) == 8;
                row.prediction_token_mask = entry.at("mandatory_current").at("raw").at(index).get<bool>();
                pending[(position-begin)*16+10] = 2;
                test_case.expected_bits[position] = entry.at("expected_activation_bits").at("raw").at(index);
            }
            if (entry.at("expected_upgrade_count").get<unsigned>())
                for (unsigned index = 0; index < entry.at("context_candidates").at("raw").size(); ++index) {
                    const auto ordinal = entry.at("context_candidates").at("raw").at(index).get<unsigned>();
                    const auto score = entry.at("context_scores").at("raw").at(index).get<unsigned>();
                    const auto offset = (test_case.expected.at(ordinal)-begin)*16;
                    pending[offset] = score; pending[offset+1] = score>>8;
                }
            test_case.pending_expected = pending;
            in_block_refresh_budget budget{};
            budget.target_token_count_x2 = test_case.expected.size()*2; budget.region_start = begin;
            const auto before = pack_in_block_refresh_budget(budget);
            test_case.extra_memory[0x60000] = {before.begin(), before.end()};
            budget.regular_steps = 1; budget.cumulative_active_token_count = test_case.expected.size();
            const auto after = pack_in_block_refresh_budget(budget);
            test_case.budget_expected = {after.begin(), after.end()};
            result.push_back(std::move(test_case));
        }
        return result;
    }
    TestCase published_state_changes(TestCase test_case) {
        test_case.name += "_published_state_changes";
        auto& config = test_case.extra_memory.at(0x8500);
        config[15] |= 2;
        const unsigned keys = config[10] | unsigned(config[11])<<8;
        const auto& budget = test_case.extra_memory.at(0x60000);
        const unsigned begin = budget[2] | unsigned(budget[3])<<8;
        const auto original = test_case.extra_memory.at(0x40000);
        std::vector<std::uint8_t> states;
        for (unsigned key = keys; key-- != 0;) {
            if (!(original[key*4+2]&1)) continue;
            const auto offset = states.size(); states.resize(offset+32);
            states[offset] = begin+key; states[offset+1] = (begin+key)>>8;
            states[offset+20] = 99;
            states[offset+25] = original[key*4+2];
            states[offset+26] = original[key*4]; states[offset+27] = original[key*4+1];
        }
        require(!states.empty(), "state change test_case needs a real changed key");
        const auto outside = states.size(); states.resize(outside+32);
        states[outside] = 255; states[outside+1] = 7; states[outside+20] = 99;
        states[outside+26] = 128; states[outside+27] = 63;
        const std::uint64_t limit = 0x40000+states.size();
        for (unsigned byte = 0; byte < 8; ++byte) config[40+byte] = limit>>(byte*8);
        test_case.extra_memory[0x40000] = std::move(states);
        return test_case;
    }
    std::vector<TestCase> regular_test_cases(const char* name, bool direct_path = false) {
        std::ifstream stream(direct_path ? std::string(name) : std::string("cases/control/")+name);
        require(stream.good(), std::string("cannot read regular reference: ")+name);
        json document; stream >> document;
        const unsigned begin = document.at("region_start"), rows = document.at("region_end").get<unsigned>()-begin;
        unsigned relation_shift = 6;
        while ((1u << relation_shift) < rows*2) ++relation_shift;
        const unsigned relation_stride = 1u << relation_shift;
        std::vector<TestCase> result;
        const auto put16 = [](std::vector<std::uint8_t>& target, unsigned at, unsigned value) {
            target.at(at) = value; target.at(at+1) = value>>8;
        };
        for (const auto& record : document.at("records")) {
            TestCase test_case;
            test_case.name = record.contains("name") ? record.at("name").get<std::string>() : "regular_step_"+record.at("step_index").dump();
            test_case.rows.resize(document.at("total_length")); test_case.target = rows;
            for (unsigned position = 0; position < test_case.rows.size(); ++position) {
                test_case.rows[position].source_index = position; test_case.rows[position].a8 = true;
            }
            refresh_score_config config{};
            config.magic = 0x314e5041; config.version = 1; config.bytes = 80;
            config.token_count = rows; config.keys = rows; config.block_end = rows; config.relation_row_shift = relation_shift;
            config.relation_base = 0x30000; config.relation_limit = 0x30000+rows*relation_stride;
            config.change_base = 0x40000; config.change_limit = 0x40000+((rows+3)/4)*16;
            config.pending_base = 0x50000; config.pending_limit = 0x50000+rows*16;
            config.regular_budget_base = 0x60000; config.regular_budget_limit = 0x60010;
            const auto packed = pack_refresh_score_config(config);
            test_case.extra_memory[0x8500] = std::vector<std::uint8_t>(packed.begin(), packed.end());
            auto& relation = test_case.extra_memory[0x30000]; relation.resize(rows*relation_stride);
            auto& changes = test_case.extra_memory[0x40000]; changes.resize(config.change_limit-config.change_base);
            auto& pending = test_case.extra_memory[0x50000]; pending.resize(rows*16); test_case.pending_expected.resize(rows*16);
            const auto& before = record.at("before"); const auto& expected = record.at("expected"); const auto& input = record.at("inputs");
            for (unsigned row = 0; row < rows; ++row) {
                for (unsigned key = 0; key < rows; ++key) put16(relation, row*relation_stride+key*2, expected.at("dependency").at("raw").at(row).at(key));
                put16(pending, row*16, before.at("pending").at("raw").at(row));
                put16(test_case.pending_expected, row*16, expected.at("pending").at("raw").at(row));
                const unsigned predicted = input.at("predicted").at("raw").at(row).get<bool>();
                pending.at(row*16+10) = unsigned(before.at("refresh").at("raw").at(row).get<bool>()) | predicted<<1;
                test_case.pending_expected.at(row*16+10) = predicted<<1;
                if (expected.at("next_refresh").at("raw").at(row).get<bool>()) test_case.expected.push_back(begin+row);
                put16(changes, row*4, input.at("changed_confidence_global").at("raw").at(0).at(begin+row));
                changes.at(row*4+2) = unsigned(input.at("changed_global").at("raw").at(0).at(begin+row).get<bool>()) |
                    unsigned(input.at("changed_remask_global").at("raw").at(0).at(begin+row).get<bool>())<<1;
            }
            if (input.contains("dependency_mean") && before.contains("dependency")) {
                test_case.dependency_expected = relation;
                for (unsigned row = 0; row < rows; ++row)
                    for (unsigned key = 0; key < rows; ++key)
                        put16(relation, row*relation_stride+key*2, before.at("dependency").at("raw").at(row).at(key));
                // Real regular jobs can exceed 12 KiB. Keep them separate from
                // the row table at 0x10000 and probability source at 0xe0000.
                test_case.relation_job_address = 0xc0000;
                auto& probabilities = test_case.extra_memory[0xe0000];
                const auto& positions = input.at("query_positions").at("raw");
                // observe() converts the reduced FP32 dependency values
                // into BF16 state. Preserve that boundary; never truncate FP32 bits.
                const auto& transport = input.at(input.contains("dependency_bf16") ? "dependency_bf16" : "dependency_mean");
                require(transport.at("dtype") == "torch.bfloat16", "dependency transport requires explicit BF16 raw bits");
                const auto& means = transport.at("raw").at(0);
                for (unsigned query = 0; query < positions.size(); ++query) {
                    const unsigned row = positions.at(query).get<unsigned>()-begin;
                    for (unsigned key = 0; key < rows; key += 8) {
                        const auto source_offset = probabilities.size();
                        probabilities.resize(source_offset+512);
                        // Identical BF16 heads preserve the test_case's supplied
                        // mean exactly while exercising the real reduction.
                        for (unsigned head = 0; head < 32; ++head)
                            for (unsigned lane = 0; lane < 8 && key+lane < rows; ++lane)
                                put16(probabilities, source_offset+head*16+lane*2, means.at(query).at(key+lane));
                        attention_dependency_job job{};
                        job.source_base = 0xe0000+source_offset; job.source_limit = job.source_base+512;
                        job.head_stride = 16; job.lane_mask = (1u << std::min(8u, rows-key))-1;
                        job.operation = 0;
                        job.output_base = 0x30000+row*relation_stride+key*2;
                        job.output_limit = job.output_base+16;
                        const auto bytes = pack_attention_dependency_job(job);
                        test_case.jobs.insert(test_case.jobs.end(), bytes.begin(), bytes.end());
                    }
                }
                test_case.extra_memory[test_case.relation_job_address] = test_case.jobs;
                require(test_case.relation_job_address+test_case.jobs.size() <= 0xe0000,
                        "regular dependency jobs overlap probability input");
            }
            in_block_refresh_budget budget{};
            budget.target_token_count_x2 = unsigned(document.at("target_active_token_count").get<double>()*2);
            budget.region_start = begin; budget.regular_steps = before.at("regular_steps"); budget.cumulative_active_token_count = before.at("cumulative_active_token_count");
            budget.token_count_x2_credit = budget.target_token_count_x2*budget.regular_steps-2*budget.cumulative_active_token_count;
            const auto before_bytes = pack_in_block_refresh_budget(budget);
            test_case.extra_memory[0x60000] = std::vector<std::uint8_t>(before_bytes.begin(), before_bytes.end());
            budget.regular_steps = expected.at("regular_steps"); budget.cumulative_active_token_count = expected.at("cumulative_active_token_count");
            budget.token_count_x2_credit = budget.target_token_count_x2*budget.regular_steps-2*budget.cumulative_active_token_count;
            const auto after_bytes = pack_in_block_refresh_budget(budget);
            test_case.budget_expected.assign(after_bytes.begin(), after_bytes.end());
            if (record.contains("selected_bits")) {
                test_case.allocate_precision = true;
                test_case.context_a8_token_count = record.at("context_a8");
                const unsigned current_begin = record.at("current_begin"), current_end = record.at("current_end");
                for (unsigned position = current_begin; position < current_end; ++position) {
                    auto& row = test_case.rows.at(position);
                    row.prediction_token_mask = input.at("predicted").at("raw").at(position-begin).get<bool>();
                    row.a8 = record.at("current_bits").at("raw").at(position-current_begin) == 8;
                }
                const auto& bits = record.at("selected_bits").at("raw");
                require(bits.size() == test_case.expected.size(), "selected precision length mismatch");
                for (unsigned i = 0; i < test_case.expected.size(); ++i)
                    test_case.expected_bits[test_case.expected[i]] = bits.at(i);
            }
            result.push_back(std::move(test_case));
        }
        return result;
    }
    void run_paired_pending_cases() {
        const auto regular = regular_test_cases("atse_attention_dependencies.json").front();
        unsigned tested = 0;
        for (const auto& cross : pending_test_cases()) {
            if (!cross.new_pending_sequence) continue;
            auto combined = regular;
            combined.name = "paired_"+cross.name;
            combined.paired_pending = true;
            combined.rows.resize(std::max(combined.rows.size(),cross.rows.size()));
            combined.cross_pending_expected = cross.pending_expected;
            combined.extra_memory[0x8550] = combined.extra_memory.at(0x8500);
            auto primary = cross.extra_memory.at(0x8500);
            const auto put64 = [&](unsigned offset, std::uint64_t value) {
                for (unsigned byte=0;byte<8;++byte) primary.at(offset+byte)=value>>(byte*8);
            };
            for (const auto region : {std::pair<unsigned,unsigned>{0x30000,16}, {0x40000,32}, {0x50000,48}}) {
                const unsigned address=region.first+0x60000;
                auto bytes=cross.extra_memory.at(region.first);
                if (region.first==0x30000) bytes=cross.dependency_expected;
                combined.extra_memory[address]=bytes;
                put64(region.second,address); put64(region.second+8,address+bytes.size());
            }
            combined.extra_memory[0x8500]=primary;
            // Two generator inputs test serial resource reuse.
            execute(combined,0,false,true);
            combined.live_consumed=true;
            combined.extra_memory.at(0x8500)[15]|=64;
            auto& pending=combined.extra_memory.at(0xb0000);
            for (unsigned pos=0;pos<pending.size()/16;++pos) {
                if (pending[pos*16+10]&1) combined.loaded_kv_positions.push_back(pos);
                pending[pos*16+10]&=~1u;
            }
            execute(combined,0,false,true);
            auto closed=combined;
            closed.name="closed_"+combined.name; closed.closeout=true;
            closed.expected.clear(); closed.expected_bits.clear();
            closed.pending_expected=closed.extra_memory.at(0x50000);
            closed.budget_expected=closed.extra_memory.at(0x60000);
            // A deliberately invalid next-selection descriptor must not be
            // fetched when the current block has no next regular forward.
            closed.joint_config.assign(64,0);
            execute(closed,0,false,true);
            execute(combined,0,false,true); // Same DUT must clear closeout.
            auto invalid=combined; invalid.extra_memory.at(0x8550)[15]|=1;
            execute(invalid,0x41,false,true);
            invalid=combined; invalid.extra_memory.at(0x8500)[15]&=~1u;
            execute(invalid,0x41,false,true);
            execute(combined,0,false,true);
            execute(regular,0,false,true);
            ++tested;
        }
        require(tested==3,"paired pending must include all three confidence modes");
        std::cout << "PASS paired_pending three_risks live_and_DDR_consumption order_errors recovery single_record_restore\n";
    }
    void run_initial_pending_cases() {
        std::vector<std::uint8_t> persisted;
        for (auto test_case:pending_test_cases()) {
            // Keep the initially tracked keys. Metadata current32..63 differs
            // from tracked keys48..79; optional80 must still be consumed.
            if (test_case.advance_dependency_block) continue;
            if (test_case.new_pending_sequence) persisted.clear();
            test_case.name="initial_selected_"+test_case.name;
            test_case.initial_selected_consumed=true;
            auto& config=test_case.extra_memory.at(0x8500);
            config[6]=96; config[15]|=128; config.resize(96);
            config[80]=test_case.consume_current_begin;
            auto& input=test_case.extra_memory.at(0x50000);
            for (unsigned pos=0;pos<input.size()/16;++pos) {
                if (input[pos*16+10]&1) test_case.loaded_kv_positions.push_back(pos);
                input[pos*16+10]&=~1u;
            }
            if (!persisted.empty()) input=persisted;
            execute(test_case,0,false,true,&persisted);
            if (test_case.new_pending_sequence) {
                auto invalid=test_case;
                invalid.extra_memory.at(0x8500)[82]=1;
                execute(invalid,0x41,false,true);
                invalid=test_case; invalid.extra_memory.at(0x8500)[15]|=64;
                execute(invalid,0x41,false,true);
                invalid=test_case; invalid.extra_memory.at(0x8500)[80]=81;
                execute(invalid,0x41,false,true);
                execute(test_case,0,false,true);
            }
        }
        std::cout << "PASS initial_key_boundary_consumption current_range_excluded selected_future_included kv_disabled_included\n";
    }
    void run_live_pending_cases() {
        std::vector<std::uint8_t> persisted;
        for (auto test_case:pending_test_cases()) {
            if (test_case.new_pending_sequence) persisted.clear();
            test_case.name="executed_"+test_case.name; test_case.live_consumed=true;
            test_case.extra_memory.at(0x8500)[15]|=64;
            auto& input=test_case.extra_memory.at(0x50000);
            for (unsigned pos=0;pos<input.size()/16;++pos) {
                if (input[pos*16+10]&1) test_case.loaded_kv_positions.push_back(pos);
                input[pos*16+10]&=~1u;
            }
            if (!persisted.empty()) {
                // Bytes12..13 are the previous update's diagnostic invalidation,
                // not one of the five persistent pending vectors.
                require(input.size()==persisted.size(),"pending region changed across forwards");
                for (unsigned pos=0;pos<input.size()/16;++pos)
                    for (unsigned byte=0;byte<11;++byte)
                        require(input[pos*16+byte]==persisted[pos*16+byte],
                            test_case.name+": previous pending differs at token="+std::to_string(pos)+
                            " byte="+std::to_string(byte));
                input=persisted;
            }
            execute(test_case,0,false,true,&persisted);
        }
        std::cout << "PASS cross_block_live_consumed actual_bitmap synthetic_metadata_events pending_state_chain\n";
    }
    std::vector<TestCase> pending_test_cases() {
        std::ifstream stream("cases/control/atse_cross_block_pending_refresh.json");
        json document; stream >> document;
        std::vector<TestCase> result;
        std::vector<json> variants{document};
        if (document.contains("variants"))
            for (const auto& variant : document.at("variants")) variants.push_back(variant);
        for (const auto& input_document : variants) {
        const unsigned rows = input_document.at("rows");
        const auto mode = input_document.value("pending_confidence_mode", std::string("all_changes"));
        const std::vector<std::string> names{"pending", "actual_remask_pending", "actual_remask_epoch_pending", "future_pending", "future_actual_remask_pending"};
        for (const auto& record : input_document.at("records")) {
            TestCase test_case;
            test_case.name = "pending_"+mode+"_step_"+record.at("step_index").dump();
            const unsigned block_start = record.value("block_start", input_document.at("block_start").get<unsigned>());
            test_case.advance_dependency_block = record.value("advance_block", false);
            test_case.new_pending_sequence = record.at("step_index") == 0;
            test_case.rows.resize(rows); test_case.rows[0].current = true; test_case.target = 1; test_case.expected = {0};
            refresh_score_config config{};
            config.magic = 0x314e5041; config.version = 1; config.bytes = 80;
            config.token_count = rows; config.keys = 32; config.block_end = block_start+32; config.relation_row_shift = 6; config.flags = 1 | ((mode == "all_changes" ? 0u : mode == "stable_unmask" ? 1u : 2u) << 3) | (test_case.advance_dependency_block ? 32u : 0u);
            config.relation_base = 0x30000; config.relation_limit = 0x30000+rows*64;
            config.change_base = 0x40000; config.change_limit = 0x40080;
            config.pending_base = 0x50000; config.pending_limit = 0x50000+rows*16;
            const auto packed = pack_refresh_score_config(config);
            test_case.extra_memory[0x8500] = std::vector<std::uint8_t>(packed.begin(), packed.end());
            auto& relation = test_case.extra_memory[0x30000]; relation.resize(rows*64);
            auto& changes = test_case.extra_memory[0x40000]; changes.resize(128);
            auto& pending = test_case.extra_memory[0x50000]; pending.resize(rows*16);
            test_case.pending_expected.resize(rows*16);
            const auto put16 = [](std::vector<std::uint8_t>& target, unsigned at, unsigned value) {
                target.at(at) = value; target.at(at+1) = value>>8;
            };
            const auto& input = record.at("inputs");
            for (unsigned row = 0; row < rows; ++row) {
                for (unsigned key = 0; key < 32; ++key)
                    put16(relation, row*64+key*2, record.at("expected").at("relation").at("raw").at(row).at(key));
                for (unsigned field = 0; field < names.size(); ++field) {
                    put16(pending, row*16+field*2, record.at("before").at(names[field]).at("raw").at(row));
                    put16(test_case.pending_expected, row*16+field*2, record.at("expected").at(names[field]).at("raw").at(row));
                }
            }
            test_case.dependency_expected = relation;
            for (unsigned row = 0; row < rows; ++row)
                for (unsigned key = 0; key < 32; ++key)
                    put16(relation, row*64+key*2, record.at("before").at("relation").at("raw").at(row).at(key));
            test_case.relation_job_address = 0xd000;
            auto& probabilities = test_case.extra_memory[0xe0000];
            const auto& positions = input.at("query_positions").at("raw");
            const auto& means = input.at("relation").at("raw").at(0);
            for (unsigned query = 0; query < positions.size(); ++query) {
                const unsigned row = positions.at(query);
                for (unsigned key = 0; key < 32; key += 8) {
                    const auto offset = probabilities.size(); probabilities.resize(offset+512);
                    for (unsigned head = 0; head < 32; ++head)
                        for (unsigned lane = 0; lane < 8; ++lane)
                            put16(probabilities, offset+head*16+lane*2, means.at(query).at(key+lane));
                    attention_dependency_job job{};
                    job.source_base = 0xe0000+offset; job.source_limit = job.source_base+512;
                    job.head_stride = 16; job.lane_mask = 255; job.operation = 0;
                    job.output_base = 0x30000+row*64+key*2; job.output_limit = job.output_base+16;
                    const auto bytes = pack_attention_dependency_job(job);
                    test_case.jobs.insert(test_case.jobs.end(), bytes.begin(), bytes.end());
                }
            }
            test_case.extra_memory[test_case.relation_job_address] = test_case.jobs;
            for (const auto row : input.at("consumed_positions").at("raw")) pending.at(row.get<unsigned>()*16+10) = 1;
            for (unsigned key = 0; key < 32; ++key) {
                put16(changes, key*4, input.at("changed_confidence_global").at("raw").at(0).at(block_start+key));
                changes.at(key*4+2) = unsigned(input.at("changed_global").at("raw").at(0).at(block_start+key).get<bool>()) |
                    unsigned(input.at("changed_remask_global").at("raw").at(0).at(block_start+key).get<bool>())<<1;
            }
            result.push_back(std::move(test_case));
        }
        }
        return result;
    }
    std::vector<TestCase> deep_test_cases() {
        std::ifstream stream("cases/control/atse_uaps_limits.json");
        json document; stream >> document;
        std::vector<TestCase> result;
        for (const auto& entry : document.at("records")) {
            if (entry.at("kind") != "deep_precision") continue;
            const auto& input = entry.at("inputs");
            const auto positions = input.at("deep_positions").at("raw").get<std::vector<unsigned>>();
            if (!std::is_sorted(positions.begin(), positions.end())) continue;
            TestCase test_case;
            test_case.name = entry.at("name"); test_case.target = positions.size(); test_case.expected = positions;
            test_case.rows.resize(positions.back()+2);
            const int limit = input.at("a8_limit");
            test_case.deep_precision = limit >= 0; test_case.deep_a8_limit = limit >= 0 ? limit : 0;
            for (unsigned row = 0; row < positions.size(); ++row) {
                auto& item = test_case.rows[positions[row]];
                item.eligible = true; item.eligible_order = row;
                item.score = input.at("score_q8").at("raw").at(row);
                item.a8 = input.at("activation_bits").at("raw").at(row) == 8;
                item.mandatory = input.at("protected").at("raw").at(row);
                item.source_index = positions[row];
                test_case.expected_bits[positions[row]] = entry.at("expected").at("activation_bits").at("raw").at(row);
            }
            const auto current = std::find_if(positions.begin(), positions.end(), [&](unsigned position) {
                return test_case.rows[position].mandatory || !test_case.rows[position].a8;
            });
            if (current == positions.end()) continue;
            test_case.rows[*current].current = true;
            test_case.rows[*current].eligible = false;
            test_case.rows[*current].mandatory = false;
            result.push_back(std::move(test_case));
        }
        return result;
    }
    std::vector<TestCase> joint_test_cases() {
        std::vector<TestCase> result;
        for (const auto* name : {"uaps_token_selection.json", "uaps_capacity_priority.json", "uaps_priority.json", "atse_uaps_in_block.json", "psme_uaps_context_precision.json", "atse_uaps_limits.json", "uaps_retry_prediction_target.json", "uaps_source_b_selection.json"}) {
            std::ifstream stream(std::string("cases/control/")+name);
            json document; stream >> document;
            for (const auto& entry : document.at("records")) {
                if (entry.contains("kind") && entry.at("kind") != "future_admission_selection") continue;
                const auto& input = entry.at("inputs");
                const auto& expected = entry.at("expected");
                const unsigned source_b_flags = entry.value("source_b_flags", 0u);
                TestCase test_case;
                const bool precision = entry.contains("precision_record_index");
                const bool regular = precision || entry.contains("regular_source");
                if (precision) {
                    test_case = precision_test_cases().at(entry.at("precision_record_index"));
                    test_case.expected.clear();
                } else if (regular) {
                    test_case = regular_test_cases(entry.at("regular_source").get<std::string>().c_str()).at(entry.at("regular_record_index"));
                    test_case.expected.clear();
                }
                test_case.name = "joint_"+entry.at("name").get<std::string>();
                test_case.rows.resize(128); test_case.target = document.at("target_joint_tokens");
                uaps_config config{};
                config.magic = 0x314a4449; config.version = 1; config.bytes = 64;
                const unsigned base_token_count = input.at("base_positions").at("raw").size();
                config.base_token_count = regular ? 0 : base_token_count;
                config.future_token_count = input.at("next_activation_bits").at("raw").size();
                config.next_block_start = input.at("next_block_start");
                config.max_next_tokens = document.at("max_next_tokens");
                const bool rank_priority = entry.contains("available");
                if (rank_priority) config.priority_control = 0x80 |
                    (entry.at("low_dependency").get<bool>() ? 0x40 : 0) | entry.at("available").get<unsigned>();
                config.base_table_base = regular ? 0 : 0xa000; config.base_table_limit = regular ? 0 : 0xa000+config.base_token_count*16;
                config.future_table_base = 0xb000; config.future_table_limit = 0xb000+config.future_token_count*16;
                config.result_base = 0x28000; config.result_limit = 0x28010;
                const auto packed_config = pack_uaps_config(config);
                test_case.joint_config.assign(packed_config.begin(), packed_config.end());
                test_case.joint_base.resize(config.base_token_count*16); test_case.joint_future.resize(config.future_token_count*16);
                for (unsigned position = 0; position < test_case.rows.size(); ++position) {
                    test_case.rows[position].source_index = (position*17)%2048;
                    if (regular && !precision) {
                        test_case.rows[position].a8 = position%3 != 0;
                        test_case.rows[position].forecast_a8 = true;
                        test_case.rows[position].current = false;
                    }
                }
                for (unsigned row = 0; row < base_token_count; ++row) {
                    const unsigned position = input.at("base_positions").at("raw").at(row);
                    test_case.expected.push_back(position);
                    if (!precision) test_case.rows[position].a8 = input.at("base_activation_bits").at("raw").at(row) == 8;
                    test_case.rows[position].current = !regular;
                    test_case.rows[position].forecast_a8 = input.at("next_step_base_activation_bits").at("raw").at(row) == 8 &&
                        (!precision || input.at("base_activation_bits").at("raw").at(row) != 8);
                    if (!regular) {
                        test_case.joint_base[row*16] = position; test_case.joint_base[row*16+1] = position >> 8;
                        test_case.joint_base[row*16+2] = unsigned(test_case.rows[position].a8) | unsigned(test_case.rows[position].forecast_a8) << 1;
                        if (row < input.value("current_prediction_tokens", 0u)) test_case.joint_base[row*16+2] |= 4;
                    }
                }
                if (!test_case.expected.empty()) test_case.rows[test_case.expected.front()].source_index = 126463;
                for (unsigned row = 0; row < config.future_token_count; ++row) {
                    const unsigned priority = rank_priority ?
                        (row < entry.at("available").get<unsigned>() ? entry.at("dependency").at("raw").at(row).get<unsigned>() : 0) :
                        input.at("next_priority").at("raw").at(row).get<unsigned>();
                    if (source_b_flags & 1) {
                        const unsigned dependency = input.at("next_dependency_score").at("raw").at(row);
                        test_case.joint_future[row*16+12] = dependency;
                        test_case.joint_future[row*16+13] = dependency >> 8;
                    }
                    const unsigned service = input.at("next_service_count").at("raw").at(row);
                    const bool a8 = input.at("next_activation_bits").at("raw").at(row) == 8;
                    const unsigned position = config.next_block_start+row;
                    if (!test_case.rows[position].current) test_case.rows[position].a8 = a8;
                    test_case.joint_future[row*16] = priority; test_case.joint_future[row*16+1] = priority >> 8;
                    test_case.joint_future[row*16+2] = unsigned(input.at("next_unresolved").at("raw").at(row).get<bool>()) |
                        unsigned(input.at("next_tentative").at("raw").at(row).get<bool>()) << 1 | unsigned(a8) << 2;
                    for (unsigned byte = 0; byte < 4; ++byte) test_case.joint_future[row*16+4+byte] = service >> (byte*8);
                    if (input.contains("next_last_confidence")) {
                        const auto confidence = input.at("next_last_confidence").at("raw").at(row).get<int>();
                        if (confidence >= 0) {
                            test_case.joint_future[row*16+8] = confidence;
                            test_case.joint_future[row*16+9] = confidence >> 8;
                            test_case.joint_future[row*16+10] = 1;
                        }
                    }
                }
                uaps_result expected_result{};
                for (const auto position : expected.at("progress_local_positions").at("raw"))
                    expected_result.future_prediction_mask |= 1u << position.get<unsigned>();
                for (const auto position : expected.at("added_local_positions").at("raw")) {
                    expected_result.added_future_token_mask |= 1u << position.get<unsigned>();
                    test_case.expected.push_back(config.next_block_start+position.get<unsigned>());
                    if (precision) test_case.expected_bits[config.next_block_start+position.get<unsigned>()] =
                        input.at("next_activation_bits").at("raw").at(position.get<unsigned>());
                }
                expected_result.future_prediction_count = __builtin_popcount(expected_result.future_prediction_mask);
                expected_result.added_future_token_count = __builtin_popcount(expected_result.added_future_token_mask);
                const auto units = [&](const char* key) {
                    return expected.at(key).at("a4_token_count").get<unsigned>() +
                           2 * expected.at(key).at("a8_token_count").get<unsigned>();
                };
                expected_result.base_activation_slots = units("base_residency");
                expected_result.joint_activation_slots = units("joint_residency");
                expected_result.next_pass_activation_slots = units("next_step_verification_residency");
                const auto packed_result = pack_uaps_result(expected_result);
                test_case.joint_result.assign(packed_result.begin(), packed_result.end());
                if (entry.contains("retry_min_confidence_bf16")) {
                    rtl_joint_input request{};
                    request.base_token_count = base_token_count; request.future_token_count = config.future_token_count;
                    request.target_token_count = test_case.target; request.max_next_tokens = config.max_next_tokens;
                    request.next_block_start = config.next_block_start;
                    request.prediction_target = std::max(0, input.value("prediction_token_target", -1));
                    request.current_prediction_tokens = input.value("current_prediction_tokens", 0u);
                    request.retry_min_confidence = entry.at("retry_min_confidence_bf16");
                    for (unsigned i = 0; i < base_token_count; ++i) {
                        request.base_positions[i] = input.at("base_positions").at("raw").at(i);
                        request.base_bits[i] = input.at("base_activation_bits").at("raw").at(i);
                        request.forecast_bits[i] = input.at("next_step_base_activation_bits").at("raw").at(i);
                    }
                    std::array<std::uint8_t, 32> allowed{};
                    for (unsigned i = 0; i < config.future_token_count; ++i) {
                        request.future_bits[i] = input.at("next_activation_bits").at("raw").at(i);
                        request.unresolved[i] = input.at("next_unresolved").at("raw").at(i).get<bool>();
                        request.tentative[i] = input.at("next_tentative").at("raw").at(i).get<bool>();
                        request.priority[i] = input.at("next_priority").at("raw").at(i);
                        request.service_count[i] = input.at("next_service_count").at("raw").at(i);
                        request.future_admission_attempts[i] = input.at("future_admission_attempts").at("raw").at(i);
                        request.last_confidence[i] = input.at("next_last_confidence").at("raw").at(i).get<int>();
                        const int limit = input.at("max_future_admission_attempts");
                        allowed[i] = limit < 0 || request.future_admission_attempts[i] < unsigned(limit);
                    }
                    rtl_joint_selection selected{};
                    require(!rtl_joint_select_allowed(&request, allowed.data(), &selected), "C11 joint rejected "+test_case.name);
                    unsigned progress = 0, added = 0;
                    for (unsigned i = 0; i < selected.future_prediction_count; ++i) progress |= 1u << selected.progress[i];
                    for (unsigned i = 0; i < selected.added_future_token_count; ++i) added |= 1u << selected.added[i];
                    require(progress == expected_result.future_prediction_mask && added == expected_result.added_future_token_mask,
                        "C11 joint selection differs from generator: "+test_case.name);
                }
                const auto retry = input.value("future_admission_retry_min_confidence", 0.0);
                const int prediction_target = input.value("prediction_token_target", -1);
                const int attempt_limit = input.value("max_future_admission_attempts", -1);
                const unsigned min_reuse_score = entry.value("min_reuse_score_bf16", 0u);
                if (attempt_limit >= 0 || retry > 0 || prediction_target >= 0 || min_reuse_score || source_b_flags) {
                    uaps_attempt_config extension{};
                    const bool track = attempt_limit >= 0 || retry > 0;
                    extension.state_base = track ? 0xc000 : 0; extension.state_limit = track ? 0xc090 : 0;
                    extension.max_attempts = attempt_limit >= 0 ? attempt_limit : 0x7fffffff;
                    extension.retry_min_confidence_bf16 = entry.value("retry_min_confidence_bf16", 0u);
                    extension.prediction_target = prediction_target < 0 ? 0 : prediction_target;
                    extension.min_reuse_score_bf16 = min_reuse_score;
                    extension.source_b_flags = source_b_flags;
                    extension.block_step_index = input.value("step_index", 0u);
                    const auto suffix = pack_uaps_attempt_config(extension);
                    test_case.extra_memory[0x9040] = {suffix.begin(), suffix.end()};
                    test_case.joint_config[6] = 96;
                    if (track) {
                    auto& state = test_case.extra_memory[0xc000]; state.resize(144);
                    auto write32 = [](auto& bytes, unsigned offset, std::uint32_t value) {
                        for (unsigned byte = 0; byte < 4; ++byte) bytes.at(offset+byte) = value>>(byte*8);
                    };
                    write32(state, 0, 0x31425441); write32(state, 4, config.next_block_start); write32(state, 8, 98);
                    for (unsigned row = 0; row < config.future_token_count; ++row)
                        write32(state, 16+row*4, input.at("future_admission_attempts").at("raw").at(row));
                    test_case.attempts_expected = state;
                    write32(test_case.attempts_expected, 8, 99);
                    write32(test_case.attempts_expected, 12, expected_result.added_future_token_mask);
                    for (unsigned row = 0; row < config.future_token_count; ++row)
                        write32(test_case.attempts_expected, 16+row*4,
                            input.at("future_admission_attempts").at("raw").at(row).get<unsigned>() + ((expected_result.added_future_token_mask>>row)&1));
                    }
                }
                std::sort(test_case.expected.begin(), test_case.expected.end());
                result.push_back(std::move(test_case));
            }
        }
        return result;
    }
    Vattention_guided_token_selector_tb dut_;
    std::vector<std::uint8_t> last_attempt_state_;
    std::uint64_t cycles_ = 0;
    TestCase scout_test_case() {
        std::ifstream stream("cases/control/atse_cross_block_scout_reduction_order.json");
        json source; stream >> source;
        TestCase test_case{"actual_cuda_scout_to_selector", std::vector<Row>(17), 6, 0, {0,1,4,5,6,7}};
        test_case.expected_scores = source.at("records").at(1).at("expected").at("score_q8").at("raw").get<std::vector<unsigned>>();
        test_case.probabilities.resize(17*512);
        for (unsigned position = 0; position < 17; ++position) {
            auto& row = test_case.rows[position];
            row.current = position >= 5 && position <= 7;
            row.eligible = !row.current; row.eligible_order = position; row.score = 200;
            attention_dependency_job job{};
            job.source_base = 0x40000+position*512; job.source_limit = job.source_base+512;
            job.head_stride = 16; job.lane_mask = row.current ? 0 : 0x12; job.operation = 7;
            job.output_base = 0x10000+position*8; job.output_limit = job.output_base+8;
            const auto packed = pack_attention_dependency_job(job);
            test_case.jobs.insert(test_case.jobs.end(), packed.begin(), packed.end());
            for (unsigned head = 0; head < 32; ++head)
                for (unsigned key = 0; key < 8; ++key) {
                    const auto raw = source.at("probabilities").at("raw").at(0).at(head).at(position).at(key).get<std::uint32_t>();
                    require((raw & 65535) == 0, "scout input retained non-BF16 precision");
                    const auto offset = position*512+head*16+key*2;
                    test_case.probabilities[offset] = raw >> 16;
                    test_case.probabilities[offset+1] = raw >> 24;
                }
        }
        return test_case;
    }
    void edge() {
        dut_.clk = 1;
        dut_.eval();
        dut_.clk = 0;
        dut_.eval();
        ++cycles_;
    }
    void execute(const TestCase& test_case, unsigned fault, bool commit = false, bool prepare = false,
                 std::vector<std::uint8_t>* pending_output = nullptr, bool replace_preparation = false,
                 bool commit_metadata = true) {
        dut_.consume_source_a = 0;
        dut_.loaded_token_valid = 0; dut_.loaded_token_position = 0; dut_.loaded_token_kv_write = 0;
        dut_.start_valid = 1;
        dut_.start_prepare = prepare;
        dut_.start_closeout = test_case.closeout;
        dut_.start_relation_only = 0;
        dut_.start_relation_l31 = 0;
        dut_.start_descriptor_address = 0x8000;
        const bool joint = !test_case.joint_config.empty();
        const bool emit_metadata = (commit && commit_metadata) || joint ||
            test_case.publish_result || !test_case.budget_expected.empty();
        dut_.start_joint_descriptor_address = joint ? 0x9000 : 0;
        token_refresh_config fields{};
        fields.magic = 0x31465241;
        fields.version = 1;
        fields.bytes = TOKEN_REFRESH_CONFIG_BYTES;
        fields.sequence_length = test_case.rows.size();
        fields.target_token_count = test_case.target;
        fields.required_quota = test_case.quota;
        fields.table_base = 0x10000;
        fields.table_limit = 0x10000 + test_case.rows.size()*8;
        fields.flags = joint ? 6 : commit ? 1 | (emit_metadata ? 2 : 0) : emit_metadata ? 2 : 0;
        if (test_case.publish_result) fields.flags |= 64;
        if (test_case.qkvo_group) fields.flags |= 0x8000;
        if (test_case.paired_pending) fields.flags |= 128 | 4;
        if (test_case.state_table_update) fields.flags |= 4;
        if (test_case.allocate_precision)
            fields.flags |= 8 | (test_case.all_a8 ? 16 : 0) | (test_case.context_a8_token_count<<8);
        if (test_case.deep_precision) fields.flags |= 32 | (test_case.deep_a8_limit<<8);
        fields.metadata_base = 0x20000;
        fields.metadata_limit = fields.metadata_base+test_case.metadata_capacity;
        const auto result_address=fields.metadata_limit-16;
        fields.metadata_version = 77;
        fields.capture_index = 99;
        fields.head_stride = 262144;
        fields.scale_head_stride = 4096;
        fields.relation_job_count = test_case.jobs.size()/64;
        fields.relation_job_base = test_case.jobs.empty() ? 0 : test_case.relation_job_address;
        const bool shortlist = !test_case.shortlist_eligible.empty();
        const bool monitor = prepare && (!test_case.jobs.empty() || shortlist || test_case.capture_p8_only);
        fields.probability_configuration_offset = monitor ? (test_case.execution_extension ? 208 : 176) : 0;
        fields.pending_configuration_offset = test_case.extra_memory.count(0x8500) ? 0x500 : 0;
        attention_probability_config probability{};
        if (shortlist) probability.shortlist_configuration_base = 0x8600;
        probability.output_base = 0x80000; probability.output_limit = 0xc0000;
        probability.batch_stride_bytes = 128; probability.head_stride_bytes = 768;
        probability.round_stride_bytes = 24576; probability.layer_mask = 1;
        if (test_case.dependency_layers) { probability.capture_p8 = test_case.dependency_p8; probability.layer_mask = 3; }
        if (test_case.capture_p8_only) probability.capture_p8 = 1;
        probability.key_groups0 = 1; probability.query_end = test_case.rows.size();
        const auto probability_descriptor = pack_attention_probability_config(probability);
        auto descriptor = pack_token_refresh_config(fields);
        if (test_case.execution_extension) descriptor[6] = 208;
        for (unsigned region = 0; region < 6; ++region) {
            const auto base = fault == 10 && region == 3 ? kCacheBases[0]+16 : kCacheBases[region];
            const auto limit = base + kCacheSizes[region%3];
            for (unsigned byte = 0; byte < 8; ++byte) {
                descriptor[TOKEN_REFRESH_CONFIG_SOURCE_K_BASE_OFFSET+region*8+byte] = base >> (byte*8);
                descriptor[TOKEN_REFRESH_CONFIG_SOURCE_K_LIMIT_OFFSET+region*8+byte] = limit >> (byte*8);
            }
        }
        dut_.done_ready = 0;
        dut_.dma_read_valid = 0;
        dut_.dma_error = 0;
        dut_.abort_request = 0;
        dut_.dma_abort_ack = 0;
        dut_.dma_write_done = 0;
        dut_.dma_write_error = 0;
        dut_.eval();
        require(dut_.start_ready, "selector not ready for " + test_case.name);
        edge();
        dut_.start_valid = 0;
        bool stream = false, held_output = false, held_read = false, injected = false, aborted = false;
        unsigned index = 0, scans = 0, held_position = 0, held_last = 0;
        unsigned drain_wait = 0;
        bool table_read = true, descriptor_read = false, metadata_read = false, metadata_write = false, writing = false;
        bool relation_read = false, relation_write = false, joint_read = false, joint_result_write = false;
        bool extra_read = false, extra_write = false;
        bool state_table_read = false, state_table_write = false;
        unsigned extra_tag = 0;
        auto extra_memory = test_case.extra_memory;
        unsigned alternate_job_reads = 0;
        if (test_case.execution_extension) {
            auto& suffix = extra_memory[0x8000+176]; suffix.resize(32);
            const auto set = [&](unsigned offset, unsigned width, std::uint64_t value) {
                for (unsigned b=0;b<width;++b) suffix[offset+b] = value >> (8*b);
            };
            set(0,8,0x31000); set(8,4,test_case.jobs.size()/64);
            set(12,4,3); set(16,4,6); set(20,2,0); set(24,4,98);
            extra_memory[0x31000] = test_case.jobs;
        }
        if (test_case.publish_result) extra_memory[result_address] = std::vector<std::uint8_t>(16,0xa5);
        auto probabilities = test_case.probabilities;
        unsigned completed_dependency_layers = 0, dependency_writes = 0;
        if (fault == 17) extra_memory.at(0x40000).at(20) = 100;
        if (fault == 18) {
            auto& records = extra_memory.at(0x40000);
            require(records.size() >= 64, "duplicate state test needs two entries");
            std::copy_n(records.begin(), 32, records.begin()+32);
        }
        auto extra_region = [&](std::uint64_t address, unsigned bytes) {
            auto found = extra_memory.upper_bound(address);
            if (found != extra_memory.begin()) {
                --found;
                if (address >= found->first && address+bytes <= found->first+found->second.size()) return found;
            }
            return extra_memory.end();
        };
        std::uint64_t read_address = 0, write_address = 0, committed_bytes = 0;
        unsigned read_size = 0, write_size = 0, write_offset = 0, write_response_delay = 0;
        std::array<std::vector<bool>, 3> written;
        if (commit) for (unsigned kind = 0; kind < 3; ++kind) written[kind].resize(kCacheSizes[kind]);
        std::vector<unsigned> actual;
        std::vector<std::uint8_t> metadata;
        std::vector<std::uint8_t> joint_result;
        bool preparing = prepare, abort_after_add = false;
        std::vector<std::uint8_t> table_bytes(test_case.rows.size()*8);
        for (unsigned position = 0; position < test_case.rows.size(); ++position)
            for (unsigned byte = 0; byte < 8; ++byte)
                table_bytes[position*8+byte] = encode(test_case.rows[position]) >> (byte*8);
        auto table_record = [&](unsigned position) {
            if (!test_case.jobs.empty() || shortlist || test_case.state_table_update) {
                std::uint64_t record = 0;
                for (unsigned byte = 0; byte < 8; ++byte) record |= std::uint64_t(table_bytes.at(position*8+byte)) << (byte*8);
                return record;
            }
            auto row = test_case.rows.at(position);
            if (preparing) row.score = (position*77)%256;
            if (fault == 8) row.source_index = 2048;
            return encode(row);
        };
        unsigned state_begin=0,state_count=0;
        if (test_case.state_table_update) {
            const auto& region=test_case.extra_memory.at(0x60000);
            const auto& config=test_case.extra_memory.at(0x8500);
            state_begin=unsigned(region[2]) | unsigned(region[3])<<8;
            state_count=unsigned(config[10]) | unsigned(config[11])<<8;
        }
        const auto started = cycles_;
        unsigned quiet_cycles = 0;
        // Serial top-K and final selection have bounded O(S*K) work. Monitor
        // accepted SRAM/DDR/arithmetic events rather than killing useful scans.
        const auto safety_cycles = 1000000ull + 8ull * test_case.rows.size() * 800;
        for (unsigned step = 0; step < safety_cycles && !dut_.done_valid && !aborted; ++step) {
            if (abort_after_add) dut_.abort_request = 1;
            dut_.dma_request_ready = fault == 5 ? injected : step % 5 != 0;
            dut_.selected_ready = step % 7 >= 3;
            dut_.dma_write_request_ready = !writing && write_response_delay == 0 && step%5 != 0;
            dut_.dma_write_ready = step%11 >= 3;
            dut_.dma_write_done = write_response_delay && --write_response_delay == 0;
            if (fault == 16 && shortlist && dut_.dma_write_done && !injected) {
                dut_.dma_write_error = 1; injected = true;
            }
            if (fault == 19 && state_table_write && dut_.dma_write_done && !injected) {
                dut_.dma_write_error = 1; injected = true;
            }
            if (fault == 21 && extra_write && write_address == 0x50000 && write_size > 16 &&
                    dut_.dma_write_done && !injected) {
                dut_.dma_write_error = 1; injected = true;
            }
            if (fault == 25 && extra_write && write_address == result_address && dut_.dma_write_done && !injected) {
                dut_.dma_write_error = 1; dut_.dma_write_done = 0; injected = true;
            }
            if (fault == 22 && extra_write && write_address == 0xc000 && dut_.dma_write_done && !injected) {
                dut_.dma_write_error = 1; injected = true;
            }
            dut_.dma_read_valid = stream && (held_read || step % 9 != 0);
            if (stream && extra_read) {
                const auto region = extra_region(read_address, read_size);
                require(region != extra_memory.end(), "pending read range disappeared");
                for (unsigned word = 0; word < 4; ++word) {
                    dut_.dma_read_data[word] = 0;
                    for (unsigned byte = 0; byte < 4; ++byte)
                        dut_.dma_read_data[word] |= std::uint32_t(region->second.at(read_address-region->first+index+word*4+byte))<<(byte*8);
                }
                dut_.dma_read_tag = extra_tag; dut_.dma_read_last = index+16 == read_size; dut_.dma_read_byte_enable = 65535;
            } else if (stream && joint_read) {
                const auto& source = read_address == 0x9000 ? test_case.joint_config :
                    read_address == 0xa000 ? test_case.joint_base : test_case.joint_future;
                for (unsigned word = 0; word < 4; ++word) {
                    dut_.dma_read_data[word] = 0;
                    for (unsigned byte = 0; byte < 4; ++byte)
                        dut_.dma_read_data[word] |= std::uint32_t(source.at(index+word*4+byte)) << (byte*8);
                }
                dut_.dma_read_tag = 0xb1; dut_.dma_read_last = index+16 == read_size; dut_.dma_read_byte_enable = 65535;
                if (read_address == 0xb000 && index == 0 && !injected) {
                    if (fault == 11) { dut_.dma_error = 1; dut_.dma_read_valid = 0; stream = false; injected = true; }
                    if (fault == 12) dut_.dma_read_data[0] = (dut_.dma_read_data[0] & 0xffff0000u) | 0x7fc0;
                }
            } else if (stream && relation_read) {
                const unsigned bytes = std::min(16u, read_size-index);
                for (unsigned word = 0; word < 4; ++word) dut_.dma_read_data[word] = 0;
                for (unsigned byte = 0; byte < bytes; ++byte) {
                    const auto address = read_address+index+byte;
                    const auto value = address >= 0x40000 ? probabilities.at(address-0x40000) :
                        address >= 0x30000 ? test_case.jobs.at(address-0x30000) : table_bytes.at(address-0x10000);
                    dut_.dma_read_data[byte/4] |= std::uint32_t(value) << ((byte%4)*8);
                }
                dut_.dma_read_tag = 0xb7; dut_.dma_read_last = index+bytes == read_size;
                dut_.dma_read_byte_enable = (1u << bytes)-1;
            } else if (stream && state_table_read) {
                const auto bytes = std::min(16u, read_size-index);
                for (unsigned word = 0; word < 4; ++word) dut_.dma_read_data[word] = 0;
                for (unsigned byte = 0; byte < bytes; ++byte)
                    dut_.dma_read_data[byte/4] |= std::uint32_t(table_bytes.at(read_address-0x10000+index+byte))<<((byte%4)*8);
                dut_.dma_read_tag = 0xba; dut_.dma_read_last = index+bytes == read_size;
                dut_.dma_read_byte_enable = (1u<<bytes)-1;
            } else if (stream && descriptor_read) {
                for (unsigned word = 0; word < 4; ++word) {
                    std::uint32_t value = 0;
                    for (unsigned byte = 0; byte < 4; ++byte) value |= std::uint32_t(
                        read_address == 0x8000 ? descriptor.at(index+word*4+byte) :
                        probability_descriptor.at(index+word*4+byte)) << (byte*8);
                    dut_.dma_read_data[word] = value;
                }
                dut_.dma_read_tag = 0xb1;
                dut_.dma_read_last = index+16 == read_size;
                dut_.dma_read_byte_enable = 65535;
            } else if (stream && table_read) {
                const auto low = table_record(index);
                const auto high = index + 1 < test_case.rows.size() ? table_record(index+1) : 0;
                dut_.dma_read_data[0] = low;
                dut_.dma_read_data[1] = low >> 32;
                dut_.dma_read_data[2] = high;
                dut_.dma_read_data[3] = high >> 32;
                dut_.dma_read_tag = 0xb1;
                dut_.dma_read_last = index + 2 >= test_case.rows.size();
                dut_.dma_read_byte_enable = index + 1 < test_case.rows.size() ? 0xffff : 0xff;
                if (!injected && scans == 1 && index == 8 && (fault == 3 || fault == 4)) {
                    dut_.dma_error = fault == 3;
                    dut_.abort_request = fault == 4;
                    injected = true;
                    if (fault == 3) { stream = false; dut_.dma_read_valid = 0; }
                }
            } else if (stream && metadata_read) {
                const auto record = table_record(unsigned((read_address-0x10000)/8));
                dut_.dma_read_data[0] = record;
                dut_.dma_read_data[1] = record >> 32;
                dut_.dma_read_data[2] = 0;
                dut_.dma_read_data[3] = 0;
                dut_.dma_read_tag = extra_tag;
                dut_.dma_read_last = 1;
                dut_.dma_read_byte_enable = 255;
                if (shortlist && extra_tag == 0xbc && !injected && (fault == 14 || fault == 15)) {
                    dut_.dma_error = fault == 14; dut_.abort_request = fault == 15;
                    dut_.dma_read_valid = 0; injected = true;
                    if (fault == 14) stream = false;
                }
            } else if (stream) {
                const auto bytes = std::min(16u, read_size-index);
                for (unsigned word = 0; word < 4; ++word) {
                    std::uint32_t value = 0;
                    for (unsigned byte = 0; byte < 4; ++byte)
                        if (word*4+byte < bytes) value |= std::uint32_t(cache_pattern(read_address+index+word*4+byte)) << (byte*8);
                    dut_.dma_read_data[word] = value;
                }
                dut_.dma_read_tag = 0xb2;
                dut_.dma_read_last = index+bytes == read_size;
                dut_.dma_read_byte_enable = (1u << bytes)-1;
            }
            dut_.eval();
            if (fault == 5 && dut_.dma_request_valid && !injected) {
                dut_.abort_request = 1;
                injected = true;
                dut_.eval();
            }
            if (dut_.abort_request && ++drain_wait == 5 && stream) {
                stream = false;
                dut_.dma_read_valid = 0;
                dut_.dma_abort_ack = 1;
                dut_.eval();
            }
            if (dut_.dma_request_valid && dut_.dma_request_ready) {
                require(!stream, "overlapping refresh DMA");
                if (dut_.dma_request_address == 0x31000) ++alternate_job_reads;
                descriptor_read = dut_.dma_request_address == 0x8000 ||
                    dut_.dma_request_address == 0x8000 + fields.probability_configuration_offset;
                joint_read = dut_.dma_request_address == 0x9000 || dut_.dma_request_address == 0xa000 || dut_.dma_request_address == 0xb000;
                extra_read = extra_region(dut_.dma_request_address, dut_.dma_request_bytes) != extra_memory.end();
                extra_tag = dut_.dma_request_tag;
                state_table_read = test_case.state_table_update && extra_tag == 0xba &&
                    dut_.dma_request_address >= 0x10000 && dut_.dma_request_address < 0x10000+table_bytes.size();
                metadata_read = dut_.dma_request_tag == 0xbc && dut_.dma_request_bytes == 8 || dut_.dma_request_tag == 0xb4 ||
                    (joint && !test_case.budget_expected.empty() && dut_.dma_request_tag == 0xb1 &&
                     dut_.dma_request_bytes == 8 && dut_.dma_request_address >= 0x10000 &&
                     dut_.dma_request_address < 0x10000+test_case.rows.size()*8);
                table_read = dut_.dma_request_tag == 0xb1 && !descriptor_read && !joint_read && !extra_read && !metadata_read;
                relation_read = dut_.dma_request_tag == 0xb7;
                if (extra_read) require((preparing && test_case.execution_extension && extra_tag == 0xb1 &&
                    dut_.dma_request_address == 0x80b0 && dut_.dma_request_bytes == 32) || (preparing && extra_tag == 0xb1 && dut_.dma_request_address == 0x8500 && dut_.dma_request_bytes == 80) ||
                    (preparing && test_case.initial_selected_consumed && extra_tag == 0xb1 && dut_.dma_request_address == 0x8550 && dut_.dma_request_bytes == 16) ||
                    (shortlist && preparing && (extra_tag == 0xb1 || extra_tag == 0xbc)) ||
                    (!preparing && (extra_tag == 0xb1 || extra_tag == 0xba ||
                        (test_case.state_table_update && extra_tag == 0xb4 && dut_.dma_request_address == 0x50000 &&
                            dut_.dma_request_bytes == test_case.pending_expected.size()) ||
                        (!test_case.jobs.empty() && extra_tag == 0xb7))), "pending ran before forward or used an invalid DMA tag");
                else if (state_table_read) require(dut_.dma_request_address == 0x10000+state_begin*8 && dut_.dma_request_bytes == state_count*8,
                    "state table was not streamed as one region");
                else if (descriptor_read) require(dut_.dma_request_bytes ==
                    (dut_.dma_request_address == 0x8000 ? descriptor.size() : probability_descriptor.size()),
                    "wrong refresh/probability descriptor length");
                else if (table_read) require(dut_.dma_request_bytes == test_case.rows.size()*8 &&
                    dut_.dma_request_address == 0x10000, "truncated table DMA");
                else if (joint_read) require(joint && !preparing, "joint inputs read before the forward completed");
                else if (metadata_read) require((emit_metadata || shortlist) && dut_.dma_request_bytes == 8, "invalid metadata row read");
                else if (relation_read) require(!test_case.jobs.empty() && !preparing, "relation ran before forward");
                else require(commit && dut_.dma_request_tag == 0xb2 && metadata.size() == dut_.metadata_bytes,
                    "unexpected refresh DMA source or metadata not completed");
                require(fault != 5, "revoked read request was accepted");
                read_address = dut_.dma_request_address;
                read_size = dut_.dma_request_bytes;
                stream = true;
                index = 0;
                if (table_read) ++scans;
            } else if (dut_.dma_read_valid && dut_.dma_read_ready) {
                index += table_read ? 2 : std::min(16u, read_size-index);
                if (index >= (table_read ? test_case.rows.size() : read_size)) stream = false;
            }
            if (dut_.dma_write_request_valid && dut_.dma_write_request_ready) {
                state_table_write = test_case.state_table_update && dut_.dma_write_request_tag == 0xbb &&
                    dut_.dma_write_request_address >= 0x10000 && dut_.dma_write_request_address < 0x10000+table_bytes.size();
                extra_write = (dut_.dma_write_request_tag == 0xbb && !state_table_write) || ((dut_.dma_write_request_tag == 0xb5 ||
                    (!test_case.jobs.empty() && dut_.dma_write_request_tag == 0xb8)) &&
                    extra_region(dut_.dma_write_request_address, dut_.dma_write_request_bytes) != extra_memory.end());
                if (test_case.dependency_layers && dut_.dma_write_request_tag == 0xb8) ++dependency_writes;
                metadata_write = dut_.dma_write_request_tag == 0xb5 && !extra_write;
                joint_result_write = metadata_write && dut_.dma_write_request_address == 0x28000;
                relation_write = dut_.dma_write_request_tag == 0xb8 || dut_.dma_write_request_tag == 0xbd || state_table_write;
                if (state_table_write) require(dut_.dma_write_request_address == 0x10000+state_begin*8 && dut_.dma_write_request_bytes == state_count*8,
                    "state table write was not streamed as one region");
                if (extra_write) require(extra_region(dut_.dma_write_request_address, dut_.dma_write_request_bytes) != extra_memory.end(), "pending write escaped DDR region");
                if (preparing && dut_.dma_write_request_tag == 0xbb)
                    require(test_case.advance_dependency_block && dut_.dma_write_request_address == 0x30000 &&
                        dut_.dma_write_request_bytes == test_case.rows.size()*64, "unexpected PREPARE dependency clear");
                require(!writing && (extra_write || relation_write || (emit_metadata && metadata_write) ||
                    (commit && dut_.dma_write_request_tag == 0xb3)), "unexpected refresh write");
                if (metadata_write) require(joint_result_write ? joint && dut_.dma_write_request_bytes == 16 &&
                    metadata.size() == dut_.metadata_bytes : dut_.dma_write_request_address == 0x20000+metadata.size(),
                    "metadata/joint result write order or address mismatch");
                write_address = dut_.dma_write_request_address;
                write_size = dut_.dma_write_request_bytes;
                write_offset = 0;
                writing = true;
            }
            if (dut_.dma_write_valid && dut_.dma_write_ready) {
                require(writing, "refresh data without write request");
                const auto bytes = std::min(16u, write_size-write_offset);
                require(dut_.dma_write_byte_enable == (1u << bytes)-1 &&
                    bool(dut_.dma_write_last) == (write_offset+bytes == write_size), "refresh write tail mismatch");
                for (unsigned byte = 0; byte < bytes; ++byte) {
                    if (extra_write) {
                        auto region = extra_region(write_address, write_size);
                        region->second.at(write_address-region->first+write_offset+byte) = dut_.dma_write_data[byte/4]>>((byte%4)*8);
                        continue;
                    }
                    if (relation_write) {
                        const auto offset = write_address+write_offset+byte-0x10000;
                        table_bytes.at(offset) = dut_.dma_write_data[byte/4] >> ((byte%4)*8);
                        continue;
                    }
                    if (metadata_write) {
                        (joint_result_write ? joint_result : metadata).push_back(dut_.dma_write_data[byte/4] >> ((byte%4)*8));
                        continue;
                    }
                    const auto address = write_address+write_offset+byte;
                    unsigned kind = 0;
                    for (; kind < 3; ++kind)
                        if (address >= kCacheBases[kind+3] && address-kCacheBases[kind+3] < kCacheSizes[kind]) break;
                    require(kind < 3, "refresh write escaped persistent cache");
                    const auto offset = address-kCacheBases[kind+3];
                    const auto head_offset = offset % (kind == 2 ? 4096 : 262144);
                    const auto position = kind == 0 ? head_offset/128 : kind == 1 ? (head_offset%16384)/8 : head_offset/2;
                    require(std::binary_search(test_case.expected.begin(), test_case.expected.end(), position),
                        "RTL selector committed an algorithm-unselected row");
                    require(!written[kind][offset], "refresh committed a duplicate cache byte");
                    written[kind][offset] = true;
                    require(((dut_.dma_write_data[byte/4] >> ((byte%4)*8)) & 255) ==
                        cache_pattern(kCacheBases[kind]+offset), "refresh committed wrong raw cache data");
                }
                write_offset += bytes;
                if (!metadata_write && !relation_write && !extra_write) committed_bytes += bytes;
                if (write_offset == write_size) { writing = false; write_response_delay = 4; }
            }
            if (held_output)
                require(dut_.selected_valid && dut_.selected_position == held_position &&
                    dut_.selected_last == held_last, "selector changed stalled position");
            held_output = dut_.selected_valid && !dut_.selected_ready;
            held_position = dut_.selected_position;
            held_last = dut_.selected_last;
            if (dut_.selected_valid && dut_.selected_ready) {
                actual.push_back(dut_.selected_position);
                require(bool(dut_.selected_last) == (actual.size() == test_case.expected.size()), "selector output last mismatch");
            }
            held_read = dut_.dma_read_valid && !dut_.dma_read_ready;
            if (fault == 13 && dut_.joint_add_observe_valid) abort_after_add = true;
            const bool progress = dut_.scratch_observe_valid || dut_.joint_add_observe_valid ||
                (dut_.dma_request_valid && dut_.dma_request_ready) ||
                (dut_.dma_read_valid && dut_.dma_read_ready) ||
                (dut_.dma_write_valid && dut_.dma_write_ready) || dut_.dma_write_done ||
                (dut_.selected_valid && dut_.selected_ready);
            quiet_cycles = progress ? 0 : quiet_cycles + 1;
            require(quiet_cycles < 20000, test_case.name + ": no accepted event; selector state=" +
                std::to_string(dut_.selector_observe_state) + " read_active=" + std::to_string(stream) +
                " write_active=" + std::to_string(writing) + " scans=" + std::to_string(scans));
            edge();
            dut_.dma_error = 0;
            dut_.dma_write_error = 0;
            if (dut_.abort_ack) aborted = true;
            if (dut_.done_valid && preparing && !dut_.error) {
                require(!dut_.error && dut_.prepared_for_forward && actual.empty() && committed_bytes == 0 && metadata.empty(),
                    "refresh preparation published selection/cache before forward");
                if (test_case.execution_extension)
                    require(dut_.source_a_valid && dut_.source_a_mask == 6 &&
                            dut_.source_a_block_start == 0 && dut_.source_a_capture_index == 98,
                            "PREPARE did not restore the captured Source A state");
                require(bool(dut_.probability_enable) == monitor, "probability enable leaked across operator_executions");
                require(dut_.relation_layer_mask == (test_case.dependency_layers ? 3u : 0u),
                    "layer callback enabled without P8 dependency jobs or retained across operator_executions");
                if (monitor) require(dut_.probability_output_base == probability.output_base &&
                    dut_.probability_query_end == probability.query_end, "probability descriptor was not retained for forward");
                if (shortlist) {
                    require(extra_memory.at(0x70000) == test_case.extra_memory.at(0x70000), "shortlist modified persistent pending state");
                    for (unsigned position = 0; position < test_case.rows.size(); ++position) {
                        std::uint64_t actual_record = 0;
                        for (unsigned byte = 0; byte < 8; ++byte) actual_record |= std::uint64_t(table_bytes[position*8+byte]) << (byte*8);
                        const bool eligible = std::binary_search(test_case.shortlist_eligible.begin(), test_case.shortlist_eligible.end(), position);
                        const unsigned order = test_case.shortlist_orders.empty() ? position : test_case.shortlist_orders.at(position);
                        auto expected_record = (encode(test_case.rows[position]) & ~(0x7ffull << 12)) |
                            (std::uint64_t(order) << 12) | (std::uint64_t(eligible) << 9);
                        if (!test_case.shortlist_mandatory.empty()) {
                            expected_record &= ~(1ull<<10);
                            expected_record |= std::uint64_t(std::binary_search(test_case.shortlist_mandatory.begin(),
                                test_case.shortlist_mandatory.end(), position))<<10;
                        }
                        require(actual_record == expected_record, "previous shortlist differs at row " + std::to_string(position));
                    }
                }
                if (test_case.advance_dependency_block) {
                    const auto& cleared = extra_memory.at(0x30000);
                    require(std::all_of(cleared.begin(), cleared.end(), [](auto byte) { return byte == 0; }),
                        "PREPARE retained stale dependency rows");
                    require(extra_memory.at(0x50000) == test_case.extra_memory.at(0x50000),
                        "PREPARE consumed or merged pending scores before boundary selection");
                }
                dut_.done_ready = 1; edge(); dut_.done_ready = 0;
                preparing = false;
                dut_.dma_read_valid = 0;
                for (unsigned idle = 0; idle < 5; ++idle) {
                    require(!dut_.dma_request_valid && !dut_.dma_write_request_valid && !dut_.selected_valid,
                        "prepared refresh issued work during forward");
                    edge();
                }
                if (test_case.initial_selected_consumed) {
                    for (unsigned pos=test_case.consume_current_begin;pos<test_case.consume_current_begin+32;++pos) {
                        dut_.loaded_token_valid=1; dut_.loaded_token_position=pos; dut_.loaded_token_kv_write=1; edge();
                    }
                    for (unsigned repeat=0;repeat<2;++repeat) for (const auto pos:test_case.loaded_kv_positions) {
                        dut_.loaded_token_valid=1; dut_.loaded_token_position=pos; dut_.loaded_token_kv_write=0; edge();
                    }
                    dut_.loaded_token_valid=0; dut_.loaded_token_kv_write=0;
                }
                if (test_case.live_consumed) {
                    // Disabled K/V writes must not consume pending; repetitions
                    // across layer metadata loads must be idempotent.
                    for (unsigned pos=0;pos<test_case.rows.size();++pos) {
                        dut_.loaded_token_valid=1;dut_.loaded_token_position=pos;dut_.loaded_token_kv_write=0;edge();
                    }
                    for (unsigned repeat=0;repeat<2;++repeat) for (const auto pos:test_case.loaded_kv_positions) {
                        dut_.loaded_token_valid=1;dut_.loaded_token_position=pos;dut_.loaded_token_kv_write=1;edge();
                    }
                    dut_.loaded_token_valid=0;dut_.loaded_token_kv_write=0;
                }
                require(dut_.start_ready, "prepared refresh did not accept forward completion");
                if (replace_preparation) {
                    require(!stream && !writing && write_response_delay == 0,
                        "new preparation reached a live transaction");
                    replace_preparation = false; preparing = true;
                    dut_.start_prepare = 1;
                    dut_.start_valid = 1; edge(); dut_.start_valid = 0;
                    continue;
                }
                dut_.start_prepare = 0;
                dut_.start_relation_only = test_case.dependency_layers != 0;
                dut_.start_valid = 1; edge(); dut_.start_valid = 0;
            } else if (dut_.done_valid && test_case.dependency_layers &&
                       completed_dependency_layers < test_case.dependency_layers && !dut_.error) {
                require(dut_.prepared_for_forward && actual.empty() && !writing && write_response_delay == 0,
                    "layer update published selection or completed before draining");
                const std::array<unsigned, 8> expected_codes{0, 0x3f00, 0x3e80, 0x3f00, 0x3b80, 0x3f00, 0x3c40, 0x3f00};
                const auto& saved = extra_memory.at(0x70000);
                for (unsigned word = 0; word < 24; ++word) {
                    const unsigned raw = saved[word*2] | unsigned(saved[word*2+1]) << 8;
                    require(raw == (word >= 8 && word < 16 ? expected_codes[word-8] : 0x3f00),
                        "layer replace/max changed an unexecuted row or inactive lane");
                }
                ++completed_dependency_layers;
                dut_.done_ready = 1; edge(); dut_.done_ready = 0;
                dut_.dma_read_valid = 0;
                for (unsigned idle = 0; idle < 5; ++idle) {
                    require(!dut_.dma_request_valid && !dut_.dma_write_request_valid && !dut_.selected_valid,
                        "layer reduction did not return to forward wait");
                    edge();
                }
                // The second layer has lower probabilities; its maximum must
                // retain the first layer, rather than restore pre-forward data.
                if (test_case.dependency_p8)
                    for (unsigned head = 0; head < 32; ++head) probabilities[head*16+8] = 0;
                else std::fill(probabilities.begin(), probabilities.end(), 0);
                dut_.start_relation_l31 = test_case.execution_extension && completed_dependency_layers == 1;
                dut_.start_relation_only = completed_dependency_layers < test_case.dependency_layers;
                dut_.start_valid = 1; edge(); dut_.start_valid = 0;
            }
        }
        if (fault == 4 || fault == 5 || fault == 13 || fault == 15) {
            require(aborted && !dut_.done_valid, "abort did not drain without successful completion");
            dut_.abort_request = 0;
            dut_.dma_abort_ack = 0;
            edge();
        } else {
            require(dut_.done_valid, "selector did not complete " + test_case.name);
            require(bool(dut_.error) == (fault != 0), "unexpected selector error " + std::to_string(dut_.error_id));
            if (!fault) require(actual == test_case.expected, "selected deep positions differ from algorithm: " + test_case.name + " expected=" + json(test_case.expected).dump() + " actual=" + json(actual).dump());
            if (!fault && !test_case.expected_scores.empty())
                for (unsigned position = 0; position < test_case.rows.size(); ++position) {
                    require(table_bytes[position*8] == test_case.expected_scores[position], "scout score differs from actual CUDA test_case");
                    for (unsigned byte = 1; byte < 8; ++byte)
                        require(table_bytes[position*8+byte] == std::uint8_t(encode(test_case.rows[position]) >> (byte*8)),
                            "relation computation changed token metadata");
                }
            else require(dut_.error_id == (fault == 8 ? 0x26u : fault == 10 ? 0x12u :
                fault == 11 ? 3u : fault == 12 ? 8u : fault == 14 ? 0x53u : fault == 16 ? 0x54u :
                fault == 17 || fault == 18 ? 0x46u : fault == 19 ? 0x44u : fault == 20 ? 8u : fault == 21 || fault == 22 || fault == 25 ? 0x25u :
                fault == 27 ? 0x23u : fault == 23 ? 0x55u : fault == 24 ? 0x51u : fault), "wrong first error classification");
            dut_.done_ready = 1;
            edge();
            dut_.done_ready = 0;
            dut_.dma_abort_ack = 0;
            if (!fault) {
                require(bool(dut_.source_a_valid) == (joint && !test_case.closeout), "Source A result valid disagrees with completed joint selection");
                if (joint && !test_case.closeout) {
                    const auto word = [&](unsigned at) {
                        unsigned value = 0;
                        for (unsigned byte = 0; byte < 4; ++byte) value |= unsigned(test_case.joint_result.at(at+byte))<<(byte*8);
                        return value;
                    };
                    require(dut_.source_a_mask == (word(0)&~word(4)) && dut_.source_a_capture_index == 99 &&
                        dut_.source_a_block_start == (unsigned(test_case.joint_config[12]) | unsigned(test_case.joint_config[13])<<8),
                        "published Source A differs from independent joint result");
                    dut_.consume_source_a = 1; edge(); dut_.consume_source_a = 0;
                    require(!dut_.source_a_valid, "consumed Source A result remained valid");
                }
            }
        }
        if (fault && test_case.paired_pending)
            require(actual.empty() && metadata.empty(), "failed paired pending published next selection or metadata");
        if (!fault && test_case.paired_pending) {
            const auto& actual_pending=extra_memory.at(0xb0000);
            for (unsigned token=0;token<actual_pending.size()/16;++token)
                for (unsigned byte=0;byte<10;++byte)
                    require(actual_pending[token*16+byte]==test_case.cross_pending_expected.at(token*16+byte),
                        test_case.name+": cross pending mismatch token="+std::to_string(token)+" byte="+std::to_string(byte));
        }
        require(!writing && write_response_delay == 0, "refresh completed before cache write response");
        if (fault == 22) require(injected, "attempt-state write error was not exercised");
        if (test_case.state_table_update && fault && fault != 21)
            require(actual.empty() && metadata.empty(), "failed state table update published next metadata");
        if (fault == 21)
            require(extra_memory.at(0x60000) == test_case.extra_memory.at(0x60000),
                "failed consumed-marker write published the next regular budget");
        if (test_case.execution_extension) require(alternate_job_reads == 1, "L31 did not read its separate relation job list");
        if (test_case.dependency_layers)
            require(completed_dependency_layers == test_case.dependency_layers && dependency_writes == test_case.dependency_layers,
                "final forward completion repeated or omitted a layer dependency update");
        if (!fault && !test_case.dependency_expected.empty())
            require(extra_memory.at(0x30000) == test_case.dependency_expected,
                test_case.name+": relation-to-pending saved dependency differs from algorithm");
        if (!fault && !test_case.expected_table.empty())
            require(table_bytes == test_case.expected_table, "published state row-table raw mismatch");
        if (!fault && !test_case.pending_expected.empty()) {
            const auto& pending = extra_memory.at(0x50000);
            for (unsigned row = 0; row < test_case.pending_expected.size()/16; ++row)
                for (unsigned byte = 0; byte < 11; ++byte)
                    require(pending.at(row*16+byte) == test_case.pending_expected.at(row*16+byte),
                        test_case.name+": parent pending raw mismatch row="+std::to_string(row)+" byte="+std::to_string(byte));
        }
        if (!fault && !test_case.budget_expected.empty())
            require(extra_memory.at(0x60000) == test_case.budget_expected, "regular budget state differs from algorithm");
        if (!fault && !test_case.attempts_expected.empty()) {
            last_attempt_state_ = extra_memory.at(0xc000);
            require(last_attempt_state_ == test_case.attempts_expected, "new future-token admission persistent state differs: "+test_case.name);
        }
        if (emit_metadata && !fault) {
            require(committed_bytes == (commit ? test_case.expected.size()*8256ull : 0), "refresh selected cache coverage incomplete");
            for (unsigned i=0;i<test_case.joint_result.size();++i)
                require(i<joint_result.size() && joint_result[i]==test_case.joint_result[i],
                    test_case.name+": joint result byte="+std::to_string(i)+" actual="+
                    std::to_string(i<joint_result.size()?joint_result[i]:999)+" expected="+std::to_string(test_case.joint_result[i]));
            require(joint_result.size()==test_case.joint_result.size(), "joint result byte count");
            require(metadata.size() == dut_.metadata_bytes, "refresh metadata byte count mismatch");
            if (test_case.publish_result) {
                const auto& result=extra_memory.at(result_address);
                const auto u32=[&](unsigned o) { return unsigned(result[o]) | unsigned(result[o+1])<<8 |
                    unsigned(result[o+2])<<16 | unsigned(result[o+3])<<24; };
                require((u32(0)&65535)==test_case.expected.size() && (u32(0)>>16)==dut_.metadata_rounds &&
                    u32(4)==metadata.size() && u32(8)==99 && u32(12)==77, "published metadata result differs from actual stream");
            }
            const auto read16 = [&](unsigned offset) { return unsigned(metadata.at(offset)) | unsigned(metadata.at(offset+1))<<8; };
            std::vector<unsigned> metadata_positions;
            unsigned offset = 0, ordinal = 0, slots = 0;
            const auto expected_a8 = [&](unsigned position) {
                return test_case.expected_bits.empty() ? test_case.rows[position].a8 : test_case.expected_bits.at(position) == 8;
            };
            for (auto position : test_case.expected) slots += expected_a8(position) ? 2 : 1;
            const auto rounds = std::max((test_case.expected.size()+47)/48, std::size_t((slots+63)/64));
            require(dut_.metadata_rounds == rounds, "refresh metadata has an unnecessary round");
            unsigned group_rounds=0, group_compute=0, group_slots=0;
            for (unsigned round = 0; round < rounds; ++round) {
                const unsigned rows = metadata.at(offset);
                require(read16(offset+4) == round && read16(offset+6) == ordinal, "refresh round identity mismatch");
                const auto inverse = offset+read16(offset+12);
                unsigned round_slots = 0;
                for (unsigned physical = 0; physical < rows; ++physical) {
                    const unsigned entry = offset+32+physical*16, position = read16(entry+4);
                    require(std::binary_search(test_case.expected.begin(), test_case.expected.end(), position), "metadata contains unselected row");
                    const auto& source = test_case.rows[position];
                    const auto source_index = test_case.expected_sources.empty() ? source.source_index : test_case.expected_sources.at(position);
                    require((read16(entry) | read16(entry+2)<<16) == source_index && read16(entry+6) == position &&
                        metadata.at(entry+10) == (expected_a8(position) ? 8 : 4) && metadata.at(entry+9) == unsigned(joint || test_case.state_table_update || test_case.paired_pending),
                        test_case.name+": refresh metadata source or precision changed position="+std::to_string(position));
                    const bool kv_disabled = (metadata.at(offset+24+physical/8) >> (physical%8)) & 1;
                    const bool expected_disabled = std::find(test_case.expected_kv_disabled.begin(),
                        test_case.expected_kv_disabled.end(), position) != test_case.expected_kv_disabled.end();
                    require(kv_disabled == expected_disabled, test_case.name+": physical KV write mask mismatch");
                    require(metadata.at(inverse+metadata.at(entry+8)) == physical, "refresh inverse mapping mismatch");
                    metadata_positions.push_back(position);
                    round_slots += expected_a8(position) ? 2 : 1;
                }
                require(rows <= 48 && round_slots <= 64, "refresh metadata exceeded SRAM activation capacity");
                ++group_rounds; group_compute+=metadata.at(offset+1); group_slots+=round_slots;
                if (test_case.qkvo_group) {
                    require(group_rounds<=6 && group_compute<=18 && group_slots<=288,
                            "published Q/K/V/O group exceeds runtime capacity");
                    const bool end=metadata.at(offset+3)&4;
                    require(round+1<rounds || end,"last group lacks its end marker");
                    if (end) { group_rounds=0; group_compute=0; group_slots=0; }
                } else require(!(metadata.at(offset+3)&4),"group flag set in ungrouped metadata");
                ordinal += rows; offset += read16(offset+8);
            }
            std::sort(metadata_positions.begin(), metadata_positions.end());
            require(metadata_positions == test_case.expected && offset == metadata.size(), "refresh metadata selection coverage mismatch");
        }
        if (fault == 27) require(extra_memory.at(result_address)==std::vector<std::uint8_t>(16,0xa5),
            "insufficient output region published a valid-looking result");
        if (fault == 25) require(injected, "result write error was not exercised");
        if (commit && fault && fault != 25) require(committed_bytes == 0, "failed metadata/configuration still committed cache");
        if (pending_output && !fault) *pending_output = extra_memory.at(0x50000);
        std::cout << "selector case=" << test_case.name << " fault=" << fault << " scans=" << scans
                  << " committed_bytes=" << committed_bytes << " cycles=" << cycles_-started << '\n';
    }
};
}
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Regression test;
        if (argc >= 3 && std::string(argv[1]) == "--observed-regular") {
            for (int i = 2; i < argc; ++i) test.run_observed_regular(argv[i]);
        } else if (argc >= 3 && std::string(argv[1]) == "--observed-boundary") {
            for (int i = 2; i < argc; ++i) test.run_observed_boundary(argv[i]);
        } else if (argc >= 3 && std::string(argv[1]) == "--precision-reference") {
            for (int i = 2; i < argc; ++i) test.run_precision(argv[i]);
        } else {
            require(argc == 2, "usage: attention_refresh <boundary index> | --observed-regular <reference>...");
            test.run(load(argv[1]));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL attention_refresh: " << error.what() << '\n';
        return 1;
    }
}
