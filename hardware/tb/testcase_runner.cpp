#include "Vsupra_top_tb.h"
#include "verilated.h"
#include "tb/dramsim3/dramsim3_backend.hpp"
#include "tb/dramsim3/ddr_idle_observer.hpp"
#include "tb/head_checkpoint_checker.hpp"
#include "tb/attention_checkpoint_checker.hpp"
#include "tb/testcase_memory_actions.hpp"
#include "tb/testcase_handoff_actions.hpp"
#include "tb/embedding_read_checker.hpp"
#include <json.hpp>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

static json read_json(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    json value; input >> value; return value;
}

static json read_backend_stats() {
    return json::parse(supra_dramsim3_stats_json());
}

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        if (argc != 4) throw std::runtime_error("usage: testcase_runner CASE_JSON OUTPUT_DIR THREADS");
        const auto input = fs::absolute(argv[1]);
        const auto output = fs::absolute(argv[2]);
        const auto cfg = read_json(input);
        const auto resolve = [&](const std::string& name) { return fs::absolute(input.parent_path() / name); };
        if (cfg.at("schema") != "supra-testcase/v1") throw std::runtime_error("unsupported testcase run schema");
        if (cfg.at("executions").empty() || cfg.at("expected").empty()) throw std::runtime_error("executions and expected must be nonempty");
        for (const auto& region : cfg.at("expected"))
            if (region.contains("execution_index") &&
                    (!region.at("execution_index").is_number_integer() ||
                     region.at("execution_index").get<int>() < 0 ||
                     region.at("execution_index").get<std::size_t>() >= cfg.at("executions").size()))
                throw std::runtime_error("expected execution_index must identify an existing execution");
        if (cfg.contains("backpressure") || cfg.contains("scope"))
            throw std::runtime_error("backpressure and scope are not supported testcase settings");
        for (const auto& execution : cfg.at("executions"))
            if (execution.contains("scope") || execution.contains("backpressure"))
                throw std::runtime_error("backpressure and scope are not supported execution settings");
        fs::create_directories(output / "dramsim3");
        auto context = std::make_unique<VerilatedContext>();
        context->threads(std::stoul(argv[3]));
        std::vector<std::string> arguments{argv[0],
            "+DDR_INITIAL_IMAGE=" + resolve(cfg.at("ddr_image")).string(),
            "+DRAMSIM3_CONFIG=" + resolve(cfg.at("dramsim3_config")).string(),
            "+DRAMSIM3_REGION_MANIFEST=" + resolve(cfg.at("memory_map")).string(),
            "+DRAMSIM3_OUTPUT_DIR=" + (output / "dramsim3").string()};
        std::vector<const char*> pointers;
        for (const auto& value : arguments) pointers.push_back(value.c_str());
        context->commandArgs(pointers.size(), pointers.data());
        auto storage = std::make_unique<Vsupra_top_tb>(context.get());
        auto& dut = *storage;
        HeadCheckpointChecker head(cfg, input.parent_path());
        AttentionCheckpointChecker attention(cfg, input.parent_path());
        EmbeddingReadChecker embedding;
        std::cout << "VERILATOR_MODEL_THREADS " << dut.threads() << '\n';
        std::uint64_t cycles = 0;
        struct CommandProfile {
            std::uint64_t cycles = 0;
            std::uint64_t axi_ar_fire = 0;
            std::uint64_t axi_r_fire = 0;
        };
        std::array<CommandProfile, 11> command_profiles{};
        std::array<std::array<std::uint64_t, 64>, 11> qkv_state_cycles{};
        std::array<std::uint64_t, 32> attention_state_cycles{};
        struct MatmulStateProfile {
            std::uint64_t cycles = 0;
            std::uint64_t panel_loading = 0;
            std::uint64_t panel_computing = 0;
            std::uint64_t panel_write_stall = 0;
            std::uint64_t panel_write_fire = 0;
            std::uint64_t dma_response_stall = 0;
            std::uint64_t dma_response_fire = 0;
            std::uint64_t read_bundle_stall = 0;
            std::uint64_t pe_stall = 0;
            std::uint64_t pe_fire = 0;
        };
        std::array<std::array<MatmulStateProfile, 16>, 11> qkv_matmul_states{};
        std::array<std::uint64_t, 32> softmax_state_cycles{};
        const bool trace_ffn = std::getenv("SUPRA_TRACE_FFN_SERVICE") != nullptr;
        std::array<std::array<std::uint64_t, 64>, 11> output_state_cycles{}, elementwise_state_cycles{};
        std::ofstream ffn_trace;
        unsigned ffn_samples = 0;
        if (trace_ffn) {
            ffn_trace.open(output / "ffn_service_events.tsv");
            if (!ffn_trace) throw std::runtime_error("cannot create FFN service trace");
            ffn_trace << "cycle\tcommand\tevent\toperation\tbatch\trow\tcolumn\tup\n";
        }
        const bool trace_softmax = std::getenv("SUPRA_TRACE_SOFTMAX") != nullptr;
        std::ofstream softmax_trace;
        if (trace_softmax) {
            softmax_trace.open(output / "softmax_events.tsv");
            if (!softmax_trace) throw std::runtime_error("cannot create Softmax trace");
            softmax_trace << "cycle\tstate\tevent\thead\trow_base\ttag\n";
        }
        const bool trace_axi_read = std::getenv("SUPRA_TRACE_AXI_READ") != nullptr &&
            std::string(std::getenv("SUPRA_TRACE_AXI_READ")) == "1";
        std::ofstream axi_ar_trace;
        if (trace_axi_read) {
            axi_ar_trace.open(output / "axi_ar_trace.tsv");
            if (!axi_ar_trace)
                throw std::runtime_error("cannot create AXI read-request trace");
            axi_ar_trace << "cycle\tcommand\taddress\tbeats\n";
        }
        const bool trace_axi_write = std::getenv("SUPRA_TRACE_AXI_WRITE") != nullptr &&
            std::string(std::getenv("SUPRA_TRACE_AXI_WRITE")) == "1";
        std::ofstream axi_aw_trace, axi_w_trace;
        if (trace_axi_write) {
            axi_aw_trace.open(output / "axi_aw_trace.tsv");
            axi_w_trace.open(output / "axi_w_trace.tsv");
            if (!axi_aw_trace || !axi_w_trace)
                throw std::runtime_error("cannot create AXI write traces");
            axi_aw_trace << "cycle\tcommand\taddress\tbeats\n";
            axi_w_trace << "cycle\tcommand\tstrb\tlast\n";
        }
        const bool trace_idle = std::getenv("SUPRA_TRACE_DDR_IDLE") != nullptr &&
            std::string(std::getenv("SUPRA_TRACE_DDR_IDLE")) == "1";
        std::ofstream idle_trace;
        std::unique_ptr<DdrIdleObserver> idle_observer;
        json post_completions = json::array();
        const auto tick = [&]() {
            const auto command = std::min<unsigned>(dut.observed_command, 10u);
            dut.clk = 0; dut.eval(); head.observe(dut); attention.observe(dut); embedding.observe(dut); context->timeInc(1);
            if (!dut.rst && dut.post_completion_observe_valid)
                post_completions.push_back(bool(dut.post_completion_observe_block_complete));
            if (trace_ffn && !dut.rst) {
                ++output_state_cycles[command].at(dut.debug_matmul_output_controller_state);
                ++elementwise_state_cycles[command].at(dut.elementwise_debug_state);
                if (command == 6u && ffn_samples < 128u) {
                    const auto emit = [&](const char* event, unsigned operation) {
                        ffn_trace << cycles << '\t' << command << '\t' << event << '\t'
                            << operation << '\t' << unsigned(dut.ffn_fragment_batch) << '\t'
                            << unsigned(dut.ffn_fragment_row) << '\t'
                            << unsigned(dut.ffn_fragment_column) << '\t'
                            << unsigned(dut.ffn_fragment_up) << '\n';
                        ++ffn_samples;
                    };
                    if (dut.ffn_fragment_valid) emit("fragment", 0);
                    if (dut.bf16_request_observe_accepted)
                        emit("bf16_request", dut.bf16_request_observe_operation);
                }
            }
            if (command == 2u && !dut.rst) {
                ++softmax_state_cycles[unsigned(dut.softmax_row_state)];
                if (trace_softmax) {
                    const auto emit = [&](const char* event, unsigned tag) {
                        softmax_trace << cycles << '\t' << unsigned(dut.softmax_row_state)
                            << '\t' << event << '\t' << unsigned(dut.softmax_source_req_head)
                            << '\t' << unsigned(dut.softmax_source_req_row_base)
                            << '\t' << tag << '\n';
                    };
                    if (dut.softmax_source_req_valid && dut.softmax_source_req_ready)
                        emit("score_request", dut.softmax_source_req_tag);
                    if (dut.softmax_source_rsp_valid && dut.softmax_source_rsp_ready)
                        emit("score_response", dut.softmax_source_rsp_tag);
                    if (dut.softmax_vector_req_valid && dut.softmax_vector_req_ready)
                        emit("vector_request", dut.softmax_vector_req_tag);
                    if (dut.softmax_vector_rsp_valid && dut.softmax_vector_rsp_ready)
                        emit("vector_response", dut.softmax_vector_rsp_tag);
                }
            }
            const unsigned bf16_clients = dut.debug_bf16_request_clients;
            if (bf16_clients && (bf16_clients & (bf16_clients - 1u)))
                std::cerr << "BF16_REQUEST_CONFLICT cycle=" << cycles
                          << " command=" << command << " clients=" << bf16_clients
                          << " qkv_state=" << unsigned(dut.qkv_state)
                          << " attention_state=" << unsigned(dut.attention_controller_state)
                          << std::endl;
            if (idle_observer)
                idle_observer->observe(cycles, dut.debug_ddr_paths_drained,
                    dut.debug_ddr_read_requested, dut.debug_ddr_write_requested);
            dut.clk = 1; dut.eval(); context->timeInc(1);
            ++command_profiles[command].cycles;
            ++qkv_state_cycles[command][unsigned(dut.qkv_state)];
            if (command == 2u)
                ++attention_state_cycles[unsigned(dut.attention_controller_state)];
            if ((command == 1u || command == 2u) &&
                    dut.qkv_state == 6u) {
                auto& state = qkv_matmul_states[command][unsigned(dut.debug_matmul_state)];
                ++state.cycles;
                state.panel_loading += dut.debug_matmul_panel_loading != 0;
                state.panel_computing += dut.debug_matmul_panel_computing != 0;
                state.panel_write_stall += dut.matmul_panel_memory_write_valid &&
                                           !dut.matmul_panel_memory_write_ready;
                state.panel_write_fire += dut.matmul_panel_memory_write_valid &&
                                          dut.matmul_panel_memory_write_ready;
                state.dma_response_stall += dut.matmul_dma_rsp_valid &&
                                            !dut.matmul_dma_rsp_ready;
                state.dma_response_fire += dut.matmul_dma_rsp_valid &&
                                           dut.matmul_dma_rsp_ready;
                state.read_bundle_stall += dut.matmul_read_bundle_valid &&
                                           !dut.debug_matmul_read_bundle_ready;
                state.pe_stall += dut.shared_pe_req_valid &&
                                  !dut.shared_pe_req_ready;
                state.pe_fire += dut.shared_pe_req_valid &&
                                 dut.shared_pe_req_ready;
            }
            command_profiles[command].axi_ar_fire += dut.axi_ar_fire;
            command_profiles[command].axi_r_fire += dut.axi_r_fire;
            if (trace_axi_read && dut.axi_ar_fire)
                axi_ar_trace << cycles << '\t' << command << '\t'
                             << dut.axi_araddr << '\t'
                             << (unsigned(dut.axi_arlen) + 1u) << '\n';
            if (trace_axi_write && dut.axi_aw_fire)
                axi_aw_trace << cycles << '\t' << command << '\t'
                             << dut.axi_awaddr << '\t'
                             << (unsigned(dut.axi_awlen) + 1u) << '\n';
            if (trace_axi_write && dut.axi_w_fire)
                axi_w_trace << cycles << '\t' << command << '\t'
                            << dut.axi_wstrb << '\t'
                            << unsigned(dut.axi_wlast) << '\n';
            dut.clk = 0; dut.eval(); ++cycles;
            if (dut.ddr_protocol_error) throw std::runtime_error("DDR protocol error");
            if (context->gotFinish()) throw std::runtime_error("RTL terminated before replay completed");
            if ((cycles & ((1ULL << 20) - 1)) == 0)
                std::cout << "TESTCASE_PROGRESS cycles=" << cycles
                    << " execution_state=" << unsigned(dut.debug_execution_state)
                    << " layer=" << unsigned(dut.observed_layer)
                    << " command=" << unsigned(dut.observed_command)
                    << " attention_state=" << unsigned(dut.attention_controller_state)
                    << " softmax_state=" << unsigned(dut.softmax_row_state)
                    << " matmul_state=" << unsigned(dut.debug_matmul_state)
                    << " elementwise_state=" << unsigned(dut.elementwise_debug_state)
                    << " r4_state=" << unsigned(dut.r4_controller_state)
                    << " read_bytes=" << dut.accepted_read_bytes
                    << " write_bytes=" << dut.accepted_write_bytes << '\n';
        };
        dut.rst = 1; dut.run_valid = 0; dut.run_scope = 2;
        dut.run_config_address = 0; dut.run_id = 0;
        dut.random_stall = 0;
        dut.performance_config = 0;
        dut.inject_read_response_error = 0; dut.inject_write_response_error = 0;
        dut.completion_ready = 0; dut.image_check_request = 0;
        for (unsigned i = 0; i < 4; ++i) tick();
        dut.rst = 0; tick();
        if (!dut.ddr_backend_idle || !supra_dramsim3_idle())
            throw std::runtime_error("initial DDR segments require idle memory");
        const auto initialized_bytes = testcase_initialize_segments(cfg, input.parent_path(),
                                                               supra_dramsim3_initialize_byte);
        std::uint64_t host_copy_bytes = 0;
        std::uint64_t embedding_mismatches = 0, handoff_action_count = 0, completion_errors = 0;
        json completed_executions = json::array();
        std::uint64_t mismatches = 0, bytes = 0;
        json comparisons = json::array();
        const auto compare_expected = [&](int after_execution) {
            for (const auto& result : testcase_compare_expected(cfg, input.parent_path(), output,
                    after_execution, supra_dramsim3_inspect_byte)) {
                mismatches += result.at("mismatches").get<std::uint64_t>();
                bytes += result.at("bytes").get<std::uint64_t>();
                comparisons.push_back(result);
            }
        };
        std::size_t execution_index = 0;
        for (const auto& execution : cfg.at("executions")) {
            post_completions = json::array();
            head.begin_execution(execution_index);
            attention.begin_execution(execution_index);
            ++execution_index;
            if (execution.contains("copies")) {
                if (!dut.ddr_backend_idle || !supra_dramsim3_idle())
                    throw std::runtime_error("inter-execution copies require drained DDR");
                host_copy_bytes += testcase_copy_memory(execution, supra_dramsim3_inspect_byte,
                                                   supra_dramsim3_initialize_byte);
            }
            const auto handoff = testcase_handoff_actions(execution, supra_dramsim3_inspect_byte,
                                                     supra_dramsim3_initialize_byte);
            handoff_action_count += execution.value("handoff_actions", json::array()).size();
            const bool check_embedding = execution.value("check_embedding_reads", false);
            std::vector<std::uint64_t> embedding_addresses;
            std::uint64_t embedding_base = 0, embedding_limit = 0;
            if (check_embedding) {
                const auto read = [](std::uint64_t address, unsigned width) {
                    std::uint64_t value = 0;
                    for (unsigned i = 0; i < width; ++i) {
                        const int byte = supra_dramsim3_inspect_byte(address + i);
                        if (byte < 0) throw std::runtime_error("embedding metadata read failed");
                        value |= std::uint64_t(byte) << (i * 8);
                    }
                    return value;
                };
                const auto address = execution.at("config_address").get<std::uint64_t>();
                embedding_base = read(address + EXECUTION_CONFIG_EMBEDDING_BASE_OFFSET, 8);
                embedding_limit = read(address + EXECUTION_CONFIG_EMBEDDING_LIMIT_OFFSET, 8);
                auto metadata = read(address + EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET, 8);
                auto limit = read(address + EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET, 8);
                if (read(address + EXECUTION_CONFIG_FLAGS_OFFSET, 4) & (1u << 13)) {
                    limit = read(metadata + 8, 8);
                    metadata = read(metadata, 8);
                }
                if (limit <= metadata || limit - metadata > 65536)
                    throw std::runtime_error("embedding metadata range invalid");
                unsigned bytes = 0;
                for (const auto& row : testcase_handoff::read_metadata(read, metadata, limit - metadata, bytes))
                    if (read(row.address + 9, 1) == FORWARD_POSTPROCESS_EMBEDDING_TOKEN)
                        embedding_addresses.push_back(embedding_base + read(row.address, 4) * 8192);
            }
            embedding.start(check_embedding, embedding_base, embedding_limit, std::move(embedding_addresses));
            const auto idle_name = "ddr_idle_launch_" + std::to_string(completed_executions.size()) + ".tsv";
            if (trace_idle) {
                idle_trace.open(output / idle_name, std::ios::trunc);
                if (!idle_trace) throw std::runtime_error("cannot create DDR idle trace");
                idle_observer = std::make_unique<DdrIdleObserver>(idle_trace);
            }
            const auto start_cycle = cycles;
            const auto start_command_profiles = command_profiles;
            const auto start_read_bytes = dut.accepted_read_bytes;
            const auto start_write_bytes = dut.accepted_write_bytes;
            const auto start_physical_read_bytes =
                supra_dramsim3_physical_read_bytes();
            const auto start_physical_write_bytes =
                supra_dramsim3_physical_write_bytes();
            const auto deadline = cycles + execution.at("max_cycles").get<std::uint64_t>();
            const auto bounded_tick = [&]() {
                if (cycles >= deadline) throw std::runtime_error("execution exceeded max_cycles");
                tick();
            };
            while (!dut.run_ready) bounded_tick();
            dut.run_config_address = execution.at("config_address").get<std::uint64_t>();
            dut.run_id = execution.at("id").get<std::uint32_t>();
            dut.run_valid = 1; bounded_tick(); dut.run_valid = 0;
            while (!dut.completion_valid) bounded_tick();
            const auto completion_error_id = dut.completion_error_id;
            if (dut.completion_error || dut.completion_id != dut.run_id) {
                std::cerr << "TESTCASE_COMPLETION_ERROR cycles=" << cycles
                    << " expected_id=" << dut.run_id << " actual_id=" << dut.completion_id
                    << " error=" << unsigned(dut.completion_error)
                    << " error_id=" << dut.completion_error_id
                    << " configuration_state=" << unsigned(dut.debug_execution_config_loader_state)
                    << " execution_state=" << unsigned(dut.debug_execution_state)
                    << " read_bytes=" << dut.accepted_read_bytes << '\n';
                ++completion_errors;
            }
            dut.completion_ready = 1; bounded_tick(); dut.completion_ready = 0;
            while (!dut.ddr_backend_idle || !supra_dramsim3_idle()) bounded_tick();
            if (idle_observer) {
                idle_observer->finish();
                idle_observer.reset();
                idle_trace.close();
            }
            const auto embedding_result = embedding.finish();
            embedding_mismatches += embedding_result.at("mismatches").get<std::uint64_t>();
            const auto end_physical_read_bytes =
                supra_dramsim3_physical_read_bytes();
            const auto end_physical_write_bytes =
                supra_dramsim3_physical_write_bytes();
            json execution_commands = json::array();
            for (unsigned command = 0; command < command_profiles.size(); ++command) {
                const auto& before = start_command_profiles[command];
                const auto& after = command_profiles[command];
                execution_commands.push_back({
                    {"command", command},
                    {"cycles", after.cycles - before.cycles},
                    {"axi_ar_fire", after.axi_ar_fire - before.axi_ar_fire},
                    {"axi_r_fire", after.axi_r_fire - before.axi_r_fire}});
            }
            completed_executions.push_back({{"id", dut.run_id}, {"start_cycle", start_cycle},
                {"end_cycle", cycles}, {"cycles", cycles - start_cycle},
                {"commands", execution_commands},
                {"read_bytes", dut.accepted_read_bytes - start_read_bytes},
                {"write_bytes", dut.accepted_write_bytes - start_write_bytes},
                {"physical_read_bytes", end_physical_read_bytes -
                    start_physical_read_bytes},
                {"physical_write_bytes", end_physical_write_bytes -
                    start_physical_write_bytes},
                {"handoff_actions", handoff}, {"embedding_reads", embedding_result}});
            if (trace_idle) completed_executions.back()["ddr_idle_trace"] = idle_name;
            completed_executions.back()["completion_error_id"] = completion_error_id;
            completed_executions.back()["post_block_completions"] = post_completions;
            if (execution.contains("expected_post_block_completions")) {
                const auto& expected = execution.at("expected_post_block_completions");
                if (!expected.is_array() || std::any_of(expected.begin(), expected.end(),
                        [](const json& value) { return !value.is_boolean(); }))
                    throw std::runtime_error("expected_post_block_completions must be a boolean array");
                if (post_completions != expected) {
                    ++mismatches;
                    std::cerr << "TESTCASE_POST_COMPLETION_MISMATCH execution=" << execution_index - 1
                        << " expected=" << expected.dump() << " actual=" << post_completions.dump() << '\n';
                }
            }
            std::cout << "TESTCASE_COMPLETED id=" << dut.run_id << " cycles=" << cycles << '\n';
            compare_expected(static_cast<int>(execution_index) - 1);
            // A rejected configuration cannot feed the next execution. Preserve
            // all captured numerical evidence before returning a failed result.
            if (completion_errors) break;
        }
        mismatches += embedding_mismatches + completion_errors;
        compare_expected(-1);
        const auto head_result = head.finish(output);
        const auto attention_result = attention.finish(output);
        mismatches += attention_result.at("mismatches").get<std::uint64_t>();
        mismatches += head_result.at("mismatches").get<std::uint64_t>();
        const auto backend_stats = read_backend_stats();
        std::ofstream command_profile(output / "command_profile.tsv");
        if (!command_profile)
            throw std::runtime_error("cannot create command profile");
        command_profile << "schema\ttestcase_command_profile_v1\n"
                        << "command\tcycles\taxi_ar_fire\taxi_r_fire\n";
        for (std::size_t command = 0; command < command_profiles.size(); ++command) {
            const auto& profile = command_profiles[command];
            command_profile << command << '\t' << profile.cycles << '\t'
                            << profile.axi_ar_fire << '\t'
                            << profile.axi_r_fire << '\n';
        }
        command_profile.close();
        if (!command_profile)
            throw std::runtime_error("command profile write failed");
        std::ofstream qkv_state_profile(output / "qkv_state_profile.tsv");
        if (!qkv_state_profile)
            throw std::runtime_error("cannot create Q/K/V state profile");
        qkv_state_profile << "command\tstate\tcycles\n";
        for (std::size_t command = 0; command < qkv_state_cycles.size(); ++command)
            for (std::size_t state = 0; state < qkv_state_cycles[command].size(); ++state)
                if (qkv_state_cycles[command][state] != 0)
                    qkv_state_profile << command << '\t' << state << '\t'
                                      << qkv_state_cycles[command][state] << '\n';
        qkv_state_profile.close();
        if (!qkv_state_profile)
            throw std::runtime_error("Q/K/V state profile write failed");
        std::ofstream softmax_state_profile(output / "softmax_state_profile.tsv");
        if (!softmax_state_profile) throw std::runtime_error("cannot create Softmax state profile");
        softmax_state_profile << "state\tcycles\n";
        for (std::size_t state = 0; state < softmax_state_cycles.size(); ++state)
            if (softmax_state_cycles[state] != 0)
                softmax_state_profile << state << '\t' << softmax_state_cycles[state] << '\n';
        softmax_state_profile.close();
        if (!softmax_state_profile) throw std::runtime_error("Softmax state profile write failed");
        if (trace_softmax) {
            softmax_trace.close();
            if (!softmax_trace) throw std::runtime_error("Softmax trace write failed");
        }
        std::ofstream attention_state_profile(output / "attention_state_profile.tsv");
        if (!attention_state_profile)
            throw std::runtime_error("cannot create Attention state profile");
        attention_state_profile << "state\tcycles\n";
        for (std::size_t state = 0; state < attention_state_cycles.size(); ++state)
            if (attention_state_cycles[state] != 0)
                attention_state_profile << state << '\t'
                                        << attention_state_cycles[state] << '\n';
        attention_state_profile.close();
        if (!attention_state_profile)
            throw std::runtime_error("Attention state profile write failed");
        std::ofstream qkv_matmul_profile(output / "qkv_matmul_profile.tsv");
        if (!qkv_matmul_profile)
            throw std::runtime_error("cannot create Q/K/V Matmul profile");
        qkv_matmul_profile << "command\tstate\tcycles\tpanel_loading\t"
                           << "panel_computing\tpanel_write_stall\tpanel_write_fire\t"
                           << "dma_response_stall\tdma_response_fire\t"
                           << "read_bundle_stall\tpe_stall\tpe_fire\n";
        for (std::size_t command : {1u, 2u})
            for (std::size_t state = 0; state < qkv_matmul_states[command].size(); ++state) {
                const auto& profile = qkv_matmul_states[command][state];
                if (profile.cycles != 0)
                    qkv_matmul_profile << command << '\t' << state << '\t'
                        << profile.cycles << '\t' << profile.panel_loading << '\t'
                        << profile.panel_computing << '\t'
                        << profile.panel_write_stall << '\t' << profile.panel_write_fire
                        << '\t' << profile.dma_response_stall << '\t'
                        << profile.dma_response_fire << '\t' << profile.read_bundle_stall
                        << '\t' << profile.pe_stall << '\t' << profile.pe_fire << '\n';
            }
        qkv_matmul_profile.close();
        if (!qkv_matmul_profile)
            throw std::runtime_error("Q/K/V Matmul profile write failed");
        if (trace_ffn) {
            ffn_trace.close();
            if (!ffn_trace) throw std::runtime_error("FFN service trace write failed");
            std::ofstream profile(output / "ffn_service_states.tsv");
            profile << "command\tmodule\tstate\tcycles\n";
            for (unsigned command = 0; command < 11; ++command)
                for (unsigned state = 0; state < 64; ++state) {
                    if (output_state_cycles[command][state])
                        profile << command << "\tmatmul_output\t" << state << '\t'
                                << output_state_cycles[command][state] << '\n';
                    if (elementwise_state_cycles[command][state])
                        profile << command << "\telementwise\t" << state << '\t'
                                << elementwise_state_cycles[command][state] << '\n';
                }
            profile.close();
            if (!profile) throw std::runtime_error("FFN state profile write failed");
        }
        if (trace_axi_read) {
            axi_ar_trace.close();
            if (!axi_ar_trace)
                throw std::runtime_error("AXI read-request trace write failed");
        }
        if (trace_axi_write) {
            axi_aw_trace.close();
            axi_w_trace.close();
            if (!axi_aw_trace || !axi_w_trace)
                throw std::runtime_error("AXI write trace failed");
        }
        json summary{{"status", mismatches ? "FAIL" : "PASS"}, {"cycles", cycles},
            {"completion_errors", completion_errors},
            {"head_checkpoints", head_result},
            {"attention_checkpoints", attention_result},
            {"host_copy_bytes", host_copy_bytes},
            {"initial_segment_bytes", initialized_bytes},
            {"handoff_action_count", handoff_action_count},
            {"executions", completed_executions},
            {"compared_bytes", bytes}, {"mismatches", mismatches}, {"memory_map", comparisons},
            {"runtime_threads", dut.threads()}, {"read_bytes", dut.accepted_read_bytes}, {"write_bytes", dut.accepted_write_bytes}};
        summary["physical_read_bytes"] = backend_stats.at("physical_read_bytes");
        summary["physical_write_bytes"] = backend_stats.at("physical_write_bytes");
        std::ofstream report(output / "summary.json"); report << summary.dump(2) << '\n';
        dut.final();
        std::cout << summary.dump() << '\n';
        return mismatches ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "SUPRA_TESTCASE_FAIL " << error.what() << '\n';
        return 1;
    }
}
