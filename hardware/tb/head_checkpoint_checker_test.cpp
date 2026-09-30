#include "tb/head_checkpoint_checker.hpp"
#include <functional>

struct Events {
    bool post_active = true;
    bool rst = false, rms_trace_sample_valid = false, post_quant_observe_valid = false;
    bool post_scale_observe_valid = false, post_int32_observe_valid = false, post_logit_observe_valid = false;
    bool post_candidate_observe_valid = false;
    unsigned post_candidate_observe_row = 0, post_candidate_observe_top1 = 0;
    unsigned post_candidate_observe_logit = 0, post_candidate_observe_confidence = 0;
    unsigned post_candidate_observe_probability = 0, post_candidate_observe_action = 0;
    unsigned post_norm_group_base = 0, rms_trace_sample_element = 0;
    unsigned post_quant_observe_row = 0, post_quant_observe_element = 0, post_scale_observe_row = 0;
    unsigned post_int32_observe_row = 0, post_int32_observe_vocab = 0;
    unsigned post_logit_observe_row = 0, post_logit_observe_vocab = 0, post_scale_observe_mask = 0;
    std::uint64_t post_quant_observe_mask = 0, post_int32_observe_mask = 0, post_logit_observe_mask = 0;
    std::array<std::uint32_t, 4> rms_trace_sample_byte_enable{}, post_scale_observe_values{};
    std::array<std::uint32_t, 16> post_quant_observe_values{};
    std::array<std::uint32_t, 32> rms_trace_sample_data{}, post_logit_observe_values{};
    std::array<std::uint32_t, 64> post_int32_observe_values{};
};

static void require(bool value) {
    if (!value) throw std::runtime_error("head checker self-test failed");
}

int main(int argc, char** argv) {
    if (argc != 2) throw std::runtime_error("supply an external self-test directory");
    const std::filesystem::path root = argv[1];
    std::filesystem::create_directories(root);
    nlohmann::json expected = nlohmann::json::array();
    const std::array<std::string, 5> names{{"norm_output", "activation_codes", "activation_scale", "accumulator", "output"}};
    const std::array<std::string, 5> dtypes{{"<u2", "|i1", "<u2", "<i4", "<u2"}};
    const std::array<unsigned, 5> widths{{2, 1, 2, 4, 2}}, counts{{4096, 4096, 1, 8, 8}};
    const std::array<std::uint32_t, 5> raw{{0x8000, 0xff, 0x3f80, 0x80000001, 0x7fc1}};
    for (unsigned i = 0; i < 5; ++i) {
        const auto filename = names[i] + ".bin";
        std::ofstream file(root / filename, std::ios::binary);
        for (unsigned element = 0; element < counts[i]; ++element)
            for (unsigned byte = 0; byte < widths[i]; ++byte) file.put(raw[i] >> (8 * byte));
        expected.push_back({{"name", names[i]}, {"path", filename}, {"dtype", dtypes[i]},
            {"bytes", counts[i] * widths[i]}, {"shape", i == 2 ? std::vector<unsigned>{1} :
                std::vector<unsigned>{1, i < 2 ? 4096u : 8u}}});
    }
    const nlohmann::json cfg{{"executions", {0}}, {"head_checkpoints", {{"tokens", 1}, {"vocab", 8}, {"expected", expected}}}};
    const auto emit = [&](HeadCheckpointChecker& checker, bool corrupt) {
        Events event;
        event.rms_trace_sample_valid = event.post_quant_observe_valid = true;
        event.rms_trace_sample_byte_enable[0] = 0xffff;
        event.post_quant_observe_mask = 0xff;
        event.rms_trace_sample_data.fill(0x80008000);
        event.post_quant_observe_values.fill(0xffffffff);
        for (unsigned column = 0; column < 4096; column += 8) {
            event.rms_trace_sample_element = event.post_quant_observe_element = column;
            checker.observe(event);
        }
        event.rms_trace_sample_valid = event.post_quant_observe_valid = false;
        event.post_scale_observe_valid = event.post_int32_observe_valid = event.post_logit_observe_valid = true;
        event.post_scale_observe_mask = 1; event.post_scale_observe_values[0] = 0x3f80;
        event.post_int32_observe_mask = event.post_logit_observe_mask = 0xff;
        event.post_int32_observe_values.fill(0x80000001);
        event.post_logit_observe_values.fill(0x7fc17fc1);
        if (corrupt) event.post_logit_observe_values[3] = 0x7fc07fc1;
        checker.observe(event);
        return event;
    };
    HeadCheckpointChecker valid(cfg, root);
    Events transformer_norm;
    transformer_norm.post_active = false;
    transformer_norm.rms_trace_sample_valid = true;
    transformer_norm.rms_trace_sample_byte_enable[0] = 0xffff;
    transformer_norm.rms_trace_sample_data.fill(0xdeadbeef);
    valid.observe(transformer_norm);
    auto last = emit(valid, false);
    require(valid.finish(root).at("status") == "PASS");
    valid.observe(last);
    require(valid.finish(root).at("mismatches") == 17);
    HeadCheckpointChecker wrong(cfg, root);
    emit(wrong, true);
    require(wrong.finish(root).at("mismatches") == 1);
    HeadCheckpointChecker wrong_norm(cfg, root);
    emit(wrong_norm, false);
    transformer_norm.post_active = true;
    wrong_norm.observe(transformer_norm);
    require(wrong_norm.finish(root).at("mismatches") == 16);
    HeadCheckpointChecker missing(cfg, root);
    require(missing.finish(root).at("mismatches") == 8209);
    auto sequence = cfg;
    sequence["executions"] = {0, 1, 2};
    sequence["head_checkpoints"]["execution_index"] = 1;
    HeadCheckpointChecker selected(sequence, root);
    selected.begin_execution(0);
    emit(selected, true);
    selected.begin_execution(1);
    emit(selected, false);
    selected.begin_execution(2);
    emit(selected, true);
    require(selected.finish(root).at("status") == "PASS");
    auto multiple = sequence;
    auto first_check = sequence["head_checkpoints"], second_check = first_check;
    first_check["execution_index"] = 0; second_check["execution_index"] = 2;
    multiple["head_checkpoints"] = {first_check, second_check};
    HeadCheckpointChecker chain(multiple, root);
    chain.begin_execution(0); emit(chain, false);
    chain.begin_execution(1); emit(chain, true);
    require(chain.finish(root).at("mismatches") == 8209);
    chain.begin_execution(2); emit(chain, false);
    auto chain_result = chain.finish(root);
    require(chain_result.at("status") == "PASS" && chain_result.at("checks").size() == 2);
    chain.observe(last);
    require(chain.finish(root).at("mismatches") == 17);
    auto candidate_cfg = cfg;
    candidate_cfg["head_checkpoints"]["candidates"] = nlohmann::json::array();
    for (const auto& name : {"candidate_top1", "candidate_logit", "candidate_confidence", "candidate_probability", "candidate_action"}) {
        const unsigned width = std::string(name) == "candidate_top1" ? 4 : 2;
        std::ofstream file(root / (std::string(name) + ".bin"), std::ios::binary);
        for (unsigned byte = 0; byte < width; ++byte) file.put(byte == 0 ? 1 : 0);
        candidate_cfg["head_checkpoints"]["candidates"].push_back({{"name", name}, {"path", std::string(name) + ".bin"},
            {"dtype", width == 4 ? "<u4" : "<u2"}, {"shape", {1}}, {"bytes", width}});
    }
    HeadCheckpointChecker candidate(candidate_cfg, root);
    emit(candidate, false);
    require(candidate.finish(root).at("mismatches") == 5);
    Events candidate_event;
    candidate_event.post_candidate_observe_valid = true;
    candidate_event.post_candidate_observe_top1 = candidate_event.post_candidate_observe_logit = 1;
    candidate_event.post_candidate_observe_confidence = candidate_event.post_candidate_observe_probability = 1;
    candidate_event.post_candidate_observe_action = 1;
    candidate.observe(candidate_event);
    require(candidate.finish(root).at("status") == "PASS");
    candidate_event.post_candidate_observe_action = 0;
    candidate.observe(candidate_event);
    require(candidate.finish(root).at("mismatches") == 6);
    for (const auto& index : nlohmann::json::array({-1, 3, true, 0.5})) {
        sequence["head_checkpoints"]["execution_index"] = index;
        bool bad_index = false;
        try { HeadCheckpointChecker invalid(sequence, root); }
        catch (const std::exception&) { bad_index = true; }
        require(bad_index);
    }
    auto malformed = cfg;
    malformed["head_checkpoints"]["expected"].erase(0);
    bool rejected = false;
    try { HeadCheckpointChecker invalid(malformed, root); }
    catch (const std::exception&) { rejected = true; }
    require(rejected);
    last.post_logit_observe_vocab = 8;
    rejected = false;
    try { wrong.observe(last); }
    catch (const std::exception&) { rejected = true; }
    require(rejected);
    std::cout << "PASS head checker raw signed-zero/NaN/int32, explicit launch selection, duplicate, missing, mismatch and out-of-range checks\n";
}
