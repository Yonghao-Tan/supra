#include "tb/attention_checkpoint_checker.hpp"

struct Events {
    bool context_bf16_observe_valid = false;
    unsigned context_bf16_observe_address = 0, context_bf16_observe_byte_enable = 0xffff;
    std::array<std::uint32_t, 4> context_bf16_observe_data{};
    bool rst = false, softmax_quantized_observe_valid = false, softmax_scale_observe_valid = false;
    bool qkv_quantized_observe_valid = false;
    bool hidden_ddr_observe_valid = false, hidden_ddr_observe_qkv = false;
    bool hidden_ddr_observe_preserve = false;
    unsigned hidden_ddr_observe_preserve_layer = 31;
    bool hidden_ddr_observe_ffn = false;
    bool hidden_local_write_observe_valid = false;
    bool hidden_local_write_observe_preserved_refill = false;
    unsigned hidden_local_write_observe_physical_row = 0;
    unsigned hidden_local_write_observe_channel_word = 0;
    unsigned hidden_local_write_observe_byte_enable = 0xffff;
    std::array<std::uint32_t, 4> hidden_local_write_observe_data{};
    unsigned hidden_ddr_observe_byte_enable = 0xffff;
    std::array<std::uint32_t, 4> hidden_ddr_observe_data{};
    unsigned observed_layer = 31, debug_row_token_batch_index = 0;
    unsigned qkv_quantized_observe_qkv_select = 0, qkv_quantized_observe_head = 0;
    unsigned qkv_quantized_observe_row = 0, qkv_quantized_observe_token_position = 0;
    unsigned qkv_quantized_observe_half = 0;
    std::uint64_t qkv_quantized_observe_lane_mask = ~std::uint64_t{0};
    std::array<std::uint32_t, 16> qkv_quantized_observe_values{};
    unsigned qkv_quantized_observe_scale = 0;
    bool attention_score_observe_valid = false, softmax_bf16_observe_valid = false;
    unsigned attention_score_observe_head = 0, attention_score_observe_query_base = 0;
    unsigned attention_score_observe_key_base = 0;
    unsigned softmax_bf16_observe_head = 0, softmax_bf16_observe_row = 0;
    unsigned softmax_bf16_observe_key = 0;
    std::array<std::uint32_t, 32> attention_score_observe_data{}, softmax_bf16_observe_data{};
    std::array<std::uint32_t, 4> attention_score_observe_byte_enable{}, softmax_bf16_observe_byte_enable{};
    unsigned softmax_quantized_observe_head = 0, softmax_quantized_observe_row = 0;
    unsigned softmax_quantized_observe_key = 0, softmax_scale_observe_head = 0;
    unsigned softmax_scale_observe_row = 0, softmax_scale_observe_row_mask = 0;
    std::uint64_t softmax_quantized_observe_byte_enable = 0;
    std::array<std::uint32_t, 16> softmax_quantized_observe_data{};
    std::array<std::uint32_t, 4> softmax_scale_observe_value{};
};

static void require(bool ok) {
    if (!ok) throw std::runtime_error("Attention checker self-test failed");
}

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--replay") {
        const std::filesystem::path case_path = argv[2], actual = argv[3];
        nlohmann::json cfg, previous;
        std::ifstream config_stream(case_path), summary_stream(actual / "summary.json");
        config_stream >> cfg; summary_stream >> previous;
        AttentionCheckpointChecker checker(cfg, case_path.parent_path());
        auto result = checker.replay_complete(actual, previous.at("attention_checkpoints"));
        std::ofstream output(argv[4]); output << result.dump(2) << '\n';
        std::cout << result.at("status") << " Attention complete raw replay\n";
        return result.at("mismatches") == 0 ? 0 : 1;
    }
    if (argc != 2) throw std::runtime_error("supply external self-test directory");
    const std::filesystem::path directory = argv[1];
    std::filesystem::create_directories(directory);
    std::ofstream codes(directory / "codes.bin", std::ios::binary), scales(directory / "scales.bin", std::ios::binary);
    std::ofstream query_codes(directory / "query_codes.bin", std::ios::binary);
    std::ofstream query_scales(directory / "query_scales.bin", std::ios::binary);
    std::ofstream scores_out(directory / "scores.bin", std::ios::binary);
    std::ofstream probabilities_out(directory / "probabilities.bin", std::ios::binary);
    std::ofstream preserved_hidden(directory / "preserved_hidden.bin", std::ios::binary);
    std::ofstream attention_residual(directory / "attention_residual.bin", std::ios::binary);
    for (unsigned token = 0; token < 3; ++token)
        for (unsigned element = 0; element < 4096; ++element) {
            const std::uint16_t value = token * 4096 + element;
            preserved_hidden.put(value); preserved_hidden.put(value >> 8);
            const std::uint16_t residual = value ^ 0x55aa;
            attention_residual.put(residual); attention_residual.put(residual >> 8);
        }
    for (unsigned head = 0; head < 32; ++head)
        for (unsigned token = 0; token < 3; ++token) {
            for (unsigned key = 0; key < 9; ++key) codes.put((head * 7 + token * 3 + key) & 255);
            scales.put(token); scales.put(0x80); // Preserve signed-zero and raw subnormal distinctions.
            for (unsigned element = 0; element < 128; ++element)
                query_codes.put((head * 11 + token * 5 + element) & 255);
            query_scales.put(token); query_scales.put(0x3f);
            for (unsigned key = 0; key < 9; ++key) {
                const auto score = head * 13 + token * 7 + key;
                const auto probability = head * 17 + token * 11 + key;
                scores_out.put(score); scores_out.put(score >> 8);
                probabilities_out.put(probability); probabilities_out.put(probability >> 8);
            }
        }
    codes.close(); scales.close(); query_codes.close(); query_scales.close();
    scores_out.close(); probabilities_out.close(); preserved_hidden.close();
    attention_residual.close();
    nlohmann::json settings{
        {"layer", 31}, {"tokens", 3}, {"heads", 32}, {"sequence", 9},
        {"token_positions", {4, 7, 8}},
        {"physical_to_reference_token", {{2, 0}, {1}}}};
    settings["expected"] = {{{"name", "probability_codes"}, {"dtype", "|i1"}, {"shape", {32, 3, 9}},
            {"path", "codes.bin"}, {"bytes", 864}},
            {{"name", "probability_scale"}, {"dtype", "<u2"}, {"shape", {32, 3, 1}},
            {"path", "scales.bin"}, {"bytes", 192}},
            {{"name", "query_codes"}, {"dtype", "|i1"}, {"shape", {32, 3, 128}},
            {"path", "query_codes.bin"}, {"bytes", 12288}},
            {{"name", "query_scale"}, {"dtype", "<u2"}, {"shape", {32, 3, 1}},
            {"path", "query_scales.bin"}, {"bytes", 192}},
            {{"name", "scores"}, {"dtype", "<u2"}, {"shape", {32, 3, 9}},
            {"path", "scores.bin"}, {"bytes", 1728}},
            {{"name", "probabilities"}, {"dtype", "<u2"}, {"shape", {32, 3, 9}},
            {"path", "probabilities.bin"}, {"bytes", 1728}},
            {{"name", "preserved_hidden"}, {"dtype", "<u2"}, {"shape", {3, 4096}},
            {"path", "preserved_hidden.bin"}, {"bytes", 24576}},
            {{"name", "preserved_hidden_refill"}, {"dtype", "<u2"},
            {"shape", {3, 4096}}, {"path", "preserved_hidden.bin"},
            {"bytes", 24576}},
            {{"name", "attention_residual"}, {"dtype", "<u2"}, {"shape", {3, 4096}},
            {"path", "attention_residual.bin"}, {"bytes", 24576}}};
    nlohmann::json cfg{{"executions", {0}}, {"attention_checkpoints", settings}};
    const auto emit = [&](AttentionCheckpointChecker& checker, bool corrupt, bool logical_preserve = false,
                          bool residual_subset = false, unsigned layer = 31,
                          bool stale_preserve_command = false) {
        Events event;
        event.observed_layer = layer;
        event.hidden_ddr_observe_preserve_layer = layer;
        const std::vector<std::vector<unsigned>> map{{2, 0}, {1}};
        const std::vector<std::vector<unsigned>> preserved = logical_preserve ?
            std::vector<std::vector<unsigned>>{{0, 2}, {1}} : map;
        for (unsigned round = 0; round < map.size(); ++round) {
            event.softmax_quantized_observe_valid = false;
            event.softmax_scale_observe_valid = false;
            event.qkv_quantized_observe_valid = false;
            event.attention_score_observe_valid = false;
            event.softmax_bf16_observe_valid = false;
            event.hidden_ddr_observe_valid = true;
            event.hidden_ddr_observe_preserve = true;
            if (stale_preserve_command && round == 0) event.observed_layer = 31;
            for (auto token : preserved[round])
                for (unsigned stripe = 0; stripe < 4096 / 8; ++stripe) {
                    event.hidden_ddr_observe_data.fill(0);
                    for (unsigned lane = 0; lane < 8; ++lane) {
                        const std::uint16_t value = token * 4096 + stripe * 8 + lane;
                        event.hidden_ddr_observe_data[lane / 2] |= value << (16 * (lane % 2));
                    }
                    checker.observe(event);
                }
            event.hidden_ddr_observe_valid = false;
            event.hidden_ddr_observe_preserve = false;
            event.observed_layer = layer;
            event.hidden_ddr_observe_valid = true;
            event.hidden_ddr_observe_ffn = true;
            for (unsigned stripe = 0; stripe < 4096 / 8; ++stripe)
                for (auto token : map[round]) {
                    if (residual_subset && round == 1) continue;
                    event.hidden_ddr_observe_data.fill(0);
                    for (unsigned lane = 0; lane < 8; ++lane) {
                        const std::uint16_t value =
                            (token * 4096 + stripe * 8 + lane) ^ 0x55aa;
                        event.hidden_ddr_observe_data[lane / 2] |= value << (16 * (lane % 2));
                    }
                    checker.observe(event);
                }
            event.hidden_ddr_observe_valid = false;
            event.hidden_ddr_observe_ffn = false;
            event.debug_row_token_batch_index = round;
            event.hidden_local_write_observe_valid = true;
            event.hidden_local_write_observe_preserved_refill = true;
            for (unsigned p = 0; p < map[round].size(); ++p)
                for (unsigned stripe = 0; stripe < 4096 / 8; ++stripe) {
                    event.hidden_local_write_observe_physical_row = p;
                    event.hidden_local_write_observe_channel_word = stripe;
                    event.hidden_local_write_observe_data.fill(0);
                    for (unsigned lane = 0; lane < 8; ++lane) {
                        const std::uint16_t value =
                            map[round][p] * 4096 + stripe * 8 + lane;
                        event.hidden_local_write_observe_data[lane / 2] |=
                            value << (16 * (lane % 2));
                    }
                    checker.observe(event);
                }
            event.hidden_local_write_observe_valid = false;
            event.hidden_local_write_observe_preserved_refill = false;
            for (unsigned head = 0; head < 32; ++head)
                for (unsigned p = 0; p < map[round].size(); ++p) {
                    event.debug_row_token_batch_index = round;
                    event.softmax_quantized_observe_valid = false;
                    event.softmax_scale_observe_valid = false;
                    event.qkv_quantized_observe_valid = true;
                    event.qkv_quantized_observe_head = head;
                    event.qkv_quantized_observe_row = p;
                    event.qkv_quantized_observe_token_position =
                        std::array<unsigned, 3>{4, 7, 8}[map[round][p]];
                    event.qkv_quantized_observe_scale = 0x3f00 + map[round][p];
                    for (unsigned half = 0; half < 2; ++half) {
                        event.qkv_quantized_observe_half = half;
                        event.qkv_quantized_observe_values.fill(0);
                        for (unsigned lane = 0; lane < 64; ++lane)
                            event.qkv_quantized_observe_values[lane / 4] |=
                                ((head * 11 + map[round][p] * 5 + half * 64 + lane) & 255)
                                << (8 * (lane % 4));
                        checker.observe(event);
                    }
                    event.qkv_quantized_observe_valid = false;
                    for (unsigned key = 0; key < 9; key += 8) {
                        const auto lanes = key ? 1u : 8u;
                        event.attention_score_observe_valid = true;
                        event.attention_score_observe_head = head;
                        event.attention_score_observe_query_base = p;
                        event.attention_score_observe_key_base = key;
                        event.attention_score_observe_data.fill(0);
                        event.attention_score_observe_byte_enable.fill(0);
                        event.softmax_bf16_observe_valid = true;
                        event.softmax_bf16_observe_head = head;
                        event.softmax_bf16_observe_row = p;
                        event.softmax_bf16_observe_key = key;
                        event.softmax_bf16_observe_data.fill(0);
                        event.softmax_bf16_observe_byte_enable.fill(0);
                        for (unsigned lane = 0; lane < lanes; ++lane) {
                            const auto score = head * 13 + map[round][p] * 7 + key + lane;
                            const auto probability = head * 17 + map[round][p] * 11 + key + lane;
                            event.attention_score_observe_data[lane / 2] |= score << (16 * (lane % 2));
                            event.softmax_bf16_observe_data[lane / 2] |= probability << (16 * (lane % 2));
                            event.attention_score_observe_byte_enable[lane / 16] |= 3u << (2 * (lane % 16));
                            event.softmax_bf16_observe_byte_enable[lane / 16] |= 3u << (2 * (lane % 16));
                        }
                        checker.observe(event);
                    }
                    event.attention_score_observe_valid = false;
                    event.softmax_bf16_observe_valid = false;
                    event.softmax_quantized_observe_head = event.softmax_scale_observe_head = head;
                    event.context_bf16_observe_valid = true;
                    for (unsigned word = 0; word < 16; ++word) {
                        event.context_bf16_observe_address = p * 16 + word;
                        event.context_bf16_observe_data.fill(0);
                        for (unsigned lane = 0; lane < 8; ++lane) {
                            const unsigned value = (head * 384 + map[round][p] * 128 + word * 8 + lane) ^ 0x8000;
                            event.context_bf16_observe_data[lane / 2] |= value << (16 * (lane % 2));
                        }
                        checker.observe(event);
                    }
                    event.context_bf16_observe_valid = false;
                    event.softmax_quantized_observe_row = event.softmax_scale_observe_row = p;
                    event.softmax_scale_observe_valid = true;
                    event.softmax_scale_observe_row_mask = 1;
                    event.softmax_scale_observe_value[0] = 0x8000 + map[round][p];
                    event.softmax_quantized_observe_valid = false;
                    checker.observe(event);
                    event.softmax_scale_observe_valid = false;
                    event.softmax_quantized_observe_valid = true;
                    for (unsigned k = 0; k < 9; k += 8) {
                        event.softmax_quantized_observe_key = k;
                        event.softmax_quantized_observe_byte_enable = k ? 1 : 255;
                        event.softmax_quantized_observe_data.fill(0);
                        for (unsigned lane = 0; lane < (k ? 1u : 8u); ++lane)
                            event.softmax_quantized_observe_data[lane / 4] |=
                                ((head * 7 + map[round][p] * 3 + k + lane) & 255) << (8 * (lane % 4));
                        if (corrupt && round == 1 && head == 31 && k == 8)
                            event.softmax_quantized_observe_data[0] ^= 1;
                        checker.observe(event);
                    }
                }
        }
        return event;
    };
    AttentionCheckpointChecker valid(cfg, directory);
    valid.begin_execution(0);
    auto last = emit(valid, false);
    const auto valid_result = valid.finish(directory);
    if (valid_result.at("status") != "PASS")
        std::cerr << valid_result.dump(2) << '\n';
    require(valid_result.at("status") == "PASS");
    valid.observe(last);
    require(valid.finish(directory).at("mismatches") == 1);
    AttentionCheckpointChecker wrong(cfg, directory);
    wrong.begin_execution(0);
    emit(wrong, true);
    require(wrong.finish(directory).at("mismatches") == 1);
    AttentionCheckpointChecker missing(cfg, directory);
    missing.begin_execution(0);
    require(missing.finish(directory).at("mismatches") == 51936);
    auto scoped_cfg = cfg;
    scoped_cfg["executions"] = {0, 1};
    scoped_cfg["attention_checkpoints"]["execution_index"] = 1;
    AttentionCheckpointChecker scoped(scoped_cfg, directory);
    scoped.begin_execution(0);
    scoped.observe(last);
    scoped.begin_execution(1);
    emit(scoped, false, false, false, 30);
    emit(scoped, false);
    require(scoped.finish(directory).at("status") == "PASS");
    auto multiple = scoped_cfg;
    auto first_check = scoped_cfg["attention_checkpoints"], second_check = first_check;
    first_check["execution_index"] = 0;
    multiple["attention_checkpoints"] = {first_check, second_check};
    AttentionCheckpointChecker chain(multiple, directory);
    chain.begin_execution(0); emit(chain, false);
    require(chain.finish(directory).at("mismatches") == 51936);
    chain.begin_execution(1); emit(chain, false);
    const auto chain_result = chain.finish(directory);
    require(chain_result.at("status") == "PASS" && chain_result.at("checks").size() == 2);
    AttentionCheckpointChecker replay_chain(multiple, directory);
    require(replay_chain.replay_complete(directory, chain_result).at("status") == "PASS");
    // The previous execution ended at L31. The new L0 preserve arrives before
    // its first command, while a separate L31 checkpoint is already active.
    auto next_l0 = second_check;
    next_l0["layer"] = 0;
    auto stale_cfg = multiple;
    stale_cfg["attention_checkpoints"] = {first_check, next_l0, second_check};
    AttentionCheckpointChecker stale(stale_cfg, directory);
    stale.begin_execution(0); emit(stale, false);
    stale.begin_execution(1);
    emit(stale, false, false, false, 0, true);
    emit(stale, false);
    const auto stale_result = stale.finish(directory / "stale_command");
    require(stale_result.at("status") == "PASS" && stale_result.at("checks").size() == 3);
    auto preserved_cfg = cfg;
    preserved_cfg["attention_checkpoints"]["preserved_hidden_token_order"] = {0, 2, 1};
    AttentionCheckpointChecker logical(preserved_cfg, directory);
    logical.begin_execution(0);
    emit(logical, false, true);
    const auto complete = logical.finish(directory);
    require(complete.at("status") == "PASS");
    AttentionCheckpointChecker replayed(preserved_cfg, directory);
    require(replayed.replay_complete(directory, complete).at("status") == "PASS_REPLAYED");
    auto incomplete = complete;
    incomplete["tensors"][0]["missing"] = 1;
    bool incomplete_rejected = false;
    try {
        AttentionCheckpointChecker invalid_replay(preserved_cfg, directory);
        invalid_replay.replay_complete(directory, incomplete);
    } catch (const std::exception&) { incomplete_rejected = true; }
    require(incomplete_rejected);
    auto subset_cfg = preserved_cfg;
    subset_cfg["attention_checkpoints"]["attention_residual_token_order"] = {{2, 0}, nlohmann::json::array()};
    AttentionCheckpointChecker subset(subset_cfg, directory);
    subset.begin_execution(0);
    emit(subset, false, true, true);
    const auto subset_result = subset.finish(directory);
    require(subset_result.at("status") == "PASS");
    AttentionCheckpointChecker subset_replay(subset_cfg, directory);
    require(subset_replay.replay_complete(directory, subset_result).at("status") == "PASS_REPLAYED");
    auto padded_result = subset_result;
    const auto residual_path = directory / "attention.attention_residual.actual.bin";
    const auto residual_bytes = std::filesystem::file_size(residual_path);
    std::filesystem::resize_file(residual_path, residual_bytes + 8192);
    for (auto& tensor : padded_result["tensors"]) if (tensor.at("name") == "attention_residual") {
        tensor["bytes"] = residual_bytes + 8192;
        tensor["missing"] = 4096;
    }
    bool padding_rejected = false;
    try {
        AttentionCheckpointChecker padded(subset_cfg, directory);
        padded.replay_complete(directory, padded_result);
    } catch (const std::exception&) { padding_rejected = true; }
    require(padding_rejected);
    std::filesystem::resize_file(residual_path, residual_bytes);
    auto missing_required = subset_result;
    for (auto& tensor : missing_required["tensors"]) if (tensor.at("name") == "attention_residual") {
        tensor["events"] = tensor["events"].get<unsigned>() - 1;
        tensor["missing"] = tensor["missing"].get<unsigned>() + 1;
    }
    bool missing_required_rejected = false;
    try {
        AttentionCheckpointChecker invalid_subset(subset_cfg, directory);
        invalid_subset.replay_complete(directory, missing_required);
    } catch (const std::exception&) { missing_required_rejected = true; }
    require(missing_required_rejected);
    auto ambiguous = scoped_cfg;
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
        auto invalid = cfg;
        for (auto& entry : invalid["attention_checkpoints"]["expected"]) {
            if (entry.at("name") != "preserved_hidden_refill") continue;
            if (mutation == 0) entry["dtype"] = "<f2";
            if (mutation == 1) entry["shape"] = {1, 12288};
            if (mutation == 2) entry["bytes"] = 24578;
            if (mutation >= 3) {
                const auto path = directory / "refill_invalid.bin";
                std::filesystem::copy_file(directory / "preserved_hidden.bin", path,
                    std::filesystem::copy_options::overwrite_existing);
                std::filesystem::resize_file(path, mutation == 3 ? 24574 : 24578);
                entry["path"] = "refill_invalid.bin";
            }
        }
        bool rejected_refill = false;
        try { AttentionCheckpointChecker bad(invalid, directory); }
        catch (const std::exception&) { rejected_refill = true; }
        require(rejected_refill);
    }
    ambiguous["attention_checkpoints"].erase("execution_index");
    bool rejected = false;
    try { AttentionCheckpointChecker bad(ambiguous, directory); }
    catch (const std::exception&) { rejected = true; }
    require(rejected);
    auto malformed = cfg;
    malformed["attention_checkpoints"]["physical_to_reference_token"] = {{2, 0}, {2}};
    rejected = false;
    try { AttentionCheckpointChecker bad(malformed, directory); }
    catch (const std::exception&) { rejected = true; }
    require(rejected);
    last.softmax_quantized_observe_key = 9;
    rejected = false;
    try { wrong.observe(last); }
    catch (const std::exception&) { rejected = true; }
    require(rejected);
    {
        std::ofstream values(directory / "context.bin", std::ios::binary);
        for (unsigned i = 0; i < 32 * 3 * 128; ++i) {
            const unsigned value = i ^ 0x8000;
            values.put(value); values.put(value >> 8);
        }
    }
    auto context_cfg = cfg;
    context_cfg["attention_checkpoints"]["expected"].push_back({
        {"name", "context"}, {"dtype", "<u2"}, {"shape", {32, 3, 128}},
        {"path", "context.bin"}, {"bytes", 24576}});
    AttentionCheckpointChecker context_checker(context_cfg, directory);
    emit(context_checker, false);
    const auto context_result = context_checker.finish(directory);
    require(context_result.at("status") == "PASS");
    AttentionCheckpointChecker context_replay(context_cfg, directory);
    require(context_replay.replay_complete(directory, context_result).at("status") == "PASS_REPLAYED");
    Events context_corrupt;
    context_corrupt.context_bf16_observe_valid = true;
    context_checker.observe(context_corrupt);
    require(context_checker.finish(directory).at("mismatches") == 16);
    std::cout << "PASS Attention checker Query/Probability raw, nonidentity token map, rounds, key tail, missing/duplicate/corrupt events\n";
}
