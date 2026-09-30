#include "tb/testcase_handoff_actions.hpp"
#include <cmath>
#include <iostream>
#include "tb/generated/token_state_entry_packer.hpp"
#include "tb/generated/forward_event_packer.hpp"
#include "tb/generated/in_block_refresh_budget_packer.hpp"
#include "tb/generated/uaps_result_packer.hpp"
extern "C" {
#include "cmodel/rtl_numeric.h"
#include "cmodel/token_refresh_model.h"
}

static std::vector<std::uint8_t> refresh_metadata(const std::vector<forward_postprocess_next_token_descriptor>& tokens,
                                                 unsigned sequence, unsigned capture) {
    std::vector<std::uint8_t> result;
    std::vector<bool> assigned(tokens.size());
    unsigned a8 = std::count_if(tokens.begin(), tokens.end(), [](const auto& token) { return token.activation_bits == 8; });
    unsigned a4 = tokens.size() - a8, ordinal = 0, round = 0;
    while (a4 + a8) {
        const auto rounds = std::max((a4 + a8 + 47) / 48, (a4 + 2 * a8 + 63) / 64);
        unsigned take8 = std::min(a8, 32u), take4 = 0;
        for (;;) {
            take4 = std::min({a4, 48 - take8, 64 - 2 * take8});
            if (a4 + a8 - take4 - take8 <= (rounds - 1) * 48 &&
                    a4 + 2 * a8 - take4 - 2 * take8 <= (rounds - 1) * 64) break;
            if (!take8) throw std::runtime_error("C11 refresh tokens do not fit round capacity");
            --take8;
        }
        a4 -= take4; a8 -= take8;
        std::vector<forward_postprocess_next_token_descriptor> selected;
        for (unsigned index = 0; index < tokens.size(); ++index) {
            auto& left = tokens[index].activation_bits == 8 ? take8 : take4;
            if (assigned[index] || !left) continue;
            assigned[index] = true; --left; selected.push_back(tokens[index]);
        }
        auto bytes = testcase_handoff::metadata(selected, sequence, capture);
        testcase_handoff::set(bytes, 4, 2, round++); testcase_handoff::set(bytes, 6, 2, ordinal);
        ordinal += selected.size(); result.insert(result.end(), bytes.begin(), bytes.end());
    }
    return result;
}

static nlohmann::json block_post_expected(const nlohmann::json& input) {
    const auto& source = input.at("states");
    const unsigned count = source.size(), vocabulary = input.at("vocabulary");
    if (!count || count > 32) throw std::runtime_error("post reference requires 1..32 tokens per block");
    std::vector<forward_postprocess_row_state> tokens(count);
    std::vector<std::uint32_t> selected(count);
    std::uint32_t masked = 0, tentative = 0, locked = 0;
    for (unsigned i = 0; i < count; ++i) {
        const auto& s = source.at(i); auto& r = tokens[i];
        r.token_id = s.at("token_id"); r.last_top1 = s.at("last_top1");
        r.precision_age = s.at("precision_age"); r.token_position = s.at("token_position");
        r.state = s.at("state"); r.commit_origin = s.at("origin"); r.activation_bits = s.at("activation_bits");
        r.cache_valid = s.at("cache_valid"); r.refresh_required = s.at("refresh_required");
        r.prediction_flag = s.at("prediction_flag"); selected[i] = r.token_id;
        if (r.state == FORWARD_POSTPROCESS_MASKED) masked |= std::uint32_t{1} << i;
        else if (r.state == FORWARD_POSTPROCESS_TENTATIVE) tentative |= std::uint32_t{1} << i;
        else if (r.state == FORWARD_POSTPROCESS_LOCKED) locked |= std::uint32_t{1} << i;
        else throw std::runtime_error("unknown transfer state");
    }
    std::vector<std::uint16_t> logits;
    if (input.contains("logits")) logits = input.at("logits").get<std::vector<std::uint16_t>>();
    else {
        const std::vector<std::uint16_t> hidden = input.at("hidden");
        const std::vector<std::int8_t> packed = input.at("packed_weights");
        const std::vector<std::uint16_t> scales = input.at("weight_scales");
        constexpr unsigned channels = 4096;
        if (hidden.size() != count * channels || packed.size() != vocabulary * channels ||
                scales.size() != vocabulary || vocabulary % 8)
            throw std::runtime_error("head reference tensor shape differs");
        std::vector<std::int8_t> weights(packed.size()), codes(hidden.size());
        std::vector<std::uint16_t> normalized(hidden.size()), activation_scales(count);
        for (unsigned output = 0; output < vocabulary; ++output)
            for (unsigned channel = 0; channel < channels; ++channel)
                weights[output * channels + channel] = packed[
                    (output / 8) * channels * 8 + (channel / 8) * 64 +
                    (channel % 8) * 8 + output % 8];
        logits.resize(count * vocabulary);
        if (forward_postprocess_final_rmsnorm_a8(hidden.data(), count, channels,
                input.at("epsilon"), normalized.data(), codes.data(), activation_scales.data()) ||
            forward_postprocess_raw_w8_matmul(codes.data(), activation_scales.data(), count,
                channels, weights.data(), scales.data(), vocabulary, logits.data(), nullptr))
            throw std::runtime_error("C11 final norm/head reference failed");
    }
    if (logits.size() != count * vocabulary) throw std::runtime_error("logit count differs");
    std::array<std::uint16_t, 256> exp{}, reciprocal{};
    for (unsigned i = 0; i < 256; ++i) {
        exp[i] = rtl_f32_to_bf16(std::exp(-16.0f + 16.0f * float(i) / 255.0f));
        reciprocal[i] = rtl_f32_to_bf16(1.0f / (1.0f + float(i) / 255.0f));
    }
    std::vector<forward_postprocess_candidate_result> candidates(count);
    const auto suppressed = input.value("suppressed_tokens", std::vector<std::uint32_t>{});
    if (forward_postprocess_candidate_bf16(logits.data(), count, vocabulary, selected.data(),
            suppressed.data(), suppressed.size(),
            exp.data(), reciprocal.data(), candidates.data())) throw std::runtime_error("candidate reference failed");
    forward_postprocess_draft_verify_config config{};
    config.mask_token_id = input.at("mask_token_id");
    config.scheduled_quota = input.at("quota"); config.remaining_forwards = input.at("remaining_forwards");
    config.maturity_age = input.at("maturity_age");
    const unsigned flags = input.value("block_flags", 32u);
    if (flags & ~39u || ((flags & 32u) && flags != 32u))
        throw std::runtime_error("post reference does not support this block update");
    config.high_confidence_threshold_bf16 = input.value("high_confidence_threshold_bf16", 0u);
    config.tail_high_confidence_threshold_bf16 = input.value("tail_high_confidence_threshold_bf16", 0u);
    config.low_confidence_threshold_bf16 = input.value("low_confidence_threshold_bf16", 0u);
    config.verify_threshold_bf16 = input.value("verify_threshold_bf16", 0u);
    config.stability_bonus_bf16 = input.value("stability_bonus_bf16", 0u);
    config.budget_scale_bf16 = flags & 2u ? 0x3f80u : input.value("budget_scale_bf16", 0u);
    config.step_index = input.value("step_index", 0u);
    config.tail_after_step = input.value("tail_after_step", 0u);
    config.tail_threshold_enable = (flags & 1u) != 0u;
    config.tail_bypass_all = (flags & 4u) != 0u;
    config.tail_bypass_stable_only = (flags & 64u) != 0u;
    forward_postprocess_draft_verify_result event{};
    const int status = flags & 32u ? forward_postprocess_transfer_update(
        tokens.data(), count, masked, tentative, locked, candidates.data(), &config, &event) :
        flags & 2u ? forward_postprocess_canonical_future_update(
            tokens.data(), count, masked, tentative, locked, input.value("observed_mask", 0u),
            candidates.data(), &config, &event) : forward_postprocess_draft_verify_update(
            tokens.data(), count, masked, tentative, locked, candidates.data(), &config, &event);
    if (status) throw std::runtime_error("post state update reference failed");
    std::vector<std::uint8_t> states;
    std::vector<forward_postprocess_next_token_descriptor> descriptors;
    nlohmann::json candidate_json = nlohmann::json::array();
    for (unsigned i = 0; i < count; ++i) {
        const auto& r = tokens[i]; const auto& c = candidates[i];
        token_state_entry s{};
        s.token_position = r.token_position; s.block_local_position = i;
        s.block_slot = input.value("block_slot", 0u);
        s.global_block_id = input.at("global_block_id"); s.state = r.state;
        s.origin = r.commit_origin; s.activation_bits = r.activation_bits; s.token_id = r.token_id;
        s.last_top1 = r.last_top1; s.precision_age = r.precision_age; s.cache_valid = r.cache_valid;
        s.refresh_required = r.refresh_required; s.capture_index = input.at("capture");
        s.prediction_flag = r.prediction_flag;
        s.change_flags = ((event.token_changed_mask >> i) & 1) | (((event.remasked_mask >> i) & 1) << 1);
        s.change_confidence_bf16 = (event.remasked_mask >> i) & 1 ? c.selected_probability_bf16 : 0x3f80;
        if (input.value("publish_action_history", false)) {
            const auto& previous = source.at(i);
            const bool observed = previous.value("candidate_observed", r.prediction_flag != 0);
            s.last_action_confidence_bf16 = observed ? c.action_confidence_bf16 :
                previous.value("last_action_confidence_bf16", 0u);
            s.last_action_confidence_valid = observed || previous.value("last_action_confidence_valid", 0u);
        }
        const auto bytes = pack_token_state_entry(s); states.insert(states.end(), bytes.begin(), bytes.end());
        descriptors.push_back({r.token_id, r.token_position, r.token_position,
            FORWARD_POSTPROCESS_EMBEDDING_TOKEN, r.activation_bits,
            input.at("predictions").at(i).at("query_group").get<std::uint8_t>(),
            input.at("predictions").at(i).at("cache_group").get<std::uint8_t>()});
        candidate_json.push_back({c.top1_token_id, c.top_logit_bf16, c.raw_confidence_bf16,
            c.selected_probability_bf16, c.action_confidence_bf16});
    }
    forward_event packed{};
    packed.block_slot = input.value("block_slot", 0u);
    packed.capture_index = input.at("capture"); packed.global_block_id = input.at("global_block_id");
    packed.position_count = count;
    packed.confirmed_mask = event.confirmed_mask; packed.remasked_mask = event.remasked_mask;
    packed.selected_mask = event.selected_mask; packed.direct_locked_mask = event.direct_locked_mask;
    packed.stable_tentative_mask = event.stable_tentative_mask; packed.fallback_tentative_mask = event.fallback_tentative_mask;
    packed.mandatory_refresh_mask = event.mandatory_refresh_mask; packed.cache_commit_mask = event.cache_commit_mask;
    packed.cache_invalidate_mask = event.cache_invalidate_mask; packed.cache_keep_mask = event.cache_keep_mask;
    packed.token_changed_mask = event.token_changed_mask; packed.tail_closed_mask = event.tail_closed_mask;
    return {{"states", states}, {"event", pack_forward_event(packed)}, {"candidates", candidate_json},
        {"metadata", testcase_handoff::metadata(descriptors, input.at("sequence"), input.at("capture"))}};
}

static nlohmann::json regular_refresh_expected(const nlohmann::json& input) {
    const auto& rows = input.at("states");
    const unsigned count = rows.size(), start = input.at("region_start");
    const unsigned block_end = input.at("block_end"), future_start = input.at("next_block_start");
    if (!count || count > RTL_REFRESH_MAX_ROWS || block_end < start ||
            block_end > start + count || future_start < start || future_start + 32 > start + count)
        throw std::runtime_error("regular reference region does not contain its current/future blocks");
    const std::vector<std::uint16_t> dependency = input.at("dependency");
    const std::vector<std::uint8_t> prior = input.at("pending");
    if (dependency.size() != count * count || prior.size() != count * 16)
        throw std::runtime_error("regular dependency/pending size differs");
    rtl_refresh_state refresh{};
    if (rtl_refresh_init(&refresh, count, input.at("target_token_count_x2"), dependency.data()))
        throw std::runtime_error("C11 regular refresh initialization failed");
    refresh.regular_steps = input.at("regular_steps");
    refresh.cumulative_active_token_count = input.at("cumulative_active_token_count");
    std::vector<std::uint8_t> predicted(count), changed(count), remasked(count);
    std::vector<std::uint16_t> confidence(count);
    for (unsigned i = 0; i < count; ++i) {
        if (rows[i].at("token_position") != start + i)
            throw std::runtime_error("regular state positions must cover the dependency region");
        predicted[i] = start + i < block_end && rows[i].at("state") != FORWARD_POSTPROCESS_LOCKED;
        changed[i] = rows[i].at("change_flags").get<unsigned>() & 1;
        remasked[i] = (rows[i].at("change_flags").get<unsigned>() >> 1) & 1;
        confidence[i] = rows[i].at("change_confidence_bf16");
        refresh.pending[i] = prior[i * 16] | unsigned(prior[i * 16 + 1]) << 8;
        refresh.refresh[i] = prior[i * 16 + 10] & 1;
    }
    if (rtl_refresh_observe(&refresh, 0, nullptr, nullptr, predicted.data(), changed.data(),
                            remasked.data(), confidence.data()))
        throw std::runtime_error("C11 regular refresh update failed");
    std::vector<forward_postprocess_next_token_descriptor> tokens;
    std::vector<std::uint8_t> bits, mandatory;
    std::vector<std::uint16_t> scores;
    auto pending = prior;
    for (unsigned i = 0; i < count; ++i) {
        testcase_handoff::set(pending, i * 16, 2, refresh.pending[i]);
        testcase_handoff::set(pending, i * 16 + 10, 1, (predicted[i] << 1) | refresh.refresh[i]);
        testcase_handoff::set(pending, i * 16 + 12, 2, refresh.new_invalidation[i]);
        if (!refresh.refresh[i]) continue;
        const auto& state = rows[i];
        tokens.push_back({state.at("token_id"), std::uint16_t(start + i), std::uint16_t(start + i),
                         FORWARD_POSTPROCESS_EMBEDDING_TOKEN, 4, 0, 0});
        bits.push_back(predicted[i] ? state.at("activation_bits").get<unsigned>() : 4);
        mandatory.push_back(predicted[i]); scores.push_back(refresh.pending[i]);
    }
    if (tokens.empty() || tokens.size() > RTL_JOINT_MAX_ROWS)
        throw std::runtime_error("regular reference selection exceeds joint capacity");
    std::vector<std::uint8_t> selected_bits(tokens.size()), upgrade_order(tokens.size());
    std::uint32_t upgrades = 0;
    if (rtl_context_precision(tokens.size(), bits.data(), mandatory.data(), scores.data(),
            input.at("context_a8_tokens"), 0, selected_bits.data(), upgrade_order.data(), &upgrades))
        throw std::runtime_error("C11 context precision failed");
    rtl_joint_input joint{};
    joint.base_token_count = tokens.size(); joint.future_token_count = 32;
    joint.target_token_count = input.at("target_tokens"); joint.max_next_tokens = input.at("max_next_tokens");
    joint.next_block_start = future_start;
    joint.prediction_target = input.at("prediction_target");
    joint.current_prediction_tokens = std::count(predicted.begin(), predicted.end(), 1);
    joint.retry_min_confidence = input.at("retry_min_confidence_bf16");
    for (unsigned i = 0; i < tokens.size(); ++i) {
        tokens[i].activation_bits = selected_bits[i];
        joint.base_positions[i] = tokens[i].token_position; joint.base_bits[i] = selected_bits[i];
        joint.forecast_bits[i] = tokens[i].token_position < block_end &&
            rows[tokens[i].token_position - start].at("state") == FORWARD_POSTPROCESS_MASKED ? 8 : selected_bits[i];
    }
    std::array<unsigned, 32> ranking{};
    for (unsigned i = 0; i < 32; ++i) ranking[i] = i;
    std::stable_sort(ranking.begin(), ranking.end(), [&](auto a, auto b) {
        return refresh.pending[future_start - start + a] < refresh.pending[future_start - start + b];
    });
    const std::vector<std::uint32_t> attempts = input.at("attempts");
    if (attempts.size() != 32) throw std::runtime_error("joint attempt count differs");
    std::array<std::uint8_t, 32> allowed{};
    for (unsigned rank = 0; rank < 32; ++rank) {
        const unsigned i = ranking[rank]; const auto& state = rows[future_start - start + i];
        joint.priority[i] = rtl_bf16_add(0x3f80, rtl_f32_to_bf16(-float(rank) / 31.0f));
        joint.future_bits[i] = state.at("activation_bits");
        joint.unresolved[i] = state.at("state") != FORWARD_POSTPROCESS_LOCKED;
        joint.tentative[i] = state.at("state") == FORWARD_POSTPROCESS_TENTATIVE;
        if (joint.tentative[i] || std::any_of(tokens.begin(), tokens.end(), [&](const auto& t) {
                return t.token_position == future_start + i; })) joint.priority[i] = 0x3f80;
        joint.future_admission_attempts[i] = attempts[i];
        // Invalid action history supplies zero to the retry comparison.
        joint.last_confidence[i] = state.value("last_action_confidence_valid", 0u) ?
            state.value("last_action_confidence_bf16", 0u) : 0u;
        allowed[i] = attempts[i] < input.at("max_attempts").get<unsigned>();
    }
    rtl_joint_selection selection{};
    if (rtl_joint_select_allowed(&joint, allowed.data(), &selection))
        throw std::runtime_error("C11 joint selection failed");
    uaps_result packed{};
    for (unsigned i = 0; i < selection.future_prediction_count; ++i)
        packed.future_prediction_mask |= 1u << selection.progress[i];
    for (unsigned i = 0; i < selection.added_future_token_count; ++i) {
        const auto local = selection.added[i]; const auto position = future_start + local;
        packed.added_future_token_mask |= 1u << local;
        tokens.push_back({rows[position - start].at("token_id"), std::uint16_t(position), std::uint16_t(position),
            FORWARD_POSTPROCESS_EMBEDDING_TOKEN, selection.added_bits[i], 0, 0});
    }
    std::sort(tokens.begin(), tokens.end(), [](const auto& a, const auto& b) {
        return a.token_position < b.token_position;
    });
    packed.future_prediction_count = selection.future_prediction_count;
    packed.added_future_token_count = selection.added_future_token_count;
    packed.base_activation_slots = selection.base.a4_token_count + 2 * selection.base.a8_token_count;
    packed.joint_activation_slots = selection.joint.a4_token_count + 2 * selection.joint.a8_token_count;
    packed.next_pass_activation_slots = selection.forecast.a4_token_count + 2 * selection.forecast.a8_token_count;
    in_block_refresh_budget budget{};
    budget.target_token_count_x2 = refresh.target_token_count_x2; budget.region_start = start;
    budget.regular_steps = refresh.regular_steps;
    budget.cumulative_active_token_count = refresh.cumulative_active_token_count;
    budget.token_count_x2_credit = refresh.target_token_count_x2 * refresh.regular_steps -
                                 2 * refresh.cumulative_active_token_count;
    std::vector<std::uint8_t> attempt_state(144);
    testcase_handoff::set(attempt_state, 0, 4, 0x31425441);
    testcase_handoff::set(attempt_state, 4, 2, future_start);
    testcase_handoff::set(attempt_state, 8, 4, input.at("capture"));
    testcase_handoff::set(attempt_state, 12, 4, packed.added_future_token_mask);
    for (unsigned i = 0; i < 32; ++i)
        testcase_handoff::set(attempt_state, 16 + i * 4, 4,
            attempts[i] + ((packed.added_future_token_mask >> i) & 1));
    std::vector<unsigned> positions, precision, base_positions;
    for (unsigned i = 0; i < joint.base_token_count; ++i) base_positions.push_back(joint.base_positions[i]);
    std::vector<unsigned> changed_keys;
    for (unsigned i = 0; i < count; ++i) changed_keys.push_back(changed[i] ? (remasked[i] ? 2 : 1) : 0);
    for (const auto& token : tokens) { positions.push_back(token.token_position); precision.push_back(token.activation_bits); }
    return {{"metadata", refresh_metadata(tokens, input.at("sequence"), input.at("capture"))},
            {"pending", pending}, {"budget", pack_in_block_refresh_budget(budget)},
            {"joint_result", pack_uaps_result(packed)}, {"attempts", attempt_state},
            {"positions", positions}, {"bits", precision}, {"base_tokens", joint.base_token_count},
            {"upgrades", upgrades}, {"base_positions", base_positions}, {"changed_keys", changed_keys}};
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected metadata, boundary, post or transfer operation");
        nlohmann::json input; std::cin >> input;
        if (std::string(argv[1]) == "handoff") {
            std::map<std::uint64_t, std::uint8_t> memory, writes;
            for (const auto &segment : input.at("segments")) {
                auto address = segment.at("address").get<std::uint64_t>();
                for (auto value : segment.at("data")) memory[address++] = value.get<std::uint8_t>();
            }
            auto actions = testcase_handoff_actions(input.at("execution"),
                [&](std::uint64_t address) -> int {
                    const auto found = memory.find(address);
                    return found == memory.end() ? -1 : found->second;
                }, [&](std::uint64_t address, std::uint8_t value) {
                    writes[address] = memory[address] = value; return 0;
                });
            nlohmann::json segments = nlohmann::json::array();
            for (auto it = writes.begin(); it != writes.end();) {
                const auto first = it->first;
                std::vector<std::uint8_t> data;
                do { data.push_back(it->second); ++it; }
                while (it != writes.end() && it->first == first + data.size());
                segments.push_back({{"address", first}, {"data", data}});
            }
            std::cout << nlohmann::json{{"actions", actions}, {"segments", segments}}.dump() << '\n';
            return 0;
        }
        if (std::string(argv[1]) == "refresh") {
            std::cout << regular_refresh_expected(input).dump() << '\n';
            return 0;
        }
        if (std::string(argv[1]) == "transfer" || std::string(argv[1]) == "post") {
            std::cout << block_post_expected(input).dump() << '\n';
            return 0;
        }
        const unsigned sequence = input.at("sequence");
        if (!sequence || sequence > 2048) throw std::runtime_error("invalid reference sequence");
        std::vector<forward_postprocess_next_token_descriptor> tokens;
        for (const auto& item : input.at("tokens")) {
            forward_postprocess_next_token_descriptor token{};
            token.token_position = token.kv_index = item.at("position");
            token.source_index = item.at("source_index"); token.source = item.at("source");
            token.activation_bits = item.at("bits"); token.query_group = item.value("query_group", 0);
            token.cache_group = item.value("cache_group", 0); tokens.push_back(token);
        }
        nlohmann::json result;
        if (std::string(argv[1]) == "boundary") {
            if (tokens.size() != sequence) throw std::runtime_error("boundary needs the full sequence");
            const std::vector<std::uint16_t> current = input.at("current"), changed = input.at("changed"), mandatory = input.at("mandatory");
            std::vector<std::uint16_t> eligible, positions(sequence), zero(sequence), probabilities(sequence);
            for (unsigned p = 0; p < sequence; ++p) {
                positions[p] = p;
                if (std::find(current.begin(), current.end(), p) == current.end()) eligible.push_back(p);
            }
            std::array<std::uint16_t, 256> exp{}, reciprocal{};
            for (unsigned i = 0; i < 256; ++i) {
                exp[i] = rtl_f32_to_bf16(std::exp(-16.0f + 16.0f * float(i) / 255.0f));
                reciprocal[i] = rtl_f32_to_bf16(1.0f / (1.0f + float(i) / 255.0f));
            }
            if (rtl_softmax_lut_bf16(zero.data(), 1, sequence, exp.data(), reciprocal.data(), probabilities.data()))
                throw std::runtime_error("C11 zero-score softmax rejected");
            std::vector<std::uint16_t> all(32 * sequence * sequence);
            for (unsigned token = 0; token < 32 * sequence; ++token)
                std::copy(probabilities.begin(), probabilities.end(), all.begin() + token * sequence);
            std::vector<std::uint8_t> scores(sequence);
            if (rtl_boundary_scout_q8(all.data(), sequence, positions.data(), current.size(), current.data(),
                    changed.size(), changed.data(), scores.data())) throw std::runtime_error("C11 scout rejected");
            rtl_boundary_selection selected{};
            if (rtl_boundary_select(sequence, scores.data(), input.at("target"), current.size(), current.data(),
                    eligible.size(), eligible.data(), mandatory.size(), mandatory.data(), 0, nullptr, 0, &selected))
                throw std::runtime_error("C11 boundary selection rejected");
            std::vector<std::uint8_t> original, selected_scores, protected_rows, bits(selected.deep_count);
            std::vector<forward_postprocess_next_token_descriptor> deep;
            for (unsigned i = 0; i < selected.deep_count; ++i) {
                const auto p = selected.deep_positions[i];
                original.push_back(tokens[p].activation_bits); selected_scores.push_back(scores[p]);
                protected_rows.push_back(std::find(current.begin(), current.end(), p) != current.end() ||
                    std::find(mandatory.begin(), mandatory.end(), p) != mandatory.end());
            }
            if (rtl_boundary_deep_precision(selected.deep_count, original.data(), selected_scores.data(),
                    protected_rows.data(), input.at("a8_limit"), bits.data())) throw std::runtime_error("C11 deep precision rejected");
            unsigned downgraded = 0;
            for (unsigned i = 0; i < selected.deep_count; ++i) {
                auto token = tokens.at(selected.deep_positions[i]);
                token.source = FORWARD_POSTPROCESS_RESIDENT_HIDDEN; token.source_index = token.token_position; token.activation_bits = bits[i];
                downgraded += original[i] == 8 && bits[i] == 4; deep.push_back(token);
            }
            tokens = deep;
            result["scores"] = scores; result["downgraded_a8"] = downgraded;
            result["positions"] = std::vector<std::uint16_t>(selected.deep_positions, selected.deep_positions + selected.deep_count);
            result["bits"] = bits;
            std::array<std::uint16_t, 128> key_zero{};
            std::array<std::int8_t, 128> key_codes{};
            std::uint16_t key_scale = 0;
            if (rtl_quantize_row_bf16(key_zero.data(), key_zero.size(), 127, key_codes.data(), &key_scale) ||
                    std::any_of(key_codes.begin(), key_codes.end(), [](auto code) { return code != 0; }))
                throw std::runtime_error("C11 zero-key quantization rejected");
            result["zero_key_scale_bf16"] = key_scale;
        } else if (std::string(argv[1]) != "metadata") throw std::runtime_error("unknown reference operation");
        result["metadata"] = std::string(argv[1]) == "boundary" ?
            refresh_metadata(tokens, sequence, input.at("capture")) : testcase_handoff::metadata(tokens, sequence, input.at("capture"));
        std::cout << result.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
