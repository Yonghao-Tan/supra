#include "token_refresh_model.h"
#include "rtl_numeric.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

int rtl_context_precision(uint32_t rows, const uint8_t *input_bits,
    const uint8_t *mandatory_current, const uint16_t *dependency,
    uint32_t context_a8_token_count, uint8_t tail_all_a8, uint8_t *output_bits,
    uint8_t *upgrade_order, uint32_t *upgrade_count) {
    uint8_t selected[RTL_REFRESH_MAX_ROWS] = {0};
    if (!rows || rows > RTL_REFRESH_MAX_ROWS || !input_bits ||
        !mandatory_current || !dependency || !output_bits || !upgrade_order ||
        !upgrade_count || tail_all_a8 > 1) return -1;
    for (uint32_t row = 0; row < rows; ++row) {
        if ((input_bits[row] != 4 && input_bits[row] != 8) ||
            mandatory_current[row] > 1 || (!mandatory_current[row] &&
            (input_bits[row] != 4 || dependency[row] > 0x3f80u))) return -1;
        output_bits[row] = input_bits[row];
    }
    *upgrade_count = 0;
    while (*upgrade_count < context_a8_token_count) {
        uint32_t best = rows;
        for (uint32_t row = 0; row < rows; ++row)
            if (!mandatory_current[row] && !selected[row] &&
                (best == rows || dependency[row] > dependency[best])) best = row;
        if (best == rows) break;
        selected[best] = 1;
        output_bits[best] = 8;
        upgrade_order[(*upgrade_count)++] = (uint8_t)best;
    }
    if (tail_all_a8) memset(output_bits, 8, rows);
    return 0;
}

int rtl_probability_relation_bf16(const uint16_t *probabilities,
    uint32_t queries, uint32_t keys, uint16_t *relation, uint16_t *all_layer_max) {
    if (!probabilities || !relation || !queries || queries > 2048 ||
        !keys || keys > 2048 || queries * keys < 2) return -1;
    for (uint32_t query = 0; query < queries; ++query) {
        for (uint32_t key = 0; key < keys; ++key) {
            float mean;
            if (rtl_probability_head_mean32(probabilities + query*keys + key, queries*keys, &mean)) return -1;
            const uint32_t index = query * keys + key;
            relation[index] = rtl_f32_to_bf16(mean);
            /* BF16 RNE is monotone here, so it commutes with the layer max
             * before TriBlockAttentionState materializes its BF16 table. */
            if (all_layer_max && relation[index] > all_layer_max[index])
                all_layer_max[index] = relation[index];
        }
    }
    return 0;
}

int rtl_p8_relation_bf16(const int8_t *codes, const uint16_t *scales,
                        uint32_t queries, uint32_t keys, uint16_t *relation,
                        uint16_t *all_layer_max) {
    if (!codes || !scales || !relation || !queries || queries > 2048 ||
        !keys || keys > 2048 || queries * keys < 2) return -1;
    for (uint32_t query = 0; query < queries; ++query) {
        for (uint32_t key = 0; key < keys; ++key) {
            uint16_t head_values[32];
            float mean;
            for (uint32_t head = 0; head < 32; ++head) {
                const int8_t code = codes[(head * queries + query) * keys + key];
                const uint16_t scale = scales[head * queries + query];
                if (code < 0 || !scale || scale >= 0x7f80u) return -1;
                const uint16_t value = rtl_bf16_mul(rtl_f32_to_bf16((float)code), scale);
                if (value > 0x3f80u) return -1;
                head_values[head] = value;
            }
            const uint32_t index = query * keys + key;
            if (rtl_probability_head_mean32(head_values, 1, &mean)) return -1;
            relation[index] = rtl_f32_to_bf16(mean);
            if (all_layer_max && relation[index] > all_layer_max[index])
                all_layer_max[index] = relation[index];
        }
    }
    return 0;
}

static int scout_mean(const uint16_t *probabilities, uint32_t sequence,
                      uint32_t query, uint32_t key, float *mean) {
    return rtl_probability_head_mean32(probabilities + query*sequence + key, sequence*sequence, mean);
}

int rtl_boundary_scout_q8(const uint16_t *probabilities, uint32_t sequence,
    const uint16_t *query_positions, uint32_t current_count,
    const uint16_t *current_positions, uint32_t transition_count,
    const uint16_t *transition_positions, uint8_t *scores) {
    uint16_t query_row[RTL_REFRESH_MAX_SEQUENCE];
    uint8_t current[RTL_REFRESH_MAX_SEQUENCE] = {0};
    uint8_t transition[RTL_REFRESH_MAX_SEQUENCE] = {0};
    if (!probabilities || !query_positions || !current_positions || !scores ||
        !sequence || sequence > RTL_REFRESH_MAX_SEQUENCE || !current_count ||
        current_count > sequence || transition_count > sequence ||
        (transition_count && !transition_positions)) return -1;
    const uint32_t reduction_outputs = (sequence - current_count) *
        (transition_count ? transition_count : current_count);
    if (reduction_outputs == 1) return -1;
    memset(query_row, 0xff, sizeof(query_row));
    for (uint32_t row = 0; row < sequence; ++row) {
        const uint16_t position = query_positions[row];
        if (position >= sequence || query_row[position] != UINT16_MAX) return -1;
        query_row[position] = (uint16_t)row;
    }
    for (uint32_t index = 0; index < current_count; ++index) {
        const uint16_t position = current_positions[index];
        if (position >= sequence || current[position]) return -1;
        current[position] = 1;
    }
    for (uint32_t index = 0; index < transition_count; ++index) {
        const uint16_t position = transition_positions[index];
        if (position >= sequence || transition[position]) return -1;
        transition[position] = 1;
    }
    for (uint32_t candidate = 0; candidate < sequence; ++candidate) {
        scores[candidate] = 0;
        if (current[candidate]) continue;
        float maximum = 0.0f;
        const uint32_t count = transition_count ? transition_count : current_count;
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t query = query_row[transition_count ? candidate : current_positions[index]];
            const uint32_t key = transition_count ? transition_positions[index] : candidate;
            float mean;
            if (scout_mean(probabilities, sequence, query, key, &mean)) return -1;
            if (mean > maximum) maximum = mean;
        }
        const float scaled = maximum * 255.0f;
        uint32_t rounded = (uint32_t)scaled;
        const float fraction = scaled - (float)rounded;
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1u))) ++rounded;
        scores[candidate] = (uint8_t)(rounded > 255u ? 255u : rounded);
    }
    return 0;
}

size_t rtl_joint_input_bytes(void) { return sizeof(struct rtl_joint_input); }
size_t rtl_joint_selection_bytes(void) { return sizeof(struct rtl_joint_selection); }

static struct rtl_activation_residency residency(uint32_t a4, uint32_t a8) {
    struct rtl_activation_residency value = {0};
    value.a4_token_count = a4;
    value.a8_token_count = a8;
    value.pe_issue_groups = (a4 + 2 * a8 + 15) / 16;
    value.issued_slice_units = value.pe_issue_groups * 16;
    value.activation_bytes[0] = 4096 * (a4 + a8);
    value.activation_bytes[1] = 2048 * a4 + 4096 * a8;
    value.activation_bytes[2] = 6144 * (a4 + 2 * a8);
    for (uint32_t i = 0; i < 3; ++i) {
        value.capacity_bytes[i] = 384 * 1024;
        value.fragments[i] = (value.activation_bytes[i] + 384 * 1024 - 1) / (384 * 1024);
    }
    return value;
}

static int future_precedes(const struct rtl_joint_input *input,
                           uint8_t left, uint8_t right, int source_b) {
    if (input->tentative[left] != input->tentative[right])
        return input->tentative[left] > input->tentative[right];
    if (input->service_count[left] != input->service_count[right])
        return input->service_count[left] < input->service_count[right];
    if (source_b && input->source_b_dependency_tie_rank) {
        if (!input->tentative[left] && input->dependency[left] != input->dependency[right])
            return input->dependency[left] < input->dependency[right];
        if (input->block_step_index >= 2) {
            const unsigned l = input->priority[left] < 0x3e80 ? 0 : input->priority[left] < 0x3f00 ? 3 : input->priority[left] < 0x3f40 ? 1 : 2;
            const unsigned r = input->priority[right] < 0x3e80 ? 0 : input->priority[right] < 0x3f00 ? 3 : input->priority[right] < 0x3f40 ? 1 : 2;
            if (l != r) return l > r;
        }
    }
    const int comparison = rtl_bf16_compare(input->priority[left], input->priority[right]);
    return comparison ? comparison > 0 : left < right;
}

static void order_future(const struct rtl_joint_input *input,
                         uint8_t *positions, uint32_t count, int source_b) {
    for (uint32_t i = 1; i < count; ++i) {
        const uint8_t position = positions[i];
        uint32_t slot = i;
        while (slot && future_precedes(input, position, positions[slot - 1], source_b)) {
            positions[slot] = positions[slot - 1];
            --slot;
        }
        positions[slot] = position;
    }
}

int rtl_joint_select(const struct rtl_joint_input *input,
                     struct rtl_joint_selection *result) {
    return rtl_joint_select_allowed(input, NULL, result);
}

int rtl_joint_select_allowed(const struct rtl_joint_input *input,
    const uint8_t *future_admission_allowed, struct rtl_joint_selection *result) {
    if (!input || !result || !input->base_token_count ||
        input->base_token_count > RTL_JOINT_MAX_ROWS || input->future_token_count > RTL_FUTURE_MAX_ROWS ||
        !input->target_token_count || input->target_token_count > RTL_JOINT_MAX_ROWS ||
        !input->max_next_tokens || input->max_next_tokens > RTL_FUTURE_MAX_ROWS ||
        input->prediction_target > 64 || input->current_prediction_tokens > input->base_token_count ||
        input->retry_min_confidence > 0x3f80 || input->min_reuse_score > 0x3f80 ||
        input->next_block_start > RTL_REFRESH_MAX_SEQUENCE - input->future_token_count) return -1;
    uint8_t existing[RTL_FUTURE_MAX_ROWS] = {0};
    uint8_t existing_order[RTL_FUTURE_MAX_ROWS], added_order[RTL_FUTURE_MAX_ROWS];
    uint8_t a4_order[RTL_FUTURE_MAX_ROWS], a8_order[RTL_FUTURE_MAX_ROWS];
    uint32_t base_a8 = 0, forecast_a8 = 0, existing_count = 0, added_future_token_count = 0;
    uint32_t total_eligible = 0;
    for (uint32_t i = 0; i < input->base_token_count; ++i) {
        if ((input->base_bits[i] != 4 && input->base_bits[i] != 8) ||
            (input->forecast_bits[i] != 4 && input->forecast_bits[i] != 8) ||
            input->base_positions[i] >= RTL_REFRESH_MAX_SEQUENCE) return -1;
        for (uint32_t j = 0; j < i; ++j)
            if (input->base_positions[i] == input->base_positions[j]) return -1;
        base_a8 += input->base_bits[i] == 8;
        forecast_a8 += input->forecast_bits[i] == 8;
        if (input->base_positions[i] >= input->next_block_start &&
            input->base_positions[i] < input->next_block_start + input->future_token_count)
            existing[input->base_positions[i] - input->next_block_start] = 1;
    }
    for (uint32_t i = 0; i < input->future_token_count; ++i) {
        if ((input->future_bits[i] != 4 && input->future_bits[i] != 8) ||
            input->unresolved[i] > 1 || input->tentative[i] > 1 ||
            (input->priority[i] & 0x7f80u) == 0x7f80u) return -1;
        if (!input->unresolved[i] || rtl_bf16_compare(input->priority[i], input->min_reuse_score) < 0) continue;
        ++total_eligible;
        if (existing[i]) existing_order[existing_count++] = (uint8_t)i;
        else if (input->tentative[i] || ((!future_admission_allowed || future_admission_allowed[i]) &&
            (!input->retry_min_confidence || !input->future_admission_attempts[i] ||
             rtl_bf16_compare(input->last_confidence[i], input->retry_min_confidence) >= 0)))
            added_order[added_future_token_count++] = (uint8_t)i;
    }
    order_future(input, existing_order, existing_count, 0);
    order_future(input, added_order, added_future_token_count, 1);
    uint32_t progress_limit = input->max_next_tokens, existing_limit = progress_limit;
    if (input->prediction_target) {
        uint32_t reused_confirmation_count = 0, added_confirmation_count = 0;
        for (uint32_t i = 0; i < existing_count; ++i) reused_confirmation_count += input->tentative[existing_order[i]];
        for (uint32_t i = 0; i < added_future_token_count; ++i) added_confirmation_count += input->tentative[added_order[i]];
        uint32_t remaining = input->prediction_target > input->current_prediction_tokens ?
            input->prediction_target - input->current_prediction_tokens : 0;
        if (remaining < reused_confirmation_count + added_confirmation_count) remaining = reused_confirmation_count + added_confirmation_count;
        if (progress_limit > remaining) progress_limit = remaining;
        existing_limit = progress_limit > added_confirmation_count ? progress_limit - added_confirmation_count : 0;
    }
    memset(result, 0, sizeof(*result));
    result->base = residency(input->base_token_count - base_a8, base_a8);
    result->joint = result->base;
    result->forecast = residency(input->base_token_count - forecast_a8, forecast_a8);
    const uint32_t forecast_fragments = result->forecast.fragments[2];
    for (uint32_t i = 0; i < existing_count && result->future_prediction_count < existing_limit; ++i) {
        const uint8_t local = existing_order[i];
        uint32_t promote = 0;
        for (uint32_t j = 0; j < input->base_token_count; ++j)
            if (input->base_positions[j] == input->next_block_start + local)
                promote += input->forecast_bits[j] == 4;
        const struct rtl_activation_residency candidate =
            residency(input->base_token_count - forecast_a8 - promote, forecast_a8 + promote);
        if (candidate.fragments[2] > forecast_fragments) {
            ++result->rejected_forecast;
            continue;
        }
        result->progress[result->future_prediction_count++] = local;
        forecast_a8 += promote;
        result->forecast = candidate;
    }
    uint32_t a4_count = 0, a8_count = 0;
    for (uint32_t i = 0; i < added_future_token_count; ++i) {
        const uint8_t local = added_order[i];
        if (input->future_bits[local] == 4) a4_order[a4_count++] = local;
        else a8_order[a8_count++] = local;
    }
    uint32_t add_limit = input->target_token_count > input->base_token_count ?
        input->target_token_count - input->base_token_count : 0;
    if (add_limit > progress_limit - result->future_prediction_count)
        add_limit = progress_limit - result->future_prediction_count;
    uint32_t max_current = 0, max_forecast = 0, best_a4 = 0, best_a8 = 0;
    int best_valid = 0;
    int32_t best_selection_key[4] = {0};
    for (uint32_t a4 = 0; a4 <= a4_count && a4 <= add_limit; ++a4)
        for (uint32_t a8 = 0; a8 <= a8_count && a4 + a8 <= add_limit; ++a8) {
            const uint32_t count = a4 + a8;
            const struct rtl_activation_residency candidate =
                residency(input->base_token_count - base_a8 + a4, base_a8 + a8);
            if (candidate.fragments[2] > result->base.fragments[2]) continue;
            if (count > max_current) max_current = count;
            const struct rtl_activation_residency forecast =
                residency(input->base_token_count - forecast_a8, forecast_a8 + count);
            if (forecast.fragments[2] > forecast_fragments) continue;
            if (count > max_forecast) max_forecast = count;
            uint32_t tentative = 0;
            uint16_t priority_sum = 0;
            /* Match the A4-prefix then A8-prefix BF16 sum before returned reordering. */
            for (uint32_t i = 0; i < count; ++i) {
                const uint8_t local = i < a4 ? a4_order[i] : a8_order[i - a4];
                tentative += input->tentative[local];
                priority_sum = rtl_bf16_add(priority_sum, input->priority[local]);
            }
            const int32_t score[4] = {(int32_t)tentative, priority_sum,
                (int32_t)count, (int32_t)(candidate.a4_token_count + 2 * candidate.a8_token_count)};
            int better = !best_valid;
            for (uint32_t i = 0; best_valid && i < 4; ++i) {
                const int comparison = i == 1 ? rtl_bf16_compare(
                    (uint16_t)score[i], (uint16_t)best_selection_key[i]) :
                    (score[i] > best_selection_key[i]) - (score[i] < best_selection_key[i]);
                if (comparison) { better = comparison > 0; break; }
            }
            if (!better) continue;
            best_valid = 1;
            memcpy(best_selection_key, score, sizeof(score));
            best_a4 = a4;
            best_a8 = a8;
            result->joint = candidate;
            result->forecast = forecast;
        }
    if (input->source_b_a4_only) {
        int reject = 0;
        for (uint32_t i = 0; i < best_a8; ++i)
            reject |= !input->tentative[a8_order[i]];
        if (reject) {
            uint32_t due4 = 0, due8 = 0;
            for (uint32_t i = 0; i < best_a4; ++i) due4 += input->tentative[a4_order[i]];
            for (uint32_t i = 0; i < best_a8; ++i) due8 += input->tentative[a8_order[i]];
            best_a4 = due4; best_a8 = due8;
            result->joint = residency(input->base_token_count - base_a8 + due4, base_a8 + due8);
            result->forecast = residency(input->base_token_count - forecast_a8, forecast_a8 + due4 + due8);
        }
    }
    uint8_t selected[RTL_FUTURE_MAX_ROWS] = {0};
    for (uint32_t i = 0; i < best_a4; ++i) selected[a4_order[i]] = 1;
    for (uint32_t i = 0; i < best_a8; ++i) selected[a8_order[i]] = 1;
    for (uint32_t i = 0; i < added_future_token_count; ++i) {
        const uint8_t local = added_order[i];
        if (!selected[local]) continue;
        result->added[result->added_future_token_count] = local;
        result->added_bits[result->added_future_token_count++] = input->future_bits[local];
        result->progress[result->future_prediction_count++] = local;
    }
    const uint32_t without_capacity = added_future_token_count < add_limit ? added_future_token_count : add_limit;
    result->rejected_current = without_capacity - max_current;
    result->rejected_forecast += max_current - max_forecast;
    const uint32_t accounted = result->future_prediction_count + result->rejected_current + result->rejected_forecast;
    result->rejected_budget = total_eligible > accounted ? total_eligible - accounted : 0;
    return 0;
}

int rtl_boundary_deep_precision(uint32_t rows, const uint8_t *input_bits,
    const uint8_t *scores, const uint8_t *protected_rows, uint32_t a8_limit,
    uint8_t *output_bits) {
    if (rows > RTL_REFRESH_MAX_SEQUENCE || !input_bits || !scores ||
        !protected_rows || !output_bits) return -1;
    uint32_t protected_a8 = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        if ((input_bits[i] != 4 && input_bits[i] != 8) || protected_rows[i] > 1) return -1;
        protected_a8 += input_bits[i] == 8 && protected_rows[i];
    }
    const uint32_t available = a8_limit > protected_a8 ? a8_limit - protected_a8 : 0;
    uint8_t result[RTL_REFRESH_MAX_SEQUENCE];
    for (uint32_t i = 0; i < rows; ++i) {
        result[i] = input_bits[i];
        if (input_bits[i] != 8 || protected_rows[i]) continue;
        uint32_t rank = 0;
        for (uint32_t j = 0; j < rows; ++j)
            if (input_bits[j] == 8 && !protected_rows[j] &&
                (scores[j] > scores[i] || (scores[j] == scores[i] && j < i))) ++rank;
        if (rank >= available) result[i] = 4;
    }
    memcpy(output_bits, result, rows);
    return 0;
}

size_t rtl_boundary_selection_bytes(void) {
    return sizeof(struct rtl_boundary_selection);
}

static void stable_score_order(uint16_t *positions, uint32_t count,
                               const uint8_t *scores) {
    for (uint32_t index = 1; index < count; ++index) {
        const uint16_t position = positions[index];
        uint32_t slot = index;
        while (slot && scores[position] > scores[positions[slot - 1]]) {
            positions[slot] = positions[slot - 1];
            --slot;
        }
        positions[slot] = position;
    }
}

int rtl_boundary_select(
    uint32_t sequence_length, const uint8_t *scores, uint32_t target_token_count,
    uint32_t current_count, const uint16_t *current,
    uint32_t eligible_count, const uint16_t *eligible,
    uint32_t mandatory_count, const uint16_t *mandatory,
    uint32_t required_count, const uint16_t *required, uint32_t required_quota,
    struct rtl_boundary_selection *result) {
    uint8_t membership[RTL_REFRESH_MAX_SEQUENCE] = {0};
    uint16_t required_order[RTL_REFRESH_MAX_SEQUENCE];
    if (!scores || !result || !current || !current_count ||
        !sequence_length || sequence_length > RTL_REFRESH_MAX_SEQUENCE ||
        current_count > target_token_count || target_token_count > sequence_length ||
        eligible_count > sequence_length || required_count > eligible_count ||
        mandatory_count > target_token_count - current_count ||
        eligible_count < target_token_count - current_count ||
        required_quota > required_count ||
        required_quota > target_token_count - current_count ||
        (eligible_count && !eligible) || (mandatory_count && !mandatory) ||
        (required_count && !required)) return -1;
    for (uint32_t i = 0; i < current_count; ++i) {
        if (current[i] >= sequence_length || membership[current[i]]) return -1;
        membership[current[i]] = 1;
    }
    for (uint32_t i = 0; i < eligible_count; ++i) {
        if (eligible[i] >= sequence_length || membership[eligible[i]]) return -1;
        membership[eligible[i]] = 2;
    }
    for (uint32_t i = 0; i < mandatory_count; ++i) {
        if (mandatory[i] >= sequence_length || membership[mandatory[i]] != 2) return -1;
        membership[mandatory[i]] |= 4;
    }
    uint32_t remaining_required = 0;
    uint32_t mandatory_required = 0;
    for (uint32_t i = 0; i < required_count; ++i) {
        const uint16_t position = required[i];
        if (position >= sequence_length || !(membership[position] & 2) ||
            (membership[position] & 8)) return -1;
        membership[position] |= 8;
        if (membership[position] & 4) ++mandatory_required;
        else required_order[remaining_required++] = position;
    }
    const uint32_t quota = required_quota > mandatory_required ?
        required_quota - mandatory_required : 0;
    const uint32_t optional_count = target_token_count - current_count;
    if (quota > remaining_required || quota > optional_count - mandatory_count)
        return -1;

    memset(result, 0, sizeof(*result));
    result->deep_count = target_token_count;
    result->optional_count = optional_count;
    result->ranked_count = eligible_count;
    for (uint32_t i = 0; i < eligible_count; ++i)
        result->ranked_candidates[i] = eligible[i];
    stable_score_order(result->ranked_candidates, eligible_count, scores);
    stable_score_order(required_order, remaining_required, scores);
    uint32_t selected = 0;
    for (uint32_t i = 0; i < mandatory_count; ++i)
        result->selected_optional[selected++] = mandatory[i];
    for (uint32_t i = 0; i < quota; ++i) {
        result->selected_optional[selected++] = required_order[i];
        membership[required_order[i]] |= 4;
    }
    for (uint32_t i = 0; i < eligible_count && selected < optional_count; ++i) {
        const uint16_t position = result->ranked_candidates[i];
        if (!(membership[position] & 4)) {
            result->selected_optional[selected++] = position;
            membership[position] |= 4;
        }
    }
    uint32_t deep = 0;
    for (uint32_t position = 0; position < sequence_length; ++position)
        if (membership[position] & 5)
            result->deep_positions[deep++] = (uint16_t)position;
    return deep == target_token_count ? 0 : -1;
}

static uint16_t dependency_value(uint16_t value) {
    if ((value & 0x7fffu) == 0) return value;
    if ((value & 0x7f80u) == 0x7f80u || (value & 0x8000u)) return 0;
    return value > 0x3f80u ? 0x3f80u : value;
}

static uint16_t confidence_value(uint16_t value) {
    if ((value & 0x7fffu) == 0) return value;
    if ((value & 0x7fffu) > 0x7f80u || (value & 0x8000u)) return 0;
    return value > 0x3f80u ? 0x3f80u : value;
}

static uint16_t invalidation_row(uint32_t keys, const uint16_t *dependency,
    const uint8_t *changed, const uint8_t *remasked, const uint16_t *confidence,
    int all_changes, uint16_t *actual_remask) {
    uint16_t ordinary_max = 0, remask_union = 0;
    uint8_t have_ordinary = 0, have_remask = 0;
    for (uint32_t key = 0; key < keys; ++key) {
        if (!changed[key]) continue;
        uint16_t risk = dependency[key];
        if (!remasked[key]) {
            if (all_changes) {
                risk = rtl_bf16_mul(risk, rtl_bf16_add(
                    0x4000u, confidence_value(confidence[key]) ^ 0x8000u));
                if (rtl_bf16_compare(risk, 0x3f80u) > 0) risk = 0x3f80u;
            }
            if (!have_ordinary || rtl_bf16_compare(risk, ordinary_max) > 0)
                ordinary_max = risk;
            have_ordinary = 1;
            continue;
        }
        const uint16_t severity = rtl_bf16_add(
            0x3f80u, confidence_value(confidence[key]) ^ 0x8000u);
        risk = rtl_bf16_mul(risk, rtl_bf16_add(0x3f80u, severity));
        if (rtl_bf16_compare(risk, 0x3f80u) > 0) risk = 0x3f80u;
        if (!have_remask) {
            remask_union = risk;
            have_remask = 1;
        } else {
            const uint16_t product = rtl_bf16_mul(remask_union, risk);
            remask_union = rtl_bf16_add(remask_union,
                rtl_bf16_add(risk, product ^ 0x8000u));
        }
    }
    *actual_remask = remask_union;
    return !have_remask || rtl_bf16_compare(ordinary_max, remask_union) >= 0 ?
        ordinary_max : remask_union;
}

size_t rtl_cross_block_state_bytes(void) { return sizeof(struct rtl_cross_block_state); }

int rtl_cross_block_init(struct rtl_cross_block_state *state,
                         uint32_t rows, uint32_t block_start) {
    if (!state || rows <= 32 || rows > RTL_REFRESH_MAX_SEQUENCE ||
        block_start > rows - 32) return -1;
    memset(state, 0, sizeof(*state));
    state->rows = rows;
    state->block_start = block_start;
    return 0;
}

int rtl_cross_block_advance(struct rtl_cross_block_state *state, uint32_t block_start) {
    if (!state || state->rows <= 32 || state->rows > RTL_REFRESH_MAX_SEQUENCE ||
        block_start < state->block_start + 32 || block_start > state->rows - 32) return -1;
    state->block_start = block_start;
    memset(state->relation, 0, sizeof(state->relation));
    for (uint32_t row = 0; row < block_start + 32; ++row) {
        if (state->future_pending[row] > state->pending[row])
            state->pending[row] = state->future_pending[row];
        if (state->future_actual_remask_pending[row] > state->actual_remask_pending[row])
            state->actual_remask_pending[row] = state->future_actual_remask_pending[row];
        state->future_pending[row] = 0;
        state->future_actual_remask_pending[row] = 0;
    }
    return 0;
}

int rtl_cross_block_observe(struct rtl_cross_block_state *state,
    uint32_t query_count, const uint16_t *query_positions,
    const uint16_t *query_relation, uint32_t consumed_count,
    const uint16_t *consumed_positions, const uint8_t *changed,
    const uint8_t *remasked, const uint16_t *confidence) {
    uint8_t seen[RTL_REFRESH_MAX_SEQUENCE] = {0};
    if (!state || !changed || !remasked || !confidence || state->rows <= 32 ||
        state->rows > RTL_REFRESH_MAX_SEQUENCE || state->block_start > state->rows - 32 ||
        query_count > state->rows || consumed_count > state->rows ||
        (query_count && (!query_positions || !query_relation)) ||
        (consumed_count && !consumed_positions)) return -1;
    for (uint32_t i = 0; i < query_count; ++i) {
        if (query_positions[i] >= state->rows || seen[query_positions[i]]) return -1;
        seen[query_positions[i]] = 1;
    }
    for (uint32_t i = 0; i < consumed_count; ++i) {
        if (consumed_positions[i] >= state->rows || (seen[consumed_positions[i]] & 2)) return -1;
        seen[consumed_positions[i]] |= 2;
    }
    for (uint32_t row = 0; row < state->rows; ++row)
        if (changed[row] > 1 || remasked[row] > 1 || (remasked[row] && !changed[row])) return -1;
    for (uint32_t i = 0; i < consumed_count; ++i) {
        const uint16_t row = consumed_positions[i];
        state->pending[row] = state->actual_remask_pending[row] = 0;
        state->actual_remask_epoch_pending[row] = state->future_pending[row] = 0;
        state->future_actual_remask_pending[row] = 0;
    }
    for (uint32_t i = 0; i < query_count; ++i)
        for (uint32_t key = 0; key < 32; ++key)
            state->relation[query_positions[i] * 32 + key] =
                dependency_value(query_relation[i * 32 + key]);
    for (uint32_t row = 0; row < state->rows; ++row) {
        uint16_t remask_risk = 0;
        const uint16_t risk = invalidation_row(32, &state->relation[row * 32],
            changed + state->block_start, remasked + state->block_start,
            confidence + state->block_start, 1, &remask_risk);
        uint16_t *pending = row < state->block_start + 32 ?
            state->pending : state->future_pending;
        uint16_t *remask_pending = row < state->block_start + 32 ?
            state->actual_remask_pending : state->future_actual_remask_pending;
        if (rtl_bf16_compare(risk, pending[row]) > 0) pending[row] = risk;
        if (rtl_bf16_compare(remask_risk, remask_pending[row]) > 0)
            remask_pending[row] = remask_risk;
        if (row < state->block_start + 32 &&
            rtl_bf16_compare(remask_risk, state->actual_remask_epoch_pending[row]) > 0)
            state->actual_remask_epoch_pending[row] = remask_risk;
    }
    return 0;
}

size_t rtl_refresh_state_bytes(void) { return sizeof(struct rtl_refresh_state); }

int rtl_refresh_init(struct rtl_refresh_state *state, uint32_t rows,
                     uint32_t target_token_count_x2, const uint16_t *dependency) {
    if (!state || !dependency || rows == 0 || rows > RTL_REFRESH_MAX_ROWS ||
        target_token_count_x2 < 2 || target_token_count_x2 > rows * 2) return -1;
    memset(state, 0, sizeof(*state));
    state->rows = rows;
    state->target_token_count_x2 = target_token_count_x2;
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t key = 0; key < rows; ++key)
            state->dependency[row * RTL_REFRESH_MAX_ROWS + key] =
                dependency_value(dependency[row * rows + key]);
    return 0;
}

int rtl_refresh_observe(struct rtl_refresh_state *state,
                        uint32_t query_count, const uint16_t *query_positions,
                        const uint16_t *query_dependency,
                        const uint8_t *predicted, const uint8_t *changed,
                        const uint8_t *remasked, const uint16_t *confidence) {
    uint8_t seen[RTL_REFRESH_MAX_ROWS] = {0};
    if (!state || !predicted || !changed || !remasked || !confidence ||
        state->rows == 0 || state->rows > RTL_REFRESH_MAX_ROWS ||
        state->target_token_count_x2 < 2 || state->target_token_count_x2 > state->rows * 2 ||
        state->regular_steps == UINT32_MAX ||
        state->cumulative_active_token_count > UINT32_MAX - state->rows ||
        query_count > state->rows ||
        (query_count && (!query_positions || !query_dependency))) return -1;
    for (uint32_t query = 0; query < query_count; ++query) {
        const uint16_t position = query_positions[query];
        if (position >= state->rows || seen[position]) return -1;
        seen[position] = 1;
    }
    for (uint32_t row = 0; row < state->rows; ++row)
        if (predicted[row] > 1 || changed[row] > 1 || remasked[row] > 1 ||
            (remasked[row] && !changed[row])) return -1;

    for (uint32_t row = 0; row < state->rows; ++row)
        if (state->refresh[row]) state->pending[row] = 0;
    for (uint32_t query = 0; query < query_count; ++query)
        for (uint32_t key = 0; key < state->rows; ++key)
            state->dependency[query_positions[query] * RTL_REFRESH_MAX_ROWS + key] =
                dependency_value(query_dependency[query * state->rows + key]);

    uint32_t selected = 0;
    for (uint32_t row = 0; row < state->rows; ++row) {
        uint16_t remask_risk = 0;
        const uint16_t invalidation = invalidation_row(state->rows,
            &state->dependency[row * RTL_REFRESH_MAX_ROWS], changed, remasked,
            confidence, 0, &remask_risk);
        state->new_invalidation[row] = invalidation;
        if (rtl_bf16_compare(invalidation, state->pending[row]) > 0)
            state->pending[row] = invalidation;
        state->mandatory[row] = predicted[row] || changed[row];
        state->refresh[row] = state->mandatory[row];
        state->optional_selected[row] = 0;
        selected += state->mandatory[row];
    }
    /* Signed division matches int(cumulative_limit - consumed), including debt. */
    int64_t allowed = ((int64_t)state->target_token_count_x2 *
        ((int64_t)state->regular_steps + 1) -
        2 * (int64_t)state->cumulative_active_token_count) / 2;
    if (allowed < selected) allowed = selected;
    if (allowed > state->rows) allowed = state->rows;
    state->allowed_active_token_count = (uint32_t)allowed;
    while (selected < state->allowed_active_token_count) {
        uint32_t best = state->rows;
        uint16_t best_selection_key = 0;
        for (uint32_t row = 0; row < state->rows; ++row)
            if (!state->refresh[row] && rtl_bf16_compare(state->pending[row], best_selection_key) > 0) {
                best = row;
                best_selection_key = state->pending[row];
            }
        if (best == state->rows) break;
        state->refresh[best] = 1;
        state->optional_selected[best] = 1;
        ++selected;
    }
    ++state->regular_steps;
    state->cumulative_active_token_count += selected;
    return 0;
}
