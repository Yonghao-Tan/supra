#include "forward_postprocess_model.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int bf16_finite(uint16_t value) {
    return (value & 0x7f80u) != 0x7f80u;
}

static int bf16_positive_finite(uint16_t value) {
    return bf16_finite(value) && (value & 0x8000u) == 0u &&
           (value & 0x7fffu) != 0u;
}

static uint16_t bf16_negate(uint16_t value) {
    return (uint16_t)(value ^ 0x8000u);
}

static uint8_t candidate_exp_lut_address(uint16_t delta) {
    unsigned exponent;
    unsigned mantissa;
    int exponent_value;
    unsigned total_shift;
    uint32_t numerator;
    uint32_t quotient;
    uint32_t remainder;
    uint32_t denominator;
    if ((delta & 0x8000u) == 0u || (delta & 0x7fffu) == 0u) return 255u;
    exponent = (delta >> 7u) & 0xffu;
    if (exponent >= 0x83u) return 0u;
    mantissa = exponent == 0u ? delta & 0x7fu : 0x80u | (delta & 0x7fu);
    exponent_value = exponent == 0u ? -126 : (int)exponent - 127;
    if (exponent_value < -5) return 255u;
    total_shift = (unsigned)(7 - exponent_value + 4);
    numerator = ((1u << total_shift) - mantissa) * 255u;
    quotient = numerator >> total_shift;
    denominator = 1u << total_shift;
    remainder = numerator & (denominator - 1u);
    if ((remainder << 1u) > denominator ||
        ((remainder << 1u) == denominator && (quotient & 1u))) ++quotient;
    return quotient > 255u ? 255u : (uint8_t)quotient;
}

static uint16_t candidate_exp(uint16_t delta, const uint16_t *exp_lut) {
    return exp_lut[candidate_exp_lut_address(delta)];
}

static uint16_t candidate_reciprocal(uint16_t sum_value,
                                     const uint16_t *reciprocal_lut) {
    unsigned exponent = (sum_value >> 7u) & 0xffu;
    unsigned numerator;
    unsigned quotient;
    unsigned remainder;
    uint16_t lut_value;
    int sum_exponent;
    int reciprocal_exponent;
    if ((sum_value & 0x8000u) != 0u || exponent == 0u || exponent == 0xffu)
        return 0x7fc0u;
    numerator = (sum_value & 0x7fu) * 255u;
    quotient = numerator >> 7u;
    remainder = numerator & 0x7fu;
    if (remainder > 64u || (remainder == 64u && (quotient & 1u))) ++quotient;
    if (quotient > 255u) quotient = 255u;
    lut_value = reciprocal_lut[quotient];
    sum_exponent = (int)exponent - 127;
    reciprocal_exponent = (int)((lut_value >> 7u) & 0xffu) - sum_exponent;
    if (((lut_value >> 7u) & 0xffu) == 0xffu) return lut_value;
    if (reciprocal_exponent >= 255)
        return (uint16_t)((lut_value & 0x8000u) | 0x7f80u);
    if (reciprocal_exponent <= 0) return (uint16_t)(lut_value & 0x8000u);
    return (uint16_t)((lut_value & 0x807fu) |
                      ((uint16_t)reciprocal_exponent << 7u));
}

static int bf16_ge(uint16_t lhs, uint16_t rhs) {
    return rtl_bf16_compare(lhs, rhs) >= 0;
}

static int bf16_lt(uint16_t lhs, uint16_t rhs) {
    return rtl_bf16_compare(lhs, rhs) < 0;
}

static unsigned popcount32(uint32_t value) {
    unsigned count = 0u;
    while (value != 0u) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

static void write_physical_row(
    const struct forward_postprocess_next_token_descriptor *token_ordinals,
    size_t logical_index, uint8_t compute_group, uint8_t pe_slot,
    uint8_t phase_mask, struct forward_postprocess_physical_row *physical_rows,
    uint8_t *logical_to_physical, size_t *physical_count) {
    const struct forward_postprocess_next_token_descriptor *source =
        &token_ordinals[logical_index];
    struct forward_postprocess_physical_row *destination =
        &physical_rows[*physical_count];
    destination->source_index = source->source_index;
    destination->token_position = source->token_position;
    destination->kv_index = source->kv_index;
    destination->token_ordinal = (uint8_t)logical_index;
    destination->source = source->source;
    destination->activation_bits = source->activation_bits;
    destination->query_group = source->query_group;
    destination->cache_group = source->cache_group;
    destination->compute_group = compute_group;
    destination->pe_slot = pe_slot;
    destination->phase_mask = phase_mask;
    logical_to_physical[logical_index] = (uint8_t)*physical_count;
    ++*physical_count;
}

int forward_postprocess_pack_next_tokens(
    const struct forward_postprocess_next_token_descriptor *token_ordinals,
    size_t row_count, uint16_t sequence_length,
    struct forward_postprocess_physical_row *physical_rows,
    uint8_t *logical_to_physical, uint8_t *compute_group_count,
    uint8_t *semantic_group_count) {
    uint8_t group_query[FORWARD_POSTPROCESS_MAX_RESIDENT_ROWS];
    uint8_t group_cache[FORWARD_POSTPROCESS_MAX_RESIDENT_ROWS];
    size_t group_count = 0u;
    size_t physical_count = 0u;
    size_t row;
    size_t group;

    if (token_ordinals == NULL || physical_rows == NULL ||
        logical_to_physical == NULL || compute_group_count == NULL ||
        semantic_group_count == NULL || row_count == 0u ||
        row_count > FORWARD_POSTPROCESS_MAX_RESIDENT_ROWS || sequence_length == 0u ||
        sequence_length > 2048u) {
        return -1;
    }
    for (row = 0u; row < row_count; ++row) {
        size_t other;
        size_t matching_group = group_count;
        if ((token_ordinals[row].source == FORWARD_POSTPROCESS_EMBEDDING_TOKEN &&
             token_ordinals[row].source_index >= 126464u) ||
            (token_ordinals[row].source == FORWARD_POSTPROCESS_RESIDENT_HIDDEN &&
             token_ordinals[row].source_index >= 2048u) ||
            token_ordinals[row].source > FORWARD_POSTPROCESS_EMBEDDING_TOKEN ||
            (token_ordinals[row].activation_bits != 4u &&
             token_ordinals[row].activation_bits != 8u) ||
            token_ordinals[row].token_position >= sequence_length ||
            token_ordinals[row].kv_index >= sequence_length) {
            return -1;
        }
        for (other = 0u; other < row; ++other) {
            if (token_ordinals[other].token_position ==
                token_ordinals[row].token_position) {
                return -1;
            }
        }
        for (group = 0u; group < group_count; ++group) {
            if (group_query[group] == token_ordinals[row].query_group &&
                group_cache[group] == token_ordinals[row].cache_group) {
                matching_group = group;
                break;
            }
        }
        if (matching_group == group_count) {
            group_query[group_count] = token_ordinals[row].query_group;
            group_cache[group_count] = token_ordinals[row].cache_group;
            ++group_count;
        }
    }

    *compute_group_count = 0u;
    *semantic_group_count = (uint8_t)group_count;
    /* Each compute group starts at the first unassigned logical token.
     * Revisit semantic groups after each physical group, as token_issue_packer
     * does; exhausting one semantic group first changes the inverse mapping. */
    memset(logical_to_physical, 0xff, row_count);
    while (physical_count < row_count) {
        size_t first = 0u;
        while (first < row_count && logical_to_physical[first] != 0xffu) ++first;
        uint8_t a8_rows[FORWARD_POSTPROCESS_MAX_RESIDENT_ROWS];
        uint8_t a4_rows[FORWARD_POSTPROCESS_MAX_RESIDENT_ROWS];
        size_t a8_count = 0u;
        size_t a4_count = 0u;
        for (row = 0u; row < row_count; ++row) {
            if (logical_to_physical[row] == 0xffu &&
                token_ordinals[row].query_group == token_ordinals[first].query_group &&
                token_ordinals[row].cache_group == token_ordinals[first].cache_group) {
                if (token_ordinals[row].activation_bits == 8u)
                    a8_rows[a8_count++] = (uint8_t)row;
                else
                    a4_rows[a4_count++] = (uint8_t)row;
            }
        }
        {
            const size_t remaining_a8 = a8_count;
            const size_t remaining_a4 = a4_count;
            const size_t selected_a8 = remaining_a8 < 8u ? remaining_a8 : 8u;
            const size_t phase_capacity = 8u - selected_a8;
            const size_t a4_capacity = selected_a8 == 0u && remaining_a4 <= 8u ?
                8u : 2u * phase_capacity;
            const size_t selected_a4 = remaining_a4 < a4_capacity ?
                remaining_a4 : a4_capacity;
            const size_t phase0_a4 = selected_a4 < phase_capacity ?
                selected_a4 : phase_capacity;
            size_t index;

            if (selected_a8 == 0u && selected_a4 == 0u) {
                return -1;
            }
            for (index = 0u; index < selected_a8; ++index) {
                write_physical_row(
                    token_ordinals, a8_rows[index],
                    *compute_group_count, (uint8_t)index, 3u,
                    physical_rows, logical_to_physical, &physical_count);
            }
            for (index = 0u; index < selected_a4; ++index) {
                const uint8_t phase_mask = index < phase0_a4 ? 1u : 2u;
                const size_t phase_index = index < phase0_a4 ?
                    index : index - phase0_a4;
                write_physical_row(
                    token_ordinals, a4_rows[index],
                    *compute_group_count,
                    (uint8_t)(selected_a8 + phase_index), phase_mask,
                    physical_rows, logical_to_physical, &physical_count);
            }
            ++*compute_group_count;
        }
    }
    return physical_count == row_count ? 0 : -1;
}

int forward_postprocess_embedding_address(uint64_t embedding_base,
                                 uint64_t embedding_limit,
                                 uint32_t token_id,
                                 uint64_t *row_address) {
    uint64_t offset;
    uint64_t address;
    uint64_t end_address;
    uint64_t table_end;
    if (row_address == NULL || token_id >= FORWARD_POSTPROCESS_EMBEDDING_VOCABULARY ||
        (embedding_base & (FORWARD_POSTPROCESS_EMBEDDING_ROW_BYTES - 1u)) != 0u ||
        embedding_base > UINT64_MAX - FORWARD_POSTPROCESS_EMBEDDING_TABLE_BYTES) {
        return -1;
    }
    table_end = embedding_base + FORWARD_POSTPROCESS_EMBEDDING_TABLE_BYTES;
    if (embedding_limit < table_end) {
        return -1;
    }
    offset = (uint64_t)token_id << 13u;
    if (embedding_base > UINT64_MAX - offset) {
        return -1;
    }
    address = embedding_base + offset;
    if (address > UINT64_MAX - FORWARD_POSTPROCESS_EMBEDDING_ROW_BYTES) {
        return -1;
    }
    end_address = address + FORWARD_POSTPROCESS_EMBEDDING_ROW_BYTES;
    if (end_address > embedding_limit) {
        return -1;
    }
    *row_address = address;
    return 0;
}

size_t forward_postprocess_token_metadata_token_batch_bytes(size_t resident_rows) {
    size_t inverse_bytes;
    if (resident_rows == 0u || resident_rows > FORWARD_POSTPROCESS_MAX_RESIDENT_ROWS) {
        return 0u;
    }
    inverse_bytes = (resident_rows + 15u) & ~(size_t)15u;
    return sizeof(struct forward_postprocess_token_batch_header) +
        resident_rows * sizeof(struct forward_postprocess_physical_row) + inverse_bytes;
}

int forward_postprocess_final_rmsnorm_a8(const uint16_t *final_hidden, size_t rows,
                                size_t channels, uint16_t epsilon_bf16,
                                uint16_t *normalized_bf16,
                                int8_t *activation_codes,
                                uint16_t *activation_scales_bf16) {
    size_t row;
    if (final_hidden == NULL || normalized_bf16 == NULL ||
        activation_codes == NULL || activation_scales_bf16 == NULL ||
        rows == 0u || rows > FORWARD_POSTPROCESS_MAX_ROWS || channels == 0u) return -1;
    if (rtl_rmsnorm_bf16(final_hidden, NULL, rows, channels, epsilon_bf16,
                         normalized_bf16) != 0) return -2;
    for (row = 0u; row < rows; ++row) {
        if (rtl_quantize_row_bf16(normalized_bf16 + row * channels, channels,
                                 127, activation_codes + row * channels,
                                 activation_scales_bf16 + row) != 0) return -3;
    }
    return 0;
}

int forward_postprocess_raw_w8_matmul(const int8_t *activation_codes,
                             const uint16_t *activation_scales_bf16,
                             size_t rows, size_t input_features,
                             const int8_t *weight_codes,
                             const uint16_t *weight_scales_bf16,
                             size_t output_features, uint16_t *logits_bf16,
                             int32_t *accumulators) {
    size_t row;
    if (activation_codes == NULL || activation_scales_bf16 == NULL ||
        weight_codes == NULL || weight_scales_bf16 == NULL ||
        logits_bf16 == NULL || rows == 0u || rows > FORWARD_POSTPROCESS_MAX_ROWS ||
        input_features == 0u || input_features > 4096u ||
        output_features == 0u) return -1;
    for (row = 0u; row < rows; ++row) {
        size_t output;
        if (!bf16_positive_finite(activation_scales_bf16[row])) return -2;
        for (output = 0u; output < output_features; ++output) {
            int64_t sum = 0;
            size_t channel;
            if (!bf16_positive_finite(weight_scales_bf16[output])) return -2;
            for (channel = 0u; channel < input_features; ++channel) {
                int activation = activation_codes[row * input_features + channel];
                int weight = weight_codes[output * input_features + channel];
                if (activation < -127 || weight < -127) return -3;
                sum += (int64_t)activation * (int64_t)weight;
            }
            if (sum < -(1ll << 26) || sum > (1ll << 26) - 1ll) return -4;
            if (accumulators != NULL)
                accumulators[row * output_features + output] = (int32_t)sum;
            logits_bf16[row * output_features + output] = rtl_bf16_mul(
                rtl_f32_to_bf16((float)sum *
                    rtl_bf16_to_f32(activation_scales_bf16[row])),
                weight_scales_bf16[output]);
        }
    }
    return 0;
}

int forward_postprocess_candidate_bf16(
    const uint16_t *logits_bf16, size_t rows, size_t vocabulary_size,
    const uint32_t *selected_token_ids, const uint32_t *suppressed_token_ids,
    size_t suppressed_token_count, const uint16_t *exp_lut,
    const uint16_t *reciprocal_lut,
    struct forward_postprocess_candidate_result *results) {
    size_t row;
    uint16_t *lane_max;
    uint16_t *lane_sum;
    uint32_t *lane_id;
    if (logits_bf16 == NULL || selected_token_ids == NULL || exp_lut == NULL ||
        reciprocal_lut == NULL || results == NULL || rows == 0u ||
        rows > FORWARD_POSTPROCESS_MAX_ROWS || vocabulary_size == 0u ||
        vocabulary_size > 126464u ||
        (suppressed_token_count != 0u && suppressed_token_ids == NULL)) return -1;
    lane_max = malloc(rows * FORWARD_POSTPROCESS_CANDIDATE_LANES * sizeof(*lane_max));
    lane_sum = malloc(rows * FORWARD_POSTPROCESS_CANDIDATE_LANES * sizeof(*lane_sum));
    lane_id = malloc(rows * FORWARD_POSTPROCESS_CANDIDATE_LANES * sizeof(*lane_id));
    if (lane_max == NULL || lane_sum == NULL || lane_id == NULL) {
        free(lane_max); free(lane_sum); free(lane_id); return -2;
    }
    for (row = 0u; row < rows; ++row) {
        size_t token;
        if (selected_token_ids[row] >= vocabulary_size) {
            free(lane_max); free(lane_sum); free(lane_id); return -3;
        }
        for (token = 0u; token < FORWARD_POSTPROCESS_CANDIDATE_LANES; ++token) {
            lane_max[row * FORWARD_POSTPROCESS_CANDIDATE_LANES + token] = 0xff80u;
            lane_sum[row * FORWARD_POSTPROCESS_CANDIDATE_LANES + token] = 0u;
            lane_id[row * FORWARD_POSTPROCESS_CANDIDATE_LANES + token] = UINT32_MAX;
        }
        results[row].selected_token_logit_bf16 = 0u;
        for (token = 0u; token < vocabulary_size; ++token) {
            size_t lane = token & (FORWARD_POSTPROCESS_CANDIDATE_LANES - 1u);
            size_t index = row * FORWARD_POSTPROCESS_CANDIDATE_LANES + lane;
            uint16_t value = logits_bf16[row * vocabulary_size + token];
            if (!bf16_finite(value)) {
                free(lane_max); free(lane_sum); free(lane_id); return -4;
            }
            if ((uint32_t)token == selected_token_ids[row])
                results[row].selected_token_logit_bf16 = value;
            if (lane_id[index] == UINT32_MAX) {
                lane_max[index] = value;
                lane_sum[index] = 0x3f80u;
                lane_id[index] = (uint32_t)token;
            } else {
                uint16_t old_max = lane_max[index];
                uint16_t new_max = rtl_bf16_compare(value, old_max) > 0 ? value : old_max;
                uint16_t scaled_old = rtl_bf16_mul(
                    lane_sum[index], candidate_exp(
                        rtl_bf16_add(old_max, bf16_negate(new_max)), exp_lut));
                uint16_t new_term = candidate_exp(
                    rtl_bf16_add(value, bf16_negate(new_max)), exp_lut);
                if (rtl_bf16_compare(value, old_max) > 0 ||
                    (rtl_bf16_compare(value, old_max) == 0 && token < lane_id[index]))
                    lane_id[index] = (uint32_t)token;
                lane_max[index] = new_max;
                lane_sum[index] = rtl_bf16_add(scaled_old, new_term);
            }
        }
    }
    for (row = 0u; row < rows; ++row) {
        uint16_t global_max = 0xff80u;
        uint32_t winner = UINT32_MAX;
        uint16_t reduced[FORWARD_POSTPROCESS_CANDIDATE_LANES];
        int global_valid = 0;
        size_t lane;
        size_t width;
        for (lane = 0u; lane < FORWARD_POSTPROCESS_CANDIDATE_LANES; ++lane) {
            size_t index = row * FORWARD_POSTPROCESS_CANDIDATE_LANES + lane;
            if (lane_id[index] != UINT32_MAX &&
                (!global_valid || rtl_bf16_compare(lane_max[index], global_max) > 0)) {
                global_max = lane_max[index];
                global_valid = 1;
            }
        }
        for (lane = 0u; lane < FORWARD_POSTPROCESS_CANDIDATE_LANES; ++lane) {
            size_t index = row * FORWARD_POSTPROCESS_CANDIDATE_LANES + lane;
            if (lane_id[index] != UINT32_MAX &&
                rtl_bf16_compare(lane_max[index], global_max) == 0 &&
                lane_id[index] < winner) winner = lane_id[index];
            reduced[lane] = lane_id[index] == UINT32_MAX ? 0u : rtl_bf16_mul(
                lane_sum[index], candidate_exp(
                    rtl_bf16_add(lane_max[index], bf16_negate(global_max)), exp_lut));
        }
        for (width = FORWARD_POSTPROCESS_CANDIDATE_LANES; width > 1u; width >>= 1u)
            for (lane = 0u; lane < width / 2u; ++lane)
                reduced[lane] = rtl_bf16_add(reduced[2u * lane], reduced[2u * lane + 1u]);
        results[row].top1_token_id = winner;
        results[row].top_logit_bf16 = global_max;
        results[row].raw_confidence_bf16 = candidate_reciprocal(reduced[0], reciprocal_lut);
        results[row].selected_probability_bf16 = rtl_bf16_mul(
            results[row].raw_confidence_bf16, candidate_exp(
                rtl_bf16_add(results[row].selected_token_logit_bf16,
                             bf16_negate(global_max)), exp_lut));
        results[row].suppressed_winner = 0u;
        for (lane = 0u; lane < suppressed_token_count; ++lane)
            if (winner == suppressed_token_ids[lane]) results[row].suppressed_winner = 1u;
        results[row].action_confidence_bf16 = results[row].suppressed_winner ?
            0x0000u : results[row].raw_confidence_bf16;
    }
    free(lane_max); free(lane_sum); free(lane_id);
    return 0;
}

static int pick_best(const struct forward_postprocess_row_state *rows,
                     const uint16_t *scores, uint32_t allowed,
                     uint32_t selected, size_t row_count) {
    int best = -1;
    size_t row;
    for (row = 0u; row < row_count; ++row) {
        uint32_t bit = UINT32_C(1) << row;
        if ((allowed & bit) == 0u || (selected & bit) != 0u) continue;
        if (best < 0 || rtl_bf16_compare(scores[row], scores[best]) > 0 ||
            (rtl_bf16_compare(scores[row], scores[best]) == 0 &&
             rows[row].token_position < rows[best].token_position)) best = (int)row;
    }
    return best;
}

static uint8_t draft_verify_original_activation_bits(
    const struct forward_postprocess_row_state *row,
    uint16_t maturity_age
) {
    return row->state == FORWARD_POSTPROCESS_MASKED ||
        (row->state == FORWARD_POSTPROCESS_LOCKED &&
         row->precision_age >= (int16_t)maturity_age) ? 4u : 8u;
}

static int forward_postprocess_update(
    struct forward_postprocess_row_state *rows, size_t row_count,
    uint32_t masked_at_start, uint32_t tentative_at_start,
    uint32_t locked_at_start,
    const struct forward_postprocess_candidate_result *candidates,
    const struct forward_postprocess_draft_verify_config *config,
    struct forward_postprocess_draft_verify_result *result,
    int canonical_future, uint32_t observed) {
    uint32_t valid_mask;
    uint32_t admission_mask;
    uint32_t observed_masked;
    uint32_t eligible = 0u;
    uint32_t high_mask = 0u;
    uint32_t stable_mask = 0u;
    uint32_t stable_low_mask = 0u;
    int inherited_a4_confirmation = 0;
    uint16_t scores[FORWARD_POSTPROCESS_MAX_DRAFT_VERIFY_POSITIONS];
    uint16_t active_tau;
    size_t row;
    unsigned active_count;
    unsigned minimum_required;
    unsigned base_quota;
    unsigned max_quota;
    uint16_t quota_product;
    uint32_t token_before[FORWARD_POSTPROCESS_MAX_DRAFT_VERIFY_POSITIONS];
    if ((canonical_future != 0 && canonical_future != 1) ||
        rows == NULL || candidates == NULL || config == NULL || result == NULL ||
        row_count == 0u || row_count > FORWARD_POSTPROCESS_MAX_DRAFT_VERIFY_POSITIONS ||
        config->source_a_handoff > 1u || (config->source_a_handoff && !canonical_future) || config->remaining_forwards == 0u || config->tail_bypass_all > 1u || config->tail_bypass_stable_only > 1u ||
        (config->tail_bypass_all && config->tail_bypass_stable_only) ||
        config->maturity_age == 0u || config->maturity_age > INT16_MAX ||
        !bf16_positive_finite(config->budget_scale_bf16) ||
        !bf16_finite(config->high_confidence_threshold_bf16) ||
        !bf16_finite(config->tail_high_confidence_threshold_bf16) ||
        !bf16_finite(config->low_confidence_threshold_bf16) ||
        !bf16_finite(config->verify_threshold_bf16) ||
        !bf16_finite(config->stability_bonus_bf16)) return -1;
    valid_mask = row_count == 32u ? UINT32_MAX : (UINT32_C(1) << row_count) - 1u;
    if (!canonical_future) observed = valid_mask;
    else if ((observed & ~valid_mask) != 0u || config->tail_threshold_enable || config->tail_bypass_all || config->tail_bypass_stable_only ||
             config->budget_scale_bf16 != 0x3f80u || config->scheduled_quota > 32u) return -1;
    if (((masked_at_start | tentative_at_start | locked_at_start) != valid_mask) ||
        (masked_at_start & tentative_at_start) != 0u ||
        (masked_at_start & locked_at_start) != 0u ||
        (tentative_at_start & locked_at_start) != 0u) return -1;
    for (row = 0u; row < row_count; ++row) {
        uint32_t bit = UINT32_C(1) << row;
        uint8_t expected_state = (masked_at_start & bit) != 0u ?
            FORWARD_POSTPROCESS_MASKED : (tentative_at_start & bit) != 0u ?
                FORWARD_POSTPROCESS_TENTATIVE : FORWARD_POSTPROCESS_LOCKED;
        uint8_t expected_prediction = expected_state != FORWARD_POSTPROCESS_LOCKED && (observed & bit) != 0u;
        if (rows[row].source_a > 1u || rows[row].source_a_pending > 1u ||
            (rows[row].source_a && !canonical_future) ||
            (rows[row].source_a_pending && (expected_state != FORWARD_POSTPROCESS_TENTATIVE ||
                (canonical_future && (!config->source_a_handoff || (observed & bit))))) ||
            rows[row].state != expected_state ||
            (rows[row].activation_bits != 4u && rows[row].activation_bits != 8u) ||
            rows[row].cache_valid > 1u || rows[row].refresh_required > 1u ||
            rows[row].prediction_flag != expected_prediction) return -1;
    }
    memset(result, 0, sizeof(*result));
    for (row = 0u; row < row_count; ++row) {
        token_before[row] = rows[row].token_id;
        inherited_a4_confirmation |= !canonical_future &&
            (tentative_at_start & (UINT32_C(1) << row)) != 0u && rows[row].activation_bits == 4u;
        if (!canonical_future) rows[row].source_a_pending = 0u;
    }
    for (row = 0u; row < row_count; ++row) {
        uint32_t bit = UINT32_C(1) << row;
        /* The engine defers inherited TENTATIVE confirmation during uniform
           A4 initialization forwards. A regular confirmation executes the token at A8. */
        if ((tentative_at_start & observed & bit) != 0u &&
            (canonical_future || rows[row].activation_bits == 8u)) {
            uint16_t keep = candidates[row].suppressed_winner ? 0u :
                candidates[row].selected_probability_bf16;
            if (candidates[row].top1_token_id == rows[row].token_id &&
                bf16_ge(keep, config->verify_threshold_bf16)) {
                rows[row].state = FORWARD_POSTPROCESS_LOCKED;
                rows[row].precision_age = 0;
                result->confirmed_mask |= bit;
                result->cache_commit_mask |= bit;
            } else {
                rows[row].state = FORWARD_POSTPROCESS_MASKED;
                rows[row].token_id = config->mask_token_id;
                rows[row].last_top1 = UINT32_MAX;
                rows[row].precision_age = -1;
                rows[row].commit_origin = FORWARD_POSTPROCESS_ORIGIN_NONE;
                result->remasked_mask |= bit;
                result->cache_invalidate_mask |= bit;
            }
        }
    }
    observed_masked = masked_at_start & observed;
    admission_mask = observed_masked & ~result->remasked_mask;
    active_tau = config->tail_threshold_enable &&
        config->step_index >= config->tail_after_step ?
        config->tail_high_confidence_threshold_bf16 : config->high_confidence_threshold_bf16;
    for (row = 0u; row < row_count; ++row) {
        uint32_t bit = UINT32_C(1) << row;
        int stable;
        int high;
        int stable_low;
        int a4_high;
        int a4_direct;
        if ((admission_mask & bit) == 0u) { scores[row] = 0u; continue; }
        stable = rows[row].last_top1 == candidates[row].top1_token_id;
        high = bf16_ge(candidates[row].action_confidence_bf16,
                       canonical_future ? config->low_confidence_threshold_bf16 : active_tau);
        stable_low = stable &&
            bf16_ge(candidates[row].action_confidence_bf16, config->low_confidence_threshold_bf16) &&
            bf16_lt(candidates[row].action_confidence_bf16, config->high_confidence_threshold_bf16);
        a4_high = high && rows[row].activation_bits == 4u;
        a4_direct = rows[row].activation_bits == 4u &&
            bf16_ge(candidates[row].action_confidence_bf16,
                    FORWARD_POSTPROCESS_A4_DIRECT_THRESHOLD_BF16);
        if (stable) stable_mask |= bit;
        if ((canonical_future && high) || (high && rows[row].activation_bits == 8u) || a4_direct) high_mask |= bit;
        if (canonical_future ? stable_low : (stable_low || (a4_high && !a4_direct))) stable_low_mask |= bit;
        if (canonical_future ? high : (high && rows[row].activation_bits == 8u) || a4_direct ||
            stable_low || a4_high) eligible |= bit;
        scores[row] = stable ? rtl_bf16_add(
            candidates[row].action_confidence_bf16,
            config->stability_bonus_bf16) : candidates[row].action_confidence_bf16;
    }
    if (canonical_future) admission_mask &= eligible;
    active_count = popcount32(admission_mask);
    minimum_required = (active_count + config->remaining_forwards - 1u) /
        config->remaining_forwards;
    base_quota = config->scheduled_quota;
    if (base_quota < 1u) base_quota = 1u;
    if (base_quota < minimum_required) base_quota = minimum_required;
    if (canonical_future && config->scheduled_quota == 0u) base_quota = 0u;
    quota_product = rtl_bf16_mul(rtl_f32_to_bf16((float)base_quota),
                                 config->budget_scale_bf16);
    max_quota = (unsigned)ceilf(rtl_bf16_to_f32(quota_product));
    if (max_quota > active_count) max_quota = active_count;
    while (popcount32(result->selected_mask) < max_quota) {
        int best = pick_best(rows, scores, eligible, result->selected_mask, row_count);
        if (best < 0) break;
        result->selected_mask |= UINT32_C(1) << (unsigned)best;
    }
    while (!canonical_future && popcount32(result->selected_mask) < base_quota) {
        int best = pick_best(rows, scores, admission_mask,
                             result->selected_mask, row_count);
        if (best < 0) break;
        result->selected_mask |= UINT32_C(1) << (unsigned)best;
    }
    for (row = 0u; row < row_count; ++row) {
        uint32_t bit = UINT32_C(1) << row;
        if ((result->selected_mask & bit) != 0u) {
            int direct = canonical_future || (rows[row].activation_bits == 4u &&
                bf16_ge(candidates[row].action_confidence_bf16,
                        FORWARD_POSTPROCESS_A4_DIRECT_THRESHOLD_BF16)) ||
                ((high_mask & bit) != 0u && rows[row].activation_bits == 8u &&
                 (bf16_ge(candidates[row].action_confidence_bf16,
                          config->high_confidence_threshold_bf16) || (stable_mask & bit) != 0u));
            rows[row].source_a_pending = canonical_future && config->source_a_handoff && rows[row].source_a;
            if (rows[row].source_a_pending) direct = 0;
            rows[row].token_id = candidates[row].top1_token_id;
            rows[row].last_top1 = UINT32_MAX;
            if (direct) {
                rows[row].state = FORWARD_POSTPROCESS_LOCKED;
                rows[row].precision_age = 0;
                rows[row].commit_origin = FORWARD_POSTPROCESS_ORIGIN_HIGH;
                result->direct_locked_mask |= bit;
            } else {
                rows[row].state = FORWARD_POSTPROCESS_TENTATIVE;
                rows[row].precision_age = -1;
                if ((stable_low_mask & bit) != 0u) {
                    rows[row].commit_origin = FORWARD_POSTPROCESS_ORIGIN_STABLE;
                    result->stable_tentative_mask |= bit;
                } else {
                    rows[row].commit_origin = FORWARD_POSTPROCESS_ORIGIN_FALLBACK;
                    result->fallback_tentative_mask |= bit;
                }
            }
            result->cache_invalidate_mask |= bit;
        }
    }
    for (row = 0u; row < row_count; ++row) {
        uint32_t bit = UINT32_C(1) << row;
        if ((observed_masked & bit) != 0u && rows[row].state == FORWARD_POSTPROCESS_MASKED)
            rows[row].last_top1 = candidates[row].top1_token_id;
        if (rows[row].state != FORWARD_POSTPROCESS_MASKED) rows[row].last_top1 = UINT32_MAX;
        if ((locked_at_start & observed & bit) != 0u && rows[row].state == FORWARD_POSTPROCESS_LOCKED &&
            rows[row].precision_age < INT16_MAX) ++rows[row].precision_age;
        if (rows[row].state == FORWARD_POSTPROCESS_MASKED)
            rows[row].precision_age = -1;
        if ((locked_at_start & observed & bit) != 0u && rows[row].refresh_required) {
            rows[row].cache_valid = 1u;
            rows[row].refresh_required = 0u;
        }
        rows[row].activation_bits = draft_verify_original_activation_bits(
            &rows[row], config->maturity_age);
        rows[row].prediction_flag =
            rows[row].state == FORWARD_POSTPROCESS_LOCKED ? 0u : 1u;
        if (rows[row].token_id != token_before[row])
            result->token_changed_mask |= bit;
        if ((observed_masked & bit) != 0u &&
            (result->selected_mask & bit) == 0u &&
            rows[row].state == FORWARD_POSTPROCESS_MASKED) {
            result->cache_keep_mask |= bit;
            rows[row].cache_valid = 1u;
            rows[row].refresh_required = 0u;
        } else if ((locked_at_start & bit) != 0u &&
                     rows[row].state == FORWARD_POSTPROCESS_LOCKED &&
                     rows[row].cache_valid && !rows[row].refresh_required) {
            result->cache_keep_mask |= bit;
        }
        if ((result->cache_commit_mask & bit) != 0u) {
            rows[row].cache_valid = 1u;
            rows[row].refresh_required = 0u;
        }
        if ((result->cache_invalidate_mask & bit) != 0u) {
            rows[row].cache_valid = 0u;
            rows[row].refresh_required = 1u;
        }
        if ((result->token_changed_mask & bit) != 0u) {
            result->mandatory_refresh_mask |= bit;
            result->cache_invalidate_mask |= bit;
            result->cache_commit_mask &= ~bit;
            result->cache_keep_mask &= ~bit;
            rows[row].cache_valid = 0u;
            rows[row].refresh_required = 1u;
        }
        if (rows[row].refresh_required)
            result->mandatory_refresh_mask |= bit;
    }
    if (config->tail_bypass_all || config->tail_bypass_stable_only) {
        int tail_blocked = inherited_a4_confirmation;
        for (row = 0u; row < row_count; ++row)
            tail_blocked |= rows[row].state == FORWARD_POSTPROCESS_MASKED ||
                (config->tail_bypass_stable_only && rows[row].state == FORWARD_POSTPROCESS_TENTATIVE &&
                 rows[row].commit_origin != FORWARD_POSTPROCESS_ORIGIN_STABLE);
        if (!tail_blocked) {
            if (forward_postprocess_draft_verify_tail_all(rows, row_count, &result->tail_closed_mask)) return -1;
            for (row = 0u; row < row_count; ++row) {
                const uint32_t bit = UINT32_C(1) << row;
                if (rows[row].cache_valid && !rows[row].refresh_required)
                    result->cache_keep_mask |= bit;
                if (rows[row].refresh_required)
                    result->mandatory_refresh_mask |= bit;
            }
        }
    }
    return 0;
}

int forward_postprocess_transfer_update(
    struct forward_postprocess_row_state *rows, size_t row_count,
    uint32_t masked_at_start, uint32_t tentative_at_start, uint32_t locked_at_start,
    const struct forward_postprocess_candidate_result *candidates,
    const struct forward_postprocess_draft_verify_config *config,
    struct forward_postprocess_draft_verify_result *result) {
    if (!rows || !candidates || !config || !result || !row_count ||
        row_count > 32 || tentative_at_start || config->scheduled_quota > 32) return -1;
    const uint32_t valid = row_count == 32 ? UINT32_MAX : (UINT32_C(1) << row_count) - 1;
    if ((masked_at_start | locked_at_start) != valid || (masked_at_start & locked_at_start)) return -1;
    uint16_t scores[32];
    for (size_t row = 0; row < row_count; ++row) {
        const uint32_t bit = UINT32_C(1) << row;
        if (rows[row].state != ((masked_at_start & bit) ? FORWARD_POSTPROCESS_MASKED : FORWARD_POSTPROCESS_LOCKED) ||
            ((masked_at_start & bit) && rows[row].token_id != config->mask_token_id) ||
            (rows[row].activation_bits != 4 && rows[row].activation_bits != 8) ||
            !bf16_finite(candidates[row].action_confidence_bf16) ||
            bf16_lt(candidates[row].action_confidence_bf16, 0)) return -1;
        scores[row] = candidates[row].action_confidence_bf16;
    }
    memset(result, 0, sizeof(*result));
    for (unsigned count = 0; count < config->scheduled_quota; ++count) {
        const int best = pick_best(rows, scores, masked_at_start, result->selected_mask, row_count);
        if (best < 0) break;
        result->selected_mask |= UINT32_C(1) << best;
    }
    for (size_t row = 0; row < row_count; ++row) {
        const uint32_t bit = UINT32_C(1) << row;
        const uint32_t token_before = rows[row].token_id;
        if (result->selected_mask & bit) {
            rows[row].token_id = candidates[row].top1_token_id;
            rows[row].state = FORWARD_POSTPROCESS_LOCKED;
            rows[row].commit_origin = FORWARD_POSTPROCESS_ORIGIN_HIGH;
            rows[row].precision_age = 0;
            rows[row].last_top1 = UINT32_MAX;
            result->direct_locked_mask |= bit;
        } else if (masked_at_start & bit) {
            rows[row].last_top1 = candidates[row].top1_token_id;
        }
        if (locked_at_start & bit) {
            if (rows[row].precision_age < INT16_MAX) ++rows[row].precision_age;
            rows[row].last_top1 = UINT32_MAX;
        }
        if (rows[row].token_id != token_before) {
            rows[row].cache_valid = 0;
            rows[row].refresh_required = 1;
            result->token_changed_mask |= bit;
            result->cache_invalidate_mask |= bit;
            result->mandatory_refresh_mask |= bit;
        } else {
            if ((masked_at_start & bit) || rows[row].refresh_required) {
                rows[row].cache_valid = 1;
                rows[row].refresh_required = 0;
            }
            if (rows[row].cache_valid && !rows[row].refresh_required)
                result->cache_keep_mask |= bit;
        }
        rows[row].activation_bits = 8;
        rows[row].prediction_flag = rows[row].state == FORWARD_POSTPROCESS_MASKED;
    }
    return 0;
}

int forward_postprocess_draft_verify_update(
    struct forward_postprocess_row_state *rows, size_t row_count,
    uint32_t masked_at_start, uint32_t tentative_at_start, uint32_t locked_at_start,
    const struct forward_postprocess_candidate_result *candidates,
    const struct forward_postprocess_draft_verify_config *config, struct forward_postprocess_draft_verify_result *result) {
    return forward_postprocess_update(rows, row_count, masked_at_start, tentative_at_start,
                             locked_at_start, candidates, config, result, 0, 0u);
}

int forward_postprocess_canonical_future_update(
    struct forward_postprocess_row_state *rows, size_t row_count,
    uint32_t masked_at_start, uint32_t tentative_at_start, uint32_t locked_at_start,
    uint32_t observed, const struct forward_postprocess_candidate_result *candidates,
    const struct forward_postprocess_draft_verify_config *config, struct forward_postprocess_draft_verify_result *result) {
    return forward_postprocess_update(rows, row_count, masked_at_start, tentative_at_start,
                             locked_at_start, candidates, config, result, 1, observed);
}

int forward_postprocess_draft_verify_closeout(
    struct forward_postprocess_row_state *rows, size_t row_count, uint8_t kind,
    const struct forward_postprocess_candidate_result *candidates,
    const struct forward_postprocess_draft_verify_config *config, struct forward_postprocess_draft_verify_result *result) {
    if (!rows || !candidates || !config || !result || row_count == 0 ||
        row_count > FORWARD_POSTPROCESS_MAX_DRAFT_VERIFY_POSITIONS || (kind != 1 && kind != 2) ||
        config->maturity_age == 0 || config->maturity_age > INT16_MAX ||
        !bf16_finite(config->verify_threshold_bf16)) return -1;
    for (size_t i = 0; i < row_count; ++i) {
        const uint8_t predicted = kind == 1 ? rows[i].state == FORWARD_POSTPROCESS_TENTATIVE : rows[i].state == FORWARD_POSTPROCESS_MASKED;
        if (rows[i].state > FORWARD_POSTPROCESS_LOCKED || (kind == 2 && rows[i].state == FORWARD_POSTPROCESS_TENTATIVE) ||
            rows[i].prediction_flag != predicted || (rows[i].activation_bits != 4 && rows[i].activation_bits != 8) ||
            rows[i].cache_valid > 1 || rows[i].refresh_required > 1 ||
            (predicted && !bf16_finite(candidates[i].selected_probability_bf16))) return -1;
    }
    memset(result, 0, sizeof(*result));
    for (size_t i = 0; i < row_count; ++i) {
        struct forward_postprocess_row_state *row = &rows[i];
        const uint32_t bit = UINT32_C(1) << i;
        const uint32_t token_before = row->token_id;
        const uint8_t state_before = row->state;
        if (state_before == FORWARD_POSTPROCESS_LOCKED) {
            if (row->precision_age < INT16_MAX) ++row->precision_age;
            if (row->refresh_required) { row->cache_valid = 1; row->refresh_required = 0; }
        } else if (kind == 1 && state_before == FORWARD_POSTPROCESS_TENTATIVE) {
            if (!candidates[i].suppressed_winner && candidates[i].top1_token_id == row->token_id &&
                bf16_ge(candidates[i].selected_probability_bf16, config->verify_threshold_bf16)) {
                row->state = FORWARD_POSTPROCESS_LOCKED; row->precision_age = 0;
                row->cache_valid = 1; row->refresh_required = 0;
                result->confirmed_mask |= bit; result->cache_commit_mask |= bit;
            } else {
                row->state = FORWARD_POSTPROCESS_MASKED; row->token_id = config->mask_token_id;
                row->precision_age = -1; row->commit_origin = FORWARD_POSTPROCESS_ORIGIN_NONE;
                row->last_top1 = UINT32_MAX; result->remasked_mask |= bit;
                row->cache_valid = 0; row->refresh_required = 1;
                result->cache_invalidate_mask |= bit;
            }
        } else if (kind == 2 && state_before == FORWARD_POSTPROCESS_MASKED) {
            row->token_id = candidates[i].top1_token_id; row->state = FORWARD_POSTPROCESS_LOCKED; row->precision_age = 0;
            result->selected_mask |= bit; result->tail_closed_mask |= bit;
        }
        if (row->token_id != token_before) {
            result->token_changed_mask |= bit; result->cache_invalidate_mask |= bit;
            row->cache_valid = 0; row->refresh_required = 1;
        }
        if (row->refresh_required) result->mandatory_refresh_mask |= bit;
        if (row->cache_valid && !row->refresh_required && !(result->cache_commit_mask & bit))
            result->cache_keep_mask |= bit;
        row->activation_bits = draft_verify_original_activation_bits(row, config->maturity_age);
        row->prediction_flag = row->state != FORWARD_POSTPROCESS_LOCKED;
    }
    return 0;
}

int forward_postprocess_draft_verify_tail_all(struct forward_postprocess_row_state *rows,
                              size_t row_count, uint32_t *closed_mask) {
    uint32_t tentative = 0u;
    size_t row;
    if (rows == NULL || closed_mask == NULL || row_count == 0u ||
        row_count > FORWARD_POSTPROCESS_MAX_DRAFT_VERIFY_POSITIONS) return -1;
    for (row = 0u; row < row_count; ++row) {
        uint32_t bit = UINT32_C(1) << row;
        if (rows[row].state == FORWARD_POSTPROCESS_MASKED) return -2;
        if (rows[row].state == FORWARD_POSTPROCESS_TENTATIVE) tentative |= bit;
    }
    for (row = 0u; row < row_count; ++row) {
        if ((tentative & (UINT32_C(1) << row)) != 0u) {
            rows[row].state = FORWARD_POSTPROCESS_LOCKED;
            rows[row].precision_age = 0;
            rows[row].activation_bits = 8u;
            rows[row].prediction_flag = 0u;
        }
    }
    *closed_mask = tentative;
    return 0;
}
