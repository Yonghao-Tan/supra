#ifndef SUPRA_ATTENTION_REFRESH_MODEL_H
#define SUPRA_ATTENTION_REFRESH_MODEL_H

#include <stddef.h>
#include <stdint.h>

#define RTL_REFRESH_MAX_ROWS 96u
#define RTL_REFRESH_MAX_SEQUENCE 2048u
#define RTL_JOINT_MAX_ROWS 48u
#define RTL_FUTURE_MAX_ROWS 32u

/* Default packed precision: preserve unresolved current bits, start context
 * at A4, and upgrade the highest dependencies, retaining input-order ties. */
int rtl_context_precision(uint32_t rows, const uint8_t *input_bits,
    const uint8_t *mandatory_current, const uint16_t *dependency,
    uint32_t context_a8_token_count, uint8_t tail_all_a8, uint8_t *output_bits,
    uint8_t *upgrade_order, uint32_t *upgrade_count);

/* Selected-row order defines stable score ties. Callers bypass when disabled. */
int rtl_boundary_deep_precision(uint32_t rows, const uint8_t *input_bits,
    const uint8_t *scores, const uint8_t *protected_rows, uint32_t a8_limit,
    uint8_t *output_bits);

/* CUDA-target reference: four FP32 partial sums, head h updates partial[h%4],
 * then ((partial[0]+partial[1])+partial[2])+partial[3], divided by 32.
 * Matches PyTorch 2.4 Reduce.cuh vt0=4 for contiguous head-outer reductions
 * with >=2 output elements. A single-output reduction is rejected: CUDA
 * uses a different warp reduction. CPU comparisons cover fixed reference data sets,
 * not every CPU reduction kernel or tensor layout.
 * Initialize the optional, separate all_layer_max array to zero each forward.
 */
int rtl_p8_relation_bf16(const int8_t *codes, const uint16_t *scales,
                        uint32_t queries, uint32_t keys, uint16_t *relation,
                        uint16_t *all_layer_max);
int rtl_probability_relation_bf16(const uint16_t *probabilities,
    uint32_t queries, uint32_t keys, uint16_t *relation, uint16_t *all_layer_max);
int rtl_boundary_scout_q8(const uint16_t *probabilities, uint32_t sequence,
    const uint16_t *query_positions, uint32_t current_count,
    const uint16_t *current_positions, uint32_t transition_count,
    const uint16_t *transition_positions, uint8_t *scores);

/* Tracks pending token changes and future risk for a 32-token block. */
struct rtl_cross_block_state {
    uint32_t rows, block_start;
    uint16_t relation[RTL_REFRESH_MAX_SEQUENCE * 32];
    uint16_t pending[RTL_REFRESH_MAX_SEQUENCE];
    uint16_t actual_remask_pending[RTL_REFRESH_MAX_SEQUENCE];
    uint16_t actual_remask_epoch_pending[RTL_REFRESH_MAX_SEQUENCE];
    uint16_t future_pending[RTL_REFRESH_MAX_SEQUENCE];
    uint16_t future_actual_remask_pending[RTL_REFRESH_MAX_SEQUENCE];
};
size_t rtl_cross_block_state_bytes(void);
int rtl_cross_block_init(struct rtl_cross_block_state *state,
                         uint32_t rows, uint32_t block_start);
int rtl_cross_block_advance(struct rtl_cross_block_state *state, uint32_t block_start);
int rtl_cross_block_observe(struct rtl_cross_block_state *state,
    uint32_t query_count, const uint16_t *query_positions,
    const uint16_t *query_relation, uint32_t consumed_count,
    const uint16_t *consumed_positions, const uint8_t *changed,
    const uint8_t *remasked, const uint16_t *confidence);

/* Canonical joint selection: default 384 KiB buffers, no segment-growth limit. */
struct rtl_joint_input {
    uint32_t base_token_count, future_token_count, target_token_count, max_next_tokens, next_block_start;
    uint16_t base_positions[RTL_JOINT_MAX_ROWS];
    uint8_t base_bits[RTL_JOINT_MAX_ROWS];
    uint8_t forecast_bits[RTL_JOINT_MAX_ROWS];
    uint8_t future_bits[RTL_FUTURE_MAX_ROWS];
    uint8_t unresolved[RTL_FUTURE_MAX_ROWS];
    uint8_t tentative[RTL_FUTURE_MAX_ROWS];
    uint16_t priority[RTL_FUTURE_MAX_ROWS];
    int32_t service_count[RTL_FUTURE_MAX_ROWS];
    uint32_t prediction_target, current_prediction_tokens;
    uint32_t future_admission_attempts[RTL_FUTURE_MAX_ROWS];
    uint16_t last_confidence[RTL_FUTURE_MAX_ROWS], retry_min_confidence, min_reuse_score;
    uint16_t dependency[RTL_FUTURE_MAX_ROWS], block_step_index;
    uint8_t source_b_dependency_tie_rank, source_b_a4_only;
};

struct rtl_activation_residency {
    uint32_t a4_token_count, a8_token_count, issued_slice_units, pe_issue_groups;
    uint32_t activation_bytes[3], capacity_bytes[3], fragments[3];
};

struct rtl_joint_selection {
    uint32_t future_prediction_count, added_future_token_count;
    uint32_t rejected_current, rejected_forecast, rejected_budget;
    uint8_t progress[RTL_FUTURE_MAX_ROWS];
    uint8_t added[RTL_FUTURE_MAX_ROWS];
    uint8_t added_bits[RTL_FUTURE_MAX_ROWS];
    struct rtl_activation_residency base, joint, forecast;
};

size_t rtl_joint_input_bytes(void);
size_t rtl_joint_selection_bytes(void);
int rtl_joint_select(const struct rtl_joint_input *input,
                     struct rtl_joint_selection *result);
/* NULL means no new future-token admission limit; reused future tokens and TENTATIVE always remain eligible. */
int rtl_joint_select_allowed(const struct rtl_joint_input *input,
    const uint8_t *future_admission_allowed, struct rtl_joint_selection *result);

struct rtl_boundary_selection {
    uint32_t deep_count;
    uint32_t optional_count;
    uint32_t ranked_count;
    uint16_t deep_positions[RTL_REFRESH_MAX_SEQUENCE];
    uint16_t selected_optional[RTL_REFRESH_MAX_SEQUENCE];
    uint16_t ranked_candidates[RTL_REFRESH_MAX_SEQUENCE];
};

size_t rtl_boundary_selection_bytes(void);
/* Explicit eligible list; stable ties retain its order, not position order. */
int rtl_boundary_select(
    uint32_t sequence_length, const uint8_t *scores, uint32_t target_token_count,
    uint32_t current_count, const uint16_t *current,
    uint32_t eligible_count, const uint16_t *eligible,
    uint32_t mandatory_count, const uint16_t *mandatory,
    uint32_t required_count, const uint16_t *required, uint32_t required_quota,
    struct rtl_boundary_selection *result);

/* Tri-region state after Attention reduction; zero initial burst/row credit. */
struct rtl_refresh_state {
    uint32_t rows;
    uint32_t target_token_count_x2;
    uint32_t regular_steps;
    uint32_t cumulative_active_token_count;
    uint32_t allowed_active_token_count;
    uint16_t dependency[RTL_REFRESH_MAX_ROWS * RTL_REFRESH_MAX_ROWS];
    uint16_t pending[RTL_REFRESH_MAX_ROWS];
    uint16_t new_invalidation[RTL_REFRESH_MAX_ROWS];
    uint8_t refresh[RTL_REFRESH_MAX_ROWS];
    uint8_t mandatory[RTL_REFRESH_MAX_ROWS];
    uint8_t optional_selected[RTL_REFRESH_MAX_ROWS];
};

size_t rtl_refresh_state_bytes(void);
int rtl_refresh_init(struct rtl_refresh_state *state, uint32_t rows,
                     uint32_t target_token_count_x2, const uint16_t *dependency);

/* Positions and masks are region-local. query_dependency is query_count x rows. */
int rtl_refresh_observe(struct rtl_refresh_state *state,
                        uint32_t query_count, const uint16_t *query_positions,
                        const uint16_t *query_dependency,
                        const uint8_t *predicted, const uint8_t *changed,
                        const uint8_t *remasked, const uint16_t *confidence);

#endif
