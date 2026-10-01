#ifndef SUPRA_FORWARD_POSTPROCESS_MODEL_H
#define SUPRA_FORWARD_POSTPROCESS_MODEL_H

#include "rtl_numeric.h"

#include <stddef.h>
#include <stdint.h>

#define FORWARD_POSTPROCESS_A4_DIRECT_THRESHOLD_BF16 0x3f66u
#define FORWARD_POSTPROCESS_MAX_ROWS 128u
#define FORWARD_POSTPROCESS_MAX_DRAFT_VERIFY_POSITIONS 32u
#define FORWARD_POSTPROCESS_CANDIDATE_LANES 64u
#define FORWARD_POSTPROCESS_MAX_RESIDENT_ROWS 48u
#define FORWARD_POSTPROCESS_EMBEDDING_VOCABULARY 126464u
#define FORWARD_POSTPROCESS_EMBEDDING_ROW_BYTES 8192u
#define FORWARD_POSTPROCESS_EMBEDDING_TABLE_BYTES UINT64_C(1035993088)

enum forward_postprocess_token_state {
    FORWARD_POSTPROCESS_MASKED = 0,
    FORWARD_POSTPROCESS_TENTATIVE = 1,
    FORWARD_POSTPROCESS_LOCKED = 2
};

enum forward_postprocess_commit_origin {
    FORWARD_POSTPROCESS_ORIGIN_NONE = 0,
    FORWARD_POSTPROCESS_ORIGIN_HIGH = 1,
    FORWARD_POSTPROCESS_ORIGIN_STABLE = 2,
    FORWARD_POSTPROCESS_ORIGIN_FALLBACK = 3
};

enum forward_postprocess_row_source {
    FORWARD_POSTPROCESS_RESIDENT_HIDDEN = 0,
    FORWARD_POSTPROCESS_EMBEDDING_TOKEN = 1
};

struct forward_postprocess_next_token_descriptor {
    uint32_t source_index;
    uint16_t token_position;
    uint16_t kv_index;
    uint8_t source;
    uint8_t activation_bits;
    uint8_t query_group;
    uint8_t cache_group;
};

struct forward_postprocess_physical_row {
    uint32_t source_index;
    uint16_t token_position;
    uint16_t kv_index;
    uint8_t token_ordinal;
    uint8_t source;
    uint8_t activation_bits;
    uint8_t query_group;
    uint8_t cache_group;
    uint8_t compute_group;
    uint8_t pe_slot;
    uint8_t phase_mask;
};

struct forward_postprocess_token_batch_header {
    uint8_t resident_token_count;
    uint8_t compute_group_count;
    uint8_t semantic_group_count;
    uint8_t flags;
    uint16_t token_batch_index;
    uint16_t first_token_ordinal;
    uint16_t token_batch_bytes;
    uint16_t token_entry_bytes;
    uint16_t inverse_offset;
    uint16_t inverse_bytes;
    uint32_t metadata_version;
    uint32_t capture_index;
    uint64_t reserved;
};

struct forward_postprocess_candidate_result {
    uint32_t top1_token_id;
    uint16_t top_logit_bf16;
    uint16_t raw_confidence_bf16;
    uint16_t selected_token_logit_bf16;
    uint16_t selected_probability_bf16;
    uint16_t action_confidence_bf16;
    uint8_t suppressed_winner;
    uint8_t reserved;
};

struct forward_postprocess_row_state {
    uint32_t token_id;
    uint32_t last_top1;
    int16_t precision_age;
    uint16_t token_position;
    uint8_t state;
    uint8_t commit_origin;
    uint8_t activation_bits;
    uint8_t cache_valid;
    uint8_t refresh_required;
    uint8_t prediction_flag;
    uint8_t source_a;
    uint8_t source_a_pending;
};

struct forward_postprocess_draft_verify_config {
    uint32_t mask_token_id;
    uint16_t high_confidence_threshold_bf16;
    uint16_t tail_high_confidence_threshold_bf16;
    uint16_t low_confidence_threshold_bf16;
    uint16_t verify_threshold_bf16;
    uint16_t stability_bonus_bf16;
    uint16_t budget_scale_bf16;
    uint16_t scheduled_quota;
    uint16_t remaining_forwards;
    uint16_t step_index;
    uint16_t tail_after_step;
    uint16_t maturity_age;
    uint8_t tail_threshold_enable;
    uint8_t tail_bypass_all;
    uint8_t tail_bypass_stable_only;
    uint8_t source_a_handoff;
};

struct forward_postprocess_draft_verify_result {
    uint32_t confirmed_mask;
    uint32_t remasked_mask;
    uint32_t selected_mask;
    uint32_t direct_locked_mask;
    uint32_t stable_tentative_mask;
    uint32_t fallback_tentative_mask;
    uint32_t mandatory_refresh_mask;
    uint32_t cache_commit_mask;
    uint32_t cache_invalidate_mask;
    uint32_t cache_keep_mask;
    uint32_t token_changed_mask;
    uint32_t tail_closed_mask;
};

int forward_postprocess_pack_next_tokens(
    const struct forward_postprocess_next_token_descriptor *token_ordinals,
    size_t row_count, uint16_t sequence_length,
    struct forward_postprocess_physical_row *physical_rows,
    uint8_t *logical_to_physical, uint8_t *compute_group_count,
    uint8_t *semantic_group_count);

int forward_postprocess_embedding_address(uint64_t embedding_base,
                                 uint64_t embedding_limit,
                                 uint32_t token_id,
                                 uint64_t *row_address);
size_t forward_postprocess_token_metadata_token_batch_bytes(size_t resident_rows);

int forward_postprocess_final_rmsnorm_a8(const uint16_t *final_hidden, size_t rows,
                                size_t channels, uint16_t epsilon_bf16,
                                uint16_t *normalized_bf16,
                                int8_t *activation_codes,
                                uint16_t *activation_scales_bf16);

int forward_postprocess_raw_w8_matmul(const int8_t *activation_codes,
                             const uint16_t *activation_scales_bf16,
                             size_t rows, size_t input_features,
                             const int8_t *weight_codes,
                             const uint16_t *weight_scales_bf16,
                             size_t output_features, uint16_t *logits_bf16,
                             int32_t *accumulators);

int forward_postprocess_candidate_bf16(
    const uint16_t *logits_bf16, size_t rows, size_t vocabulary_size,
    const uint32_t *selected_token_ids, const uint32_t *suppressed_token_ids,
    size_t suppressed_token_count, const uint16_t *exp_lut,
    const uint16_t *reciprocal_lut,
    struct forward_postprocess_candidate_result *results);

int forward_postprocess_draft_verify_update(
    struct forward_postprocess_row_state *rows, size_t row_count,
    uint32_t masked_at_start, uint32_t tentative_at_start,
    uint32_t locked_at_start,
    const struct forward_postprocess_candidate_result *candidates,
    const struct forward_postprocess_draft_verify_config *config,
    struct forward_postprocess_draft_verify_result *result);

/* Scheduled quota on raw action confidence, with equal scores ordered by logical position. */
int forward_postprocess_transfer_update(
    struct forward_postprocess_row_state *rows, size_t row_count,
    uint32_t masked_at_start, uint32_t tentative_at_start, uint32_t locked_at_start,
    const struct forward_postprocess_candidate_result *candidates,
    const struct forward_postprocess_draft_verify_config *config,
    struct forward_postprocess_draft_verify_result *result);

int forward_postprocess_draft_verify_tail_all(struct forward_postprocess_row_state *rows,
                              size_t row_count, uint32_t *closed_mask);

/* Canonical future direct uses low_confidence_threshold for admission and direct locking,
 * unit budget_scale and no tail override; only observed rows update history/age.
 * Newly remasked positions cannot be admitted with this forward's candidate.
 */
int forward_postprocess_canonical_future_update(
    struct forward_postprocess_row_state *rows, size_t row_count,
    uint32_t masked_at_start, uint32_t tentative_at_start, uint32_t locked_at_start,
    uint32_t observed, const struct forward_postprocess_candidate_result *candidates,
    const struct forward_postprocess_draft_verify_config *config, struct forward_postprocess_draft_verify_result *result);

/* kind 1: confirmation only; kind 2: forced finish after confirmation. */
int forward_postprocess_draft_verify_closeout(
    struct forward_postprocess_row_state *rows, size_t row_count, uint8_t kind,
    const struct forward_postprocess_candidate_result *candidates,
    const struct forward_postprocess_draft_verify_config *config, struct forward_postprocess_draft_verify_result *result);

#endif
