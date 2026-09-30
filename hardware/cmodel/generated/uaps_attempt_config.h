#ifndef UAPS_ATTEMPT_CONFIG_H
#define UAPS_ATTEMPT_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define UAPS_ATTEMPT_CONFIG_BYTES 32u
#define UAPS_ATTEMPT_CONFIG_ALIGNMENT 16u
#define UAPS_ATTEMPT_CONFIG_STATE_BASE_OFFSET 0u
#define UAPS_ATTEMPT_CONFIG_STATE_LIMIT_OFFSET 8u
#define UAPS_ATTEMPT_CONFIG_MAX_ATTEMPTS_OFFSET 16u
#define UAPS_ATTEMPT_CONFIG_RETRY_MIN_CONFIDENCE_BF16_OFFSET 20u
#define UAPS_ATTEMPT_CONFIG_PREDICTION_TARGET_OFFSET 22u
#define UAPS_ATTEMPT_CONFIG_SOURCE_B_FLAGS_OFFSET 23u
#define UAPS_ATTEMPT_CONFIG_MIN_REUSE_SCORE_BF16_OFFSET 24u
#define UAPS_ATTEMPT_CONFIG_MAX_CURRENT_UNRESOLVED_OFFSET 26u
#define UAPS_ATTEMPT_CONFIG_MAX_HANDOFF_TOKENS_OFFSET 27u
#define UAPS_ATTEMPT_CONFIG_ADMISSION_BUDGET_OFFSET 28u
#define UAPS_ATTEMPT_CONFIG_STATE_CONTROL_OFFSET 29u
#define UAPS_ATTEMPT_CONFIG_BLOCK_STEP_INDEX_OFFSET 30u

struct uaps_attempt_config {
    uint64_t state_base;
    uint64_t state_limit;
    uint32_t max_attempts;
    uint16_t retry_min_confidence_bf16;
    uint8_t prediction_target;
    uint8_t source_b_flags;
    uint16_t min_reuse_score_bf16;
    uint8_t max_current_unresolved;
    uint8_t max_handoff_tokens;
    uint8_t admission_budget;
    uint8_t state_control;
    uint16_t block_step_index;
};

UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(sizeof(struct uaps_attempt_config) == UAPS_ATTEMPT_CONFIG_BYTES,
               "uaps_attempt_config size mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, state_base) ==
               UAPS_ATTEMPT_CONFIG_STATE_BASE_OFFSET,
               "state_base offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, state_limit) ==
               UAPS_ATTEMPT_CONFIG_STATE_LIMIT_OFFSET,
               "state_limit offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, max_attempts) ==
               UAPS_ATTEMPT_CONFIG_MAX_ATTEMPTS_OFFSET,
               "max_attempts offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, retry_min_confidence_bf16) ==
               UAPS_ATTEMPT_CONFIG_RETRY_MIN_CONFIDENCE_BF16_OFFSET,
               "retry_min_confidence_bf16 offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, prediction_target) ==
               UAPS_ATTEMPT_CONFIG_PREDICTION_TARGET_OFFSET,
               "prediction_target offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, source_b_flags) ==
               UAPS_ATTEMPT_CONFIG_SOURCE_B_FLAGS_OFFSET,
               "source_b_flags offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, min_reuse_score_bf16) ==
               UAPS_ATTEMPT_CONFIG_MIN_REUSE_SCORE_BF16_OFFSET,
               "min_reuse_score_bf16 offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, max_current_unresolved) ==
               UAPS_ATTEMPT_CONFIG_MAX_CURRENT_UNRESOLVED_OFFSET,
               "max_current_unresolved offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, max_handoff_tokens) ==
               UAPS_ATTEMPT_CONFIG_MAX_HANDOFF_TOKENS_OFFSET,
               "max_handoff_tokens offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, admission_budget) ==
               UAPS_ATTEMPT_CONFIG_ADMISSION_BUDGET_OFFSET,
               "admission_budget offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, state_control) ==
               UAPS_ATTEMPT_CONFIG_STATE_CONTROL_OFFSET,
               "state_control offset mismatch");
UAPS_ATTEMPT_CONFIG_STATIC_ASSERT(offsetof(struct uaps_attempt_config, block_step_index) ==
               UAPS_ATTEMPT_CONFIG_BLOCK_STEP_INDEX_OFFSET,
               "block_step_index offset mismatch");

#undef UAPS_ATTEMPT_CONFIG_STATIC_ASSERT

#endif
