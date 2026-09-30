#ifndef IN_BLOCK_REFRESH_BUDGET_H
#define IN_BLOCK_REFRESH_BUDGET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define IN_BLOCK_REFRESH_BUDGET_BYTES 16u
#define IN_BLOCK_REFRESH_BUDGET_ALIGNMENT 16u
#define IN_BLOCK_REFRESH_BUDGET_TARGET_TOKEN_COUNT_X2_OFFSET 0u
#define IN_BLOCK_REFRESH_BUDGET_SELECTION_CONTROL_OFFSET 1u
#define IN_BLOCK_REFRESH_BUDGET_REGION_START_OFFSET 2u
#define IN_BLOCK_REFRESH_BUDGET_TOKEN_COUNT_X2_CREDIT_OFFSET 4u
#define IN_BLOCK_REFRESH_BUDGET_REGULAR_STEPS_OFFSET 8u
#define IN_BLOCK_REFRESH_BUDGET_CUMULATIVE_ACTIVE_TOKEN_COUNT_OFFSET 12u

struct in_block_refresh_budget {
    uint8_t target_token_count_x2;
    uint8_t selection_control;
    uint16_t region_start;
    uint32_t token_count_x2_credit;
    uint32_t regular_steps;
    uint32_t cumulative_active_token_count;
};

IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT(sizeof(struct in_block_refresh_budget) == IN_BLOCK_REFRESH_BUDGET_BYTES,
               "in_block_refresh_budget size mismatch");
IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT(offsetof(struct in_block_refresh_budget, target_token_count_x2) ==
               IN_BLOCK_REFRESH_BUDGET_TARGET_TOKEN_COUNT_X2_OFFSET,
               "target_token_count_x2 offset mismatch");
IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT(offsetof(struct in_block_refresh_budget, selection_control) ==
               IN_BLOCK_REFRESH_BUDGET_SELECTION_CONTROL_OFFSET,
               "selection_control offset mismatch");
IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT(offsetof(struct in_block_refresh_budget, region_start) ==
               IN_BLOCK_REFRESH_BUDGET_REGION_START_OFFSET,
               "region_start offset mismatch");
IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT(offsetof(struct in_block_refresh_budget, token_count_x2_credit) ==
               IN_BLOCK_REFRESH_BUDGET_TOKEN_COUNT_X2_CREDIT_OFFSET,
               "token_count_x2_credit offset mismatch");
IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT(offsetof(struct in_block_refresh_budget, regular_steps) ==
               IN_BLOCK_REFRESH_BUDGET_REGULAR_STEPS_OFFSET,
               "regular_steps offset mismatch");
IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT(offsetof(struct in_block_refresh_budget, cumulative_active_token_count) ==
               IN_BLOCK_REFRESH_BUDGET_CUMULATIVE_ACTIVE_TOKEN_COUNT_OFFSET,
               "cumulative_active_token_count offset mismatch");

#undef IN_BLOCK_REFRESH_BUDGET_STATIC_ASSERT

#endif
