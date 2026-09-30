#ifndef CROSS_BLOCK_SHORTLIST_CONFIG_H
#define CROSS_BLOCK_SHORTLIST_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define CROSS_BLOCK_SHORTLIST_CONFIG_BYTES 32u
#define CROSS_BLOCK_SHORTLIST_CONFIG_ALIGNMENT 16u
#define CROSS_BLOCK_SHORTLIST_CONFIG_PENDING_BASE_OFFSET 0u
#define CROSS_BLOCK_SHORTLIST_CONFIG_PENDING_LIMIT_OFFSET 8u
#define CROSS_BLOCK_SHORTLIST_CONFIG_SHORTLIST_TOKEN_COUNT_OFFSET 16u
#define CROSS_BLOCK_SHORTLIST_CONFIG_FLAGS_OFFSET 18u
#define CROSS_BLOCK_SHORTLIST_CONFIG_RELATIVE_SCORE_FLOOR_BF16_OFFSET 20u
#define CROSS_BLOCK_SHORTLIST_CONFIG_PROTECTED_BEGIN_OFFSET 22u
#define CROSS_BLOCK_SHORTLIST_CONFIG_PROTECTED_END_OFFSET 24u
#define CROSS_BLOCK_SHORTLIST_CONFIG_RESERVED0_OFFSET 26u
#define CROSS_BLOCK_SHORTLIST_CONFIG_RESERVED1_OFFSET 28u

struct cross_block_shortlist_config {
    uint64_t pending_base;
    uint64_t pending_limit;
    uint16_t shortlist_token_count;
    uint16_t flags;
    uint16_t relative_score_floor_bf16;
    uint16_t protected_begin;
    uint16_t protected_end;
    uint16_t reserved0;
    uint32_t reserved1;
};

CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(sizeof(struct cross_block_shortlist_config) == CROSS_BLOCK_SHORTLIST_CONFIG_BYTES,
               "cross_block_shortlist_config size mismatch");
CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(offsetof(struct cross_block_shortlist_config, pending_base) ==
               CROSS_BLOCK_SHORTLIST_CONFIG_PENDING_BASE_OFFSET,
               "pending_base offset mismatch");
CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(offsetof(struct cross_block_shortlist_config, pending_limit) ==
               CROSS_BLOCK_SHORTLIST_CONFIG_PENDING_LIMIT_OFFSET,
               "pending_limit offset mismatch");
CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(offsetof(struct cross_block_shortlist_config, shortlist_token_count) ==
               CROSS_BLOCK_SHORTLIST_CONFIG_SHORTLIST_TOKEN_COUNT_OFFSET,
               "shortlist_token_count offset mismatch");
CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(offsetof(struct cross_block_shortlist_config, flags) ==
               CROSS_BLOCK_SHORTLIST_CONFIG_FLAGS_OFFSET,
               "flags offset mismatch");
CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(offsetof(struct cross_block_shortlist_config, relative_score_floor_bf16) ==
               CROSS_BLOCK_SHORTLIST_CONFIG_RELATIVE_SCORE_FLOOR_BF16_OFFSET,
               "relative_score_floor_bf16 offset mismatch");
CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(offsetof(struct cross_block_shortlist_config, protected_begin) ==
               CROSS_BLOCK_SHORTLIST_CONFIG_PROTECTED_BEGIN_OFFSET,
               "protected_begin offset mismatch");
CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(offsetof(struct cross_block_shortlist_config, protected_end) ==
               CROSS_BLOCK_SHORTLIST_CONFIG_PROTECTED_END_OFFSET,
               "protected_end offset mismatch");
CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(offsetof(struct cross_block_shortlist_config, reserved0) ==
               CROSS_BLOCK_SHORTLIST_CONFIG_RESERVED0_OFFSET,
               "reserved0 offset mismatch");
CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT(offsetof(struct cross_block_shortlist_config, reserved1) ==
               CROSS_BLOCK_SHORTLIST_CONFIG_RESERVED1_OFFSET,
               "reserved1 offset mismatch");

#undef CROSS_BLOCK_SHORTLIST_CONFIG_STATIC_ASSERT

#endif
