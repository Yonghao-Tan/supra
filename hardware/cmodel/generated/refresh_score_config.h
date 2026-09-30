#ifndef REFRESH_SCORE_CONFIG_H
#define REFRESH_SCORE_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define REFRESH_SCORE_CONFIG_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define REFRESH_SCORE_CONFIG_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define REFRESH_SCORE_CONFIG_BYTES 80u
#define REFRESH_SCORE_CONFIG_ALIGNMENT 16u
#define REFRESH_SCORE_CONFIG_MAGIC_OFFSET 0u
#define REFRESH_SCORE_CONFIG_VERSION_OFFSET 4u
#define REFRESH_SCORE_CONFIG_BYTES_OFFSET 6u
#define REFRESH_SCORE_CONFIG_TOKEN_COUNT_OFFSET 8u
#define REFRESH_SCORE_CONFIG_KEYS_OFFSET 10u
#define REFRESH_SCORE_CONFIG_BLOCK_END_OFFSET 12u
#define REFRESH_SCORE_CONFIG_RELATION_ROW_SHIFT_OFFSET 14u
#define REFRESH_SCORE_CONFIG_FLAGS_OFFSET 15u
#define REFRESH_SCORE_CONFIG_RELATION_BASE_OFFSET 16u
#define REFRESH_SCORE_CONFIG_RELATION_LIMIT_OFFSET 24u
#define REFRESH_SCORE_CONFIG_CHANGE_BASE_OFFSET 32u
#define REFRESH_SCORE_CONFIG_CHANGE_LIMIT_OFFSET 40u
#define REFRESH_SCORE_CONFIG_PENDING_BASE_OFFSET 48u
#define REFRESH_SCORE_CONFIG_PENDING_LIMIT_OFFSET 56u
#define REFRESH_SCORE_CONFIG_REGULAR_BUDGET_BASE_OFFSET 64u
#define REFRESH_SCORE_CONFIG_REGULAR_BUDGET_LIMIT_OFFSET 72u

struct refresh_score_config {
    uint32_t magic;
    uint16_t version;
    uint16_t bytes;
    uint16_t token_count;
    uint16_t keys;
    uint16_t block_end;
    uint8_t relation_row_shift;
    uint8_t flags;
    uint64_t relation_base;
    uint64_t relation_limit;
    uint64_t change_base;
    uint64_t change_limit;
    uint64_t pending_base;
    uint64_t pending_limit;
    uint64_t regular_budget_base;
    uint64_t regular_budget_limit;
};

REFRESH_SCORE_CONFIG_STATIC_ASSERT(sizeof(struct refresh_score_config) == REFRESH_SCORE_CONFIG_BYTES,
               "refresh_score_config size mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, magic) ==
               REFRESH_SCORE_CONFIG_MAGIC_OFFSET,
               "magic offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, version) ==
               REFRESH_SCORE_CONFIG_VERSION_OFFSET,
               "version offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, bytes) ==
               REFRESH_SCORE_CONFIG_BYTES_OFFSET,
               "bytes offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, token_count) ==
               REFRESH_SCORE_CONFIG_TOKEN_COUNT_OFFSET,
               "token_count offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, keys) ==
               REFRESH_SCORE_CONFIG_KEYS_OFFSET,
               "keys offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, block_end) ==
               REFRESH_SCORE_CONFIG_BLOCK_END_OFFSET,
               "block_end offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, relation_row_shift) ==
               REFRESH_SCORE_CONFIG_RELATION_ROW_SHIFT_OFFSET,
               "relation_row_shift offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, flags) ==
               REFRESH_SCORE_CONFIG_FLAGS_OFFSET,
               "flags offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, relation_base) ==
               REFRESH_SCORE_CONFIG_RELATION_BASE_OFFSET,
               "relation_base offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, relation_limit) ==
               REFRESH_SCORE_CONFIG_RELATION_LIMIT_OFFSET,
               "relation_limit offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, change_base) ==
               REFRESH_SCORE_CONFIG_CHANGE_BASE_OFFSET,
               "change_base offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, change_limit) ==
               REFRESH_SCORE_CONFIG_CHANGE_LIMIT_OFFSET,
               "change_limit offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, pending_base) ==
               REFRESH_SCORE_CONFIG_PENDING_BASE_OFFSET,
               "pending_base offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, pending_limit) ==
               REFRESH_SCORE_CONFIG_PENDING_LIMIT_OFFSET,
               "pending_limit offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, regular_budget_base) ==
               REFRESH_SCORE_CONFIG_REGULAR_BUDGET_BASE_OFFSET,
               "regular_budget_base offset mismatch");
REFRESH_SCORE_CONFIG_STATIC_ASSERT(offsetof(struct refresh_score_config, regular_budget_limit) ==
               REFRESH_SCORE_CONFIG_REGULAR_BUDGET_LIMIT_OFFSET,
               "regular_budget_limit offset mismatch");

#undef REFRESH_SCORE_CONFIG_STATIC_ASSERT

#endif
