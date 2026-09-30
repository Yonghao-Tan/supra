#ifndef UAPS_CONFIG_H
#define UAPS_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define UAPS_CONFIG_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define UAPS_CONFIG_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define UAPS_CONFIG_BYTES 64u
#define UAPS_CONFIG_ALIGNMENT 16u
#define UAPS_CONFIG_MAGIC_OFFSET 0u
#define UAPS_CONFIG_VERSION_OFFSET 4u
#define UAPS_CONFIG_BYTES_OFFSET 6u
#define UAPS_CONFIG_BASE_TOKEN_COUNT_OFFSET 8u
#define UAPS_CONFIG_FUTURE_TOKEN_COUNT_OFFSET 10u
#define UAPS_CONFIG_NEXT_BLOCK_START_OFFSET 12u
#define UAPS_CONFIG_MAX_NEXT_TOKENS_OFFSET 14u
#define UAPS_CONFIG_PRIORITY_CONTROL_OFFSET 15u
#define UAPS_CONFIG_BASE_TABLE_BASE_OFFSET 16u
#define UAPS_CONFIG_BASE_TABLE_LIMIT_OFFSET 24u
#define UAPS_CONFIG_FUTURE_TABLE_BASE_OFFSET 32u
#define UAPS_CONFIG_FUTURE_TABLE_LIMIT_OFFSET 40u
#define UAPS_CONFIG_RESULT_BASE_OFFSET 48u
#define UAPS_CONFIG_RESULT_LIMIT_OFFSET 56u

struct uaps_config {
    uint32_t magic;
    uint16_t version;
    uint16_t bytes;
    uint16_t base_token_count;
    uint16_t future_token_count;
    uint16_t next_block_start;
    uint8_t max_next_tokens;
    uint8_t priority_control;
    uint64_t base_table_base;
    uint64_t base_table_limit;
    uint64_t future_table_base;
    uint64_t future_table_limit;
    uint64_t result_base;
    uint64_t result_limit;
};

UAPS_CONFIG_STATIC_ASSERT(sizeof(struct uaps_config) == UAPS_CONFIG_BYTES,
               "uaps_config size mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, magic) ==
               UAPS_CONFIG_MAGIC_OFFSET,
               "magic offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, version) ==
               UAPS_CONFIG_VERSION_OFFSET,
               "version offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, bytes) ==
               UAPS_CONFIG_BYTES_OFFSET,
               "bytes offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, base_token_count) ==
               UAPS_CONFIG_BASE_TOKEN_COUNT_OFFSET,
               "base_token_count offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, future_token_count) ==
               UAPS_CONFIG_FUTURE_TOKEN_COUNT_OFFSET,
               "future_token_count offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, next_block_start) ==
               UAPS_CONFIG_NEXT_BLOCK_START_OFFSET,
               "next_block_start offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, max_next_tokens) ==
               UAPS_CONFIG_MAX_NEXT_TOKENS_OFFSET,
               "max_next_tokens offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, priority_control) ==
               UAPS_CONFIG_PRIORITY_CONTROL_OFFSET,
               "priority_control offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, base_table_base) ==
               UAPS_CONFIG_BASE_TABLE_BASE_OFFSET,
               "base_table_base offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, base_table_limit) ==
               UAPS_CONFIG_BASE_TABLE_LIMIT_OFFSET,
               "base_table_limit offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, future_table_base) ==
               UAPS_CONFIG_FUTURE_TABLE_BASE_OFFSET,
               "future_table_base offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, future_table_limit) ==
               UAPS_CONFIG_FUTURE_TABLE_LIMIT_OFFSET,
               "future_table_limit offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, result_base) ==
               UAPS_CONFIG_RESULT_BASE_OFFSET,
               "result_base offset mismatch");
UAPS_CONFIG_STATIC_ASSERT(offsetof(struct uaps_config, result_limit) ==
               UAPS_CONFIG_RESULT_LIMIT_OFFSET,
               "result_limit offset mismatch");

#undef UAPS_CONFIG_STATIC_ASSERT

#endif
