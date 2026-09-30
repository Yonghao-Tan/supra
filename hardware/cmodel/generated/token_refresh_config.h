#ifndef TOKEN_REFRESH_CONFIG_H
#define TOKEN_REFRESH_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define TOKEN_REFRESH_CONFIG_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define TOKEN_REFRESH_CONFIG_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define TOKEN_REFRESH_CONFIG_BYTES 176u
#define TOKEN_REFRESH_CONFIG_ALIGNMENT 16u
#define TOKEN_REFRESH_CONFIG_MAGIC_OFFSET 0u
#define TOKEN_REFRESH_CONFIG_VERSION_OFFSET 4u
#define TOKEN_REFRESH_CONFIG_BYTES_OFFSET 6u
#define TOKEN_REFRESH_CONFIG_FLAGS_OFFSET 8u
#define TOKEN_REFRESH_CONFIG_PENDING_CONFIGURATION_OFFSET_OFFSET 10u
#define TOKEN_REFRESH_CONFIG_SEQUENCE_LENGTH_OFFSET 12u
#define TOKEN_REFRESH_CONFIG_TARGET_TOKEN_COUNT_OFFSET 14u
#define TOKEN_REFRESH_CONFIG_REQUIRED_QUOTA_OFFSET 16u
#define TOKEN_REFRESH_CONFIG_PROBABILITY_CONFIGURATION_OFFSET_OFFSET 18u
#define TOKEN_REFRESH_CONFIG_HEAD_STRIDE_OFFSET 20u
#define TOKEN_REFRESH_CONFIG_SCALE_HEAD_STRIDE_OFFSET 24u
#define TOKEN_REFRESH_CONFIG_METADATA_VERSION_OFFSET 28u
#define TOKEN_REFRESH_CONFIG_TABLE_BASE_OFFSET 32u
#define TOKEN_REFRESH_CONFIG_TABLE_LIMIT_OFFSET 40u
#define TOKEN_REFRESH_CONFIG_METADATA_BASE_OFFSET 48u
#define TOKEN_REFRESH_CONFIG_METADATA_LIMIT_OFFSET 56u
#define TOKEN_REFRESH_CONFIG_SOURCE_K_BASE_OFFSET 64u
#define TOKEN_REFRESH_CONFIG_SOURCE_V_BASE_OFFSET 72u
#define TOKEN_REFRESH_CONFIG_SOURCE_SCALE_BASE_OFFSET 80u
#define TOKEN_REFRESH_CONFIG_DESTINATION_K_BASE_OFFSET 88u
#define TOKEN_REFRESH_CONFIG_DESTINATION_V_BASE_OFFSET 96u
#define TOKEN_REFRESH_CONFIG_DESTINATION_SCALE_BASE_OFFSET 104u
#define TOKEN_REFRESH_CONFIG_SOURCE_K_LIMIT_OFFSET 112u
#define TOKEN_REFRESH_CONFIG_SOURCE_V_LIMIT_OFFSET 120u
#define TOKEN_REFRESH_CONFIG_SOURCE_SCALE_LIMIT_OFFSET 128u
#define TOKEN_REFRESH_CONFIG_DESTINATION_K_LIMIT_OFFSET 136u
#define TOKEN_REFRESH_CONFIG_DESTINATION_V_LIMIT_OFFSET 144u
#define TOKEN_REFRESH_CONFIG_DESTINATION_SCALE_LIMIT_OFFSET 152u
#define TOKEN_REFRESH_CONFIG_CAPTURE_INDEX_OFFSET 160u
#define TOKEN_REFRESH_CONFIG_RELATION_JOB_COUNT_OFFSET 164u
#define TOKEN_REFRESH_CONFIG_RELATION_JOB_BASE_OFFSET 168u

struct token_refresh_config {
    uint32_t magic;
    uint16_t version;
    uint16_t bytes;
    uint16_t flags;
    uint16_t pending_configuration_offset;
    uint16_t sequence_length;
    uint16_t target_token_count;
    uint16_t required_quota;
    uint16_t probability_configuration_offset;
    uint32_t head_stride;
    uint32_t scale_head_stride;
    uint32_t metadata_version;
    uint64_t table_base;
    uint64_t table_limit;
    uint64_t metadata_base;
    uint64_t metadata_limit;
    uint64_t source_k_base;
    uint64_t source_v_base;
    uint64_t source_scale_base;
    uint64_t destination_k_base;
    uint64_t destination_v_base;
    uint64_t destination_scale_base;
    uint64_t source_k_limit;
    uint64_t source_v_limit;
    uint64_t source_scale_limit;
    uint64_t destination_k_limit;
    uint64_t destination_v_limit;
    uint64_t destination_scale_limit;
    uint32_t capture_index;
    uint32_t relation_job_count;
    uint64_t relation_job_base;
};

TOKEN_REFRESH_CONFIG_STATIC_ASSERT(sizeof(struct token_refresh_config) == TOKEN_REFRESH_CONFIG_BYTES,
               "token_refresh_config size mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, magic) ==
               TOKEN_REFRESH_CONFIG_MAGIC_OFFSET,
               "magic offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, version) ==
               TOKEN_REFRESH_CONFIG_VERSION_OFFSET,
               "version offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, bytes) ==
               TOKEN_REFRESH_CONFIG_BYTES_OFFSET,
               "bytes offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, flags) ==
               TOKEN_REFRESH_CONFIG_FLAGS_OFFSET,
               "flags offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, pending_configuration_offset) ==
               TOKEN_REFRESH_CONFIG_PENDING_CONFIGURATION_OFFSET_OFFSET,
               "pending_configuration_offset offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, sequence_length) ==
               TOKEN_REFRESH_CONFIG_SEQUENCE_LENGTH_OFFSET,
               "sequence_length offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, target_token_count) ==
               TOKEN_REFRESH_CONFIG_TARGET_TOKEN_COUNT_OFFSET,
               "target_token_count offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, required_quota) ==
               TOKEN_REFRESH_CONFIG_REQUIRED_QUOTA_OFFSET,
               "required_quota offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, probability_configuration_offset) ==
               TOKEN_REFRESH_CONFIG_PROBABILITY_CONFIGURATION_OFFSET_OFFSET,
               "probability_configuration_offset offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, head_stride) ==
               TOKEN_REFRESH_CONFIG_HEAD_STRIDE_OFFSET,
               "head_stride offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, scale_head_stride) ==
               TOKEN_REFRESH_CONFIG_SCALE_HEAD_STRIDE_OFFSET,
               "scale_head_stride offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, metadata_version) ==
               TOKEN_REFRESH_CONFIG_METADATA_VERSION_OFFSET,
               "metadata_version offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, table_base) ==
               TOKEN_REFRESH_CONFIG_TABLE_BASE_OFFSET,
               "table_base offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, table_limit) ==
               TOKEN_REFRESH_CONFIG_TABLE_LIMIT_OFFSET,
               "table_limit offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, metadata_base) ==
               TOKEN_REFRESH_CONFIG_METADATA_BASE_OFFSET,
               "metadata_base offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, metadata_limit) ==
               TOKEN_REFRESH_CONFIG_METADATA_LIMIT_OFFSET,
               "metadata_limit offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, source_k_base) ==
               TOKEN_REFRESH_CONFIG_SOURCE_K_BASE_OFFSET,
               "source_k_base offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, source_v_base) ==
               TOKEN_REFRESH_CONFIG_SOURCE_V_BASE_OFFSET,
               "source_v_base offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, source_scale_base) ==
               TOKEN_REFRESH_CONFIG_SOURCE_SCALE_BASE_OFFSET,
               "source_scale_base offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, destination_k_base) ==
               TOKEN_REFRESH_CONFIG_DESTINATION_K_BASE_OFFSET,
               "destination_k_base offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, destination_v_base) ==
               TOKEN_REFRESH_CONFIG_DESTINATION_V_BASE_OFFSET,
               "destination_v_base offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, destination_scale_base) ==
               TOKEN_REFRESH_CONFIG_DESTINATION_SCALE_BASE_OFFSET,
               "destination_scale_base offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, source_k_limit) ==
               TOKEN_REFRESH_CONFIG_SOURCE_K_LIMIT_OFFSET,
               "source_k_limit offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, source_v_limit) ==
               TOKEN_REFRESH_CONFIG_SOURCE_V_LIMIT_OFFSET,
               "source_v_limit offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, source_scale_limit) ==
               TOKEN_REFRESH_CONFIG_SOURCE_SCALE_LIMIT_OFFSET,
               "source_scale_limit offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, destination_k_limit) ==
               TOKEN_REFRESH_CONFIG_DESTINATION_K_LIMIT_OFFSET,
               "destination_k_limit offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, destination_v_limit) ==
               TOKEN_REFRESH_CONFIG_DESTINATION_V_LIMIT_OFFSET,
               "destination_v_limit offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, destination_scale_limit) ==
               TOKEN_REFRESH_CONFIG_DESTINATION_SCALE_LIMIT_OFFSET,
               "destination_scale_limit offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, capture_index) ==
               TOKEN_REFRESH_CONFIG_CAPTURE_INDEX_OFFSET,
               "capture_index offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, relation_job_count) ==
               TOKEN_REFRESH_CONFIG_RELATION_JOB_COUNT_OFFSET,
               "relation_job_count offset mismatch");
TOKEN_REFRESH_CONFIG_STATIC_ASSERT(offsetof(struct token_refresh_config, relation_job_base) ==
               TOKEN_REFRESH_CONFIG_RELATION_JOB_BASE_OFFSET,
               "relation_job_base offset mismatch");

#undef TOKEN_REFRESH_CONFIG_STATIC_ASSERT

#endif
