#ifndef EXECUTION_CONFIG_H
#define EXECUTION_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define EXECUTION_CONFIG_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define EXECUTION_CONFIG_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define EXECUTION_CONFIG_BYTES 320u
#define EXECUTION_CONFIG_ALIGNMENT 16u
#define EXECUTION_CONFIG_MAGIC_OFFSET 0u
#define EXECUTION_CONFIG_VERSION_OFFSET 4u
#define EXECUTION_CONFIG_HEADER_BYTES_OFFSET 6u
#define EXECUTION_CONFIG_TOTAL_BYTES_OFFSET 8u
#define EXECUTION_CONFIG_FLAGS_OFFSET 12u
#define EXECUTION_CONFIG_LAYER_COUNT_OFFSET 16u
#define EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET 18u
#define EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET 20u
#define EXECUTION_CONFIG_TOKEN_METADATA_FORMAT_OFFSET 22u
#define EXECUTION_CONFIG_START_LAYER_OFFSET 24u
#define EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET 26u
#define EXECUTION_CONFIG_ROTATION_ENABLE_MASK_OFFSET 28u
#define EXECUTION_CONFIG_FFN_PAIR_FIRST_BATCH_PLUS1_OFFSET 29u
#define EXECUTION_CONFIG_JOINT_CONFIGURATION_OFFSET_OFFSET 30u
#define EXECUTION_CONFIG_NEXT_TOKEN_METADATA_BASE_OFFSET 32u
#define EXECUTION_CONFIG_NEXT_TOKEN_METADATA_LIMIT_OFFSET 40u
#define EXECUTION_CONFIG_OUTPUT_HIDDEN_BASE_OFFSET 48u
#define EXECUTION_CONFIG_OUTPUT_HIDDEN_LIMIT_OFFSET 56u
#define EXECUTION_CONFIG_BF16_TEMPORARY_BASE_OFFSET 64u
#define EXECUTION_CONFIG_BF16_TEMPORARY_LIMIT_OFFSET 72u
#define EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET 80u
#define EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_LIMIT_OFFSET 88u
#define EXECUTION_CONFIG_CURRENT_K_CACHE_BASE_OFFSET 96u
#define EXECUTION_CONFIG_CURRENT_K_CACHE_LIMIT_OFFSET 104u
#define EXECUTION_CONFIG_CURRENT_V_CACHE_BASE_OFFSET 112u
#define EXECUTION_CONFIG_CURRENT_V_CACHE_LIMIT_OFFSET 120u
#define EXECUTION_CONFIG_RETAINED_K_CACHE_BASE_OFFSET 128u
#define EXECUTION_CONFIG_RETAINED_K_CACHE_LIMIT_OFFSET 136u
#define EXECUTION_CONFIG_RETAINED_V_CACHE_BASE_OFFSET 144u
#define EXECUTION_CONFIG_RETAINED_V_CACHE_LIMIT_OFFSET 152u
#define EXECUTION_CONFIG_K_SCALE_BASE_OFFSET 160u
#define EXECUTION_CONFIG_K_SCALE_LIMIT_OFFSET 168u
#define EXECUTION_CONFIG_V_SCALE_BASE_OFFSET 176u
#define EXECUTION_CONFIG_V_SCALE_LIMIT_OFFSET 184u
#define EXECUTION_CONFIG_ATTENTION_WORKSPACE_BASE_OFFSET 192u
#define EXECUTION_CONFIG_ATTENTION_WORKSPACE_LIMIT_OFFSET 200u
#define EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET 208u
#define EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET 216u
#define EXECUTION_CONFIG_EMBEDDING_BASE_OFFSET 224u
#define EXECUTION_CONFIG_EMBEDDING_LIMIT_OFFSET 232u
#define EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BASE_OFFSET 240u
#define EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_LIMIT_OFFSET 248u
#define EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET 256u
#define EXECUTION_CONFIG_PREDICTION_TABLE_LIMIT_OFFSET 264u
#define EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET 272u
#define EXECUTION_CONFIG_EMBEDDING_ROW_BYTES_OFFSET 276u
#define EXECUTION_CONFIG_DEPLOYMENT_ARTIFACT_ID_OFFSET 280u
#define EXECUTION_CONFIG_TOKEN_METADATA_ENTRY_BYTES_OFFSET 288u
#define EXECUTION_CONFIG_TOKEN_METADATA_BATCH_HEADER_BYTES_OFFSET 290u
#define EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BYTES_OFFSET 292u
#define EXECUTION_CONFIG_PREDICTION_ENTRY_BYTES_OFFSET 294u
#define EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_FORMAT_OFFSET 296u
#define EXECUTION_CONFIG_PREDICTION_TABLE_FORMAT_OFFSET 298u
#define EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET 300u
#define EXECUTION_CONFIG_GENERATION_BLOCK_COUNT_OFFSET 302u
#define EXECUTION_CONFIG_SUPPRESSED_TOKEN_COUNT_OFFSET 303u
#define EXECUTION_CONFIG_RETAINED_K_SCALE_BASE_OFFSET 304u
#define EXECUTION_CONFIG_RETAINED_K_SCALE_LIMIT_OFFSET 312u

struct execution_config {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint32_t total_bytes;
    uint32_t flags;
    uint16_t layer_count;
    uint16_t total_token_count;
    uint16_t sequence_length;
    uint16_t token_metadata_format;
    uint16_t start_layer;
    uint16_t refresh_configuration_offset;
    uint8_t rotation_enable_mask;
    uint8_t ffn_pair_first_batch_plus1;
    uint16_t joint_configuration_offset;
    uint64_t next_token_metadata_base;
    uint64_t next_token_metadata_limit;
    uint64_t output_hidden_base;
    uint64_t output_hidden_limit;
    uint64_t bf16_temporary_base;
    uint64_t bf16_temporary_limit;
    uint64_t layer_weight_table_base;
    uint64_t layer_weight_table_limit;
    uint64_t current_k_cache_base;
    uint64_t current_k_cache_limit;
    uint64_t current_v_cache_base;
    uint64_t current_v_cache_limit;
    uint64_t retained_k_cache_base;
    uint64_t retained_k_cache_limit;
    uint64_t retained_v_cache_base;
    uint64_t retained_v_cache_limit;
    uint64_t k_scale_base;
    uint64_t k_scale_limit;
    uint64_t v_scale_base;
    uint64_t v_scale_limit;
    uint64_t attention_workspace_base;
    uint64_t attention_workspace_limit;
    uint64_t token_metadata_base;
    uint64_t token_metadata_limit;
    uint64_t embedding_base;
    uint64_t embedding_limit;
    uint64_t forward_postprocess_configuration_base;
    uint64_t forward_postprocess_configuration_limit;
    uint64_t prediction_table_base;
    uint64_t prediction_table_limit;
    uint32_t layer_weight_entry_stride;
    uint32_t embedding_row_bytes;
    uint64_t deployment_artifact_id;
    uint16_t token_metadata_entry_bytes;
    uint16_t token_metadata_batch_header_bytes;
    uint16_t forward_postprocess_configuration_bytes;
    uint16_t prediction_entry_bytes;
    uint16_t forward_postprocess_configuration_format;
    uint16_t prediction_table_format;
    uint16_t prediction_count;
    uint8_t generation_block_count;
    uint8_t suppressed_token_count;
    uint64_t retained_k_scale_base;
    uint64_t retained_k_scale_limit;
};

EXECUTION_CONFIG_STATIC_ASSERT(sizeof(struct execution_config) == EXECUTION_CONFIG_BYTES,
               "execution_config size mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, magic) ==
               EXECUTION_CONFIG_MAGIC_OFFSET,
               "magic offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, version) ==
               EXECUTION_CONFIG_VERSION_OFFSET,
               "version offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, header_bytes) ==
               EXECUTION_CONFIG_HEADER_BYTES_OFFSET,
               "header_bytes offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, total_bytes) ==
               EXECUTION_CONFIG_TOTAL_BYTES_OFFSET,
               "total_bytes offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, flags) ==
               EXECUTION_CONFIG_FLAGS_OFFSET,
               "flags offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, layer_count) ==
               EXECUTION_CONFIG_LAYER_COUNT_OFFSET,
               "layer_count offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, total_token_count) ==
               EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET,
               "total_token_count offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, sequence_length) ==
               EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET,
               "sequence_length offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, token_metadata_format) ==
               EXECUTION_CONFIG_TOKEN_METADATA_FORMAT_OFFSET,
               "token_metadata_format offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, start_layer) ==
               EXECUTION_CONFIG_START_LAYER_OFFSET,
               "start_layer offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, refresh_configuration_offset) ==
               EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET,
               "refresh_configuration_offset offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, rotation_enable_mask) ==
               EXECUTION_CONFIG_ROTATION_ENABLE_MASK_OFFSET,
               "rotation_enable_mask offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, ffn_pair_first_batch_plus1) ==
               EXECUTION_CONFIG_FFN_PAIR_FIRST_BATCH_PLUS1_OFFSET,
               "ffn_pair_first_batch_plus1 offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, joint_configuration_offset) ==
               EXECUTION_CONFIG_JOINT_CONFIGURATION_OFFSET_OFFSET,
               "joint_configuration_offset offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, next_token_metadata_base) ==
               EXECUTION_CONFIG_NEXT_TOKEN_METADATA_BASE_OFFSET,
               "next_token_metadata_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, next_token_metadata_limit) ==
               EXECUTION_CONFIG_NEXT_TOKEN_METADATA_LIMIT_OFFSET,
               "next_token_metadata_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, output_hidden_base) ==
               EXECUTION_CONFIG_OUTPUT_HIDDEN_BASE_OFFSET,
               "output_hidden_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, output_hidden_limit) ==
               EXECUTION_CONFIG_OUTPUT_HIDDEN_LIMIT_OFFSET,
               "output_hidden_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, bf16_temporary_base) ==
               EXECUTION_CONFIG_BF16_TEMPORARY_BASE_OFFSET,
               "bf16_temporary_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, bf16_temporary_limit) ==
               EXECUTION_CONFIG_BF16_TEMPORARY_LIMIT_OFFSET,
               "bf16_temporary_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, layer_weight_table_base) ==
               EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET,
               "layer_weight_table_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, layer_weight_table_limit) ==
               EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_LIMIT_OFFSET,
               "layer_weight_table_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, current_k_cache_base) ==
               EXECUTION_CONFIG_CURRENT_K_CACHE_BASE_OFFSET,
               "current_k_cache_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, current_k_cache_limit) ==
               EXECUTION_CONFIG_CURRENT_K_CACHE_LIMIT_OFFSET,
               "current_k_cache_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, current_v_cache_base) ==
               EXECUTION_CONFIG_CURRENT_V_CACHE_BASE_OFFSET,
               "current_v_cache_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, current_v_cache_limit) ==
               EXECUTION_CONFIG_CURRENT_V_CACHE_LIMIT_OFFSET,
               "current_v_cache_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, retained_k_cache_base) ==
               EXECUTION_CONFIG_RETAINED_K_CACHE_BASE_OFFSET,
               "retained_k_cache_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, retained_k_cache_limit) ==
               EXECUTION_CONFIG_RETAINED_K_CACHE_LIMIT_OFFSET,
               "retained_k_cache_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, retained_v_cache_base) ==
               EXECUTION_CONFIG_RETAINED_V_CACHE_BASE_OFFSET,
               "retained_v_cache_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, retained_v_cache_limit) ==
               EXECUTION_CONFIG_RETAINED_V_CACHE_LIMIT_OFFSET,
               "retained_v_cache_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, k_scale_base) ==
               EXECUTION_CONFIG_K_SCALE_BASE_OFFSET,
               "k_scale_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, k_scale_limit) ==
               EXECUTION_CONFIG_K_SCALE_LIMIT_OFFSET,
               "k_scale_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, v_scale_base) ==
               EXECUTION_CONFIG_V_SCALE_BASE_OFFSET,
               "v_scale_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, v_scale_limit) ==
               EXECUTION_CONFIG_V_SCALE_LIMIT_OFFSET,
               "v_scale_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, attention_workspace_base) ==
               EXECUTION_CONFIG_ATTENTION_WORKSPACE_BASE_OFFSET,
               "attention_workspace_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, attention_workspace_limit) ==
               EXECUTION_CONFIG_ATTENTION_WORKSPACE_LIMIT_OFFSET,
               "attention_workspace_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, token_metadata_base) ==
               EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET,
               "token_metadata_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, token_metadata_limit) ==
               EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET,
               "token_metadata_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, embedding_base) ==
               EXECUTION_CONFIG_EMBEDDING_BASE_OFFSET,
               "embedding_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, embedding_limit) ==
               EXECUTION_CONFIG_EMBEDDING_LIMIT_OFFSET,
               "embedding_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, forward_postprocess_configuration_base) ==
               EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BASE_OFFSET,
               "forward_postprocess_configuration_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, forward_postprocess_configuration_limit) ==
               EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_LIMIT_OFFSET,
               "forward_postprocess_configuration_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, prediction_table_base) ==
               EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET,
               "prediction_table_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, prediction_table_limit) ==
               EXECUTION_CONFIG_PREDICTION_TABLE_LIMIT_OFFSET,
               "prediction_table_limit offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, layer_weight_entry_stride) ==
               EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET,
               "layer_weight_entry_stride offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, embedding_row_bytes) ==
               EXECUTION_CONFIG_EMBEDDING_ROW_BYTES_OFFSET,
               "embedding_row_bytes offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, deployment_artifact_id) ==
               EXECUTION_CONFIG_DEPLOYMENT_ARTIFACT_ID_OFFSET,
               "deployment_artifact_id offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, token_metadata_entry_bytes) ==
               EXECUTION_CONFIG_TOKEN_METADATA_ENTRY_BYTES_OFFSET,
               "token_metadata_entry_bytes offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, token_metadata_batch_header_bytes) ==
               EXECUTION_CONFIG_TOKEN_METADATA_BATCH_HEADER_BYTES_OFFSET,
               "token_metadata_batch_header_bytes offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, forward_postprocess_configuration_bytes) ==
               EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BYTES_OFFSET,
               "forward_postprocess_configuration_bytes offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, prediction_entry_bytes) ==
               EXECUTION_CONFIG_PREDICTION_ENTRY_BYTES_OFFSET,
               "prediction_entry_bytes offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, forward_postprocess_configuration_format) ==
               EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_FORMAT_OFFSET,
               "forward_postprocess_configuration_format offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, prediction_table_format) ==
               EXECUTION_CONFIG_PREDICTION_TABLE_FORMAT_OFFSET,
               "prediction_table_format offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, prediction_count) ==
               EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET,
               "prediction_count offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, generation_block_count) ==
               EXECUTION_CONFIG_GENERATION_BLOCK_COUNT_OFFSET,
               "generation_block_count offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, suppressed_token_count) ==
               EXECUTION_CONFIG_SUPPRESSED_TOKEN_COUNT_OFFSET,
               "suppressed_token_count offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, retained_k_scale_base) ==
               EXECUTION_CONFIG_RETAINED_K_SCALE_BASE_OFFSET,
               "retained_k_scale_base offset mismatch");
EXECUTION_CONFIG_STATIC_ASSERT(offsetof(struct execution_config, retained_k_scale_limit) ==
               EXECUTION_CONFIG_RETAINED_K_SCALE_LIMIT_OFFSET,
               "retained_k_scale_limit offset mismatch");

#undef EXECUTION_CONFIG_STATIC_ASSERT

#endif
