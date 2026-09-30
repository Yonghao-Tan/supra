#ifndef LAYER_ADDRESS_TABLE_H
#define LAYER_ADDRESS_TABLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define LAYER_ADDRESS_TABLE_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define LAYER_ADDRESS_TABLE_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define LAYER_ADDRESS_TABLE_BYTES 624u
#define LAYER_ADDRESS_TABLE_ALIGNMENT 16u
#define LAYER_ADDRESS_TABLE_MAGIC_OFFSET 0u
#define LAYER_ADDRESS_TABLE_VERSION_OFFSET 4u
#define LAYER_ADDRESS_TABLE_HEADER_BYTES_OFFSET 6u
#define LAYER_ADDRESS_TABLE_ENTRY_BYTES_OFFSET 8u
#define LAYER_ADDRESS_TABLE_FLAGS_OFFSET 12u
#define LAYER_ADDRESS_TABLE_HIDDEN_SIZE_OFFSET 16u
#define LAYER_ADDRESS_TABLE_FFN_SIZE_OFFSET 20u
#define LAYER_ADDRESS_TABLE_HEAD_COUNT_OFFSET 24u
#define LAYER_ADDRESS_TABLE_HEAD_DIMENSION_OFFSET 26u
#define LAYER_ADDRESS_TABLE_MAXIMUM_SEQUENCE_LENGTH_OFFSET 28u
#define LAYER_ADDRESS_TABLE_QUERY_CLIP_RATIO_BF16_OFFSET 30u
#define LAYER_ADDRESS_TABLE_KV_HEAD_STRIDE_BYTES_OFFSET 32u
#define LAYER_ADDRESS_TABLE_KV_TOKEN_STRIDE_BYTES_OFFSET 36u
#define LAYER_ADDRESS_TABLE_KV_CHUNK_STRIDE_BYTES_OFFSET 40u
#define LAYER_ADDRESS_TABLE_K_SCALE_HEAD_STRIDE_BYTES_OFFSET 44u
#define LAYER_ADDRESS_TABLE_K_SCALE_TOKEN_STRIDE_BYTES_OFFSET 48u
#define LAYER_ADDRESS_TABLE_KEY_CLIP_RATIO_BF16_OFFSET 52u
#define LAYER_ADDRESS_TABLE_VALUE_CLIP_RATIO_BF16_OFFSET 54u
#define LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_CLIP_RATIO_BF16_OFFSET 56u
#define LAYER_ADDRESS_TABLE_FFN_GATE_CLIP_RATIO_BF16_OFFSET 58u
#define LAYER_ADDRESS_TABLE_CONTEXT_ROW_STRIDE_BYTES_OFFSET 60u
#define LAYER_ADDRESS_TABLE_QUERY_BASE_WEIGHT_BASE_OFFSET 64u
#define LAYER_ADDRESS_TABLE_QUERY_BASE_WEIGHT_LIMIT_OFFSET 72u
#define LAYER_ADDRESS_TABLE_QUERY_ENHANCEMENT_WEIGHT_BASE_OFFSET 80u
#define LAYER_ADDRESS_TABLE_QUERY_ENHANCEMENT_WEIGHT_LIMIT_OFFSET 88u
#define LAYER_ADDRESS_TABLE_QUERY_WEIGHT_SCALE_BASE_OFFSET 96u
#define LAYER_ADDRESS_TABLE_QUERY_WEIGHT_SCALE_LIMIT_OFFSET 104u
#define LAYER_ADDRESS_TABLE_KEY_BASE_WEIGHT_BASE_OFFSET 112u
#define LAYER_ADDRESS_TABLE_KEY_BASE_WEIGHT_LIMIT_OFFSET 120u
#define LAYER_ADDRESS_TABLE_KEY_ENHANCEMENT_WEIGHT_BASE_OFFSET 128u
#define LAYER_ADDRESS_TABLE_KEY_ENHANCEMENT_WEIGHT_LIMIT_OFFSET 136u
#define LAYER_ADDRESS_TABLE_KEY_WEIGHT_SCALE_BASE_OFFSET 144u
#define LAYER_ADDRESS_TABLE_KEY_WEIGHT_SCALE_LIMIT_OFFSET 152u
#define LAYER_ADDRESS_TABLE_VALUE_BASE_WEIGHT_BASE_OFFSET 160u
#define LAYER_ADDRESS_TABLE_VALUE_BASE_WEIGHT_LIMIT_OFFSET 168u
#define LAYER_ADDRESS_TABLE_VALUE_ENHANCEMENT_WEIGHT_BASE_OFFSET 176u
#define LAYER_ADDRESS_TABLE_VALUE_ENHANCEMENT_WEIGHT_LIMIT_OFFSET 184u
#define LAYER_ADDRESS_TABLE_VALUE_WEIGHT_SCALE_BASE_OFFSET 192u
#define LAYER_ADDRESS_TABLE_VALUE_WEIGHT_SCALE_LIMIT_OFFSET 200u
#define LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_BASE_WEIGHT_BASE_OFFSET 208u
#define LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_BASE_WEIGHT_LIMIT_OFFSET 216u
#define LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_ENHANCEMENT_WEIGHT_BASE_OFFSET 224u
#define LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_ENHANCEMENT_WEIGHT_LIMIT_OFFSET 232u
#define LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_WEIGHT_SCALE_BASE_OFFSET 240u
#define LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_WEIGHT_SCALE_LIMIT_OFFSET 248u
#define LAYER_ADDRESS_TABLE_FFN_GATE_BASE_WEIGHT_BASE_OFFSET 256u
#define LAYER_ADDRESS_TABLE_FFN_GATE_BASE_WEIGHT_LIMIT_OFFSET 264u
#define LAYER_ADDRESS_TABLE_FFN_GATE_ENHANCEMENT_WEIGHT_BASE_OFFSET 272u
#define LAYER_ADDRESS_TABLE_FFN_GATE_ENHANCEMENT_WEIGHT_LIMIT_OFFSET 280u
#define LAYER_ADDRESS_TABLE_FFN_GATE_WEIGHT_SCALE_BASE_OFFSET 288u
#define LAYER_ADDRESS_TABLE_FFN_GATE_WEIGHT_SCALE_LIMIT_OFFSET 296u
#define LAYER_ADDRESS_TABLE_FFN_UP_BASE_WEIGHT_BASE_OFFSET 304u
#define LAYER_ADDRESS_TABLE_FFN_UP_BASE_WEIGHT_LIMIT_OFFSET 312u
#define LAYER_ADDRESS_TABLE_FFN_UP_ENHANCEMENT_WEIGHT_BASE_OFFSET 320u
#define LAYER_ADDRESS_TABLE_FFN_UP_ENHANCEMENT_WEIGHT_LIMIT_OFFSET 328u
#define LAYER_ADDRESS_TABLE_FFN_UP_WEIGHT_SCALE_BASE_OFFSET 336u
#define LAYER_ADDRESS_TABLE_FFN_UP_WEIGHT_SCALE_LIMIT_OFFSET 344u
#define LAYER_ADDRESS_TABLE_FFN_DOWN_BASE_WEIGHT_BASE_OFFSET 352u
#define LAYER_ADDRESS_TABLE_FFN_DOWN_BASE_WEIGHT_LIMIT_OFFSET 360u
#define LAYER_ADDRESS_TABLE_FFN_DOWN_ENHANCEMENT_WEIGHT_BASE_OFFSET 368u
#define LAYER_ADDRESS_TABLE_FFN_DOWN_ENHANCEMENT_WEIGHT_LIMIT_OFFSET 376u
#define LAYER_ADDRESS_TABLE_FFN_DOWN_WEIGHT_SCALE_BASE_OFFSET 384u
#define LAYER_ADDRESS_TABLE_FFN_DOWN_WEIGHT_SCALE_LIMIT_OFFSET 392u
#define LAYER_ADDRESS_TABLE_ATTENTION_RMS_GAMMA_BASE_OFFSET 400u
#define LAYER_ADDRESS_TABLE_ATTENTION_RMS_GAMMA_LIMIT_OFFSET 408u
#define LAYER_ADDRESS_TABLE_FFN_RMS_GAMMA_BASE_OFFSET 416u
#define LAYER_ADDRESS_TABLE_FFN_RMS_GAMMA_LIMIT_OFFSET 424u
#define LAYER_ADDRESS_TABLE_ATTENTION_RMS_EPSILON_BF16_OFFSET 432u
#define LAYER_ADDRESS_TABLE_FFN_RMS_EPSILON_BF16_OFFSET 434u
#define LAYER_ADDRESS_TABLE_FFN_UP_CLIP_RATIO_BF16_OFFSET 436u
#define LAYER_ADDRESS_TABLE_FFN_DOWN_CLIP_RATIO_BF16_OFFSET 438u
#define LAYER_ADDRESS_TABLE_ROPE_COS_LUT_BASE_OFFSET 440u
#define LAYER_ADDRESS_TABLE_ROPE_COS_LUT_LIMIT_OFFSET 448u
#define LAYER_ADDRESS_TABLE_ROPE_SIN_LUT_BASE_OFFSET 456u
#define LAYER_ADDRESS_TABLE_ROPE_SIN_LUT_LIMIT_OFFSET 464u
#define LAYER_ADDRESS_TABLE_CURRENT_K_CACHE_BASE_OFFSET 472u
#define LAYER_ADDRESS_TABLE_CURRENT_K_CACHE_LIMIT_OFFSET 480u
#define LAYER_ADDRESS_TABLE_CURRENT_V_CACHE_BASE_OFFSET 488u
#define LAYER_ADDRESS_TABLE_CURRENT_V_CACHE_LIMIT_OFFSET 496u
#define LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_BASE_OFFSET 504u
#define LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_LIMIT_OFFSET 512u
#define LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_BASE_OFFSET 520u
#define LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_LIMIT_OFFSET 528u
#define LAYER_ADDRESS_TABLE_K_SCALE_BASE_OFFSET 536u
#define LAYER_ADDRESS_TABLE_K_SCALE_LIMIT_OFFSET 544u
#define LAYER_ADDRESS_TABLE_V_SCALE_BASE_OFFSET 552u
#define LAYER_ADDRESS_TABLE_V_SCALE_LIMIT_OFFSET 560u
#define LAYER_ADDRESS_TABLE_CONTEXT_BASE_OFFSET 568u
#define LAYER_ADDRESS_TABLE_CONTEXT_LIMIT_OFFSET 576u
#define LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET 584u
#define LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_LIMIT_OFFSET 592u
#define LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET 600u
#define LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_LIMIT_OFFSET 608u
#define LAYER_ADDRESS_TABLE_RETAINED_K_SCALE_BASE_OFFSET 616u

struct layer_address_table {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint32_t entry_bytes;
    uint32_t flags;
    uint32_t hidden_size;
    uint32_t ffn_size;
    uint16_t head_count;
    uint16_t head_dimension;
    uint16_t maximum_sequence_length;
    uint16_t query_clip_ratio_bf16;
    uint32_t kv_head_stride_bytes;
    uint32_t kv_token_stride_bytes;
    uint32_t kv_chunk_stride_bytes;
    uint32_t k_scale_head_stride_bytes;
    uint32_t k_scale_token_stride_bytes;
    uint16_t key_clip_ratio_bf16;
    uint16_t value_clip_ratio_bf16;
    uint16_t attention_output_clip_ratio_bf16;
    uint16_t ffn_gate_clip_ratio_bf16;
    uint32_t context_row_stride_bytes;
    uint64_t query_base_weight_base;
    uint64_t query_base_weight_limit;
    uint64_t query_enhancement_weight_base;
    uint64_t query_enhancement_weight_limit;
    uint64_t query_weight_scale_base;
    uint64_t query_weight_scale_limit;
    uint64_t key_base_weight_base;
    uint64_t key_base_weight_limit;
    uint64_t key_enhancement_weight_base;
    uint64_t key_enhancement_weight_limit;
    uint64_t key_weight_scale_base;
    uint64_t key_weight_scale_limit;
    uint64_t value_base_weight_base;
    uint64_t value_base_weight_limit;
    uint64_t value_enhancement_weight_base;
    uint64_t value_enhancement_weight_limit;
    uint64_t value_weight_scale_base;
    uint64_t value_weight_scale_limit;
    uint64_t attention_output_base_weight_base;
    uint64_t attention_output_base_weight_limit;
    uint64_t attention_output_enhancement_weight_base;
    uint64_t attention_output_enhancement_weight_limit;
    uint64_t attention_output_weight_scale_base;
    uint64_t attention_output_weight_scale_limit;
    uint64_t ffn_gate_base_weight_base;
    uint64_t ffn_gate_base_weight_limit;
    uint64_t ffn_gate_enhancement_weight_base;
    uint64_t ffn_gate_enhancement_weight_limit;
    uint64_t ffn_gate_weight_scale_base;
    uint64_t ffn_gate_weight_scale_limit;
    uint64_t ffn_up_base_weight_base;
    uint64_t ffn_up_base_weight_limit;
    uint64_t ffn_up_enhancement_weight_base;
    uint64_t ffn_up_enhancement_weight_limit;
    uint64_t ffn_up_weight_scale_base;
    uint64_t ffn_up_weight_scale_limit;
    uint64_t ffn_down_base_weight_base;
    uint64_t ffn_down_base_weight_limit;
    uint64_t ffn_down_enhancement_weight_base;
    uint64_t ffn_down_enhancement_weight_limit;
    uint64_t ffn_down_weight_scale_base;
    uint64_t ffn_down_weight_scale_limit;
    uint64_t attention_rms_gamma_base;
    uint64_t attention_rms_gamma_limit;
    uint64_t ffn_rms_gamma_base;
    uint64_t ffn_rms_gamma_limit;
    uint16_t attention_rms_epsilon_bf16;
    uint16_t ffn_rms_epsilon_bf16;
    uint16_t ffn_up_clip_ratio_bf16;
    uint16_t ffn_down_clip_ratio_bf16;
    uint64_t rope_cos_lut_base;
    uint64_t rope_cos_lut_limit;
    uint64_t rope_sin_lut_base;
    uint64_t rope_sin_lut_limit;
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
    uint64_t context_base;
    uint64_t context_limit;
    uint64_t ffn_gate_up_workspace_base;
    uint64_t ffn_gate_up_workspace_limit;
    uint64_t ffn_residual_workspace_base;
    uint64_t ffn_residual_workspace_limit;
    uint64_t retained_k_scale_base;
};

LAYER_ADDRESS_TABLE_STATIC_ASSERT(sizeof(struct layer_address_table) == LAYER_ADDRESS_TABLE_BYTES,
               "layer_address_table size mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, magic) ==
               LAYER_ADDRESS_TABLE_MAGIC_OFFSET,
               "magic offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, version) ==
               LAYER_ADDRESS_TABLE_VERSION_OFFSET,
               "version offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, header_bytes) ==
               LAYER_ADDRESS_TABLE_HEADER_BYTES_OFFSET,
               "header_bytes offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, entry_bytes) ==
               LAYER_ADDRESS_TABLE_ENTRY_BYTES_OFFSET,
               "entry_bytes offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, flags) ==
               LAYER_ADDRESS_TABLE_FLAGS_OFFSET,
               "flags offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, hidden_size) ==
               LAYER_ADDRESS_TABLE_HIDDEN_SIZE_OFFSET,
               "hidden_size offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_size) ==
               LAYER_ADDRESS_TABLE_FFN_SIZE_OFFSET,
               "ffn_size offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, head_count) ==
               LAYER_ADDRESS_TABLE_HEAD_COUNT_OFFSET,
               "head_count offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, head_dimension) ==
               LAYER_ADDRESS_TABLE_HEAD_DIMENSION_OFFSET,
               "head_dimension offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, maximum_sequence_length) ==
               LAYER_ADDRESS_TABLE_MAXIMUM_SEQUENCE_LENGTH_OFFSET,
               "maximum_sequence_length offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, query_clip_ratio_bf16) ==
               LAYER_ADDRESS_TABLE_QUERY_CLIP_RATIO_BF16_OFFSET,
               "query_clip_ratio_bf16 offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, kv_head_stride_bytes) ==
               LAYER_ADDRESS_TABLE_KV_HEAD_STRIDE_BYTES_OFFSET,
               "kv_head_stride_bytes offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, kv_token_stride_bytes) ==
               LAYER_ADDRESS_TABLE_KV_TOKEN_STRIDE_BYTES_OFFSET,
               "kv_token_stride_bytes offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, kv_chunk_stride_bytes) ==
               LAYER_ADDRESS_TABLE_KV_CHUNK_STRIDE_BYTES_OFFSET,
               "kv_chunk_stride_bytes offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, k_scale_head_stride_bytes) ==
               LAYER_ADDRESS_TABLE_K_SCALE_HEAD_STRIDE_BYTES_OFFSET,
               "k_scale_head_stride_bytes offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, k_scale_token_stride_bytes) ==
               LAYER_ADDRESS_TABLE_K_SCALE_TOKEN_STRIDE_BYTES_OFFSET,
               "k_scale_token_stride_bytes offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, key_clip_ratio_bf16) ==
               LAYER_ADDRESS_TABLE_KEY_CLIP_RATIO_BF16_OFFSET,
               "key_clip_ratio_bf16 offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, value_clip_ratio_bf16) ==
               LAYER_ADDRESS_TABLE_VALUE_CLIP_RATIO_BF16_OFFSET,
               "value_clip_ratio_bf16 offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_output_clip_ratio_bf16) ==
               LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_CLIP_RATIO_BF16_OFFSET,
               "attention_output_clip_ratio_bf16 offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_gate_clip_ratio_bf16) ==
               LAYER_ADDRESS_TABLE_FFN_GATE_CLIP_RATIO_BF16_OFFSET,
               "ffn_gate_clip_ratio_bf16 offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, context_row_stride_bytes) ==
               LAYER_ADDRESS_TABLE_CONTEXT_ROW_STRIDE_BYTES_OFFSET,
               "context_row_stride_bytes offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, query_base_weight_base) ==
               LAYER_ADDRESS_TABLE_QUERY_BASE_WEIGHT_BASE_OFFSET,
               "query_base_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, query_base_weight_limit) ==
               LAYER_ADDRESS_TABLE_QUERY_BASE_WEIGHT_LIMIT_OFFSET,
               "query_base_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, query_enhancement_weight_base) ==
               LAYER_ADDRESS_TABLE_QUERY_ENHANCEMENT_WEIGHT_BASE_OFFSET,
               "query_enhancement_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, query_enhancement_weight_limit) ==
               LAYER_ADDRESS_TABLE_QUERY_ENHANCEMENT_WEIGHT_LIMIT_OFFSET,
               "query_enhancement_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, query_weight_scale_base) ==
               LAYER_ADDRESS_TABLE_QUERY_WEIGHT_SCALE_BASE_OFFSET,
               "query_weight_scale_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, query_weight_scale_limit) ==
               LAYER_ADDRESS_TABLE_QUERY_WEIGHT_SCALE_LIMIT_OFFSET,
               "query_weight_scale_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, key_base_weight_base) ==
               LAYER_ADDRESS_TABLE_KEY_BASE_WEIGHT_BASE_OFFSET,
               "key_base_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, key_base_weight_limit) ==
               LAYER_ADDRESS_TABLE_KEY_BASE_WEIGHT_LIMIT_OFFSET,
               "key_base_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, key_enhancement_weight_base) ==
               LAYER_ADDRESS_TABLE_KEY_ENHANCEMENT_WEIGHT_BASE_OFFSET,
               "key_enhancement_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, key_enhancement_weight_limit) ==
               LAYER_ADDRESS_TABLE_KEY_ENHANCEMENT_WEIGHT_LIMIT_OFFSET,
               "key_enhancement_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, key_weight_scale_base) ==
               LAYER_ADDRESS_TABLE_KEY_WEIGHT_SCALE_BASE_OFFSET,
               "key_weight_scale_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, key_weight_scale_limit) ==
               LAYER_ADDRESS_TABLE_KEY_WEIGHT_SCALE_LIMIT_OFFSET,
               "key_weight_scale_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, value_base_weight_base) ==
               LAYER_ADDRESS_TABLE_VALUE_BASE_WEIGHT_BASE_OFFSET,
               "value_base_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, value_base_weight_limit) ==
               LAYER_ADDRESS_TABLE_VALUE_BASE_WEIGHT_LIMIT_OFFSET,
               "value_base_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, value_enhancement_weight_base) ==
               LAYER_ADDRESS_TABLE_VALUE_ENHANCEMENT_WEIGHT_BASE_OFFSET,
               "value_enhancement_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, value_enhancement_weight_limit) ==
               LAYER_ADDRESS_TABLE_VALUE_ENHANCEMENT_WEIGHT_LIMIT_OFFSET,
               "value_enhancement_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, value_weight_scale_base) ==
               LAYER_ADDRESS_TABLE_VALUE_WEIGHT_SCALE_BASE_OFFSET,
               "value_weight_scale_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, value_weight_scale_limit) ==
               LAYER_ADDRESS_TABLE_VALUE_WEIGHT_SCALE_LIMIT_OFFSET,
               "value_weight_scale_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_output_base_weight_base) ==
               LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_BASE_WEIGHT_BASE_OFFSET,
               "attention_output_base_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_output_base_weight_limit) ==
               LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_BASE_WEIGHT_LIMIT_OFFSET,
               "attention_output_base_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_output_enhancement_weight_base) ==
               LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_ENHANCEMENT_WEIGHT_BASE_OFFSET,
               "attention_output_enhancement_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_output_enhancement_weight_limit) ==
               LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_ENHANCEMENT_WEIGHT_LIMIT_OFFSET,
               "attention_output_enhancement_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_output_weight_scale_base) ==
               LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_WEIGHT_SCALE_BASE_OFFSET,
               "attention_output_weight_scale_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_output_weight_scale_limit) ==
               LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_WEIGHT_SCALE_LIMIT_OFFSET,
               "attention_output_weight_scale_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_gate_base_weight_base) ==
               LAYER_ADDRESS_TABLE_FFN_GATE_BASE_WEIGHT_BASE_OFFSET,
               "ffn_gate_base_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_gate_base_weight_limit) ==
               LAYER_ADDRESS_TABLE_FFN_GATE_BASE_WEIGHT_LIMIT_OFFSET,
               "ffn_gate_base_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_gate_enhancement_weight_base) ==
               LAYER_ADDRESS_TABLE_FFN_GATE_ENHANCEMENT_WEIGHT_BASE_OFFSET,
               "ffn_gate_enhancement_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_gate_enhancement_weight_limit) ==
               LAYER_ADDRESS_TABLE_FFN_GATE_ENHANCEMENT_WEIGHT_LIMIT_OFFSET,
               "ffn_gate_enhancement_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_gate_weight_scale_base) ==
               LAYER_ADDRESS_TABLE_FFN_GATE_WEIGHT_SCALE_BASE_OFFSET,
               "ffn_gate_weight_scale_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_gate_weight_scale_limit) ==
               LAYER_ADDRESS_TABLE_FFN_GATE_WEIGHT_SCALE_LIMIT_OFFSET,
               "ffn_gate_weight_scale_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_up_base_weight_base) ==
               LAYER_ADDRESS_TABLE_FFN_UP_BASE_WEIGHT_BASE_OFFSET,
               "ffn_up_base_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_up_base_weight_limit) ==
               LAYER_ADDRESS_TABLE_FFN_UP_BASE_WEIGHT_LIMIT_OFFSET,
               "ffn_up_base_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_up_enhancement_weight_base) ==
               LAYER_ADDRESS_TABLE_FFN_UP_ENHANCEMENT_WEIGHT_BASE_OFFSET,
               "ffn_up_enhancement_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_up_enhancement_weight_limit) ==
               LAYER_ADDRESS_TABLE_FFN_UP_ENHANCEMENT_WEIGHT_LIMIT_OFFSET,
               "ffn_up_enhancement_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_up_weight_scale_base) ==
               LAYER_ADDRESS_TABLE_FFN_UP_WEIGHT_SCALE_BASE_OFFSET,
               "ffn_up_weight_scale_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_up_weight_scale_limit) ==
               LAYER_ADDRESS_TABLE_FFN_UP_WEIGHT_SCALE_LIMIT_OFFSET,
               "ffn_up_weight_scale_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_down_base_weight_base) ==
               LAYER_ADDRESS_TABLE_FFN_DOWN_BASE_WEIGHT_BASE_OFFSET,
               "ffn_down_base_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_down_base_weight_limit) ==
               LAYER_ADDRESS_TABLE_FFN_DOWN_BASE_WEIGHT_LIMIT_OFFSET,
               "ffn_down_base_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_down_enhancement_weight_base) ==
               LAYER_ADDRESS_TABLE_FFN_DOWN_ENHANCEMENT_WEIGHT_BASE_OFFSET,
               "ffn_down_enhancement_weight_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_down_enhancement_weight_limit) ==
               LAYER_ADDRESS_TABLE_FFN_DOWN_ENHANCEMENT_WEIGHT_LIMIT_OFFSET,
               "ffn_down_enhancement_weight_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_down_weight_scale_base) ==
               LAYER_ADDRESS_TABLE_FFN_DOWN_WEIGHT_SCALE_BASE_OFFSET,
               "ffn_down_weight_scale_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_down_weight_scale_limit) ==
               LAYER_ADDRESS_TABLE_FFN_DOWN_WEIGHT_SCALE_LIMIT_OFFSET,
               "ffn_down_weight_scale_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_rms_gamma_base) ==
               LAYER_ADDRESS_TABLE_ATTENTION_RMS_GAMMA_BASE_OFFSET,
               "attention_rms_gamma_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_rms_gamma_limit) ==
               LAYER_ADDRESS_TABLE_ATTENTION_RMS_GAMMA_LIMIT_OFFSET,
               "attention_rms_gamma_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_rms_gamma_base) ==
               LAYER_ADDRESS_TABLE_FFN_RMS_GAMMA_BASE_OFFSET,
               "ffn_rms_gamma_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_rms_gamma_limit) ==
               LAYER_ADDRESS_TABLE_FFN_RMS_GAMMA_LIMIT_OFFSET,
               "ffn_rms_gamma_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, attention_rms_epsilon_bf16) ==
               LAYER_ADDRESS_TABLE_ATTENTION_RMS_EPSILON_BF16_OFFSET,
               "attention_rms_epsilon_bf16 offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_rms_epsilon_bf16) ==
               LAYER_ADDRESS_TABLE_FFN_RMS_EPSILON_BF16_OFFSET,
               "ffn_rms_epsilon_bf16 offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_up_clip_ratio_bf16) ==
               LAYER_ADDRESS_TABLE_FFN_UP_CLIP_RATIO_BF16_OFFSET,
               "ffn_up_clip_ratio_bf16 offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_down_clip_ratio_bf16) ==
               LAYER_ADDRESS_TABLE_FFN_DOWN_CLIP_RATIO_BF16_OFFSET,
               "ffn_down_clip_ratio_bf16 offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, rope_cos_lut_base) ==
               LAYER_ADDRESS_TABLE_ROPE_COS_LUT_BASE_OFFSET,
               "rope_cos_lut_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, rope_cos_lut_limit) ==
               LAYER_ADDRESS_TABLE_ROPE_COS_LUT_LIMIT_OFFSET,
               "rope_cos_lut_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, rope_sin_lut_base) ==
               LAYER_ADDRESS_TABLE_ROPE_SIN_LUT_BASE_OFFSET,
               "rope_sin_lut_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, rope_sin_lut_limit) ==
               LAYER_ADDRESS_TABLE_ROPE_SIN_LUT_LIMIT_OFFSET,
               "rope_sin_lut_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, current_k_cache_base) ==
               LAYER_ADDRESS_TABLE_CURRENT_K_CACHE_BASE_OFFSET,
               "current_k_cache_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, current_k_cache_limit) ==
               LAYER_ADDRESS_TABLE_CURRENT_K_CACHE_LIMIT_OFFSET,
               "current_k_cache_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, current_v_cache_base) ==
               LAYER_ADDRESS_TABLE_CURRENT_V_CACHE_BASE_OFFSET,
               "current_v_cache_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, current_v_cache_limit) ==
               LAYER_ADDRESS_TABLE_CURRENT_V_CACHE_LIMIT_OFFSET,
               "current_v_cache_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, retained_k_cache_base) ==
               LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_BASE_OFFSET,
               "retained_k_cache_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, retained_k_cache_limit) ==
               LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_LIMIT_OFFSET,
               "retained_k_cache_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, retained_v_cache_base) ==
               LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_BASE_OFFSET,
               "retained_v_cache_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, retained_v_cache_limit) ==
               LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_LIMIT_OFFSET,
               "retained_v_cache_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, k_scale_base) ==
               LAYER_ADDRESS_TABLE_K_SCALE_BASE_OFFSET,
               "k_scale_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, k_scale_limit) ==
               LAYER_ADDRESS_TABLE_K_SCALE_LIMIT_OFFSET,
               "k_scale_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, v_scale_base) ==
               LAYER_ADDRESS_TABLE_V_SCALE_BASE_OFFSET,
               "v_scale_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, v_scale_limit) ==
               LAYER_ADDRESS_TABLE_V_SCALE_LIMIT_OFFSET,
               "v_scale_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, context_base) ==
               LAYER_ADDRESS_TABLE_CONTEXT_BASE_OFFSET,
               "context_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, context_limit) ==
               LAYER_ADDRESS_TABLE_CONTEXT_LIMIT_OFFSET,
               "context_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_gate_up_workspace_base) ==
               LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET,
               "ffn_gate_up_workspace_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_gate_up_workspace_limit) ==
               LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_LIMIT_OFFSET,
               "ffn_gate_up_workspace_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_residual_workspace_base) ==
               LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET,
               "ffn_residual_workspace_base offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, ffn_residual_workspace_limit) ==
               LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_LIMIT_OFFSET,
               "ffn_residual_workspace_limit offset mismatch");
LAYER_ADDRESS_TABLE_STATIC_ASSERT(offsetof(struct layer_address_table, retained_k_scale_base) ==
               LAYER_ADDRESS_TABLE_RETAINED_K_SCALE_BASE_OFFSET,
               "retained_k_scale_base offset mismatch");

#undef LAYER_ADDRESS_TABLE_STATIC_ASSERT

#endif
