#ifndef ATTENTION_PROBABILITY_CONFIG_H
#define ATTENTION_PROBABILITY_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define ATTENTION_PROBABILITY_CONFIG_BYTES 80u
#define ATTENTION_PROBABILITY_CONFIG_ALIGNMENT 16u
#define ATTENTION_PROBABILITY_CONFIG_OUTPUT_BASE_OFFSET 0u
#define ATTENTION_PROBABILITY_CONFIG_OUTPUT_LIMIT_OFFSET 8u
#define ATTENTION_PROBABILITY_CONFIG_BATCH_STRIDE_BYTES_OFFSET 16u
#define ATTENTION_PROBABILITY_CONFIG_HEAD_STRIDE_BYTES_OFFSET 20u
#define ATTENTION_PROBABILITY_CONFIG_ROUND_STRIDE_BYTES_OFFSET 24u
#define ATTENTION_PROBABILITY_CONFIG_LAYER_MASK_OFFSET 28u
#define ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS0_OFFSET 32u
#define ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS1_OFFSET 40u
#define ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS2_OFFSET 48u
#define ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS3_OFFSET 56u
#define ATTENTION_PROBABILITY_CONFIG_QUERY_BEGIN_OFFSET 64u
#define ATTENTION_PROBABILITY_CONFIG_QUERY_END_OFFSET 66u
#define ATTENTION_PROBABILITY_CONFIG_CAPTURE_P8_OFFSET 68u
#define ATTENTION_PROBABILITY_CONFIG_SHORTLIST_CONFIGURATION_BASE_OFFSET 72u

struct attention_probability_config {
    uint64_t output_base;
    uint64_t output_limit;
    uint32_t batch_stride_bytes;
    uint32_t head_stride_bytes;
    uint32_t round_stride_bytes;
    uint32_t layer_mask;
    uint64_t key_groups0;
    uint64_t key_groups1;
    uint64_t key_groups2;
    uint64_t key_groups3;
    uint16_t query_begin;
    uint16_t query_end;
    uint32_t capture_p8;
    uint64_t shortlist_configuration_base;
};

ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(sizeof(struct attention_probability_config) == ATTENTION_PROBABILITY_CONFIG_BYTES,
               "attention_probability_config size mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, output_base) ==
               ATTENTION_PROBABILITY_CONFIG_OUTPUT_BASE_OFFSET,
               "output_base offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, output_limit) ==
               ATTENTION_PROBABILITY_CONFIG_OUTPUT_LIMIT_OFFSET,
               "output_limit offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, batch_stride_bytes) ==
               ATTENTION_PROBABILITY_CONFIG_BATCH_STRIDE_BYTES_OFFSET,
               "batch_stride_bytes offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, head_stride_bytes) ==
               ATTENTION_PROBABILITY_CONFIG_HEAD_STRIDE_BYTES_OFFSET,
               "head_stride_bytes offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, round_stride_bytes) ==
               ATTENTION_PROBABILITY_CONFIG_ROUND_STRIDE_BYTES_OFFSET,
               "round_stride_bytes offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, layer_mask) ==
               ATTENTION_PROBABILITY_CONFIG_LAYER_MASK_OFFSET,
               "layer_mask offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, key_groups0) ==
               ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS0_OFFSET,
               "key_groups0 offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, key_groups1) ==
               ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS1_OFFSET,
               "key_groups1 offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, key_groups2) ==
               ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS2_OFFSET,
               "key_groups2 offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, key_groups3) ==
               ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS3_OFFSET,
               "key_groups3 offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, query_begin) ==
               ATTENTION_PROBABILITY_CONFIG_QUERY_BEGIN_OFFSET,
               "query_begin offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, query_end) ==
               ATTENTION_PROBABILITY_CONFIG_QUERY_END_OFFSET,
               "query_end offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, capture_p8) ==
               ATTENTION_PROBABILITY_CONFIG_CAPTURE_P8_OFFSET,
               "capture_p8 offset mismatch");
ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT(offsetof(struct attention_probability_config, shortlist_configuration_base) ==
               ATTENTION_PROBABILITY_CONFIG_SHORTLIST_CONFIGURATION_BASE_OFFSET,
               "shortlist_configuration_base offset mismatch");

#undef ATTENTION_PROBABILITY_CONFIG_STATIC_ASSERT

#endif
