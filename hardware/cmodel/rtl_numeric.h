#ifndef SUPRA_RTL_NUMERIC_H
#define SUPRA_RTL_NUMERIC_H

#include <stddef.h>
#include <stdint.h>

enum rtl_matmul_mode {
    RTL_MODE_W4A4 = 0,
    RTL_MODE_W4A8 = 1,
    RTL_MODE_W8A8 = 2
};

#define RTL_H12288_FEATURES 12288u
#define RTL_H12288_STAGE_COUNT 12u

enum rtl_h12288_stage {
    RTL_H12288_H1024_STAGE_1 = 0,
    RTL_H12288_H1024_STAGE_10 = 9,
    RTL_H12288_H12_STAGE = 10,
    RTL_H12288_NORMALIZED_STAGE = 11
};

typedef int (*rtl_h12288_observer)(
    void *context, size_t row, uint8_t stage,
    const uint16_t *values, size_t value_count);

uint16_t rtl_f32_to_bf16(float value);
float rtl_bf16_to_f32(uint16_t value);
/* Four FP32 recurrences and left-associated combine, matching the established
 * 32-head CUDA reduction. Callers validate the original reduction shape.
 */
static inline int rtl_probability_head_mean32(const uint16_t *values,
                                              size_t head_stride, float *mean) {
    float partial[4] = {0};
    if (!values || !head_stride || !mean) return -1;
    for (size_t head = 0; head < 32; ++head) {
        const uint16_t raw = values[head*head_stride];
        if ((raw & 0x7fffu) > 0x3f80u || ((raw & 0x8000u) && (raw & 0x7fffu))) return -1;
        partial[head % 4] += rtl_bf16_to_f32(raw);
    }
    float sum = partial[0] + partial[1];
    sum = sum + partial[2];
    sum = sum + partial[3];
    *mean = sum * 0.03125f;
    return 0;
}
uint16_t rtl_bf16_add(uint16_t lhs, uint16_t rhs);
uint16_t rtl_bf16_mul(uint16_t lhs, uint16_t rhs);
uint16_t rtl_bf16_mul_sign(uint16_t value, uint8_t negative);
int rtl_bf16_compare(uint16_t lhs, uint16_t rhs);
uint16_t rtl_bf16_rsqrt_newton(uint16_t positive_value);
uint16_t rtl_bf16_silu_pwl(uint16_t value);
uint16_t rtl_activation_scale_bf16(uint16_t maximum_bf16, int code_max);
int rtl_clip_row_bf16(const uint16_t *values, size_t count, uint16_t ratio_bf16,
                      uint16_t *output, uint16_t *limit_bf16);
int rtl_clip_row_bf16_trace(const uint16_t *values, size_t count, uint16_t ratio_bf16,
                           uint16_t *output, uint16_t *limit_bf16,
                           float *row_max, float *limit_fp32);

int rtl_quantize_row_bf16(const uint16_t *values, size_t count, int code_max,
                         int8_t *codes, uint16_t *scale_bf16);
int rtl_quantize_bf16_with_scale(const uint16_t *values, size_t count,
                                int code_max, uint16_t scale_bf16,
                                int8_t *codes);
int rtl_h12288_bf16(const uint16_t *values, size_t rows, uint16_t *output);
int rtl_h12288_bf16_trace(const uint16_t *values, size_t rows,
                          uint16_t *output, rtl_h12288_observer observer,
                          void *observer_context);

int rtl_validate_activation_codes(const int8_t *codes, size_t count, int code_max);
int rtl_validate_weight_codes(const int8_t *codes, size_t count);

int32_t rtl_partial_product(int8_t activation, int8_t base, int8_t enhancement,
                           enum rtl_matmul_mode mode);

int rtl_dot_scalar(const int8_t *activation, const int8_t *base,
                  const int8_t *enhancement, size_t count,
                  enum rtl_matmul_mode mode, int32_t *result);

int rtl_dot_grouped(const int8_t *activation, const int8_t *base,
                   const int8_t *enhancement, size_t count,
                   enum rtl_matmul_mode mode, size_t k_step,
                   int32_t *partial_sums, size_t partial_sum_capacity,
                   size_t *partial_sum_count, int32_t *result);

uint16_t rtl_bf16_div16_rne(uint16_t value);
uint16_t rtl_effective_weight_scale_bf16(uint16_t base_scale,
                                        enum rtl_matmul_mode mode);

uint16_t rtl_rescale_bf16(int32_t accumulator, uint16_t activation_scale,
                         uint16_t base_weight_scale,
                         enum rtl_matmul_mode mode);

uint16_t rtl_qk_rescale_bf16(int32_t accumulator, uint16_t query_scale,
                            uint16_t key_scale, uint16_t inverse_sqrt);
uint16_t rtl_pv_rescale_bf16(int32_t accumulator, uint16_t probability_scale,
                            uint16_t value_scale);

int rtl_rmsnorm_bf16(const uint16_t *values, const uint16_t *gamma,
                    size_t rows, size_t channels, uint16_t epsilon,
                    uint16_t *output);
int rtl_rope_bf16(const uint16_t *values, const uint16_t *sine,
                 const uint16_t *cosine, size_t rows, uint16_t *output);
int rtl_softmax_lut_bf16(const uint16_t *scores, size_t rows, size_t columns,
                        const uint16_t *exp_lut,
                        const uint16_t *reciprocal_lut, uint16_t *output);
int rtl_qk_bf16(const int8_t *query_codes, const uint16_t *query_scales,
               const int8_t *key_codes, const uint16_t *key_scales,
               size_t query_tokens, size_t key_tokens, size_t head_dim,
               uint16_t inverse_sqrt, uint16_t *scores);
int rtl_pv_bf16(const int8_t *probability_codes,
               const uint16_t *probability_scales,
               const int8_t *value_codes, uint16_t value_scale,
               size_t query_tokens, size_t key_tokens, size_t head_dim,
               uint16_t *context);
int rtl_elementwise_add_bf16(const uint16_t *lhs, const uint16_t *rhs,
                            size_t count, uint16_t *output);
int rtl_silu_multiply_bf16(const uint16_t *gate, const uint16_t *up,
                          size_t count, uint16_t *output);
int rtl_quantize_v_heads_bf16(const uint16_t *values, size_t rows,
                              size_t heads, size_t head_features,
                              const uint16_t *head_scales_bf16,
                              int8_t *output_codes);

#endif
