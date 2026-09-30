#include "rtl_numeric.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint32_t float_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_float(uint32_t bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t rtl_f32_to_bf16(float value) {
    uint32_t bits = float_bits(value);
    uint32_t exponent = bits & 0x7f800000u;
    uint32_t mantissa = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0u) {
        return (uint16_t)((bits >> 16u) | 0x0040u);
    }
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

float rtl_bf16_to_f32(uint16_t value) {
    return bits_float(((uint32_t)value) << 16u);
}

static int bf16_is_nan(uint16_t value) {
    return (value & 0x7f80u) == 0x7f80u && (value & 0x007fu) != 0u;
}

uint16_t rtl_bf16_add(uint16_t lhs, uint16_t rhs) {
    if (bf16_is_nan(lhs)) return (uint16_t)(lhs | 0x0040u);
    if (bf16_is_nan(rhs)) return (uint16_t)(rhs | 0x0040u);
    return rtl_f32_to_bf16(rtl_bf16_to_f32(lhs) + rtl_bf16_to_f32(rhs));
}

uint16_t rtl_bf16_mul(uint16_t lhs, uint16_t rhs) {
    if (bf16_is_nan(lhs)) return (uint16_t)(lhs | 0x0040u);
    if (bf16_is_nan(rhs)) return (uint16_t)(rhs | 0x0040u);
    return rtl_f32_to_bf16(rtl_bf16_to_f32(lhs) * rtl_bf16_to_f32(rhs));
}

uint16_t rtl_bf16_mul_sign(uint16_t value, uint8_t negative) {
    const uint16_t exponent = (uint16_t)((value >> 7u) & 0xffu);
    const uint16_t fraction = (uint16_t)(value & 0x007fu);
    if (exponent == 0xffu && fraction != 0u) {
        return (uint16_t)(value | 0x0040u);
    }
    return negative != 0u ? (uint16_t)(value ^ 0x8000u) : value;
}

int rtl_bf16_compare(uint16_t lhs, uint16_t rhs) {
    float left = rtl_bf16_to_f32(lhs);
    float right = rtl_bf16_to_f32(rhs);
    return (left > right) - (left < right);
}

static uint16_t bf16_negate(uint16_t value) {
    return (uint16_t)(value ^ 0x8000u);
}

uint16_t rtl_bf16_rsqrt_newton(uint16_t positive_value) {
    float x = rtl_bf16_to_f32(positive_value);
    uint32_t bits;
    uint16_t estimate;
    uint16_t half_x;
    uint16_t estimate_squared;
    uint16_t product;
    uint16_t correction;
    if (!(x > 0.0f) || !isfinite(x)) {
        return rtl_f32_to_bf16(NAN);
    }
    bits = float_bits(x);
    estimate = rtl_f32_to_bf16(bits_float(0x5f3759dfu - (bits >> 1u)));
    half_x = rtl_bf16_mul(positive_value, 0x3f00u);
    estimate_squared = rtl_bf16_mul(estimate, estimate);
    product = rtl_bf16_mul(half_x, estimate_squared);
    correction = rtl_bf16_add(0x3fc0u, bf16_negate(product));
    return rtl_bf16_mul(estimate, correction);
}

uint16_t rtl_bf16_silu_pwl(uint16_t value) {
    static const uint16_t breakpoints[17] = {
        0xc100u, 0xc0c0u, 0xc080u, 0xc040u, 0xc000u, 0xbfc0u,
        0xbf80u, 0xbf00u, 0x0000u, 0x3f00u, 0x3f80u, 0x3fc0u,
        0x4000u, 0x4040u, 0x4080u, 0x40c0u, 0x4100u
    };
#ifdef SUPRA_SILU_SEGMENT_0
    static const uint32_t coefficients[16] = {
        SUPRA_SILU_SEGMENT_0, SUPRA_SILU_SEGMENT_1, SUPRA_SILU_SEGMENT_2, SUPRA_SILU_SEGMENT_3,
        SUPRA_SILU_SEGMENT_4, SUPRA_SILU_SEGMENT_5, SUPRA_SILU_SEGMENT_6, SUPRA_SILU_SEGMENT_7,
        SUPRA_SILU_SEGMENT_8, SUPRA_SILU_SEGMENT_9, SUPRA_SILU_SEGMENT_10, SUPRA_SILU_SEGMENT_11,
        SUPRA_SILU_SEGMENT_12, SUPRA_SILU_SEGMENT_13, SUPRA_SILU_SEGMENT_14, SUPRA_SILU_SEGMENT_15
    };
#elif defined(SUPRA_SILU_MONOTONE)
    static const uint32_t coefficients[16] = {
        0xbd5bbbdau, 0xbe40bcf7u, 0xbeb5bd92u, 0xbedfbdc9u,
        0xbec5bd94u, 0xbe883c09u, 0xbdf43e22u, 0x00003eccu,
        0x00003f17u, 0xbde63f56u, 0xbe843f7du, 0xbec33f89u,
        0xbeda3f8cu, 0xbeb33f89u, 0xbe293f83u, 0x00003f80u
    };
#else
    static const uint16_t slopes[16] = {
        0xbbd6u, 0xbcefu, 0xbd92u, 0xbdc9u, 0xbd94u, 0x3c09u,
        0x3e22u, 0x3eccu, 0x3f17u, 0x3f56u, 0x3f7du, 0x3f89u,
        0x3f8cu, 0x3f89u, 0x3f83u, 0x3f80u
    };
    static const uint16_t intercepts[16] = {
        0xbd58u, 0xbe3bu, 0xbeb5u, 0xbedfu, 0xbec5u, 0xbe88u,
        0xbdf4u, 0x0000u, 0x0000u, 0xbde6u, 0xbe84u, 0xbec3u,
        0xbedau, 0xbeb3u, 0xbe29u, 0xa6fcu
    };
#endif
    size_t segment = 0u;
    float x = rtl_bf16_to_f32(value);
    if (isnan(x)) {
        return (uint16_t)(value | 0x0040u);
    }
    if (x <= rtl_bf16_to_f32(breakpoints[0])) {
        return 0x0000u;
    }
    if (x >= rtl_bf16_to_f32(breakpoints[16])) {
        return value;
    }
    while (segment + 1u < 16u &&
           x >= rtl_bf16_to_f32(breakpoints[segment + 1u])) {
        ++segment;
    }
#if defined(SUPRA_SILU_SEGMENT_0) || defined(SUPRA_SILU_MONOTONE)
    return rtl_bf16_add(rtl_bf16_mul(value, (uint16_t)coefficients[segment]),
                       (uint16_t)(coefficients[segment] >> 16));
#else
    return rtl_bf16_add(rtl_bf16_mul(value, slopes[segment]),
                       intercepts[segment]);
#endif
}

int rtl_quantize_bf16_with_scale(const uint16_t *values, size_t count,
                                int code_max, uint16_t scale_bf16,
                                int8_t *codes) {
    float scale;
    size_t index;
    if (values == NULL || codes == NULL || count == 0u ||
        (code_max != 7 && code_max != 127) || (scale_bf16 & 0x8000u) != 0u ||
        (scale_bf16 & 0x7fffu) == 0u || (scale_bf16 & 0x7f80u) == 0x7f80u) {
        return -1;
    }
    scale = rtl_bf16_to_f32(scale_bf16);
    for (index = 0u; index < count; ++index) {
        float scaled = rtl_bf16_to_f32(values[index]) / scale;
        long rounded;
        if (scaled >= (float)code_max) {
            rounded = (long)code_max;
        } else if (scaled <= (float)-code_max) {
            rounded = (long)-code_max;
        } else {
            rounded = lrintf(scaled);
        }
        codes[index] = (int8_t)rounded;
    }
    return 0;
}

uint16_t rtl_activation_scale_bf16(uint16_t maximum_bf16, int code_max) {
    uint16_t scale;
    if ((code_max != 7 && code_max != 127) ||
        (maximum_bf16 & 0x8000u) != 0u ||
        (maximum_bf16 & 0x7f80u) == 0x7f80u) {
        return 0x7fc0u;
    }
    if ((maximum_bf16 & 0x7fffu) == 0u) {
        return 0x3f80u;
    }
    scale = rtl_f32_to_bf16(
        rtl_bf16_to_f32(maximum_bf16) / (float)code_max);
    return (scale & 0x7fffu) == 0u ? 0x3f80u : scale;
}

int rtl_clip_row_bf16_trace(const uint16_t *values, size_t count, uint16_t ratio_bf16,
                           uint16_t *output, uint16_t *limit_bf16,
                           float *row_max, float *limit_fp32) {
    float maximum = 0.0f;
    if (values == NULL || output == NULL || limit_bf16 == NULL || count == 0 ||
        ratio_bf16 == 0 || ratio_bf16 > 0x3f80u) return -1;
    for (size_t i = 0; i < count; ++i) {
        const float magnitude = fabsf(rtl_bf16_to_f32(values[i]));
        if (!isfinite(magnitude)) return -1;
        if (magnitude > maximum) maximum = magnitude;
    }
    const float product = maximum * rtl_bf16_to_f32(ratio_bf16);
    *limit_bf16 = rtl_f32_to_bf16(product);
    if (row_max != NULL) *row_max = maximum;
    if (limit_fp32 != NULL) *limit_fp32 = product;
    for (size_t i = 0; i < count; ++i)
        output[i] = (values[i] & 0x7fffu) > *limit_bf16 ?
            (values[i] & 0x8000u) | *limit_bf16 : values[i];
    return 0;
}

int rtl_clip_row_bf16(const uint16_t *values, size_t count, uint16_t ratio_bf16,
                      uint16_t *output, uint16_t *limit_bf16) {
    return rtl_clip_row_bf16_trace(values, count, ratio_bf16, output, limit_bf16, NULL, NULL);
}

int rtl_quantize_row_bf16(const uint16_t *values, size_t count, int code_max,
                         int8_t *codes, uint16_t *scale_bf16) {
    float maximum = 0.0f;
    size_t index;
    if (values == NULL || codes == NULL || scale_bf16 == NULL || count == 0u ||
        (code_max != 7 && code_max != 127)) {
        return -1;
    }
    for (index = 0u; index < count; ++index) {
        float magnitude = fabsf(rtl_bf16_to_f32(values[index]));
        if (magnitude > maximum) {
            maximum = magnitude;
        }
    }
    *scale_bf16 = rtl_activation_scale_bf16(
        rtl_f32_to_bf16(maximum), code_max);
    return rtl_quantize_bf16_with_scale(values, count, code_max,
                                       *scale_bf16, codes);
}

int rtl_quantize_v_heads_bf16(const uint16_t *values, size_t rows,
                              size_t heads, size_t head_features,
                              const uint16_t *head_scales_bf16,
                              int8_t *output_codes) {
    size_t head;
    if (values == NULL || head_scales_bf16 == NULL || output_codes == NULL ||
        rows == 0u || heads == 0u || head_features == 0u ||
        heads > SIZE_MAX / head_features ||
        rows > SIZE_MAX / (heads * head_features)) {
        return -1;
    }
    for (head = 0u; head < heads; ++head) {
        size_t row;
        for (row = 0u; row < rows; ++row) {
            if (rtl_quantize_bf16_with_scale(
                    values + (row * heads + head) * head_features,
                    head_features, 127, head_scales_bf16[head],
                    output_codes + (head * rows + row) * head_features) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static uint8_t paley_h12_negative(size_t output_block, size_t input_block) {
    static const uint8_t quadratic_residue[11] = {
        0u, 1u, 0u, 1u, 1u, 1u, 0u, 0u, 0u, 1u, 0u
    };
    size_t row;
    size_t column;
    size_t difference;

    if (output_block == 0u || input_block == 0u) {
        return 0u;
    }
    row = output_block - 1u;
    column = input_block - 1u;
    if (row == column) {
        return 1u;
    }
    difference = (row + 11u - column) % 11u;
    return quadratic_residue[difference] == 0u;
}

int rtl_h12288_bf16_trace(const uint16_t *values, size_t rows,
                          uint16_t *output, rtl_h12288_observer observer,
                          void *observer_context) {
    uint16_t *stage;
    uint16_t *h12_output;
    size_t row;

    if (values == NULL || output == NULL || rows == 0u ||
        rows > SIZE_MAX / RTL_H12288_FEATURES) {
        return -1;
    }
    stage = malloc(RTL_H12288_FEATURES * sizeof(*stage));
    h12_output = malloc(RTL_H12288_FEATURES * sizeof(*h12_output));
    if (stage == NULL || h12_output == NULL) {
        free(stage);
        free(h12_output);
        return -2;
    }

    for (row = 0u; row < rows; ++row) {
        size_t stride;
        uint8_t stage_index = 0u;
        memcpy(stage, values + row * RTL_H12288_FEATURES,
               RTL_H12288_FEATURES * sizeof(*stage));

        for (stride = 1u; stride < 1024u; stride <<= 1u) {
            size_t block;
            for (block = 0u; block < 12u; ++block) {
                size_t base;
                const size_t block_base = block * 1024u;
                for (base = 0u; base < 1024u; base += 2u * stride) {
                    size_t index;
                    for (index = 0u; index < stride; ++index) {
                        const size_t lhs_index = block_base + base + index;
                        const size_t rhs_index = lhs_index + stride;
                        const uint16_t lhs = stage[lhs_index];
                        const uint16_t rhs = stage[rhs_index];
                        stage[lhs_index] = rtl_bf16_add(lhs, rhs);
                        stage[rhs_index] = rtl_bf16_add(lhs, bf16_negate(rhs));
                    }
                }
            }
            if (observer != NULL && observer(
                    observer_context, row, stage_index, stage,
                    RTL_H12288_FEATURES) != 0) {
                free(stage);
                free(h12_output);
                return -3;
            }
            ++stage_index;
        }

        {
            size_t output_block;
            for (output_block = 0u; output_block < 12u; ++output_block) {
                size_t position;
                for (position = 0u; position < 1024u; ++position) {
                    uint16_t terms[16] = {0u};
                    size_t input_block;
                    size_t term_count;
                    for (input_block = 0u; input_block < 12u; ++input_block) {
                        terms[input_block] = rtl_bf16_mul_sign(
                            stage[input_block * 1024u + position],
                            paley_h12_negative(output_block, input_block));
                    }
                    for (term_count = 16u; term_count > 1u; term_count >>= 1u) {
                        size_t term;
                        for (term = 0u; term < term_count / 2u; ++term) {
                            terms[term] = rtl_bf16_add(
                                terms[2u * term], terms[2u * term + 1u]);
                        }
                    }
                    h12_output[output_block * 1024u + position] = terms[0];
                }
            }
        }
        if (observer != NULL && observer(
                observer_context, row, RTL_H12288_H12_STAGE, h12_output,
                RTL_H12288_FEATURES) != 0) {
            free(stage);
            free(h12_output);
            return -3;
        }

        {
            size_t index;
            uint16_t *row_output = output + row * RTL_H12288_FEATURES;
            for (index = 0u; index < RTL_H12288_FEATURES; ++index) {
                row_output[index] = rtl_bf16_mul(h12_output[index], 0x3c14u);
            }
            if (observer != NULL && observer(
                    observer_context, row, RTL_H12288_NORMALIZED_STAGE,
                    row_output, RTL_H12288_FEATURES) != 0) {
                free(stage);
                free(h12_output);
                return -3;
            }
        }
    }

    free(stage);
    free(h12_output);
    return 0;
}

int rtl_h12288_bf16(const uint16_t *values, size_t rows, uint16_t *output) {
    return rtl_h12288_bf16_trace(values, rows, output, NULL, NULL);
}

static int mode_parameters(enum rtl_matmul_mode mode, int *activation_max,
                           size_t *required_k_step) {
    switch (mode) {
    case RTL_MODE_W4A4:
        *activation_max = 7;
        *required_k_step = 32u;
        return 0;
    case RTL_MODE_W4A8:
        *activation_max = 127;
        *required_k_step = 16u;
        return 0;
    case RTL_MODE_W8A8:
        *activation_max = 127;
        *required_k_step = 8u;
        return 0;
    default:
        return -1;
    }
}

int rtl_validate_activation_codes(const int8_t *codes, size_t count, int code_max) {
    size_t index;
    if (codes == NULL || (code_max != 7 && code_max != 127)) {
        return -1;
    }
    for (index = 0u; index < count; ++index) {
        if (codes[index] < -code_max || codes[index] > code_max) {
            return -1;
        }
    }
    return 0;
}

int rtl_validate_weight_codes(const int8_t *codes, size_t count) {
    size_t index;
    if (codes == NULL) {
        return -1;
    }
    for (index = 0u; index < count; ++index) {
        if (codes[index] < -8 || codes[index] > 7) {
            return -1;
        }
    }
    return 0;
}

int32_t rtl_partial_product(int8_t activation, int8_t base, int8_t enhancement,
                           enum rtl_matmul_mode mode) {
    if (mode == RTL_MODE_W4A8 || mode == RTL_MODE_W8A8) {
        uint8_t activation_bits = (uint8_t)activation;
        int32_t high_signed = (int32_t)(activation_bits >> 4u);
        int32_t low_unsigned = (int32_t)(activation_bits & 0x0fu);
        if (high_signed >= 8) {
            high_signed -= 16;
        }
        if (mode == RTL_MODE_W8A8) {
            return 256 * high_signed * (int32_t)base +
                   16 * high_signed * (int32_t)enhancement +
                   16 * low_unsigned * (int32_t)base +
                   low_unsigned * (int32_t)enhancement;
        }
        return 16 * high_signed * (int32_t)base + low_unsigned * (int32_t)base;
    }
    return (int32_t)activation * (int32_t)base;
}

int rtl_dot_scalar(const int8_t *activation, const int8_t *base,
                  const int8_t *enhancement, size_t count,
                  enum rtl_matmul_mode mode, int32_t *result) {
    int activation_max;
    size_t required_k_step;
    size_t index;
    int64_t sum = 0;
    if (result == NULL || mode_parameters(mode, &activation_max, &required_k_step) != 0 ||
        rtl_validate_activation_codes(activation, count, activation_max) != 0 ||
        rtl_validate_weight_codes(base, count) != 0 ||
        (mode == RTL_MODE_W8A8 && rtl_validate_weight_codes(enhancement, count) != 0)) {
        return -1;
    }
    (void)required_k_step;
    for (index = 0u; index < count; ++index) {
        sum += rtl_partial_product(activation[index], base[index],
                                  mode == RTL_MODE_W8A8 ? enhancement[index] : 0, mode);
    }
    if (sum < INT32_MIN || sum > INT32_MAX) {
        return -2;
    }
    *result = (int32_t)sum;
    return 0;
}

int rtl_dot_grouped(const int8_t *activation, const int8_t *base,
                   const int8_t *enhancement, size_t count,
                   enum rtl_matmul_mode mode, size_t k_step,
                   int32_t *partial_sums, size_t partial_sum_capacity,
                   size_t *partial_sum_count, int32_t *result) {
    int activation_max;
    size_t required_k_step;
    size_t base_index;
    size_t partial_sum_index = 0u;
    int64_t sum = 0;
    if (result == NULL || partial_sum_count == NULL || partial_sums == NULL ||
        mode_parameters(mode, &activation_max, &required_k_step) != 0 ||
        k_step != required_k_step ||
        rtl_validate_activation_codes(activation, count, activation_max) != 0 ||
        rtl_validate_weight_codes(base, count) != 0 ||
        (mode == RTL_MODE_W8A8 && rtl_validate_weight_codes(enhancement, count) != 0)) {
        return -1;
    }
    for (base_index = 0u; base_index < count; base_index += k_step) {
        size_t lane;
        size_t limit = count - base_index < k_step ? count - base_index : k_step;
        if (partial_sum_index >= partial_sum_capacity) {
            return -3;
        }
        for (lane = 0u; lane < limit; ++lane) {
            size_t index = base_index + lane;
            sum += rtl_partial_product(activation[index], base[index],
                                      mode == RTL_MODE_W8A8 ? enhancement[index] : 0, mode);
        }
        if (sum < INT32_MIN || sum > INT32_MAX) {
            return -2;
        }
        partial_sums[partial_sum_index++] = (int32_t)sum;
    }
    *partial_sum_count = partial_sum_index;
    *result = (int32_t)sum;
    return 0;
}

static unsigned highest_set_bit_u32(uint32_t value) {
    unsigned bit = 0u;
    while (value >>= 1u) {
        ++bit;
    }
    return bit;
}

static uint32_t int32_to_fp32_rne_bits(int32_t value) {
    uint32_t raw = (uint32_t)value;
    uint32_t sign = raw >> 31u;
    uint32_t magnitude = sign != 0u ? (~raw + 1u) : raw;
    uint32_t quotient;
    unsigned leading_bit;
    unsigned exponent;
    if (magnitude == 0u) {
        return sign << 31u;
    }
    leading_bit = highest_set_bit_u32(magnitude);
    exponent = leading_bit + 127u;
    if (leading_bit <= 23u) {
        quotient = magnitude << (23u - leading_bit);
    } else {
        unsigned shift = leading_bit - 23u;
        uint32_t remainder;
        uint32_t halfway;
        quotient = magnitude >> shift;
        remainder = magnitude & ((UINT32_C(1) << shift) - 1u);
        halfway = UINT32_C(1) << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (quotient & 1u) != 0u)) {
            ++quotient;
        }
        if ((quotient & UINT32_C(0x01000000)) != 0u) {
            quotient >>= 1u;
            ++exponent;
        }
    }
    return (sign << 31u) | (exponent << 23u) |
           (quotient & UINT32_C(0x007fffff));
}

static unsigned normalization_shift24_u32(uint32_t significand) {
    return 23u - highest_set_bit_u32(significand);
}

static uint32_t fp32_multiply_rne_bits(uint32_t lhs, uint32_t rhs) {
    uint32_t exponent_lhs = (lhs >> 23u) & 0xffu;
    uint32_t exponent_rhs = (rhs >> 23u) & 0xffu;
    uint32_t fraction_lhs = lhs & UINT32_C(0x007fffff);
    uint32_t fraction_rhs = rhs & UINT32_C(0x007fffff);
    uint32_t result_sign = ((lhs ^ rhs) >> 31u) & 1u;
    uint32_t significand_lhs;
    uint32_t significand_rhs;
    int unbiased_lhs;
    int unbiased_rhs;
    int exponent_sum;
    int result_exponent;
    uint64_t product;
    uint64_t quotient;
    uint64_t remainder;
    uint64_t halfway;
    uint64_t rounded;
    unsigned shift;

    if (exponent_lhs == 0xffu && fraction_lhs != 0u) {
        return lhs | UINT32_C(0x00400000);
    }
    if (exponent_rhs == 0xffu && fraction_rhs != 0u) {
        return rhs | UINT32_C(0x00400000);
    }
    if ((exponent_lhs == 0xffu && (rhs & UINT32_C(0x7fffffff)) == 0u) ||
        (exponent_rhs == 0xffu && (lhs & UINT32_C(0x7fffffff)) == 0u)) {
        return UINT32_C(0xffc00000);
    }
    if (exponent_lhs == 0xffu || exponent_rhs == 0xffu) {
        return (result_sign << 31u) | UINT32_C(0x7f800000);
    }
    if ((lhs & UINT32_C(0x7fffffff)) == 0u ||
        (rhs & UINT32_C(0x7fffffff)) == 0u) {
        return result_sign << 31u;
    }

    significand_lhs = exponent_lhs == 0u ? fraction_lhs :
        UINT32_C(0x00800000) | fraction_lhs;
    significand_rhs = exponent_rhs == 0u ? fraction_rhs :
        UINT32_C(0x00800000) | fraction_rhs;
    unbiased_lhs = exponent_lhs == 0u ? -126 : (int)exponent_lhs - 127;
    unbiased_rhs = exponent_rhs == 0u ? -126 : (int)exponent_rhs - 127;
    if (exponent_lhs == 0u) {
        unsigned normalization_shift = normalization_shift24_u32(significand_lhs);
        significand_lhs <<= normalization_shift;
        unbiased_lhs -= (int)normalization_shift;
    }
    if (exponent_rhs == 0u) {
        unsigned normalization_shift = normalization_shift24_u32(significand_rhs);
        significand_rhs <<= normalization_shift;
        unbiased_rhs -= (int)normalization_shift;
    }

    exponent_sum = unbiased_lhs + unbiased_rhs;
    product = (uint64_t)significand_lhs * (uint64_t)significand_rhs;
    result_exponent = exponent_sum + ((product >> 47u) != 0u ? 1 : 0);
    if (result_exponent > 127) {
        return (result_sign << 31u) | UINT32_C(0x7f800000);
    }
    if (result_exponent >= -126) {
        int biased_exponent;
        shift = (product >> 47u) != 0u ? 24u : 23u;
        quotient = product >> shift;
        remainder = product & ((UINT64_C(1) << shift) - 1u);
        halfway = UINT64_C(1) << (shift - 1u);
        rounded = quotient;
        if (remainder > halfway ||
            (remainder == halfway && (quotient & 1u) != 0u)) {
            ++rounded;
        }
        if ((rounded & UINT64_C(0x01000000)) != 0u) {
            quotient = rounded >> 1u;
            ++result_exponent;
        } else {
            quotient = rounded;
        }
        biased_exponent = result_exponent + 127;
        if (biased_exponent >= 255) {
            return (result_sign << 31u) | UINT32_C(0x7f800000);
        }
        return (result_sign << 31u) | ((uint32_t)biased_exponent << 23u) |
               ((uint32_t)quotient & UINT32_C(0x007fffff));
    }

    shift = (unsigned)(-exponent_sum - 103);
    if (shift > 48u) {
        return result_sign << 31u;
    }
    quotient = product >> shift;
    remainder = product & ((UINT64_C(1) << shift) - 1u);
    halfway = UINT64_C(1) << (shift - 1u);
    rounded = quotient;
    if (remainder > halfway ||
        (remainder == halfway && (quotient & 1u) != 0u)) {
        ++rounded;
    }
    if ((rounded & UINT64_C(0x00800000)) != 0u) {
        return (result_sign << 31u) | UINT32_C(0x00800000);
    }
    return (result_sign << 31u) |
           ((uint32_t)rounded & UINT32_C(0x007fffff));
}

static uint16_t fp32_bits_to_bf16_rne(uint32_t value) {
    uint32_t exponent = value & UINT32_C(0x7f800000);
    uint32_t fraction = value & UINT32_C(0x007fffff);
    uint16_t upper = (uint16_t)(value >> 16u);
    uint16_t lower = (uint16_t)value;
    if (exponent == UINT32_C(0x7f800000) && fraction != 0u) {
        return (uint16_t)(upper | UINT16_C(0x0040));
    }
    if (lower > UINT16_C(0x8000) ||
        (lower == UINT16_C(0x8000) && (upper & 1u) != 0u)) {
        ++upper;
    }
    return upper;
}

static uint32_t multiply_fp32_by_bf16(uint32_t lhs, uint16_t rhs) {
    return fp32_multiply_rne_bits(lhs, (uint32_t)rhs << 16u);
}

uint16_t rtl_rescale_bf16(int32_t accumulator, uint16_t activation_scale,
                         uint16_t base_weight_scale,
                         enum rtl_matmul_mode mode) {
    uint32_t activation_product;
    uint16_t activation_scaled;
    uint16_t effective_weight_scale =
        rtl_effective_weight_scale_bf16(base_weight_scale, mode);
    activation_product = multiply_fp32_by_bf16(
        int32_to_fp32_rne_bits(accumulator), activation_scale);
    activation_scaled = fp32_bits_to_bf16_rne(activation_product);
    return fp32_bits_to_bf16_rne(multiply_fp32_by_bf16(
        (uint32_t)activation_scaled << 16u, effective_weight_scale));
}

uint16_t rtl_bf16_div16_rne(uint16_t value) {
    uint16_t sign = value & 0x8000u;
    uint16_t exponent = (value >> 7u) & 0xffu;
    uint16_t fraction = value & 0x007fu;
    uint16_t significand;
    unsigned shift;
    uint16_t quotient;
    uint16_t remainder;
    uint16_t halfway;

    if (exponent == 0xffu) {
        return value;
    }
    if (exponent >= 5u) {
        return (uint16_t)(sign | ((exponent - 4u) << 7u) | fraction);
    }

    significand = exponent == 0u ? fraction : (uint16_t)(0x0080u | fraction);
    shift = exponent == 0u ? 4u : 5u - exponent;
    quotient = (uint16_t)(significand >> shift);
    remainder = (uint16_t)(significand & ((1u << shift) - 1u));
    halfway = (uint16_t)(1u << (shift - 1u));
    if (remainder > halfway || (remainder == halfway && (quotient & 1u))) {
        quotient = (uint16_t)(quotient + 1u);
    }
    return (uint16_t)(sign | quotient);
}

uint16_t rtl_effective_weight_scale_bf16(uint16_t base_scale,
                                        enum rtl_matmul_mode mode) {
    switch (mode) {
    case RTL_MODE_W4A4:
    case RTL_MODE_W4A8:
        return base_scale;
    case RTL_MODE_W8A8:
        return rtl_bf16_div16_rne(base_scale);
    default:
        return 0u;
    }
}

uint16_t rtl_qk_rescale_bf16(int32_t accumulator, uint16_t query_scale,
                            uint16_t key_scale, uint16_t inverse_sqrt) {
    uint32_t stage0 = multiply_fp32_by_bf16(
        int32_to_fp32_rne_bits(accumulator), query_scale);
    uint32_t stage1 = multiply_fp32_by_bf16(stage0, key_scale);
    uint32_t stage2 = multiply_fp32_by_bf16(stage1, inverse_sqrt);
    return fp32_bits_to_bf16_rne(stage2);
}

uint16_t rtl_pv_rescale_bf16(int32_t accumulator, uint16_t probability_scale,
                            uint16_t value_scale) {
    uint32_t stage0 = multiply_fp32_by_bf16(
        int32_to_fp32_rne_bits(accumulator), probability_scale);
    uint32_t stage1 = multiply_fp32_by_bf16(stage0, value_scale);
    return fp32_bits_to_bf16_rne(stage1);
}

static size_t next_power_of_two(size_t value) {
    size_t power = 1u;
    while (power < value) {
        if (power > SIZE_MAX / 2u) {
            return 0u;
        }
        power <<= 1u;
    }
    return power;
}

int rtl_rmsnorm_bf16(const uint16_t *values, const uint16_t *gamma,
                    size_t rows, size_t channels, uint16_t epsilon,
                    uint16_t *output) {
    size_t padded;
    uint16_t *reduction;
    size_t row;
    if (values == NULL || output == NULL || rows == 0u || channels == 0u) {
        return -1;
    }
    padded = next_power_of_two(channels);
    if (padded == 0u || padded > SIZE_MAX / sizeof(*reduction)) {
        return -1;
    }
    reduction = malloc(padded * sizeof(*reduction));
    if (reduction == NULL) {
        return -2;
    }
    for (row = 0u; row < rows; ++row) {
        size_t index;
        size_t width;
        uint16_t mean;
        uint16_t inverse;
        for (index = 0u; index < channels; ++index) {
            uint16_t value = values[row * channels + index];
            reduction[index] = rtl_bf16_mul(value, value);
        }
        for (; index < padded; ++index) {
            reduction[index] = 0x0000u;
        }
        for (width = padded; width > 1u; width >>= 1u) {
            for (index = 0u; index < width / 2u; ++index) {
                reduction[index] = rtl_bf16_add(
                    reduction[2u * index], reduction[2u * index + 1u]);
            }
        }
        mean = rtl_f32_to_bf16(
            rtl_bf16_to_f32(reduction[0]) / (float)channels);
        inverse = rtl_bf16_rsqrt_newton(rtl_bf16_add(mean, epsilon));
        for (index = 0u; index < channels; ++index) {
            uint16_t normalized = rtl_bf16_mul(
                values[row * channels + index], inverse);
            output[row * channels + index] = gamma == NULL ? normalized :
                rtl_bf16_mul(normalized, gamma[index]);
        }
    }
    free(reduction);
    return 0;
}

int rtl_rope_bf16(const uint16_t *values, const uint16_t *sine,
                 const uint16_t *cosine, size_t rows, uint16_t *output) {
    size_t row;
    if (values == NULL || sine == NULL || cosine == NULL || output == NULL ||
        rows == 0u) {
        return -1;
    }
    for (row = 0u; row < rows; ++row) {
        size_t channel;
        for (channel = 0u; channel < 128u; ++channel) {
            size_t paired = channel < 64u ? channel + 64u : channel - 64u;
            uint16_t rotated = values[row * 128u + paired];
            if (channel < 64u) {
                rotated = bf16_negate(rotated);
            }
            output[row * 128u + channel] = rtl_bf16_add(
                rtl_bf16_mul(values[row * 128u + channel],
                            cosine[row * 128u + channel]),
                rtl_bf16_mul(rotated, sine[row * 128u + channel]));
        }
    }
    return 0;
}

static uint8_t softmax_exp_lut_address(uint16_t delta) {
    unsigned exponent;
    unsigned mantissa;
    int exponent_value;
    unsigned total_shift;
    uint32_t unscaled_numerator;
    uint32_t numerator;
    uint32_t quotient;
    uint32_t remainder;
    uint32_t denominator;
    if ((delta & 0x8000u) == 0u || (delta & 0x7fffu) == 0u) {
        return 255u;
    }
    exponent = (delta >> 7u) & 0xffu;
    if (exponent >= 0x83u) {
        return 0u;
    }
    mantissa = exponent == 0u ? delta & 0x7fu : 0x80u | (delta & 0x7fu);
    exponent_value = exponent == 0u ? -126 : (int)exponent - 127;
    if (exponent_value < -5) {
        return 255u;
    }
    total_shift = (unsigned)(7 - exponent_value + 4);
    unscaled_numerator = (1u << total_shift) - mantissa;
    numerator = unscaled_numerator * 255u;
    quotient = numerator >> total_shift;
    denominator = 1u << total_shift;
    remainder = numerator & (denominator - 1u);
    if ((remainder << 1u) > denominator ||
        ((remainder << 1u) == denominator && (quotient & 1u))) {
        ++quotient;
    }
    return quotient > 255u ? 255u : (uint8_t)quotient;
}

static uint16_t softmax_reciprocal(uint16_t sum_value,
                                   const uint16_t *reciprocal_lut) {
    unsigned exponent = (sum_value >> 7u) & 0xffu;
    unsigned numerator;
    unsigned quotient;
    unsigned remainder;
    uint16_t lut_value;
    int sum_exponent;
    int reciprocal_exponent;
    if ((sum_value & 0x8000u) != 0u || exponent == 0u || exponent == 0xffu) {
        return 0x7fc0u;
    }
    numerator = (sum_value & 0x7fu) * 255u;
    quotient = numerator >> 7u;
    remainder = numerator & 0x7fu;
    if (remainder > 64u || (remainder == 64u && (quotient & 1u))) {
        ++quotient;
    }
    if (quotient > 255u) {
        quotient = 255u;
    }
    lut_value = reciprocal_lut[quotient];
    sum_exponent = (int)exponent - 127;
    reciprocal_exponent = (int)((lut_value >> 7u) & 0xffu) - sum_exponent;
    if (((lut_value >> 7u) & 0xffu) == 0xffu) {
        return lut_value;
    }
    if (reciprocal_exponent >= 255) {
        return (uint16_t)((lut_value & 0x8000u) | 0x7f80u);
    }
    if (reciprocal_exponent <= 0) {
        return (uint16_t)(lut_value & 0x8000u);
    }
    return (uint16_t)((lut_value & 0x807fu) |
                      ((uint16_t)reciprocal_exponent << 7u));
}

int rtl_softmax_lut_bf16(const uint16_t *scores, size_t rows, size_t columns,
                        const uint16_t *exp_lut,
                        const uint16_t *reciprocal_lut, uint16_t *output) {
    size_t padded;
    uint16_t *exponents;
    uint16_t *reduction;
    size_t row;
    if (scores == NULL || exp_lut == NULL || reciprocal_lut == NULL ||
        output == NULL || rows == 0u || columns == 0u || columns > 2048u) {
        return -1;
    }
    padded = next_power_of_two(columns);
    if (padded == 0u) {
        return -1;
    }
    exponents = malloc(columns * sizeof(*exponents));
    reduction = malloc(padded * sizeof(*reduction));
    if (exponents == NULL || reduction == NULL) {
        free(exponents);
        free(reduction);
        return -2;
    }
    for (row = 0u; row < rows; ++row) {
        uint16_t maximum = 0xff80u;
        uint16_t reciprocal;
        size_t column;
        size_t width;
        for (column = 0u; column < columns; ++column) {
            uint16_t value = scores[row * columns + column];
            if (column == 0u || rtl_bf16_compare(value, maximum) > 0) {
                maximum = value;
            }
        }
        for (column = 0u; column < columns; ++column) {
            uint16_t delta = rtl_bf16_add(
                scores[row * columns + column], bf16_negate(maximum));
            exponents[column] = exp_lut[softmax_exp_lut_address(delta)];
            reduction[column] = exponents[column];
        }
        for (; column < padded; ++column) {
            reduction[column] = 0x0000u;
        }
        for (width = padded; width > 1u; width >>= 1u) {
            for (column = 0u; column < width / 2u; ++column) {
                reduction[column] = rtl_bf16_add(
                    reduction[2u * column], reduction[2u * column + 1u]);
            }
        }
        reciprocal = softmax_reciprocal(reduction[0], reciprocal_lut);
        for (column = 0u; column < columns; ++column) {
            output[row * columns + column] =
                rtl_bf16_mul(exponents[column], reciprocal);
        }
    }
    free(exponents);
    free(reduction);
    return 0;
}

int rtl_qk_bf16(const int8_t *query_codes, const uint16_t *query_scales,
               const int8_t *key_codes, const uint16_t *key_scales,
               size_t query_tokens, size_t key_tokens, size_t head_dim,
               uint16_t inverse_sqrt, uint16_t *scores) {
    size_t query;
    if (query_codes == NULL || query_scales == NULL || key_codes == NULL ||
        key_scales == NULL || scores == NULL || query_tokens == 0u ||
        key_tokens == 0u || head_dim == 0u || head_dim > 128u) {
        return -1;
    }
    for (query = 0u; query < query_tokens; ++query) {
        size_t key;
        for (key = 0u; key < key_tokens; ++key) {
            int32_t sum = 0;
            size_t channel;
            for (channel = 0u; channel < head_dim; ++channel) {
                sum += (int32_t)query_codes[query * head_dim + channel] *
                       (int32_t)key_codes[key * head_dim + channel];
            }
            scores[query * key_tokens + key] = rtl_qk_rescale_bf16(
                sum, query_scales[query], key_scales[key], inverse_sqrt);
        }
    }
    return 0;
}

int rtl_pv_bf16(const int8_t *probability_codes,
               const uint16_t *probability_scales,
               const int8_t *value_codes, uint16_t value_scale,
               size_t query_tokens, size_t key_tokens, size_t head_dim,
               uint16_t *context) {
    size_t query;
    if (probability_codes == NULL || probability_scales == NULL ||
        value_codes == NULL || context == NULL || query_tokens == 0u ||
        key_tokens == 0u || head_dim == 0u || head_dim > 128u) {
        return -1;
    }
    for (query = 0u; query < query_tokens; ++query) {
        size_t channel;
        for (channel = 0u; channel < head_dim; ++channel) {
            int32_t sum = 0;
            size_t key;
            for (key = 0u; key < key_tokens; ++key) {
                sum += (int32_t)probability_codes[query * key_tokens + key] *
                       (int32_t)value_codes[key * head_dim + channel];
            }
            context[query * head_dim + channel] = rtl_pv_rescale_bf16(
                sum, probability_scales[query], value_scale);
        }
    }
    return 0;
}

int rtl_elementwise_add_bf16(const uint16_t *lhs, const uint16_t *rhs,
                            size_t count, uint16_t *output) {
    size_t index;
    if (lhs == NULL || rhs == NULL || output == NULL) {
        return -1;
    }
    for (index = 0u; index < count; ++index) {
        output[index] = rtl_bf16_add(lhs[index], rhs[index]);
    }
    return 0;
}

int rtl_silu_multiply_bf16(const uint16_t *gate, const uint16_t *up,
                          size_t count, uint16_t *output) {
    size_t index;
    if (gate == NULL || up == NULL || output == NULL) {
        return -1;
    }
    for (index = 0u; index < count; ++index) {
        output[index] = rtl_bf16_mul(rtl_bf16_silu_pwl(gate[index]), up[index]);
    }
    return 0;
}
