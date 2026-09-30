#include "rtl_numeric.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TRANSFORMER_MAX_K 12288u
#define ATTENTION_MAX_K 2048u

struct product_vector {
    int8_t activation;
    int8_t base;
    int8_t enhancement;
    enum rtl_matmul_mode mode;
    int32_t expected;
};

struct bf16_vector {
    uint16_t input;
    uint16_t expected;
};

static int failures;
static uint64_t exhaustive_product_cases;
static uint64_t exhaustive_bf16_div16_cases;
static uint64_t exhaustive_bf16_sign_cases;

static void expect_i32(const char *name, int32_t actual, int32_t expected) {
    if (actual != expected) {
        fprintf(stderr, "FAIL %s actual=%" PRId32 " expected=%" PRId32 "\n",
                name, actual, expected);
        failures += 1;
    }
}

static void expect_u16(const char *name, uint16_t actual, uint16_t expected) {
    if (actual != expected) {
        fprintf(stderr, "FAIL %s actual=0x%04" PRIx16 " expected=0x%04" PRIx16 "\n",
                name, actual, expected);
        failures += 1;
    }
}

static void test_hand_calculated_products(void) {
    static const struct product_vector vectors[] = {
        {-7, -7, 0, RTL_MODE_W4A4, 49},
        {7, -7, 0, RTL_MODE_W4A4, -49},
        {-127, 7, 0, RTL_MODE_W4A8, -889},
        {-16, -7, 0, RTL_MODE_W4A8, 112},
        {-1, -7, 0, RTL_MODE_W4A8, 7},
        {127, 7, 0, RTL_MODE_W4A8, 889},
        {-127, 7, -7, RTL_MODE_W8A8, -13335},
        {-1, -7, 7, RTL_MODE_W8A8, 105},
        {1, -7, 7, RTL_MODE_W8A8, -105},
        {127, 7, 7, RTL_MODE_W8A8, 15113},
    };
    size_t index;
    for (index = 0u; index < sizeof(vectors) / sizeof(vectors[0]); ++index) {
        expect_i32("hand_product",
                   rtl_partial_product(vectors[index].activation, vectors[index].base,
                                      vectors[index].enhancement, vectors[index].mode),
                   vectors[index].expected);
    }
}

static void test_exhaustive_valid_products(void) {
    int activation;
    int base;
    int enhancement;
    for (activation = -127; activation <= 127; ++activation) {
        for (base = -8; base <= 7; ++base) {
            int32_t expected_w4a8 = activation * base;
            expect_i32("exhaustive_w4a8",
                       rtl_partial_product((int8_t)activation, (int8_t)base, 0,
                                          RTL_MODE_W4A8),
                       expected_w4a8);
            exhaustive_product_cases += 1u;
            for (enhancement = -8; enhancement <= 7; ++enhancement) {
                int32_t expected_w8a8 =
                    activation * (16 * base + enhancement);
                expect_i32("exhaustive_w8a8",
                           rtl_partial_product((int8_t)activation, (int8_t)base,
                                              (int8_t)enhancement, RTL_MODE_W8A8),
                           expected_w8a8);
                exhaustive_product_cases += 1u;
            }
        }
    }
    for (activation = -7; activation <= 7; ++activation) {
        for (base = -8; base <= 7; ++base) {
            expect_i32("exhaustive_w4a4",
                       rtl_partial_product((int8_t)activation, (int8_t)base, 0,
                                          RTL_MODE_W4A4),
                       activation * base);
            exhaustive_product_cases += 1u;
        }
    }
}

struct h12288_observer_state {
    size_t callback_count;
};

static int observe_h12288_impulse(void *context, size_t row, uint8_t stage,
                                  const uint16_t *values,
                                  size_t value_count) {
    struct h12288_observer_state *state = context;
    expect_i32("h12288_observer_row", (int32_t)row, 0);
    expect_i32("h12288_observer_stage", (int32_t)stage,
               (int32_t)state->callback_count);
    expect_i32("h12288_observer_count", (int32_t)value_count,
               (int32_t)RTL_H12288_FEATURES);
    if (stage == RTL_H12288_H1024_STAGE_1) {
        expect_u16("h12288_stage1_value0", values[0], 0x3f80u);
        expect_u16("h12288_stage1_value1", values[1], 0x3f80u);
        expect_u16("h12288_stage1_value2", values[2], 0x0000u);
    } else if (stage == RTL_H12288_H1024_STAGE_10) {
        expect_u16("h12288_stage10_value0", values[0], 0x3f80u);
        expect_u16("h12288_stage10_value1023", values[1023], 0x3f80u);
        expect_u16("h12288_stage10_value1024", values[1024], 0x0000u);
    } else if (stage == RTL_H12288_H12_STAGE) {
        expect_u16("h12288_h12_value0", values[0], 0x3f80u);
        expect_u16("h12288_h12_last", values[RTL_H12288_FEATURES - 1u],
                   0x3f80u);
    } else if (stage == RTL_H12288_NORMALIZED_STAGE) {
        expect_u16("h12288_normalized_value0", values[0], 0x3c14u);
        expect_u16("h12288_normalized_last",
                   values[RTL_H12288_FEATURES - 1u], 0x3c14u);
    }
    ++state->callback_count;
    return 0;
}

static int reject_h12288_stage(void *context, size_t row, uint8_t stage,
                               const uint16_t *values, size_t value_count) {
    (void)context;
    (void)row;
    (void)stage;
    (void)values;
    (void)value_count;
    return -1;
}

static void test_h12288(void) {
    static uint16_t input[RTL_H12288_FEATURES];
    static uint16_t output[RTL_H12288_FEATURES];
    struct h12288_observer_state observer_state = {0u};
    size_t index;

    for (index = 0u; index <= UINT16_MAX; ++index) {
        const uint16_t value = (uint16_t)index;
        expect_u16("bf16_positive_sign", rtl_bf16_mul_sign(value, 0u),
                   rtl_bf16_mul(value, 0x3f80u));
        expect_u16("bf16_negative_sign", rtl_bf16_mul_sign(value, 1u),
                   rtl_bf16_mul(value, 0xbf80u));
        exhaustive_bf16_sign_cases += 2u;
    }

    memset(input, 0, sizeof(input));
    input[0] = 0x3f80u;
    expect_i32("h12288_trace_status", rtl_h12288_bf16_trace(
                   input, 1u, output, observe_h12288_impulse,
                   &observer_state), 0);
    expect_i32("h12288_observer_callbacks",
               (int32_t)observer_state.callback_count,
               (int32_t)RTL_H12288_STAGE_COUNT);
    for (index = 0u; index < RTL_H12288_FEATURES; ++index) {
        expect_u16("h12288_impulse", output[index], 0x3c14u);
    }

    memcpy(input, output, sizeof(input));
    expect_i32("h12288_in_place_status",
               rtl_h12288_bf16(input, 1u, input), 0);
    expect_i32("h12288_null_input",
               rtl_h12288_bf16(NULL, 1u, output), -1);
    expect_i32("h12288_zero_rows",
               rtl_h12288_bf16(input, 0u, output), -1);
    expect_i32("h12288_observer_reject", rtl_h12288_bf16_trace(
                   input, 1u, output, reject_h12288_stage, NULL), -3);
}

static void test_bf16_div16(void) {
    static const struct bf16_vector vectors[] = {
        {0x0000u, 0x0000u}, {0x8000u, 0x8000u},
        {0x0001u, 0x0000u}, {0x0008u, 0x0000u},
        {0x0009u, 0x0001u}, {0x0018u, 0x0002u},
        {0x007fu, 0x0008u}, {0x0080u, 0x0008u},
        {0x0088u, 0x0008u}, {0x0098u, 0x000au},
        {0x027fu, 0x0080u}, {0x0280u, 0x0080u},
        {0x3f80u, 0x3d80u}, {0xbf80u, 0xbd80u},
        {0x7f7fu, 0x7d7fu}, {0xff7fu, 0xfd7fu},
        {0x7f80u, 0x7f80u}, {0xff80u, 0xff80u},
        {0x7fc1u, 0x7fc1u},
    };
    size_t index;
    for (index = 0u; index < sizeof(vectors) / sizeof(vectors[0]); ++index) {
        expect_u16("bf16_div16", rtl_bf16_div16_rne(vectors[index].input),
                   vectors[index].expected);
    }

    for (index = 0u; index <= UINT16_MAX; ++index) {
        uint16_t input = (uint16_t)index;
        uint16_t exponent = (input >> 7u) & 0xffu;
        if (exponent != 0xffu) {
            uint16_t expected = rtl_f32_to_bf16(
                rtl_bf16_to_f32(input) / 16.0f);
            expect_u16("exhaustive_bf16_div16",
                       rtl_bf16_div16_rne(input), expected);
            exhaustive_bf16_div16_cases += 1u;
        }
    }
}

static void test_bf16_nan_operand_priority(void) {
    expect_u16("mul_left_nan_priority",
               rtl_bf16_mul(0x7f81u, 0x7fc3u), 0x7fc1u);
    expect_u16("mul_right_nan_payload",
               rtl_bf16_mul(0x3f80u, 0xff83u), 0xffc3u);
    expect_u16("add_left_nan_priority",
               rtl_bf16_add(0xff81u, 0x7fc3u), 0xffc1u);
    expect_u16("add_right_nan_payload",
               rtl_bf16_add(0x3f80u, 0x7f83u), 0x7fc3u);
}

static void test_shared_base_scale_rescale(void) {
    static const enum rtl_matmul_mode modes[] = {
        RTL_MODE_W4A4, RTL_MODE_W4A8, RTL_MODE_W8A8
    };
    static const int32_t accumulators[] = {112, 112, 1792};
    static const uint16_t expected_effective[] = {0x3f80u, 0x3f80u, 0x3d80u};
    static const uint16_t expected_outputs[] = {0x42e0u, 0x42e0u, 0x42e0u};
    const uint16_t activation_scale = 0x3f80u;
    const uint16_t shared_base_scale = 0x3f80u;
    size_t index;
    for (index = 0u; index < sizeof(modes) / sizeof(modes[0]); ++index) {
        expect_u16("effective_weight_scale",
                   rtl_effective_weight_scale_bf16(shared_base_scale, modes[index]),
                   expected_effective[index]);
        expect_u16("shared_base_rescale",
                   rtl_rescale_bf16(accumulators[index], activation_scale,
                                   shared_base_scale, modes[index]),
                   expected_outputs[index]);
    }
    expect_u16("invalid_mode_scale",
               rtl_effective_weight_scale_bf16(shared_base_scale,
                                               (enum rtl_matmul_mode)3),
               0u);
    expect_u16("pv_lhs_nan_payload_priority",
               rtl_pv_rescale_bf16(17, 0x7feeu, 0xffd9u), 0x7feeu);
    expect_u16("qk_first_nan_scale_priority",
               rtl_qk_rescale_bf16(17, 0x3f80u, 0x7fc3u, 0xffd9u), 0x7fc3u);
    expect_u16("pv_infinity_times_zero_nan",
               rtl_pv_rescale_bf16(17, 0x7f80u, 0x0000u), 0xffc0u);
}

static void test_supported_accumulator_ranges(void) {
    static int8_t activation[TRANSFORMER_MAX_K];
    static int8_t base[TRANSFORMER_MAX_K];
    static int8_t enhancement[TRANSFORMER_MAX_K];
    static int32_t partial_sums[TRANSFORMER_MAX_K / 8u];
    static const enum rtl_matmul_mode transformer_modes[] = {
        RTL_MODE_W4A4, RTL_MODE_W4A8
    };
    static const int8_t activation_values[] = {7, 127};
    static const size_t k_steps[] = {32u, 16u};
    static const int32_t expected[] = {602112, 10924032};
    size_t mode_index;
    for (mode_index = 0u;
         mode_index < sizeof(transformer_modes) / sizeof(transformer_modes[0]);
         ++mode_index) {
        size_t index;
        size_t partial_sum_count = 0u;
        int32_t scalar = 0;
        int32_t grouped = 0;
        for (index = 0u; index < TRANSFORMER_MAX_K; ++index) {
            activation[index] = activation_values[mode_index];
            base[index] = 7;
            enhancement[index] = 7;
        }
        expect_i32("production_scalar_status",
                   rtl_dot_scalar(activation, base, enhancement,
                                 TRANSFORMER_MAX_K,
                                 transformer_modes[mode_index], &scalar),
                   0);
        expect_i32("production_grouped_status",
                   rtl_dot_grouped(activation, base, enhancement,
                                  TRANSFORMER_MAX_K,
                                  transformer_modes[mode_index],
                                  k_steps[mode_index], partial_sums,
                                  TRANSFORMER_MAX_K / 8u, &partial_sum_count,
                                  &grouped),
                   0);
        expect_i32("production_scalar_bound", scalar, expected[mode_index]);
        expect_i32("production_grouped_bound", grouped, expected[mode_index]);
        expect_i32("production_partial_sum_count", (int32_t)partial_sum_count,
                   (int32_t)(TRANSFORMER_MAX_K / k_steps[mode_index]));
        expect_i32("production_final_partial_sum",
                   partial_sums[partial_sum_count - 1u], expected[mode_index]);
    }

    memset(activation, 127, ATTENTION_MAX_K);
    memset(base, -8, ATTENTION_MAX_K);
    memset(enhancement, 0, ATTENTION_MAX_K);
    {
        size_t partial_sum_count = 0u;
        int32_t scalar = 0;
        int32_t grouped = 0;
        const int32_t expected_attention = -33292288;
        expect_i32("attention_scalar_status",
                   rtl_dot_scalar(activation, base, enhancement,
                                  ATTENTION_MAX_K, RTL_MODE_W8A8,
                                  &scalar), 0);
        expect_i32("attention_grouped_status",
                   rtl_dot_grouped(activation, base, enhancement,
                                   ATTENTION_MAX_K, RTL_MODE_W8A8, 8u,
                                   partial_sums, TRANSFORMER_MAX_K / 8u,
                                   &partial_sum_count, &grouped), 0);
        expect_i32("attention_scalar_range", scalar, expected_attention);
        expect_i32("attention_grouped_range", grouped, expected_attention);
        expect_i32("attention_partial_sum_count",
                   (int32_t)partial_sum_count,
                   (int32_t)(ATTENTION_MAX_K / 8u));
    }
}

static void test_block_operator_numeric_schedule(void) {
    uint16_t ones[128];
    uint16_t zeros[128];
    uint16_t rope_output[128];
    uint16_t rms_output[4];
    uint16_t softmax_output;
    uint16_t exp_lut[256] = {0};
    uint16_t reciprocal_lut[256];
    int8_t query_codes[2] = {1, 2};
    int8_t key_codes[2] = {3, 4};
    uint16_t unit_scale = 0x3f80u;
    uint16_t qk_output;
    int8_t probability_codes[2] = {1, 2};
    int8_t value_codes[4] = {3, 4, 5, 6};
    uint16_t pv_output[2];
    uint16_t lhs[1] = {0x3f80u};
    uint16_t rhs[1] = {0x4000u};
    uint16_t vector_output[1];
    size_t index;

    expect_i32("bf16_compare", rtl_bf16_compare(0x4000u, 0x3f80u), 1);
    expect_u16("fast_rsqrt_one", rtl_bf16_rsqrt_newton(0x3f80u), 0x3f7fu);
    expect_u16("silu_low_tail", rtl_bf16_silu_pwl(0xc100u), 0x0000u);
    expect_u16("silu_zero", rtl_bf16_silu_pwl(0x0000u), 0x0000u);
    expect_u16("silu_high_tail", rtl_bf16_silu_pwl(0x4100u), 0x4100u);

    for (index = 0u; index < 128u; ++index) {
        ones[index] = 0x3f80u;
        zeros[index] = 0x0000u;
    }
    expect_i32("rope_status", rtl_rope_bf16(ones, zeros, ones, 1u,
                                             rope_output), 0);
    for (index = 0u; index < 128u; ++index) {
        expect_u16("rope_identity", rope_output[index], 0x3f80u);
    }

    expect_i32("rmsnorm_status", rtl_rmsnorm_bf16(
                   ones, ones, 1u, 4u, 0x0000u, rms_output), 0);
    for (index = 0u; index < 4u; ++index) {
        expect_u16("rmsnorm_unit_row", rms_output[index], 0x3f7fu);
    }

    for (index = 0u; index < 256u; ++index) {
        reciprocal_lut[index] = 0x3f80u;
    }
    exp_lut[255] = 0x3f80u;
    expect_i32("softmax_status", rtl_softmax_lut_bf16(
                   ones, 1u, 1u, exp_lut, reciprocal_lut,
                   &softmax_output), 0);
    expect_u16("softmax_singleton", softmax_output, 0x3f80u);

    expect_i32("qk_status", rtl_qk_bf16(
                   query_codes, &unit_scale, key_codes, &unit_scale,
                   1u, 1u, 2u, 0x3f80u, &qk_output), 0);
    expect_u16("qk_unit_scale", qk_output, 0x4130u);

    expect_i32("pv_status", rtl_pv_bf16(
                   probability_codes, &unit_scale, value_codes, unit_scale,
                   1u, 2u, 2u, pv_output), 0);
    expect_u16("pv_channel0", pv_output[0], 0x4150u);
    expect_u16("pv_channel1", pv_output[1], 0x4180u);

    expect_i32("elementwise_add_status", rtl_elementwise_add_bf16(
                   lhs, rhs, 1u, vector_output), 0);
    expect_u16("elementwise_add", vector_output[0], 0x4040u);
    lhs[0] = 0x4100u;
    expect_i32("silu_multiply_status", rtl_silu_multiply_bf16(
                   lhs, rhs, 1u, vector_output), 0);
    expect_u16("silu_multiply", vector_output[0], 0x4180u);
}

static void test_per_head_v_scale(void) {
    enum { ROWS = 2, HEADS = 32, FEATURES = 4 };
    uint16_t values[ROWS * HEADS * FEATURES];
    uint16_t scales[HEADS];
    int8_t output[ROWS * HEADS * FEATURES];
    size_t head;
    size_t row;
    size_t feature;

    for (head = 0u; head < HEADS; ++head) {
        scales[head] = rtl_f32_to_bf16((float)(head + 1u) / 32.0f);
        for (row = 0u; row < ROWS; ++row) {
            for (feature = 0u; feature < FEATURES; ++feature) {
                const uint16_t magnitude = scales[head];
                values[(row * HEADS + head) * FEATURES + feature] =
                    row == 0u ? magnitude : (uint16_t)(magnitude ^ 0x8000u);
            }
        }
    }
    expect_i32("per_head_v_status", rtl_quantize_v_heads_bf16(
                   values, ROWS, HEADS, FEATURES, scales, output), 0);
    for (head = 0u; head < HEADS; ++head) {
        for (feature = 0u; feature < FEATURES; ++feature) {
            expect_i32("per_head_v_positive",
                       output[(head * ROWS) * FEATURES + feature], 1);
            expect_i32("per_head_v_negative",
                       output[(head * ROWS + 1u) * FEATURES + feature], -1);
        }
    }
    scales[17] = 0u;
    expect_i32("per_head_v_zero_scale", rtl_quantize_v_heads_bf16(
                   values, ROWS, HEADS, FEATURES, scales, output), -1);
}

static void test_activation_scale_boundaries(void) {
    expect_u16("activation_scale_zero_a4",
               rtl_activation_scale_bf16(0x0000u, 7), 0x3f80u);
    expect_u16("activation_scale_zero_a8",
               rtl_activation_scale_bf16(0x0000u, 127), 0x3f80u);
    expect_u16("activation_scale_min_subnormal_a4",
               rtl_activation_scale_bf16(0x0001u, 7), 0x3f80u);
    expect_u16("activation_scale_min_subnormal_a8",
               rtl_activation_scale_bf16(0x0001u, 127), 0x3f80u);
    expect_u16("activation_scale_min_normal_a4",
               rtl_activation_scale_bf16(0x0080u, 7), 0x0012u);
    expect_u16("activation_scale_min_normal_a8",
               rtl_activation_scale_bf16(0x0080u, 127), 0x0001u);
}

int main(void) {
    test_hand_calculated_products();
    test_exhaustive_valid_products();
    test_bf16_div16();
    test_bf16_nan_operand_priority();
    test_shared_base_scale_rescale();
    test_h12288();
    test_supported_accumulator_ranges();
    test_block_operator_numeric_schedule();
    test_per_head_v_scale();
    test_activation_scale_boundaries();
    if (failures != 0) {
        fprintf(stderr, "RTL numeric FAIL failures=%d cases=%" PRIu64 "\n",
                failures, exhaustive_product_cases);
        return 1;
    }
    printf("RTL numeric PASS exhaustive_product_cases=%" PRIu64
           " hand_product_cases=10 bf16_div16_boundary_cases=19"
           " exhaustive_bf16_div16_cases=%" PRIu64
           " exhaustive_bf16_sign_cases=%" PRIu64
           " activation_scale_boundary_cases=6"
           " shared_scale_modes=3 transformer_k_modes=2 attention_k_modes=1\n",
           exhaustive_product_cases, exhaustive_bf16_div16_cases,
           exhaustive_bf16_sign_cases);
    return 0;
}
