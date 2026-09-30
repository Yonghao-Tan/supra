#include "Vmixed_precision_pe.h"
#include "verilated.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

double sc_time_stamp() { return 0.0; }

namespace {
constexpr unsigned ROWS = 8;
constexpr unsigned COLS = 8;
constexpr unsigned LANES = ROWS * COLS;

template <typename Wide>
void set_bits(Wide &words, unsigned bit_index, unsigned width, uint32_t value) {
    for (unsigned bit = 0; bit < width; ++bit) {
        const unsigned target = bit_index + bit;
        const uint32_t mask = uint32_t{1} << (target % 32);
        if ((value >> bit) & 1u) words[target / 32] |= mask;
        else words[target / 32] &= ~mask;
    }
}

template <typename Wide>
uint32_t get_bits(const Wide &words, unsigned bit_index, unsigned width) {
    uint32_t value = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const unsigned target = bit_index + bit;
        value |= ((words[target / 32] >> (target % 32)) & 1u) << bit;
    }
    return value;
}

class Regression {
public:
    Regression() {
        dut_.clk = 0;
        dut_.rst = 1;
        dut_.abort_request = 0;
        dut_.req_valid = 0;
        dut_.req_mixed_phase = 0;
        dut_.req_mixed_phase_first = 0;
        dut_.req_mixed_a8_rows = 0;
        dut_.accum_result_ready = 0;
        tick();
        tick();
        dut_.rst = 0;
        tick();
    }

    void run() {
        run_mode(0, 67, 5, 6);
        run_mode(1, 35, 8, 7);
        run_plain_signed_a8w8();
        run_mixed_groups();
        run_nibble_signedness_exhaustive();
        run_sparse_masks();
        run_pipeline_latency_and_ii();
        run_accum_result_fifo_backpressure();
        run_abort_recovery();
        dut_.final();
        std::cout << "PASS mixed_precision_pe: modes=4 K32/K16/K8=1"
                  << " runtime_W8A8=0 fixed_A8W8=1 tails=2"
                  << " mixed_groups=72 mixed_phase_ii1=1"
                  << " nibble_signedness_cases=1536"
                  << " sparse_masks=3"
                  << " signed27_accumulator_mismatch=0 accum_result_elastic_depth=1"
                  << " pipeline_latency=3 ii1_requests=4"
                  << " burst_backpressure=1 stalled_payload=1"
                  << " held_abort_single_ack=1 abort_rearm=1 abort_recovery=1\n";
    }

private:
    Vmixed_precision_pe dut_;
    uint16_t next_tag_ = 0x6100u;

    void tick() {
        dut_.clk = 0;
        dut_.eval();
        dut_.clk = 1;
        dut_.eval();
        dut_.clk = 0;
        dut_.eval();
    }

    void clear_request() {
        dut_.req_activation_payload = {};
        dut_.req_weight_payload = {};
        dut_.req_mixed_phase = 0;
        dut_.req_mixed_phase_first = 0;
        dut_.req_mixed_a8_rows = 0;
        dut_.req_activation_scales = {};
        dut_.req_weight_scales = {};
        for (unsigned row = 0; row < ROWS; ++row)
            set_bits(dut_.req_activation_scales, row * 16, 16,
                     0x3f80u + (row & 3u));
        for (unsigned column = 0; column < COLS; ++column)
            set_bits(dut_.req_weight_scales, column * 16, 16,
                     0x3e80u + column);
    }

    void accept_request() {
        dut_.req_valid = 1;
        for (unsigned wait = 0; wait < 256; ++wait) {
            dut_.eval();
            if (dut_.req_ready) {
                tick();
                dut_.req_valid = 0;
                return;
            }
            tick();
        }
        throw std::runtime_error("PE request acceptance timeout");
    }

    void expect_accum_result(const std::array<int32_t, LANES> &expected,
                           uint16_t tag, bool last, uint64_t mask) {
        dut_.accum_result_ready = 0;
        for (unsigned wait = 0; wait < 256 && !dut_.accum_result_valid; ++wait) tick();
        if (!dut_.accum_result_valid) throw std::runtime_error("PE accum_result timeout");
        if (dut_.accum_result_tag != tag || dut_.accum_result_last_k_step != last ||
            dut_.accum_result_mask != mask)
            throw std::runtime_error("PE accum_result metadata mismatch");
        std::array<uint32_t, LANES> held{};
        for (unsigned lane = 0; lane < LANES; ++lane) {
            held[lane] = get_bits(dut_.accum_result_accumulators, lane * 32, 32);
            if (((mask >> lane) & 1u) &&
                static_cast<int32_t>(held[lane]) != expected[lane]) {
                std::cerr << "accum_result mismatch tag=0x" << std::hex << tag
                          << std::dec << " lane=" << lane
                          << " expected=" << expected[lane]
                          << " actual=" << static_cast<int32_t>(held[lane]) << '\n';
                throw std::runtime_error("PE INT32 accum_result mismatch");
            }
        }
        for (unsigned row = 0; row < ROWS; ++row)
            if (get_bits(dut_.accum_result_activation_scales, row * 16, 16) !=
                    0x3f80u + (row & 3u))
                throw std::runtime_error("PE accum_result activation scale mismatch");
        for (unsigned column = 0; column < COLS; ++column)
            if (get_bits(dut_.accum_result_weight_scales, column * 16, 16) !=
                    0x3e80u + column)
                throw std::runtime_error("PE accum_result weight scale mismatch");
        for (unsigned cycle = 0; cycle < 3; ++cycle) {
            tick();
            if (!dut_.accum_result_valid)
                throw std::runtime_error("PE stalled accum_result dropped");
            for (unsigned lane = 0; lane < LANES; ++lane)
                if (get_bits(dut_.accum_result_accumulators, lane * 32, 32) != held[lane])
                    throw std::runtime_error("PE stalled accum_result changed");
        }
        dut_.accum_result_ready = 1;
        tick();
        dut_.accum_result_ready = 0;
    }

    void run_mode(unsigned mode, unsigned total_k, unsigned active_token_count,
                  unsigned active_cols) {
        const unsigned k_step = mode == 0 ? 32 : mode == 1 ? 16 : 8;
        const uint16_t tag = next_tag_++;
        const uint64_t mask = [&]() {
            uint64_t columns = 0;
            for (unsigned row = 0; row < active_token_count; ++row)
                columns |= ((uint64_t{1} << active_cols) - 1u) << (row * COLS);
            return columns;
        }();
        std::array<int32_t, LANES> expected{};
        unsigned base_k = 0;
        unsigned step_index = 0;
        while (base_k < total_k) {
            const unsigned count = std::min(k_step, total_k - base_k);
            clear_request();
            for (unsigned row = 0; row < ROWS; ++row)
                for (unsigned k = 0; k < count; ++k) {
                    const int8_t activation = mode == 0 ?
                        static_cast<int8_t>(((row * 5 + base_k + k) % 15) - 7) :
                        static_cast<int8_t>(((row * 17 + base_k + k) % 255) - 127);
                    const unsigned activation_bit = mode == 0 ?
                        (row * 32 + k) * 4 : (row * 16 + k) * 8;
                    set_bits(dut_.req_activation_payload, activation_bit,
                             mode == 0 ? 4 : 8,
                             static_cast<uint8_t>(activation));
                    for (unsigned column = 0; column < COLS; ++column) {
                        const int8_t base = static_cast<int8_t>(
                            ((base_k + k + column * 3) % 15) - 7);
                        set_bits(dut_.req_weight_payload,
                                 (k * COLS + column) * 4, 4,
                                 static_cast<uint8_t>(base) & 0xfu);
                        if (row < active_token_count && column < active_cols)
                            expected[row * COLS + column] += activation * base;
                    }
                }
            dut_.req_mode = mode;
            dut_.req_row_mask = static_cast<uint8_t>((1u << active_token_count) - 1u);
            dut_.req_k_mask = count == 32 ? 0xffffffffu : (1u << count) - 1u;
            dut_.req_col_mask = static_cast<uint8_t>((1u << active_cols) - 1u);
            dut_.req_first_k_step = step_index == 0;
            dut_.req_last_k_step = base_k + count == total_k;
            dut_.req_tag = tag;
            accept_request();
            expect_accum_result(expected, tag, dut_.req_last_k_step, mask);
            base_k += count;
            ++step_index;
        }
    }

    void run_plain_signed_a8w8() {
        clear_request();
        std::array<int32_t, LANES> expected{};
        for (unsigned row = 0; row < ROWS; ++row)
            for (unsigned k = 0; k < 8; ++k) {
                const int8_t activation = static_cast<int8_t>(row * 19 + k * 23 - 101);
                set_bits(dut_.req_activation_payload, (row * 8 + k) * 8, 8,
                         static_cast<uint8_t>(activation));
                for (unsigned column = 0; column < COLS; ++column) {
                    const int8_t weight = static_cast<int8_t>(k * 29 + column * 31 - 113);
                    const uint8_t raw = static_cast<uint8_t>(weight);
                    set_bits(dut_.req_weight_payload,
                             (k * COLS + column) * 8, 8, raw);
                    expected[row * COLS + column] += activation * weight;
                }
            }
        const uint16_t tag = next_tag_++;
        dut_.req_mode = 2;
        dut_.req_row_mask = 0xffu;
        dut_.req_k_mask = 0xffu;
        dut_.req_col_mask = 0xffu;
        dut_.req_first_k_step = 1;
        dut_.req_last_k_step = 1;
        dut_.req_tag = tag;
        accept_request();
        expect_accum_result(expected, tag, true, ~uint64_t{0});
    }

    static uint64_t row_lane_mask(uint8_t rows, uint8_t columns = 0xffu) {
        uint64_t mask = 0;
        for (unsigned row = 0; row < ROWS; ++row)
            if ((rows >> row) & 1u)
                mask |= uint64_t{columns} << (row * COLS);
        return mask;
    }

    void prepare_mixed_phase(unsigned a8_rows, unsigned phase0_a4_rows,
                             unsigned phase1_a4_rows, bool phase,
                             uint16_t tag) {
        clear_request();
        const unsigned phase_a4_rows = phase ? phase1_a4_rows : phase0_a4_rows;
        const uint8_t a8_mask = static_cast<uint8_t>((1u << a8_rows) - 1u);
        const uint8_t a4_mask = static_cast<uint8_t>(
            ((1u << phase_a4_rows) - 1u) << a8_rows);
        const uint8_t row_mask = a8_mask | a4_mask;

        for (unsigned k = 0; k < 32; ++k) {
            for (unsigned column = 0; column < COLS; ++column) {
                const int weight = (k + column) % 2 == 0 ? -8 : 7;
                set_bits(dut_.req_weight_payload,
                         (k * COLS + column) * 4, 4,
                         static_cast<uint8_t>(weight) & 0xfu);
            }
        }
        for (unsigned row = 0; row < ROWS; ++row) {
            if ((a8_mask >> row) & 1u) {
                for (unsigned k = 0; k < 16; ++k) {
                    const int activation =
                        ((row + k + (phase ? 1u : 0u)) & 1u) ? -127 : 127;
                    set_bits(dut_.req_activation_payload,
                             (row * 16 + k) * 8, 8,
                             static_cast<uint8_t>(activation));
                }
            } else if ((a4_mask >> row) & 1u) {
                for (unsigned k = 0; k < 32; ++k) {
                    const int activation = (row + k) & 1u ? -7 : 7;
                    set_bits(dut_.req_activation_payload,
                             (row * 32 + k) * 4, 4,
                             static_cast<uint8_t>(activation) & 0xfu);
                }
            }
        }
        dut_.req_mode = 3;
        dut_.req_row_mask = row_mask;
        dut_.req_k_mask = 0xffffffffu;
        dut_.req_col_mask = 0xffu;
        dut_.req_first_k_step = !phase;
        dut_.req_last_k_step = phase;
        dut_.req_mixed_phase = phase;
        dut_.req_mixed_phase_first = 1;
        dut_.req_mixed_a8_rows = a8_mask;
        dut_.req_tag = tag;
    }

    void calculate_mixed_expected(
        unsigned a8_rows, unsigned phase0_a4_rows,
        unsigned phase1_a4_rows,
        std::array<int32_t, LANES> &phase0_expected,
        std::array<int32_t, LANES> &final_low_expected,
        std::array<int32_t, LANES> &final_high_expected) {
        for (unsigned column = 0; column < COLS; ++column) {
            for (unsigned row = 0; row < a8_rows; ++row) {
                for (unsigned k = 0; k < 32; ++k) {
                    const bool phase = k >= 16;
                    const unsigned local_k = k & 15u;
                    const int activation =
                        ((row + local_k + (phase ? 1u : 0u)) & 1u) ?
                        -127 : 127;
                    const int weight = (k + column) % 2 == 0 ? -8 : 7;
                    final_low_expected[row * COLS + column] +=
                        activation * weight;
                    if (!phase)
                        phase0_expected[row * COLS + column] +=
                            activation * weight;
                }
            }
            for (unsigned index = 0; index < phase0_a4_rows; ++index) {
                const unsigned slot = a8_rows + index;
                for (unsigned k = 0; k < 32; ++k) {
                    const int activation = (slot + k) & 1u ? -7 : 7;
                    const int weight = (k + column) % 2 == 0 ? -8 : 7;
                    const int product = activation * weight;
                    phase0_expected[slot * COLS + column] += product;
                    final_low_expected[slot * COLS + column] += product;
                }
            }
            for (unsigned index = 0; index < phase1_a4_rows; ++index) {
                const unsigned slot = a8_rows + index;
                for (unsigned k = 0; k < 32; ++k) {
                    const int activation = (slot + k) & 1u ? -7 : 7;
                    const int weight = (k + column) % 2 == 0 ? -8 : 7;
                    final_high_expected[slot * COLS + column] +=
                        activation * weight;
                }
            }
        }
    }

    void run_mixed_groups() {
        unsigned group_count = 0;
        for (unsigned a8_rows = 0; a8_rows <= 8; ++a8_rows) {
            const unsigned phase_capacity = 8 - a8_rows;
            for (unsigned a4_rows = 0; a4_rows <= 2 * phase_capacity;
                 ++a4_rows) {
                if ((a8_rows == 0 && a4_rows <= 8) ||
                    a8_rows + a4_rows == 0)
                    continue;
                const unsigned phase0_a4_rows =
                    std::min(a4_rows, phase_capacity);
                const unsigned phase1_a4_rows = a4_rows - phase0_a4_rows;
                const uint8_t a8_mask = static_cast<uint8_t>(
                    (1u << a8_rows) - 1u);
                const uint8_t phase0_a4_mask = static_cast<uint8_t>(
                    ((1u << phase0_a4_rows) - 1u) << a8_rows);
                const uint8_t phase1_a4_mask = static_cast<uint8_t>(
                    ((1u << phase1_a4_rows) - 1u) << a8_rows);
                std::array<int32_t, LANES> phase0_expected{};
                std::array<int32_t, LANES> final_low_expected{};
                std::array<int32_t, LANES> final_high_expected{};
                calculate_mixed_expected(
                    a8_rows, phase0_a4_rows, phase1_a4_rows,
                    phase0_expected, final_low_expected, final_high_expected);
                const uint16_t tag = next_tag_++;

                prepare_mixed_phase(a8_rows, phase0_a4_rows,
                                    phase1_a4_rows, false, tag);
                dut_.accum_result_ready = 1;
                dut_.req_valid = 1;
                dut_.eval();
                if (!dut_.req_ready)
                    throw std::runtime_error(
                        "PE mixed phase 0 did not accept at II=1");
                tick();
                prepare_mixed_phase(a8_rows, phase0_a4_rows,
                                    phase1_a4_rows, true, tag);
                dut_.req_valid = 1;
                dut_.eval();
                if (!dut_.req_ready)
                    throw std::runtime_error(
                        "PE mixed phase 1 did not accept after phase 0");
                tick();
                dut_.req_valid = 0;
                dut_.accum_result_ready = 0;

                expect_accum_result(
                    phase0_expected, tag, false,
                    row_lane_mask(a8_mask | phase0_a4_mask));
                expect_accum_result(
                    final_low_expected, tag, true,
                    row_lane_mask(a8_mask | phase0_a4_mask));
                if (phase1_a4_rows != 0)
                    expect_accum_result(
                        final_high_expected, tag, true,
                        row_lane_mask(phase1_a4_mask));
                ++group_count;
            }
        }
        if (group_count != 72)
            throw std::runtime_error("PE mixed group coverage count mismatch");
    }

    static int signed_nibble(unsigned raw) {
        return (raw & 8u) ? static_cast<int>(raw) - 16 : static_cast<int>(raw);
    }

    static int signed_byte(uint8_t raw) {
        return (raw & 0x80u) ? static_cast<int>(raw) - 256 :
            static_cast<int>(raw);
    }

    void expect_single_product(unsigned mode, uint8_t activation,
                               uint8_t weight, int32_t expected_value) {
        clear_request();
        set_bits(dut_.req_activation_payload, 0, mode == 0 ? 4 : 8, activation);
        set_bits(dut_.req_weight_payload, 0, mode == 2 ? 8 : 4, weight);
        const uint16_t tag = next_tag_++;
        dut_.req_mode = mode;
        dut_.req_row_mask = 1;
        dut_.req_k_mask = 1;
        dut_.req_col_mask = 1;
        dut_.req_first_k_step = 1;
        dut_.req_last_k_step = 1;
        dut_.req_tag = tag;
        accept_request();
        std::array<int32_t, LANES> expected{};
        expected[0] = expected_value;
        expect_accum_result(expected, tag, true, 1);
    }

    void run_nibble_signedness_exhaustive() {
        for (unsigned lhs = 0; lhs < 16; ++lhs) {
            for (unsigned rhs = 0; rhs < 16; ++rhs) {
                const int lhs_signed = signed_nibble(lhs);
                const int rhs_signed = signed_nibble(rhs);
                const uint8_t lhs_high = static_cast<uint8_t>(
                    (lhs << 4) | (lhs == 8 ? 1 : 0));
                const uint8_t rhs_high = static_cast<uint8_t>(
                    (rhs << 4) | (rhs == 8 ? 1 : 0));
                expect_single_product(0, lhs, rhs,
                                      lhs_signed * rhs_signed);
                expect_single_product(1, static_cast<uint8_t>(lhs), rhs,
                                      static_cast<int>(lhs) * rhs_signed);
                expect_single_product(2, lhs_high, rhs_high,
                                      signed_byte(lhs_high) *
                                          signed_byte(rhs_high));
                expect_single_product(2, lhs_high,
                                      static_cast<uint8_t>(rhs),
                                      signed_byte(lhs_high) *
                                          static_cast<int>(rhs));
                expect_single_product(2, static_cast<uint8_t>(lhs),
                                      rhs_high,
                                      static_cast<int>(lhs) *
                                          signed_byte(rhs_high));
                expect_single_product(2, static_cast<uint8_t>(lhs),
                                      static_cast<uint8_t>(rhs),
                                      static_cast<int>(lhs * rhs));
            }
        }
    }

    void run_sparse_masks() {
        constexpr uint8_t ROW_MASK = 0xa5u;
        constexpr uint8_t COL_MASK = 0x5au;
        constexpr std::array<uint32_t, 3> K_MASK = {
            0xa5a55aa5u, 0x0000a55au, 0x000000a5u
        };

        for (unsigned mode = 0; mode < 3; ++mode) {
            const unsigned k_step = mode == 0 ? 32 : mode == 1 ? 16 : 8;
            clear_request();
            std::array<int32_t, LANES> expected{};
            for (unsigned row = 0; row < ROWS; ++row) {
                for (unsigned k = 0; k < k_step; ++k) {
                    const uint8_t activation_raw = mode == 0 ?
                        uint8_t((row * 7 + k * 3 + 1) & 0xfu) :
                        uint8_t(row * 29 + k * 17 + 0x53u);
                    const int activation = mode == 0 ?
                        signed_nibble(activation_raw) :
                        signed_byte(activation_raw);
                    set_bits(dut_.req_activation_payload,
                             mode == 0 ? (row * 32 + k) * 4 :
                                 mode == 1 ? (row * 16 + k) * 8 :
                                             (row * 8 + k) * 8,
                             mode == 0 ? 4 : 8, activation_raw);
                    for (unsigned column = 0; column < COLS; ++column) {
                        const uint8_t weight_raw = mode == 2 ?
                            uint8_t(k * 31 + column * 19 + 0x6du) :
                            uint8_t((k * 5 + column * 3 + 2) & 0xfu);
                        const int weight = mode == 2 ?
                            signed_byte(weight_raw) : signed_nibble(weight_raw);
                        set_bits(dut_.req_weight_payload,
                                 mode == 2 ? (k * COLS + column) * 8 :
                                             (k * COLS + column) * 4,
                                 mode == 2 ? 8 : 4, weight_raw);
                        if (((ROW_MASK >> row) & 1u) &&
                            ((COL_MASK >> column) & 1u) &&
                            ((K_MASK[mode] >> k) & 1u))
                            expected[row * COLS + column] +=
                                activation * weight;
                    }
                }
            }

            const uint16_t tag = next_tag_++;
            dut_.req_mode = mode;
            dut_.req_row_mask = ROW_MASK;
            dut_.req_k_mask = K_MASK[mode];
            dut_.req_col_mask = COL_MASK;
            dut_.req_first_k_step = 1;
            dut_.req_last_k_step = 1;
            dut_.req_tag = tag;
            accept_request();

            uint64_t lane_mask = 0;
            for (unsigned row = 0; row < ROWS; ++row)
                if ((ROW_MASK >> row) & 1u)
                    lane_mask |= uint64_t{COL_MASK} << (row * COLS);
            expect_accum_result(expected, tag, true, lane_mask);
        }
    }

    void prepare_single_step(uint16_t tag) {
        clear_request();
        for (unsigned row = 0; row < ROWS; ++row)
            for (unsigned k = 0; k < 8; ++k)
                set_bits(dut_.req_activation_payload, (row * 8 + k) * 8, 8, 1);
        for (unsigned k = 0; k < 8; ++k)
            for (unsigned column = 0; column < COLS; ++column)
                set_bits(dut_.req_weight_payload, (k * COLS + column) * 8, 8, 1);
        dut_.req_mode = 2;
        dut_.req_row_mask = 0xffu;
        dut_.req_k_mask = 0xffu;
        dut_.req_col_mask = 0xffu;
        dut_.req_first_k_step = 1;
        dut_.req_last_k_step = 1;
        dut_.req_tag = tag;
    }

    void run_pipeline_latency_and_ii() {
        constexpr unsigned REQUESTS = 4;
        constexpr uint16_t TAG = 0x6f00u;
        unsigned issued = 0;
        unsigned received = 0;
        unsigned first_accept_cycle = 0;
        unsigned first_accum_result_cycle = 0;
        dut_.accum_result_ready = 1;

        for (unsigned cycle = 0; cycle < 32 && received < REQUESTS; ++cycle) {
            if (issued < REQUESTS) {
                prepare_single_step(TAG);
                dut_.req_first_k_step = issued == 0;
                dut_.req_last_k_step = issued + 1 == REQUESTS;
                dut_.req_valid = 1;
            } else {
                dut_.req_valid = 0;
            }
            dut_.eval();
            const bool request_fire = dut_.req_valid && dut_.req_ready;
            const bool accum_result_fire = dut_.accum_result_valid && dut_.accum_result_ready;

            if (issued < REQUESTS && !request_fire)
                throw std::runtime_error("PE inserted a bubble in the II=1 request stream");
            if (request_fire) {
                if (issued == 0) first_accept_cycle = cycle;
                ++issued;
            }
            if (accum_result_fire) {
                if (received == 0) first_accum_result_cycle = cycle;
                if (dut_.accum_result_tag != TAG ||
                    static_cast<bool>(dut_.accum_result_last_k_step) !=
                        (received + 1 == REQUESTS))
                    throw std::runtime_error("PE II=1 accum_result metadata mismatch");
                const int32_t expected = static_cast<int32_t>((received + 1) * 8);
                for (unsigned lane = 0; lane < LANES; ++lane)
                    if (static_cast<int32_t>(get_bits(
                            dut_.accum_result_accumulators, lane * 32, 32)) != expected)
                        throw std::runtime_error("PE II=1 accum_result value mismatch");
                ++received;
            }
            tick();
        }
        dut_.req_valid = 0;
        dut_.accum_result_ready = 0;
        if (issued != REQUESTS || received != REQUESTS)
            throw std::runtime_error("PE II=1 stream did not complete");
        if (first_accum_result_cycle - first_accept_cycle != 3)
            throw std::runtime_error("PE K-step pipeline latency is not three cycles");
    }

    void run_accum_result_fifo_backpressure() {
        dut_.accum_result_ready = 0;
        unsigned accepted = 0;
        bool backpressure_seen = false;
        for (unsigned cycle = 0; cycle < 64; ++cycle) {
            prepare_single_step(static_cast<uint16_t>(0x7000u + cycle));
            dut_.req_valid = 1;
            dut_.eval();
            const bool handshake = dut_.req_ready;
            tick();
            if (handshake) ++accepted;
            else backpressure_seen = true;
        }
        dut_.req_valid = 0;
        if (accepted < 1 || !backpressure_seen)
            throw std::runtime_error("PE accum_result elastic entry did not apply backpressure");
        dut_.accum_result_ready = 1;
        for (unsigned wait = 0; wait < 512 && !dut_.idle; ++wait) tick();
        if (!dut_.idle) throw std::runtime_error("PE accum_result FIFO did not drain");
        dut_.accum_result_ready = 0;
    }

    void run_abort_recovery() {
        prepare_single_step(0x7f00u);
        dut_.req_last_k_step = 0;
        accept_request();
        for (unsigned cycle = 0; cycle < 3; ++cycle) tick();

        dut_.abort_request = 1;
        unsigned acknowledgement_count = 0;
        for (unsigned wait = 0; wait < 256 && acknowledgement_count == 0; ++wait) {
            tick();
            acknowledgement_count += dut_.abort_ack ? 1u : 0u;
        }
        if (acknowledgement_count != 1 || dut_.accum_result_valid)
            throw std::runtime_error("PE abort did not drain and discard accum_results");

        for (unsigned hold = 0; hold < 6; ++hold) {
            tick();
            acknowledgement_count += dut_.abort_ack ? 1u : 0u;
        }
        if (acknowledgement_count != 1)
            throw std::runtime_error("PE held abort produced duplicate acknowledgement");

        dut_.abort_request = 0;
        tick();
        if (!dut_.idle) throw std::runtime_error("PE did not return idle after abort");

        prepare_single_step(0x7f01u);
        accept_request();
        std::array<int32_t, LANES> expected{};
        expected.fill(8);
        expect_accum_result(expected, 0x7f01u, true, ~uint64_t{0});

        dut_.abort_request = 1;
        acknowledgement_count = 0;
        for (unsigned wait = 0; wait < 16 && acknowledgement_count == 0; ++wait) {
            tick();
            acknowledgement_count += dut_.abort_ack ? 1u : 0u;
        }
        for (unsigned hold = 0; hold < 3; ++hold) {
            tick();
            acknowledgement_count += dut_.abort_ack ? 1u : 0u;
        }
        if (acknowledgement_count != 1)
            throw std::runtime_error("PE did not re-arm exactly once for a new abort");
        dut_.abort_request = 0;
        tick();
    }
};
}  // namespace

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Regression regression;
        regression.run();
    } catch (const std::exception &error) {
        std::cerr << "FAIL mixed_precision_pe: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
