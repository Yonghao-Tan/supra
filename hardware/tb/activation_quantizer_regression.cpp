#include "Vactivation_quantizer.h"
#include "verilated.h"

extern "C" {
#include "../cmodel/rtl_numeric.h"
}

#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <vector>

double sc_time_stamp() { return 0.0; }

namespace {
constexpr unsigned ROWS = 8;
constexpr unsigned ELEMENTS_PER_TILE = 8;
constexpr unsigned LANES = ROWS * ELEMENTS_PER_TILE;
constexpr unsigned MAX_ELEMENTS = 32;

template <typename Words>
void set_u16(Words &words, unsigned lane, uint16_t value) {
    const unsigned word = lane / 2;
    const unsigned shift = (lane % 2) * 16;
    words[word] = (words[word] & ~(0xffffu << shift)) |
                  (static_cast<uint32_t>(value) << shift);
}

template <typename Words>
uint16_t get_u16(const Words &words, unsigned lane) {
    return static_cast<uint16_t>(
        (words[lane / 2] >> ((lane % 2) * 16)) & 0xffffu);
}

template <typename Words>
int8_t get_i8(const Words &words, unsigned lane) {
    return static_cast<int8_t>(
        (words[lane / 4] >> ((lane % 4) * 8)) & 0xffu);
}

struct ExpectedTile {
    std::array<int8_t, LANES> values{};
    uint64_t lane_mask = 0;
    uint16_t tag = 0;
};

class Regression {
public:
    Regression() {
        dut_.clk = 0;
        dut_.rst = 1;
        dut_.abort_request = 0;
        dut_.scale_req_valid = 0;
        dut_.scale_req_clip_ratio_bf16 = 0;
        dut_.scale_rsp_ready = 0;
        dut_.quantized_req_valid = 0;
        dut_.quantized_rsp_ready = 0;
        tick();
        tick();
        dut_.rst = 0;
        tick();
    }

    void run() {
        run_case(false, false, 8, 32, 0x1000u);
        run_case(true, false, 3, 17, 0x2000u);
        run_case(false, true, 8, 24, 0x3000u);
        run_case_mask(0x05u, false, 3, 32, 0x3800u);
        run_case_mask(0x55u, false, 8, 32, 0x3900u, 0x3f4du);
        run_case_mask(0x05u, false, 3, 17, 0x3a00u, 0x3f1au);
        run_case_mask(0x05u, false, 3, 17, 0x3b00u, 0x3f80u);
        run_case_mask(0x00u, false, 3, 17, 0x3c00u, 0x3f4du);
        run_case_mask(0x05u, false, 3, 17, 0x3c40u, 0x3f60u);
        run_case_mask(0x05u, false, 3, 17, 0x3c80u, 0x3f4du);
        run_case_mask(0x55u, false, 8, 32, 0x3d00u, 0x3f4du, true);
        run_case_mask(0x55u, false, 8, 32, 0x3e00u, 0x0001u, true);
        dut_.scale_req_clip_ratio_bf16 = 0;
        run_scale_boundaries(false);
        run_scale_boundaries(true);
        run_all_finite_values(false);
        run_all_finite_values(true);
        run_abort_recovery();
        dut_.final();
        std::cout << "PASS activation_quantizer: rows=8 lanes=64 setups=12"
                  << " dynamic_a8=1 dynamic_a4=1 static_row_scales=1"
                  << " mixed_a4_row_mask=0x05"
                  << " clipping_cases=8 clipping_subnormal=1 clipping_all_a8=1 clipping_deep_normal_reload=1"
                  << " accepted_quantized_ii=1 row_tail=1 element_tail=1"
                  << " scale_mismatch=0 quantized_mismatch=0 response_stall=1"
                  << " dynamic_scale_boundaries=16"
                  << " finite_raw_values=130560 abort_recovery=1\n";
    }

private:
    Vactivation_quantizer dut_;
    std::uint64_t ticks_ = 0;

    void tick() {
        dut_.clk = 0;
        dut_.eval();
        dut_.clk = 1;
        dut_.eval();
        dut_.clk = 0;
        dut_.eval();
        ++ticks_;
    }

    static uint16_t row_max_abs(
        const std::array<uint16_t, MAX_ELEMENTS> &values, unsigned count) {
        uint16_t maximum = 0;
        for (unsigned element = 0; element < count; ++element) {
            const uint16_t magnitude = values[element] & 0x7fffu;
            if (rtl_bf16_compare(magnitude, maximum) > 0) maximum = magnitude;
        }
        return maximum;
    }

    void run_case(bool a4, bool use_static_scale, unsigned active_token_count,
                  unsigned element_count, uint16_t tag_base) {
        run_case_mask(a4 ? 0xffu : 0x00u, use_static_scale, active_token_count,
                      element_count, tag_base);
    }

    void run_scale_boundaries(bool a4) {
        static constexpr std::array<uint16_t, ROWS> MAXIMUMS{{
            0x0000u, 0x0001u, 0x0003u, 0x007fu,
            0x0080u, 0x0081u, 0x3f80u, 0x7f7fu,
        }};
        dut_.scale_req_a4_row_mask = a4 ? 0xffu : 0x00u;
        dut_.scale_req_use_static_scale = 0;
        dut_.scale_req_row_mask = 0xffu;
        for (unsigned row = 0; row < ROWS; ++row) {
            set_u16(dut_.scale_req_row_max_abs, row, MAXIMUMS[row]);
            set_u16(dut_.scale_req_static_scales_bf16, row, 0);
        }
        dut_.scale_req_valid = 1;
        while (!dut_.scale_req_ready) tick();
        tick();
        dut_.scale_req_valid = 0;
        for (unsigned wait = 0; wait < 128 && !dut_.scale_rsp_valid; ++wait)
            tick();
        if (!dut_.scale_rsp_valid)
            throw std::runtime_error("dynamic scale boundary response timeout");
        for (unsigned row = 0; row < ROWS; ++row) {
            const uint16_t expected = rtl_activation_scale_bf16(
                MAXIMUMS[row], a4 ? 7 : 127);
            const uint16_t actual = get_u16(dut_.scale_rsp_values_bf16, row);
            if (actual != expected) {
                std::cerr << "dynamic scale mismatch a4=" << a4
                          << " maximum=0x" << std::hex << MAXIMUMS[row]
                          << " actual=0x" << actual
                          << " expected=0x" << expected << std::dec << '\n';
                throw std::runtime_error("dynamic scale boundary mismatch");
            }
        }
        dut_.scale_rsp_ready = 1;
        tick();
        dut_.scale_rsp_ready = 0;
    }

    void run_case_mask(uint8_t a4_rows, bool use_static_scale,
                       unsigned active_token_count, unsigned element_count,
                       uint16_t tag_base, uint16_t clip_ratio = 0, bool subnormal = false) {
        static constexpr std::array<uint16_t, 16> PATTERN{{
            0x0000u, 0x8000u, 0x3c80u, 0xbc80u,
            0x3d00u, 0xbd00u, 0x3e00u, 0xbe00u,
            0x3e80u, 0xbe80u, 0x3f00u, 0xbf00u,
            0x3f40u, 0xbf40u, 0x3f80u, 0xbf80u,
        }};
        std::array<std::array<uint16_t, MAX_ELEMENTS>, ROWS> values{};
        std::array<std::array<int8_t, MAX_ELEMENTS>, ROWS> expected_values{};
        std::array<uint16_t, ROWS> expected_scales{};

        for (unsigned row = 0; row < ROWS; ++row) {
            const int quantized_max = a4_rows >> row & 1u ? 7 : 127;
            for (unsigned element = 0; element < element_count; ++element) {
                uint16_t value = PATTERN[(row * 3 + element * 5 + tag_base) % PATTERN.size()];
                if ((value & 0x7f80u) != 0 && row < 4)
                    value = static_cast<uint16_t>(
                        (value & 0x807fu) | ((value + row * 0x80u) & 0x7f80u));
                if (subnormal) value = (element & 1u ? 0x8000u : 0) | (element % 6u);
                values[row][element] = value;
            }
            expected_scales[row] = use_static_scale ?
                static_cast<uint16_t>(0x3d80u + row * 0x40u) : 0x3f80u;
            if (row >= active_token_count) {
                expected_scales[row] = 0x3f80u;
                continue;
            }
            auto quantized_input = values[row];
            if (clip_ratio && (a4_rows >> row & 1u) && !use_static_scale) {
                const uint16_t limit = rtl_f32_to_bf16(
                    rtl_bf16_to_f32(row_max_abs(values[row], element_count)) *
                    rtl_bf16_to_f32(clip_ratio));
                for (unsigned element = 0; element < element_count; ++element)
                    if ((quantized_input[element] & 0x7fffu) > limit)
                        quantized_input[element] = (quantized_input[element] & 0x8000u) | limit;
            }
            if (use_static_scale) {
                if (rtl_quantize_bf16_with_scale(
                        values[row].data(), element_count, quantized_max,
                        expected_scales[row], expected_values[row].data()) != 0)
                    throw std::runtime_error("static C11 oracle rejected scale");
            } else if (rtl_quantize_row_bf16(
                           quantized_input.data(), element_count, quantized_max,
                           expected_values[row].data(), &expected_scales[row]) != 0) {
                throw std::runtime_error("dynamic C11 oracle rejected row");
            }
        }

        for (unsigned word = 0; word < 4; ++word) {
            dut_.scale_req_row_max_abs[word] = 0;
            dut_.scale_req_static_scales_bf16[word] = 0;
        }
        for (unsigned row = 0; row < ROWS; ++row) {
            set_u16(dut_.scale_req_row_max_abs, row,
                    row_max_abs(values[row], element_count));
            set_u16(dut_.scale_req_static_scales_bf16, row, expected_scales[row]);
        }
        dut_.scale_req_a4_row_mask = a4_rows;
        dut_.scale_req_clip_ratio_bf16 = clip_ratio;
        dut_.scale_req_use_static_scale = use_static_scale;
        dut_.scale_req_row_mask = static_cast<uint8_t>((1u << active_token_count) - 1u);
        dut_.scale_req_valid = 1;
        while (!dut_.scale_req_ready) tick();
        const auto setup_request_cycle = ticks_ + 1;
        tick();
        dut_.scale_req_valid = 0;

        for (unsigned wait = 0; wait < 128 && !dut_.scale_rsp_valid; ++wait) tick();
        if (!dut_.scale_rsp_valid)
            throw std::runtime_error("scale response timeout");
        const auto setup_cycles = ticks_ - setup_request_cycle;
        for (unsigned row = 0; row < ROWS; ++row)
            if (get_u16(dut_.scale_rsp_values_bf16, row) != expected_scales[row])
                throw std::runtime_error("scale raw mismatch");
        dut_.scale_rsp_ready = 1;
        tick();
        dut_.scale_rsp_ready = 0;

        const unsigned tile_count = (element_count + ELEMENTS_PER_TILE - 1) /
            ELEMENTS_PER_TILE;
        std::deque<ExpectedTile> expected_queue;
        std::array<unsigned, MAX_ELEMENTS / ELEMENTS_PER_TILE> accepted_cycles{};
        unsigned next_tile = 0;
        unsigned responses = 0;
        unsigned cycle = 0;
        unsigned first_response_cycle = 0;
        ExpectedTile driven;
        bool request_loaded = false;

        while (responses != tile_count) {
            if (!request_loaded && next_tile < tile_count) {
                for (unsigned word = 0; word < 32; ++word)
                    dut_.quantized_req_values_bf16[word] = 0;
                driven = ExpectedTile{};
                driven.tag = static_cast<uint16_t>(tag_base + next_tile);
                for (unsigned row = 0; row < ROWS; ++row) {
                    for (unsigned column = 0; column < ELEMENTS_PER_TILE; ++column) {
                        const unsigned element = next_tile * ELEMENTS_PER_TILE + column;
                        const unsigned lane = row * ELEMENTS_PER_TILE + column;
                        if (row < active_token_count && element < element_count) {
                            set_u16(dut_.quantized_req_values_bf16, lane,
                                    values[row][element]);
                            driven.values[lane] = expected_values[row][element];
                            driven.lane_mask |= uint64_t{1} << lane;
                        }
                    }
                }
                dut_.quantized_req_lane_mask = driven.lane_mask;
                dut_.quantized_req_tag = driven.tag;
                request_loaded = true;
            }

            dut_.quantized_req_valid = request_loaded;
            dut_.quantized_rsp_ready = !(cycle == 13 || cycle == 14 || cycle == 22);
            dut_.eval();
            const bool request_fire = dut_.quantized_req_valid && dut_.quantized_req_ready;
            const bool response_fire = dut_.quantized_rsp_valid && dut_.quantized_rsp_ready;
            if (response_fire) {
                if (responses == 0) first_response_cycle = cycle;
                if (expected_queue.empty())
                    throw std::runtime_error("quantized response without request");
                const ExpectedTile &expected = expected_queue.front();
                if (dut_.quantized_rsp_tag != expected.tag ||
                    dut_.quantized_rsp_lane_mask != expected.lane_mask)
                    throw std::runtime_error("quantized response metadata mismatch");
                for (unsigned lane = 0; lane < LANES; ++lane)
                    if ((expected.lane_mask >> lane & 1u) != 0 &&
                        get_i8(dut_.quantized_rsp_values, lane) != expected.values[lane]) {
                        std::cerr << "mismatch a4_rows=0x" << std::hex
                                  << static_cast<unsigned>(a4_rows) << std::dec
                                  << " static=" << use_static_scale
                                  << " tag=" << expected.tag
                                  << " lane=" << lane
                                  << " actual=" << static_cast<int>(
                                         get_i8(dut_.quantized_rsp_values, lane))
                                  << " expected=" << static_cast<int>(expected.values[lane])
                                  << "\n";
                        throw std::runtime_error("quantized raw mismatch");
                    }
                expected_queue.pop_front();
                ++responses;
            }
            if (request_fire) {
                accepted_cycles[next_tile] = cycle;
                expected_queue.push_back(driven);
                ++next_tile;
                request_loaded = false;
            }
            tick();
            if (++cycle > 256)
                throw std::runtime_error("quantized pipeline timeout");
        }
        dut_.quantized_req_valid = 0;
        dut_.quantized_rsp_ready = 1;
        tick();
        dut_.quantized_rsp_ready = 0;
        for (unsigned tile = 1; tile < tile_count; ++tile)
            if (accepted_cycles[tile] != accepted_cycles[tile - 1] + 1)
                throw std::runtime_error("quantized accepted interval is not one cycle");
        std::cout << "QUANT_TIMING a4_mask=" << unsigned(a4_rows)
                  << " rows=" << active_token_count << " static=" << use_static_scale
                  << " ratio_raw=" << clip_ratio << " setup_cycles=" << setup_cycles
                  << " first_code_latency=" << first_response_cycle - accepted_cycles[0] << '\n';
    }

    void run_abort_recovery() {
        dut_.scale_req_a4_row_mask = 0;
        dut_.scale_req_use_static_scale = 0;
        dut_.scale_req_row_mask = 0xffu;
        for (unsigned row = 0; row < ROWS; ++row)
            set_u16(dut_.scale_req_row_max_abs, row, 0x3f80u);
        dut_.scale_req_valid = 1;
        while (!dut_.scale_req_ready) tick();
        tick();
        dut_.scale_req_valid = 0;
        tick();
        dut_.abort_request = 1;
        tick();
        if (!dut_.abort_ack || dut_.scale_rsp_valid || dut_.quantized_rsp_valid)
            throw std::runtime_error("abort did not discard quantizer state");
        tick();
        if (dut_.abort_ack)
            throw std::runtime_error("held abort produced duplicate acknowledgement");
        dut_.abort_request = 0;
        tick();
        if (!dut_.idle || !dut_.scale_req_ready)
            throw std::runtime_error("quantizer did not recover after abort");
    }

    void run_all_finite_values(bool a4) {
        static constexpr std::array<uint16_t, ROWS> SCALES{{
            0x0080u, 0x3c80u, 0x3f00u, 0x3f80u,
            0x3fc0u, 0x4000u, 0x4040u, 0x42feu,
        }};
        std::vector<uint16_t> finite_values;
        finite_values.reserve(65280);
        for (unsigned raw = 0; raw <= 0xffffu; ++raw)
            if ((raw & 0x7f80u) != 0x7f80u)
                finite_values.push_back(static_cast<uint16_t>(raw));

        dut_.scale_req_a4_row_mask = a4 ? 0xff : 0x00;
        dut_.scale_req_use_static_scale = 1;
        dut_.scale_req_row_mask = 0xffu;
        for (unsigned row = 0; row < ROWS; ++row)
            set_u16(dut_.scale_req_static_scales_bf16, row, SCALES[row]);
        dut_.scale_req_valid = 1;
        while (!dut_.scale_req_ready) tick();
        tick();
        dut_.scale_req_valid = 0;
        while (!dut_.scale_rsp_valid) tick();
        dut_.scale_rsp_ready = 1;
        tick();
        dut_.scale_rsp_ready = 0;

        const unsigned tile_count =
            (finite_values.size() + LANES - 1) / LANES;
        std::deque<ExpectedTile> expected_queue;
        unsigned next_tile = 0;
        unsigned responses = 0;
        bool request_loaded = false;
        ExpectedTile driven;
        unsigned cycles = 0;
        dut_.quantized_rsp_ready = 1;

        while (responses != tile_count) {
            if (!request_loaded && next_tile < tile_count) {
                for (unsigned word = 0; word < 32; ++word)
                    dut_.quantized_req_values_bf16[word] = 0;
                driven = ExpectedTile{};
                driven.tag = static_cast<uint16_t>((a4 ? 0x8000u : 0x4000u) + next_tile);
                for (unsigned lane = 0; lane < LANES; ++lane) {
                    const size_t index = static_cast<size_t>(next_tile) * LANES + lane;
                    if (index >= finite_values.size()) continue;
                    const uint16_t value = finite_values[index];
                    set_u16(dut_.quantized_req_values_bf16, lane, value);
                    driven.lane_mask |= uint64_t{1} << lane;
                    if (rtl_quantize_bf16_with_scale(
                            &value, 1, a4 ? 7 : 127, SCALES[lane / 8],
                            &driven.values[lane]) != 0)
                        throw std::runtime_error("finite C11 oracle rejected input");
                }
                dut_.quantized_req_lane_mask = driven.lane_mask;
                dut_.quantized_req_tag = driven.tag;
                request_loaded = true;
            }
            dut_.quantized_req_valid = request_loaded;
            dut_.eval();
            const bool request_fire = dut_.quantized_req_valid && dut_.quantized_req_ready;
            const bool response_fire = dut_.quantized_rsp_valid && dut_.quantized_rsp_ready;
            if (response_fire) {
                if (expected_queue.empty())
                    throw std::runtime_error("finite response without request");
                const ExpectedTile &expected = expected_queue.front();
                if (dut_.quantized_rsp_tag != expected.tag ||
                    dut_.quantized_rsp_lane_mask != expected.lane_mask)
                    throw std::runtime_error("finite response metadata mismatch");
                for (unsigned lane = 0; lane < LANES; ++lane)
                    if ((expected.lane_mask >> lane & 1u) != 0 &&
                        get_i8(dut_.quantized_rsp_values, lane) != expected.values[lane]) {
                        std::cerr << "finite mismatch a4=" << a4
                                  << " tag=" << expected.tag
                                  << " lane=" << lane
                                  << " actual=" << static_cast<int>(
                                         get_i8(dut_.quantized_rsp_values, lane))
                                  << " expected=" << static_cast<int>(expected.values[lane])
                                  << "\n";
                        throw std::runtime_error("finite quantized raw mismatch");
                    }
                expected_queue.pop_front();
                ++responses;
            }
            if (request_fire) {
                expected_queue.push_back(driven);
                ++next_tile;
                request_loaded = false;
            }
            tick();
            if (++cycles > tile_count + 128)
                throw std::runtime_error("finite quantized scan timeout");
        }
        dut_.quantized_req_valid = 0;
        dut_.quantized_rsp_ready = 0;
    }
};
}  // namespace

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Regression regression;
        regression.run();
    } catch (const std::exception &error) {
        std::cerr << "FAIL activation_quantizer: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
