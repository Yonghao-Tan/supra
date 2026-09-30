#include "Vsoftmax_row_regression_top.h"
#include "verilated.h"

extern "C" {
#include "../cmodel/rtl_numeric.h"
}

#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

double sc_time_stamp() { return 0.0; }

namespace {
constexpr unsigned kRows = 8;
constexpr unsigned kElements = 8;

template <typename Wide>
void set_u16(Wide &words, unsigned index, uint16_t value) {
    const unsigned word = index / 2;
    const unsigned shift = (index % 2) * 16;
    words[word] = (words[word] & ~(0xffffu << shift)) |
                  (static_cast<uint32_t>(value) << shift);
}

template <typename Wide>
uint16_t get_u16(const Wide &words, unsigned index) {
    return static_cast<uint16_t>(
        (words[index / 2] >> ((index % 2) * 16)) & 0xffffu);
}

template <typename Wide>
int8_t get_i8(const Wide &words, unsigned index) {
    return static_cast<int8_t>(
        (words[index / 4] >> ((index % 4) * 8)) & 0xffu);
}

std::array<std::array<uint16_t, 256>, 2> load_luts() {
    std::ifstream stream("rtl/config/softmax_lut_pkg.sv");
    if (!stream)
        throw std::runtime_error("cannot read Softmax LUT package");
    const std::string text((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    const std::regex value_pattern("16'h([0-9a-fA-F]{4})");
    std::vector<uint16_t> values;
    for (std::sregex_iterator match(text.begin(), text.end(), value_pattern), end;
         match != end; ++match)
        values.push_back(static_cast<uint16_t>(
            std::stoul((*match)[1].str(), nullptr, 16)));
    if (values.size() < 512)
        throw std::runtime_error("Softmax LUT package is incomplete");
    std::array<std::array<uint16_t, 256>, 2> result{};
    for (unsigned index = 0; index < 256; ++index) {
        result[0][index] = values[index];
        result[1][index] = values[index + 256];
    }
    return result;
}

struct PendingSource {
    uint8_t pass = 0;
    unsigned key = 0;
    uint16_t tag = 0;
};

struct TestCase {
    unsigned rows = 0;
    unsigned columns = 0;
    std::vector<uint16_t> scores;
    std::vector<uint16_t> probabilities;
    std::vector<uint16_t> stored_probabilities;
    std::vector<uint16_t> exponents;
    std::vector<uint16_t> scales;
    std::vector<int8_t> values;
};

class Regression {
public:
    Regression() : luts_(load_luts()) {
        dut_.clk = 0;
        dut_.rst = 1;
        dut_.abort_request = 0;
        dut_.hold_scratch_response = 0;
        dut_.start_valid = 0;
        dut_.source_req_ready = 0;
        dut_.source_rsp_valid = 0;
        dut_.bf16_write_ready = 0;
        dut_.quantized_write_ready = 0;
        dut_.scale_write_ready = 0;
        dut_.capture_probability_enable = 0;
        dut_.capture_probability_ready = 0;
        tick();
        tick();
        dut_.rst = 0;
        tick();
    }

    void run() {
        test_start_blocked_by_abort();
        run_case(make_case(1, 1), false);
        run_case(make_case(3, 8), false);
        run_case(make_case(8, 24), false);
        run_case(make_case(3, 17), true, true);
        run_case(make_case(8, 2048), false);
        run_case(make_case(3, 24), false, false, 1);
        run_case(make_case(3, 8), false, false, 2);
        run_case(make_case(1, 8), false);
        dut_.final();
        std::cout << "PASS softmax_tile_pipeline: cases=6 rows=1/3/8/3/8/1"
                  << " S=1/8/24/17/2048/8"
                  << " tile=8x8 four_pass=1 main_pass_accepted_ii=1"
                  << " raw_mismatch=0 row_tail=1 real_sync_scratch=1"
                  << " output_stall=1 blocked_start=1 pre_p8_capture=1"
                  << " scratch_abort_drain=1 scale_abort_drain=1 restart=1\n";
    }

private:
    Vsoftmax_row_regression_top dut_;
    std::array<std::array<uint16_t, 256>, 2> luts_;

    void tick() {
        dut_.clk = 0;
        dut_.eval();
        dut_.clk = 1;
        dut_.eval();
        dut_.clk = 0;
        dut_.eval();
    }

    TestCase make_case(unsigned rows, unsigned columns) {
        static constexpr std::array<uint16_t, 16> samples{{
            0x3f00u, 0xbe80u, 0x3e80u, 0xbf40u,
            0x4000u, 0x3d80u, 0xc000u, 0x3f80u,
            0x3e00u, 0xbf00u, 0x3f40u, 0xbd80u,
            0x4040u, 0xbe00u, 0x3d00u, 0xc040u,
        }};
        TestCase item;
        item.rows = rows;
        item.columns = columns;
        item.scores.resize(rows * columns);
        item.probabilities.resize(rows * columns);
        item.stored_probabilities.resize(rows * columns, 0xffff);
        item.exponents.assign(rows * columns, 0);
        item.scales.resize(rows);
        item.values.resize(rows * columns);
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned column = 0; column < columns; ++column)
                item.scores[row * columns + column] =
                    samples[(row * 7 + column * 3 + 1) % samples.size()];
        if (rtl_softmax_lut_bf16(item.scores.data(), rows, columns,
                                luts_[0].data(), luts_[1].data(),
                                item.probabilities.data()) != 0)
            throw std::runtime_error("C11 Softmax failed");
        for (unsigned row = 0; row < rows; ++row)
            if (rtl_quantize_row_bf16(
                    &item.probabilities[row * columns], columns, 127,
                    &item.values[row * columns], &item.scales[row]) != 0)
                throw std::runtime_error("C11 probability quantization failed");
        return item;
    }

    void drive_source(const TestCase &item, const PendingSource &pending) {
        const std::vector<uint16_t> &source = pending.pass < 2 ? item.scores :
            pending.pass == 2 ? item.exponents : item.stored_probabilities;
        uint64_t mask = 0;
        for (unsigned row = 0; row < kRows; ++row) {
            for (unsigned element = 0; element < kElements; ++element) {
                const unsigned column = pending.key + element;
                const bool active = row < item.rows && column < item.columns;
                set_u16(dut_.source_rsp_values, row * kElements + element,
                        active ? source[row * item.columns + column] : 0);
                if (active)
                    mask |= uint64_t{1} << (row * kElements + element);
            }
        }
        dut_.source_rsp_lane_mask = mask;
        dut_.source_rsp_tag = pending.tag;
        dut_.source_rsp_valid = 1;
    }

    void capture_bf16_write(TestCase &item) {
        const unsigned key = dut_.bf16_write_key;
        for (unsigned row = 0; row < kRows; ++row) {
            for (unsigned element = 0; element < kElements; ++element) {
                const unsigned lane = row * kElements + element;
                const unsigned column = key + element;
                const bool active = row < item.rows && column < item.columns;
                if (((dut_.bf16_write_lane_mask >> lane) & 1u) != active)
                    throw std::runtime_error("Softmax BF16 write lane mask mismatch");
                if (!active)
                    continue;
                const uint16_t value = get_u16(dut_.bf16_write_values, lane);
                if (dut_.bf16_write_probability) {
                    if (value != item.probabilities[row * item.columns + column])
                        throw std::runtime_error("Softmax probability raw BF16 mismatch");
                    item.stored_probabilities[row * item.columns + column] = value;
                } else {
                    item.exponents[row * item.columns + column] = value;
                }
            }
        }
    }

    void check_quantized_write(const TestCase &item) {
        const unsigned key = dut_.quantized_write_key;
        for (unsigned row = 0; row < kRows; ++row) {
            for (unsigned element = 0; element < kElements; ++element) {
                const unsigned lane = row * kElements + element;
                const unsigned column = key + element;
                const bool active = row < item.rows && column < item.columns;
                if (((dut_.quantized_write_lane_mask >> lane) & 1u) != active)
                    throw std::runtime_error("Softmax quantized lane mask mismatch");
                if (active && get_i8(dut_.quantized_write_values, lane) !=
                                  item.values[row * item.columns + column])
                    throw std::runtime_error("Softmax probability quantized mismatch");
            }
        }
    }

    void test_start_blocked_by_abort() {
        const uint64_t source_before = dut_.accepted_source_tile_count;
        const uint64_t bf16_before = dut_.accepted_bf16_write_count;
        const uint64_t quantized_before = dut_.accepted_quantized_write_count;
        const uint64_t scratch_read_before = dut_.accepted_scratch_read_count;
        const uint64_t scratch_write_before = dut_.accepted_scratch_write_count;
        const uint64_t vector_before = dut_.accepted_vector_request_count;
        const uint64_t max_before = dut_.accepted_max_request_count;
        const uint64_t reduction_before = dut_.accepted_reduction_request_count;
        const uint64_t quant_before = dut_.accepted_quantized_value_count;

        dut_.row_count = 1;
        dut_.row_length = 8;
        dut_.abort_request = 1;
        dut_.start_valid = 1;
        for (unsigned cycle = 0; cycle < 3; ++cycle) {
            dut_.eval();
            if (dut_.start_ready || dut_.done || dut_.source_req_valid ||
                dut_.bf16_write_valid || dut_.quantized_write_valid ||
                dut_.scale_write_valid)
                throw std::runtime_error(
                    "Softmax row pipeline acted on a blocked start");
            tick();
        }
        if (dut_.accepted_source_tile_count != source_before ||
            dut_.accepted_bf16_write_count != bf16_before ||
            dut_.accepted_quantized_write_count != quantized_before ||
            dut_.accepted_scratch_read_count != scratch_read_before ||
            dut_.accepted_scratch_write_count != scratch_write_before ||
            dut_.accepted_vector_request_count != vector_before ||
            dut_.accepted_max_request_count != max_before ||
            dut_.accepted_reduction_request_count != reduction_before ||
            dut_.accepted_quantized_value_count != quant_before)
            throw std::runtime_error(
                "Softmax row pipeline counters changed without a start handshake");

        dut_.abort_request = 0;
        dut_.eval();
        if (!dut_.start_ready)
            throw std::runtime_error(
                "Softmax row pipeline did not accept held start after abort");
        tick();
        dut_.start_valid = 0;
        if (dut_.start_ready)
            throw std::runtime_error(
                "Softmax row pipeline accepted held start more than once");

        dut_.abort_request = 1;
        bool abort_seen = false;
        for (unsigned cycle = 0; cycle < 8; ++cycle) {
            dut_.eval();
            if (dut_.abort_ack) {
                abort_seen = true;
                break;
            }
            tick();
        }
        if (!abort_seen)
            throw std::runtime_error(
                "Softmax row pipeline did not acknowledge focused abort");
        dut_.abort_request = 0;
        tick();
        if (!dut_.start_ready)
            throw std::runtime_error(
                "Softmax row pipeline did not return to idle after focused abort");
    }

    void run_case(TestCase item, bool stalls, bool capture = false, unsigned abort_point = 0) {
        dut_.hold_scratch_response = abort_point == 1;
        dut_.capture_probability_enable = capture;
        dut_.capture_probability_ready = 0;
        dut_.row_count = item.rows;
        dut_.row_length = item.columns;
        dut_.start_valid = 1;
        dut_.source_req_ready = 1;
        dut_.bf16_write_ready = 1;
        dut_.quantized_write_ready = 1;
        dut_.scale_write_ready = 1;
        dut_.eval();
        if (!dut_.start_ready)
            throw std::runtime_error("Softmax tile pipeline did not accept start");
        tick();
        dut_.start_valid = 0;

        std::deque<PendingSource> responses;
        std::array<std::vector<uint64_t>, 4> request_cycles;
        unsigned scale_writes = 0;
        unsigned capture_cycles = 0;
        bool abort_started = false;
        uint64_t abort_cycle = 0;
        unsigned scratch_pending = 0, scale_pending = 0;
        const uint64_t source_before = dut_.accepted_source_tile_count;
        const uint64_t bf16_before = dut_.accepted_bf16_write_count;
        const uint64_t quantized_before = dut_.accepted_quantized_write_count;
        const uint64_t scratch_read_before = dut_.accepted_scratch_read_count;
        const uint64_t scratch_write_before = dut_.accepted_scratch_write_count;
        const uint64_t vector_before = dut_.accepted_vector_request_count;
        const uint64_t max_before = dut_.accepted_max_request_count;
        const uint64_t reduction_before = dut_.accepted_reduction_request_count;
        const uint64_t quant_before = dut_.accepted_quantized_value_count;

        for (uint64_t cycle = 0; cycle < 200000; ++cycle) {
            if (!abort_started && ((abort_point == 1 && scratch_pending) ||
                                  (abort_point == 2 && scale_pending))) {
                abort_started = true;
                abort_cycle = cycle;
                dut_.abort_request = 1;
            }
            if (abort_started && cycle >= abort_cycle + 5)
                dut_.hold_scratch_response = 0;
            dut_.capture_probability_ready = capture_cycles == 7;
            dut_.source_req_ready = 1;
            dut_.bf16_write_ready = !stalls || (cycle % 17) != 3;
            dut_.quantized_write_ready = !stalls || (cycle % 13) != 4;
            dut_.scale_write_ready = !stalls || (cycle % 11) != 2;
            if (responses.empty())
                dut_.source_rsp_valid = 0;
            else
                drive_source(item, responses.front());
            dut_.eval();

            if (dut_.capture_probability_valid) {
                if (!capture || !responses.empty() || dut_.source_req_valid || dut_.quantized_write_valid ||
                    item.stored_probabilities != item.probabilities)
                    throw std::runtime_error("capture started before BF16 writes drained or overlapped P8 overwrite");
                ++capture_cycles;
            }

            const bool request_fire = dut_.source_req_valid && dut_.source_req_ready;
            const bool response_fire = dut_.source_rsp_valid && dut_.source_rsp_ready;
            const bool bf16_fire = dut_.bf16_write_valid && dut_.bf16_write_ready;
            const bool quantized_fire = dut_.quantized_write_valid && dut_.quantized_write_ready;
            const bool scale_fire = dut_.scale_write_valid && dut_.scale_write_ready;
            scratch_pending += unsigned(dut_.scratch_read_accepted);
            scratch_pending -= unsigned(dut_.scratch_response_accepted);
            scale_pending += unsigned(dut_.quant_scale_accepted);
            scale_pending -= unsigned(dut_.quant_scale_response_accepted);
            if (bf16_fire)
                capture_bf16_write(item);
            if (quantized_fire)
                check_quantized_write(item);
            if (scale_fire) {
                ++scale_writes;
                const uint8_t expected_mask = static_cast<uint8_t>((1u << item.rows) - 1u);
                if (dut_.scale_write_row_mask != expected_mask)
                    throw std::runtime_error("Softmax scale row mask mismatch");
                for (unsigned row = 0; row < item.rows; ++row)
                    if (get_u16(dut_.scale_write_values, row) != item.scales[row])
                        throw std::runtime_error("Softmax row scale raw BF16 mismatch");
            }
            if (response_fire)
                responses.pop_front();
            if (request_fire) {
                request_cycles[dut_.source_req_pass].push_back(cycle);
                responses.push_back(PendingSource{
                    static_cast<uint8_t>(dut_.source_req_pass),
                    static_cast<unsigned>(dut_.source_req_key),
                    static_cast<uint16_t>(dut_.source_req_tag)});
            }
            tick();
            if (dut_.abort_ack) {
                if (!abort_started || scratch_pending || scale_pending || !responses.empty())
                    throw std::runtime_error("Softmax acknowledged abort before accepted responses drained");
                dut_.abort_request = 0;
                dut_.hold_scratch_response = 0;
                dut_.source_rsp_valid = 0;
                tick();
                if (!dut_.start_ready)
                    throw std::runtime_error("Softmax did not restart after resource drain");
                return;
            }
            if (dut_.error)
                throw std::runtime_error("Softmax tile pipeline reported error");
            if (dut_.done) {
                if (capture_cycles != (capture ? 8u : 0u))
                    throw std::runtime_error("probability capture handshake count mismatch");
                const unsigned tiles = (item.columns + 7) / 8;
                unsigned padded = 1;
                while (padded < tiles)
                    padded <<= 1;
                if (scale_writes != 1 ||
                    dut_.accepted_source_tile_count - source_before != 4u * tiles ||
                    dut_.accepted_bf16_write_count - bf16_before != 2u * tiles ||
                    dut_.accepted_quantized_write_count - quantized_before != tiles ||
                    dut_.accepted_scratch_read_count - scratch_read_before != padded - 1u ||
                    dut_.accepted_scratch_write_count - scratch_write_before != 2u * padded - 1u ||
                    dut_.accepted_vector_request_count - vector_before != 2u * tiles ||
                    dut_.accepted_max_request_count - max_before != 2u * tiles ||
                    dut_.accepted_reduction_request_count - reduction_before !=
                        tiles + padded - 1u ||
                    dut_.accepted_quantized_value_count - quant_before != tiles)
                    throw std::runtime_error("Softmax accepted-event count mismatch");
                for (const auto &pass : request_cycles) {
                    if (pass.size() != tiles)
                        throw std::runtime_error("Softmax source pass tile count mismatch");
                    for (size_t index = 1; index < pass.size(); ++index)
                        if (pass[index] != pass[index - 1] + 1)
                            throw std::runtime_error("Softmax source pass II is not one");
                }
                tick();
                return;
            }
        }
        throw std::runtime_error("Softmax tile pipeline timeout");
    }
};
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::threadContextp()->threads(1);
    try {
        Regression regression;
        regression.run();
    } catch (const std::exception &exception) {
        std::cerr << "FAIL softmax_tile_pipeline: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
