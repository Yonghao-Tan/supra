#include "Vfinal_output_sequencer.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

double sc_time_stamp() { return 0.0; }

namespace {

void put_u16(std::vector<uint8_t> &bytes, unsigned offset, uint16_t value) {
    bytes[offset] = value & 0xff;
    bytes[offset + 1] = value >> 8;
}

void put_u32(std::vector<uint8_t> &bytes, unsigned offset, uint32_t value) {
    for (unsigned byte = 0; byte < 4; ++byte)
        bytes[offset + byte] = (value >> (byte * 8)) & 0xff;
}

void put_u64(std::vector<uint8_t> &bytes, unsigned offset, uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte)
        bytes[offset + byte] = (value >> (byte * 8)) & 0xff;
}

template <typename Wide>
void set_beat(Wide &target, const uint8_t *bytes) {
    std::memset(&target, 0, sizeof(target));
    for (unsigned byte = 0; byte < 16; ++byte)
        target[byte / 4] |= uint32_t{bytes[byte]} << ((byte % 4) * 8);
}

template <typename Wide>
uint64_t get_bits(const Wide &source, unsigned lsb, unsigned width) {
    uint64_t value = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const unsigned source_bit = lsb + bit;
        if ((source[source_bit / 32] >> (source_bit % 32)) & 1u)
            value |= uint64_t{1} << bit;
    }
    return value;
}

uint64_t get_bits(uint64_t source, unsigned lsb, unsigned width) {
    const uint64_t mask = width == 64 ? ~uint64_t{0} :
        (uint64_t{1} << width) - 1;
    return (source >> lsb) & mask;
}

class Regression {
public:
    Regression() : dut_("TOP") {
        dut_.clk = 0;
        dut_.rst = 1;
        clear_inputs();
        tick();
        tick();
        dut_.rst = 0;
        tick();
    }

    void run() {
        std::cout << "VERILATOR_MODEL_THREADS " << dut_.threads() << '\n';
        run_case(1, false);
        run_case(7, false);
        run_case(8, false);
        run_case(9, false);
        run_case(16, false);
        run_case(49, false);
        run_case(96, false);
        run_case(16, true);
        run_locked_case();
        run_invalid_case(0, false);
        run_invalid_case(5, false);
        run_invalid_case(15, false);
        run_invalid_case(15, true);
        run_post_dma_abort();
        run_prediction_dma_abort();
        run_stage_abort(3);
        run_stage_abort(4);
        run_stage_abort(5);
        run_case(16, false);
        dut_.final();
        std::cout << "PASS final_output_sequencer rows=1/7/8/9/16/49/96 "
                  << "groups=2/7/12 two_round_96=PASS "
                  << "descriptor_raw=PASS descriptor_stall=PASS "
                  << "locked=PASS invalid_drain=PASS duplicate_source=PASS "
                  << "abort_dma_hidden_rms_quant=PASS restart=PASS "
                  << "gamma_bypass=PASS threads=" << dut_.threads() << '\n';
    }

private:
    static constexpr uint64_t kPostBase = 0x00100000;
    static constexpr uint64_t kPredictionBase = 0x00200000;
    static constexpr uint64_t kHiddenBase = 0x10000000;
    Vfinal_output_sequencer dut_;
    std::vector<uint8_t> prediction_records_;
    bool descriptor_lookup_pending_ = false;
    unsigned descriptor_lookup_index_ = 0;

    void tick() {
        dut_.clk = 0;
        dut_.descriptor_lookup_rsp_valid = descriptor_lookup_pending_;
        std::memset(&dut_.descriptor_lookup_rsp_data, 0,
                    sizeof(dut_.descriptor_lookup_rsp_data));
        if (descriptor_lookup_pending_) {
            for (unsigned byte = 0; byte < 32; ++byte)
                dut_.descriptor_lookup_rsp_data[byte / 4] |=
                    uint32_t{prediction_records_.at(
                        descriptor_lookup_index_ * 32 + byte)} <<
                    ((byte % 4) * 8);
        }
        dut_.eval();
        const bool lookup_request = dut_.descriptor_lookup_valid &&
            dut_.descriptor_lookup_ready;
        const bool lookup_response = dut_.descriptor_lookup_rsp_valid &&
            dut_.descriptor_lookup_rsp_ready;
        const unsigned requested_index = dut_.descriptor_lookup_index;
        dut_.clk = 1;
        dut_.eval();
        if (lookup_response)
            descriptor_lookup_pending_ = false;
        if (lookup_request) {
            descriptor_lookup_pending_ = true;
            descriptor_lookup_index_ = requested_index;
        }
        dut_.clk = 0;
        dut_.eval();
    }

    void clear_inputs() {
        dut_.start_valid = 0;
        dut_.abort_request = 0;
        dut_.dma_request_ready = 0;
        dut_.dma_response_valid = 0;
        dut_.dma_response_byte_enable = 0;
        dut_.dma_response_last = 0;
        dut_.dma_done_pulse = 0;
        dut_.dma_error = 0;
        dut_.descriptor_ready = 1;
        dut_.descriptor_lookup_ready = 1;
        dut_.descriptor_lookup_rsp_valid = 0;
        std::memset(&dut_.descriptor_lookup_rsp_data, 0,
                    sizeof(dut_.descriptor_lookup_rsp_data));
        descriptor_lookup_pending_ = false;
        dut_.hidden_start_ready = 1;
        dut_.hidden_done_valid = 0;
        dut_.hidden_error = 0;
        dut_.rms_start_ready = 1;
        dut_.rms_done_valid = 0;
        dut_.rms_error = 0;
        dut_.quant_start_ready = 1;
        dut_.quant_done_valid = 0;
        dut_.quant_error = 0;
        dut_.done_ready = 0;
    }

    static std::vector<uint8_t> make_post_config() {
        std::vector<uint8_t> bytes(192, 0);
        put_u32(bytes, 0, 0x31434250);
        put_u16(bytes, 4, 1);
        put_u16(bytes, 6, 192);
        put_u32(bytes, 8, 192);
        put_u16(bytes, 14, 64);
        put_u32(bytes, 16, 126464);
        put_u16(bytes, 20, 4096);
        put_u16(bytes, 22, 64);
        put_u32(bytes, 24, 1);
        put_u16(bytes, 28, 1);
        put_u16(bytes, 30, 1);
        for (unsigned region = 0; region < 7; ++region) {
            const unsigned offset = 32 + region * 16;
            put_u64(bytes, offset, 0x01000000ull + region * 0x00100000ull);
            put_u64(bytes, offset + 8,
                    0x01080000ull + region * 0x00100000ull);
        }
        put_u32(bytes, 144, 126336);
        put_u32(bytes, 148, 7);
        put_u16(bytes, 152, 0x3f00);
        put_u16(bytes, 154, 0x3e80);
        put_u16(bytes, 156, 0x3e00);
        put_u16(bytes, 158, 0x3f00);
        put_u16(bytes, 160, 0x3d80);
        put_u16(bytes, 162, 0x3f80);
        put_u16(bytes, 164, 128);
        put_u16(bytes, 166, 32);
        put_u16(bytes, 168, 0x3a83);
        put_u16(bytes, 170, 32);
        put_u16(bytes, 172, 4);
        put_u16(bytes, 174, 16);
        put_u64(bytes, 176, 0x02000000);
        put_u64(bytes, 184, 0x02001000);
        return bytes;
    }

    static std::vector<uint8_t> make_prediction_records(
            unsigned rows, bool duplicate,
            unsigned locked_row = std::numeric_limits<unsigned>::max()) {
        std::vector<uint8_t> bytes(rows * 32, 0);
        for (unsigned row = 0; row < rows; ++row) {
            const unsigned offset = row * 32;
            put_u16(bytes, offset, 100 + row);
            bytes[offset + 2] = row;
            bytes[offset + 3] = row & 3;
            bytes[offset + 4] = row & 63;
            bytes[offset + 5] = row & 31;
            bytes[offset + 6] = 1;
            bytes[offset + 7] = row == locked_row ? 2 : row % 2;
            put_u32(bytes, offset + 8, row);
            put_u32(bytes, offset + 12, row + 1);
            bytes[offset + 16] = row < 48 ? 0 : 1;
            bytes[offset + 17] = row % 48;
            bytes[offset + 22] = row % 2;
            bytes[offset + 23] = row % 2;
            put_u16(bytes, offset + 18, 1);
            put_u16(bytes, offset + 20,
                    duplicate && row == 1 ? 0 : row * 2);
            put_u32(bytes, offset + 24, row);
            put_u32(bytes, offset + 28, row + 1);
        }
        return bytes;
    }

    void wait_for_request(uint64_t address, uint32_t bytes, uint8_t tag) {
        for (unsigned cycle = 0; cycle < 64; ++cycle) {
            dut_.dma_request_ready = 1;
            dut_.eval();
            if (dut_.dma_request_valid) {
                if (dut_.dma_request_address != address ||
                    dut_.dma_request_bytes != bytes ||
                    dut_.dma_request_tag != tag)
                    throw std::runtime_error("final-output DMA request mismatch");
                tick();
                dut_.dma_request_ready = 0;
                return;
            }
            tick();
        }
        throw std::runtime_error("final-output DMA request timeout");
    }

    void stream_bytes(const std::vector<uint8_t> &bytes) {
        for (unsigned beat = 0; beat < bytes.size() / 16; ++beat) {
            set_beat(dut_.dma_response_data, bytes.data() + beat * 16);
            dut_.dma_response_valid = 1;
            dut_.dma_response_byte_enable = 0xffff;
            dut_.dma_response_last = beat + 1 == bytes.size() / 16;
            dut_.dma_done_pulse = dut_.dma_response_last;
            dut_.eval();
            if (!dut_.dma_response_ready)
                throw std::runtime_error("final-output DMA response stalled");
            tick();
        }
        dut_.dma_response_valid = 0;
        dut_.dma_response_byte_enable = 0;
        dut_.dma_response_last = 0;
        dut_.dma_done_pulse = 0;
    }

    void expect_descriptor(const uint8_t *expected) {
        for (unsigned byte = 0; byte < 32; ++byte) {
            const uint8_t actual = get_bits(
                dut_.descriptor_data, byte * 8, 8);
            if (actual != expected[byte])
                throw std::runtime_error(
                    "final-output descriptor raw field mismatch");
        }
    }

    void stream_prediction_records(const std::vector<uint8_t> &bytes,
                                   unsigned first_beat = 0,
                                   unsigned invalid_row =
                                       std::numeric_limits<unsigned>::max(),
                                   bool apply_stall = false,
                                   bool invalid_byte_enable = false) {
        prediction_records_ = bytes;
        const unsigned beats = bytes.size() / 16;
        for (unsigned beat = first_beat; beat < beats; ++beat) {
            const unsigned row = beat / 2;
            const bool second_half = (beat & 1u) != 0;
            const bool expected_valid = second_half && row < invalid_row;
            set_beat(dut_.dma_response_data, bytes.data() + beat * 16);
            dut_.dma_response_valid = 1;
            dut_.dma_response_byte_enable =
                invalid_byte_enable && row == invalid_row && second_half ?
                    0x7fff : 0xffff;
            dut_.dma_response_last = beat + 1 == beats;
            dut_.dma_done_pulse = 0;
            dut_.descriptor_ready = row < invalid_row;

            if (apply_stall && row == 3 && second_half) {
                dut_.descriptor_ready = 0;
                for (unsigned stalled = 0; stalled < 3; ++stalled) {
                    dut_.eval();
                    if (!dut_.descriptor_valid || dut_.dma_response_ready)
                        throw std::runtime_error(
                            "descriptor backpressure was not propagated");
                    expect_descriptor(bytes.data() + row * 32);
                    tick();
                }
                dut_.descriptor_ready = 1;
            }

            dut_.dma_done_pulse = dut_.dma_response_last;
            dut_.eval();
            if (!dut_.dma_response_ready)
                throw std::runtime_error(
                    "final-output prediction DMA response stalled");
            if (bool(dut_.descriptor_valid) != expected_valid)
                throw std::runtime_error(
                    "final-output descriptor valid mismatch at beat " +
                    std::to_string(beat) + " row " +
                    std::to_string(row) + " actual " +
                    std::to_string(unsigned(dut_.descriptor_valid)) +
                    " expected " + std::to_string(unsigned(expected_valid)));
            if (expected_valid)
                expect_descriptor(bytes.data() + row * 32);
            tick();
        }
        dut_.dma_response_valid = 0;
        dut_.dma_response_byte_enable = 0;
        dut_.dma_response_last = 0;
        dut_.dma_done_pulse = 0;
        dut_.descriptor_ready = 1;
    }

    void start(unsigned rows) {
        dut_.start_post_config_base = kPostBase;
        dut_.start_post_config_limit = kPostBase + 192;
        dut_.start_prediction_base = kPredictionBase;
        dut_.start_prediction_limit = kPredictionBase + rows * 32;
        dut_.start_prediction_count = rows;
        dut_.start_final_hidden_base = kHiddenBase;
        dut_.start_final_hidden_limit = kHiddenBase + 2048ull * 8192ull;
        dut_.start_valid = 1;
        dut_.eval();
        if (!dut_.start_ready)
            throw std::runtime_error("final-output start was not ready");
        tick();
        dut_.start_valid = 0;
    }

    void run_children(unsigned rows) {
        const unsigned groups = (rows + 7) / 8;
        unsigned hidden_groups = 0;
        unsigned rms_groups = 0;
        unsigned quant_groups = 0;

        for (unsigned cycle = 0; cycle < 12000; ++cycle) {
            dut_.eval();
            const bool hidden_start = dut_.hidden_start_valid &&
                dut_.hidden_start_ready;
            const bool rms_start = dut_.rms_start_valid && dut_.rms_start_ready;
            const bool quant_start = dut_.quant_start_valid &&
                dut_.quant_start_ready;
            const bool hidden_done = dut_.hidden_done_valid &&
                dut_.hidden_done_ready;
            const bool rms_done = dut_.rms_done_valid && dut_.rms_done_ready;
            const bool quant_done = dut_.quant_done_valid &&
                dut_.quant_done_ready;

            if (hidden_start) {
                const unsigned expected_rows =
                    hidden_groups + 1 == groups ? rows - hidden_groups * 8 : 8;
                if (dut_.hidden_start_rows != expected_rows ||
                    dut_.hidden_start_bytes != expected_rows * 8192)
                    throw std::runtime_error("hidden group shape mismatch");
                for (unsigned lane = 0; lane < expected_rows; ++lane) {
                    const unsigned prediction = hidden_groups * 8 + lane;
                    if (get_bits(dut_.hidden_start_ddr_row_index,
                                 lane * 11, 11) != prediction * 2 ||
                        get_bits(dut_.hidden_start_source_round,
                                 lane * 7, 7) != (prediction < 48 ? 0 : 1) ||
                        get_bits(dut_.hidden_start_source_row,
                                 lane * 6, 6) != prediction % 48)
                        throw std::runtime_error("hidden source mapping mismatch");
                }
                ++hidden_groups;
            }
            if (rms_start) {
                if (!dut_.rms_start_gamma_bypass ||
                    dut_.rms_start_elements != 4096)
                    throw std::runtime_error("RMSNorm start mismatch");
                ++rms_groups;
            }
            if (quant_start) {
                if (dut_.quant_start_group != quant_groups ||
                    dut_.quant_start_prediction_base != quant_groups * 8)
                    throw std::runtime_error("quantizer group mapping mismatch");
                ++quant_groups;
            }

            tick();
            dut_.hidden_done_valid = hidden_start;
            dut_.rms_done_valid = rms_start;
            dut_.quant_done_valid = quant_start;
            if (hidden_done) dut_.hidden_done_valid = 0;
            if (rms_done) dut_.rms_done_valid = 0;
            if (quant_done) dut_.quant_done_valid = 0;

            if (dut_.done_valid) {
                if (dut_.error || hidden_groups != groups ||
                    rms_groups != groups || quant_groups != groups)
                    throw std::runtime_error("final-output group completion mismatch");
                dut_.done_ready = 1;
                tick();
                dut_.done_ready = 0;
                return;
            }
        }
        throw std::runtime_error("final-output group schedule timeout");
    }

    void load_prediction_table(unsigned rows,
                               const std::vector<uint8_t> &records,
                               unsigned invalid_row =
                                   std::numeric_limits<unsigned>::max(),
                               bool apply_stall = false) {
        start(rows);
        wait_for_request(kPostBase, 192, 0xb0);
        stream_bytes(make_post_config());
        wait_for_request(kPredictionBase, rows * 32, 0xb1);
        stream_prediction_records(
            records, 0, invalid_row, apply_stall);
    }

    void release_abort() {
        if (!dut_.abort_ack)
            throw std::runtime_error("final-output abort was not acknowledged");
        if (dut_.dma_request_valid || dut_.hidden_start_valid ||
            dut_.rms_start_valid || dut_.quant_start_valid)
            throw std::runtime_error("new work appeared with abort acknowledged");
        tick();
        if (dut_.abort_ack)
            throw std::runtime_error("final-output abort acknowledge was not a pulse");
        dut_.abort_request = 0;
        tick();
        dut_.eval();
        if (!dut_.start_ready)
            throw std::runtime_error("final-output did not restart after abort");
    }

    void run_locked_case() {
        constexpr unsigned kLockedRow = 5;
        const auto records = make_prediction_records(16, false, kLockedRow);
        load_prediction_table(16, records);
        run_children(16);
    }

    void run_invalid_case(unsigned invalid_row, bool invalid_byte_enable) {
        auto records = make_prediction_records(16, false);
        if (!invalid_byte_enable)
            records[invalid_row * 32 + 7] = 3;
        start(16);
        wait_for_request(kPostBase, 192, 0xb0);
        stream_bytes(make_post_config());
        wait_for_request(kPredictionBase, 16 * 32, 0xb1);
        stream_prediction_records(records, 0, invalid_row, false,
                                  invalid_byte_enable);
        for (unsigned cycle = 0; cycle < 512; ++cycle) {
            if (!dut_.done_valid) {
                tick();
                continue;
            }
            if (!dut_.error || dut_.error_id != (invalid_byte_enable ? 0x04 : 0x05) ||
                dut_.hidden_start_valid)
                throw std::runtime_error("malformed prediction was not rejected");
            dut_.done_ready = 1;
            tick();
            dut_.done_ready = 0;
            return;
        }
        throw std::runtime_error("malformed-prediction drain timeout");
    }

    void run_post_dma_abort() {
        const auto bytes = make_post_config();
        start(16);
        wait_for_request(kPostBase, 192, 0xb0);

        set_beat(dut_.dma_response_data, bytes.data());
        dut_.dma_response_valid = 1;
        dut_.dma_response_byte_enable = 0xffff;
        dut_.dma_response_last = 0;
        tick();

        dut_.abort_request = 1;
        dut_.eval();
        if (dut_.abort_ack || dut_.abort_wait_status != 1)
            throw std::runtime_error("post DMA abort wait status mismatch");
        for (unsigned beat = 1; beat < bytes.size() / 16; ++beat) {
            set_beat(dut_.dma_response_data, bytes.data() + beat * 16);
            dut_.dma_response_last = beat + 1 == bytes.size() / 16;
            dut_.dma_done_pulse = dut_.dma_response_last;
            dut_.eval();
            if (!dut_.dma_response_ready)
                throw std::runtime_error("post DMA did not drain during abort");
            tick();
        }
        dut_.dma_response_valid = 0;
        dut_.dma_response_byte_enable = 0;
        dut_.dma_response_last = 0;
        dut_.dma_done_pulse = 0;
        if (dut_.dma_request_valid)
            throw std::runtime_error("prediction DMA started after post abort");
        release_abort();
    }

    void run_prediction_dma_abort() {
        const auto records = make_prediction_records(16, false);
        start(16);
        wait_for_request(kPostBase, 192, 0xb0);
        stream_bytes(make_post_config());
        wait_for_request(kPredictionBase, 16 * 32, 0xb1);

        set_beat(dut_.dma_response_data, records.data());
        dut_.dma_response_valid = 1;
        dut_.dma_response_byte_enable = 0xffff;
        dut_.dma_response_last = 0;
        tick();

        set_beat(dut_.dma_response_data, records.data() + 16);
        dut_.descriptor_ready = 0;
        dut_.abort_request = 1;
        dut_.eval();
        if (!dut_.descriptor_valid || dut_.dma_response_ready ||
            dut_.abort_wait_status != 6 || dut_.abort_ack)
            throw std::runtime_error("prediction descriptor abort stall mismatch");
        expect_descriptor(records.data());
        tick();
        dut_.eval();
        if (!dut_.descriptor_valid || dut_.dma_response_ready ||
            dut_.abort_wait_status != 6)
            throw std::runtime_error("prediction descriptor changed during abort");
        expect_descriptor(records.data());

        dut_.descriptor_ready = 1;
        dut_.eval();
        if (!dut_.dma_response_ready)
            throw std::runtime_error("prediction descriptor did not resume");
        tick();
        stream_prediction_records(records, 2);
        if (dut_.hidden_start_valid)
            throw std::runtime_error("hidden stage started after prediction abort");
        release_abort();
    }

    void wait_for_hidden_start() {
        for (unsigned cycle = 0; cycle < 12000; ++cycle) {
            dut_.eval();
            if (dut_.hidden_start_valid) {
                tick();
                return;
            }
            tick();
        }
        throw std::runtime_error("hidden stage start timeout");
    }

    void run_stage_abort(unsigned wait_status) {
        const auto records = make_prediction_records(16, false);
        load_prediction_table(16, records);
        wait_for_hidden_start();

        if (wait_status >= 4) {
            dut_.hidden_done_valid = 1;
            tick();
            dut_.hidden_done_valid = 0;
            for (unsigned cycle = 0; cycle < 16; ++cycle) {
                dut_.eval();
                if (dut_.rms_start_valid) {
                    tick();
                    break;
                }
                tick();
            }
        }
        if (wait_status == 5) {
            dut_.rms_done_valid = 1;
            tick();
            dut_.rms_done_valid = 0;
            for (unsigned cycle = 0; cycle < 16; ++cycle) {
                dut_.eval();
                if (dut_.quant_start_valid) {
                    tick();
                    break;
                }
                tick();
            }
        }

        dut_.abort_request = 1;
        dut_.eval();
        if (dut_.abort_ack || dut_.abort_wait_status != wait_status)
            throw std::runtime_error("stage abort wait status mismatch");
        tick();
        if (dut_.abort_ack)
            throw std::runtime_error("stage abort acknowledged before done");

        if (wait_status == 3)
            dut_.hidden_done_valid = 1;
        else if (wait_status == 4)
            dut_.rms_done_valid = 1;
        else
            dut_.quant_done_valid = 1;
        tick();
        dut_.hidden_done_valid = 0;
        dut_.rms_done_valid = 0;
        dut_.quant_done_valid = 0;
        release_abort();
    }

    void run_case(unsigned rows, bool duplicate) {
        const auto records = make_prediction_records(rows, duplicate);
        load_prediction_table(rows, records,
            std::numeric_limits<unsigned>::max(), !duplicate);

        if (!duplicate) {
            run_children(rows);
            return;
        }
        for (unsigned cycle = 0; cycle < 512; ++cycle) {
            tick();
            if (!dut_.done_valid)
                continue;
            if (!dut_.error || dut_.error_id != 0x06 ||
                dut_.hidden_start_valid)
                throw std::runtime_error("duplicate source was not rejected");
            dut_.done_ready = 1;
            tick();
            dut_.done_ready = 0;
            return;
        }
        throw std::runtime_error("duplicate-source check timeout");
    }
};

}  // namespace

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Regression regression;
        regression.run();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL final_output_sequencer: " << error.what() << '\n';
        return 1;
    }
}
