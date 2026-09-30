#include "Vsoftmax_engine_start_top.h"
#include "verilated.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

double sc_time_stamp() { return 0.0; }

namespace {
class Regression {
public:
    Regression() {
        dut_.clk = 0;
        dut_.rst = 1;
        dut_.abort_request = 0;
        dut_.start_valid = 0;
        dut_.start_row_count = 1;
        dut_.start_head_count = 1;
        dut_.start_row_length = 8;
        dut_.tile_read_req_ready = 0;
        tick();
        tick();
        dut_.rst = 0;
        tick();
    }

    void run() {
        dut_.abort_request = 1;
        dut_.start_valid = 1;
        for (unsigned cycle = 0; cycle < 3; ++cycle) {
            dut_.eval();
            if (dut_.start_ready || dut_.busy || dut_.done || dut_.error ||
                dut_.tile_read_req_valid || dut_.bf16_write_valid ||
                dut_.quantized_write_valid || dut_.scale_write_valid ||
                dut_.completed_command_count != 0)
                throw std::runtime_error(
                    "Softmax engine acted on a blocked start");
            tick();
        }

        dut_.abort_request = 0;
        dut_.eval();
        if (!dut_.start_ready)
            throw std::runtime_error(
                "Softmax engine did not accept held start after abort");
        tick();
        dut_.start_valid = 0;
        if (!dut_.busy || dut_.start_ready)
            throw std::runtime_error(
                "Softmax engine did not accept held start exactly once");

        tick();
        dut_.abort_request = 1;
        bool abort_seen = false;
        for (unsigned cycle = 0; cycle < 16; ++cycle) {
            dut_.eval();
            if (dut_.abort_ack) {
                abort_seen = true;
                break;
            }
            tick();
        }
        if (!abort_seen)
            throw std::runtime_error(
                "Softmax engine did not acknowledge focused abort");

        // Attention keeps abort asserted until its children report drained.
        // The acknowledged engine must be non-busy even before abort goes low;
        // it still must not accept another start or issue a memory operation.
        for (unsigned cycle = 0; cycle < 4; ++cycle) {
            dut_.eval();
            if (dut_.busy || dut_.start_ready || dut_.done ||
                dut_.tile_read_req_valid || dut_.bf16_write_valid ||
                dut_.quantized_write_valid || dut_.scale_write_valid)
                throw std::runtime_error(
                    "Softmax acknowledged abort but still blocks parent drain");
            tick();
            if (dut_.abort_ack)
                throw std::runtime_error("Softmax repeated held-abort acknowledgement");
        }

        dut_.abort_request = 0;
        tick();
        if (dut_.busy || !dut_.start_ready || dut_.done || dut_.error ||
            dut_.completed_command_count != 0)
            throw std::runtime_error(
                "Softmax engine did not return to idle after focused abort");

        run_normal_case(47, 1);
        run_normal_case(48, 2);
        start(1, 1);
        dut_.abort_request = 1;
        tick();
        if (!dut_.abort_ack || dut_.busy || dut_.tile_read_req_valid)
            throw std::runtime_error("Softmax pre-batch abort waited for an idle child");
        dut_.abort_request = 0;
        tick();
        run_configuration_error_case();

        dut_.final();
        std::cout << "PASS softmax_engine_start_handshake: blocked_cycles=3"
                  << " accepted_after_abort=1 duplicate_accept=0"
                  << " abort_drain=1 pre_batch_abort=1 held_abort_drained=1 normal_commands=2"
                  << " row_counts=47/48 heads=1/2"
                  << " batch_row_bases=0/8/16/24/32/40"
                  << " query_32_47=PASS address_mismatch=0"
                  << " configuration_error=PASS\n";
    }

private:
    Vsoftmax_engine_start_top dut_;

    void start(unsigned rows, unsigned heads) {
        dut_.start_row_count = rows;
        dut_.start_head_count = heads;
        dut_.start_row_length = 24;
        dut_.start_valid = 1;
        dut_.eval();
        if (!dut_.start_ready)
            throw std::runtime_error("Softmax engine normal start was not ready");
        tick();
        dut_.start_valid = 0;
    }

    void run_normal_case(unsigned rows, unsigned heads) {
        const uint64_t command_count_before = dut_.completed_command_count;
        start(rows, heads);
        const unsigned batches_per_head = (rows + 7U) / 8U;
        const unsigned expected_requests = heads * batches_per_head;
        unsigned request_count = 0;
        bool stalled_request = false;
        uint32_t stalled_group_base = 0;
        uint8_t stalled_head = 0;
        uint8_t stalled_row_base = 0;

        for (unsigned cycle = 0; cycle < 1000; ++cycle) {
            dut_.tile_read_req_ready = (cycle % 3U) != 0U;
            dut_.eval();
            if (stalled_request) {
                if (!dut_.tile_read_req_valid ||
                    dut_.tile_read_group_base != stalled_group_base ||
                    dut_.tile_read_head != stalled_head ||
                    dut_.tile_read_row_base != stalled_row_base)
                    throw std::runtime_error(
                        "Softmax engine changed a stalled batch request");
                stalled_request = false;
            }
            if (dut_.tile_read_req_valid && !dut_.tile_read_req_ready) {
                stalled_request = true;
                stalled_group_base = dut_.tile_read_group_base;
                stalled_head = dut_.tile_read_head;
                stalled_row_base = dut_.tile_read_row_base;
            }
            if (dut_.tile_read_req_valid && dut_.tile_read_req_ready) {
                const unsigned expected_head = request_count / batches_per_head;
                const unsigned expected_batch = request_count % batches_per_head;
                const unsigned expected_row_base = expected_batch * 8U;
                const uint32_t expected_group_base = 0x010000U +
                    expected_head * 0x003000U + expected_row_base * 0x000100U;
                if (dut_.tile_read_pass != 0 || dut_.tile_read_key != 0 ||
                    dut_.tile_read_tag != 0xa55a ||
                    dut_.tile_read_head != expected_head ||
                    dut_.tile_read_row_base != expected_row_base ||
                    dut_.tile_read_group_base != expected_group_base)
                    throw std::runtime_error(
                        "Softmax batch metadata mismatch at request " +
                        std::to_string(request_count));
                ++request_count;
            }
            const bool done = dut_.done;
            tick();
            if (done) {
                if (dut_.error || request_count != expected_requests ||
                    dut_.completed_command_count != command_count_before + 1U)
                    throw std::runtime_error(
                        "Softmax engine normal completion mismatch");
                dut_.tile_read_req_ready = 0;
                tick();
                if (!dut_.start_ready)
                    throw std::runtime_error(
                        "Softmax engine did not return to idle after completion");
                return;
            }
        }
        throw std::runtime_error("Softmax engine normal case timeout");
    }

    void run_configuration_error_case() {
        const uint64_t command_count_before = dut_.completed_command_count;
        dut_.start_row_count = 49;
        dut_.start_head_count = 1;
        dut_.start_row_length = 24;
        dut_.start_valid = 1;
        dut_.eval();
        if (!dut_.start_ready)
            throw std::runtime_error(
                "Softmax invalid-configuration start was not ready");
        tick();
        dut_.start_valid = 0;
        if (!dut_.done || !dut_.error || dut_.error_id != 1 || dut_.busy ||
            dut_.tile_read_req_valid ||
            dut_.completed_command_count != command_count_before)
            throw std::runtime_error(
                "Softmax row_count=49 did not report configuration error");
        tick();
        if (!dut_.start_ready)
            throw std::runtime_error(
                "Softmax engine did not recover from configuration error");
    }

    void tick() {
        dut_.clk = 0;
        dut_.eval();
        dut_.clk = 1;
        dut_.eval();
        dut_.clk = 0;
        dut_.eval();
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
        std::cerr << "FAIL softmax_engine_start_handshake: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
