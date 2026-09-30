#include "Vmatmul_activation_controller.h"
#include "verilated.h"

#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class Regression {
    Vmatmul_activation_controller dut;
    std::deque<unsigned> source, maxima, quantized;
    bool scale_pending = false;
    unsigned max_sent = 0, max_received = 0, scale_received = 0, output_received = 0;

    void tick() {
        dut.clk = 0; dut.eval();
        dut.clk = 1; dut.eval();
        dut.clk = 0; dut.eval();
    }

    void start() {
        require(source.empty() && maxima.empty() && quantized.empty() && !scale_pending,
                "restart has an unconsumed response");
        dut.abort_request = 0;
        dut.start_valid = 1;
        dut.eval();
        require(dut.start_ready, "restart was not accepted");
        tick();
        dut.start_valid = 0;
        max_sent = max_received = scale_received = output_received = 0;
    }

    // Model ordered responses with independently controlled return timing.
    void step(bool return_source, bool return_max, bool abort) {
        dut.abort_request = abort;
        dut.source_response_valid = return_source && !source.empty();
        dut.max_rsp_valid = return_max && !maxima.empty();
        dut.max_rsp_tag = maxima.empty() ? 0 : maxima.front();
        dut.max_rsp_row_mask = 7;
        for (unsigned i = 0; i < 4; ++i) dut.max_rsp_values[i] = 0x40004000;
        dut.source_response_lane_mask = 0x00ffffff;
        for (unsigned i = 0; i < 32; ++i) dut.source_response_values[i] = 0x3f803f80;
        dut.quant_scale_rsp_valid = scale_pending && !abort;
        for (unsigned i = 0; i < 4; ++i) dut.quant_scale_rsp_values_bf16[i] = 0x3c813c81;
        dut.quant_values_rsp_valid = !quantized.empty() && !abort;
        dut.quant_values_rsp_tag = quantized.empty() ? 0 : quantized.front();
        dut.quant_values_rsp_lane_mask = 0x00ffffff;
        for (unsigned i = 0; i < 16; ++i) dut.quant_values_rsp_values[i] = 0x40404040;
        dut.eval();
        const bool src_req = dut.source_request_valid && dut.source_request_ready;
        const bool src_rsp = dut.source_response_valid && dut.source_response_ready;
        const bool max_req = dut.max_req_valid && dut.max_req_ready;
        const bool max_rsp = dut.max_rsp_valid && dut.max_rsp_ready;
        const bool scale_req = dut.quant_scale_req_valid && dut.quant_scale_req_ready;
        const bool scale_rsp = dut.scale_valid && dut.scale_ready;
        const bool quant_req = dut.quant_values_req_valid && dut.quant_values_req_ready;
        const bool quant_rsp = dut.quantized_valid && dut.quantized_ready;
        const unsigned element = dut.source_request_element;
        const unsigned max_tag = dut.max_req_tag;
        const unsigned quant_tag = dut.quant_values_req_tag;
        if (abort) require(!src_req && !max_req && !scale_req && !quant_req,
                           "request accepted during abort");
        if (scale_req) require(maxima.empty(), "scale requested before max responses drained");
        if (scale_rsp) {
            require(dut.scale_row_base == 4 && dut.scale_row_mask == 7,
                    "restart scale metadata mismatch");
            require(dut.scale_values_bf16[0] == 0x3c813c81, "restart scale payload mismatch");
        }
        if (quant_rsp) {
            require(dut.quantized_row_base == 4 && dut.quantized_tag == output_received &&
                    dut.quantized_element == output_received * 8 &&
                    dut.quantized_lane_mask == 0x00ffffff &&
                    dut.quantized_values[0] == 0x40404040, "restart quantized response mismatch");
        }
        tick();
        if (src_rsp) source.pop_front();
        if (src_req) source.push_back(element);
        if (max_rsp) { maxima.pop_front(); ++max_received; }
        if (max_req) { maxima.push_back(max_tag); ++max_sent; }
        if (scale_rsp) { scale_pending = false; ++scale_received; }
        if (scale_req) scale_pending = true;
        if (quant_rsp) { quantized.pop_front(); ++output_received; }
        if (quant_req) quantized.push_back(quant_tag);
        if (abort) { scale_pending = false; quantized.clear(); }
        require(!dut.error, "controller reported an error");
        if (dut.abort_ack)
            require(source.empty() && maxima.empty(), "abort acknowledged before accepted responses drained");
    }

    void finish() {
        bool complete = false;
        for (unsigned cycle = 0; cycle < 100; ++cycle) {
            step(true, true, false);
            if (dut.done_pulse) { complete = true; break; }
        }
        require(complete, "restart did not complete");
        require(max_sent == 3 && max_received == 3 && scale_received == 1 && output_received == 3,
                "restart accepted-event counts mismatch");
        require(dut.accepted_source_tiles == 6 && dut.accepted_max_tiles == 3 &&
                dut.accepted_quantized_tiles == 3, "restart counters mismatch");
        step(true, true, false);
    }

public:
    Regression() {
        dut.rst = 1;
        dut.source_request_ready = dut.max_req_ready = dut.quant_scale_req_ready = 1;
        dut.quant_values_req_ready = dut.scale_ready = dut.quantized_ready = 1;
        dut.row_base = 4; dut.row_count = 3; dut.element_count = 24;
        tick(); tick(); dut.rst = 0; tick();
    }

    void run() {
        // Vary max returns across the abort edge, and retain a delayed SRAM return.
        for (unsigned scenario = 0; scenario < 3; ++scenario) {
            start();
            for (unsigned cycle = 0; max_sent < 2 && cycle < 30; ++cycle)
                step(max_sent == 0, scenario == 2 && max_sent != 0, false);
            require(max_sent == 2 && !maxima.empty(), "abort did not target outstanding max requests");
            require(!source.empty(), "abort did not target an outstanding source request");
            step(false, scenario == 1, true);
            for (unsigned cycle = 0; cycle < 4; ++cycle) step(false, false, true);
            require(!dut.abort_ack, "abort completed while responses were withheld");
            bool acknowledged = false;
            for (unsigned cycle = 0; cycle < 20; ++cycle) {
                step(scenario == 0 || cycle >= 3, scenario != 0 || cycle >= 3, true);
                if (dut.abort_ack) { acknowledged = true; break; }
            }
            require(acknowledged, "abort did not drain outstanding max/source responses");
            for (unsigned cycle = 0; cycle < 3; ++cycle) {
                step(true, true, true);
                require(!dut.start_ready && !dut.abort_ack, "held abort accepted a restart or repeated acknowledgment");
            }
            start();
            finish();
        }
        dut.final();
        std::cout << "PASS matmul_activation_controller: abort_cases=3 restart_cases=3 "
                     "delayed_max_and_source=1 abort_edge_max_return=1 simultaneous_max_request_return=1 "
                     "rows=3 elements=24 runtime_threads=" << dut.threads() << '\n';
    }
};
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try { Regression test; test.run(); }
    catch (const std::exception& error) {
        std::cerr << "FAIL matmul_activation_controller: " << error.what() << '\n';
        return 1;
    }
}
