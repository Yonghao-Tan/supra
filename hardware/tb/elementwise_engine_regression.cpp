#include "Velementwise_engine.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Ordered arithmetic responses isolate accepted-event and drain behavior.
// Product SRAM retains the exact raw payload written by the controller.
class Regression {
    Velementwise_engine dut;
    using Product = std::array<std::uint32_t, 17>;
    std::deque<unsigned> arithmetic, maxima, quantized;
    std::deque<Product> products;
    std::map<unsigned, Product> scratch;
    bool dma_pending = false, scale_pending = false;
    unsigned dma_tag = 0, dma_requests = 0, output_tiles = 0;
    bool writer_complete = false;

    void tick() {
        dut.clk = 0; dut.eval();
        dut.clk = 1; dut.eval();
        dut.clk = 0; dut.eval();
    }

    bool drained() const {
        return !dma_pending && arithmetic.empty() && maxima.empty() &&
            products.empty() && quantized.empty() && !scale_pending;
    }

    void start() {
        require(drained(), "restart has an unconsumed response");
        dut.abort_request = dut.read_request_error = dut.writer_error = 0;
        dut.read_data_valid = dut.read_request_done = 0;
        dut.arithmetic_rsp_valid = dut.max_rsp_valid = 0;
        dut.product_memory_rsp_valid = dut.quant_scale_rsp_valid = dut.quant_values_rsp_valid = 0;
        dut.writer_done_pulse = 0;
        dut.start_valid = 1;
        dut.eval();
        require(dut.start_ready, "controller did not accept restart");
        tick();
        dut.start_valid = 0;
        dma_requests = output_tiles = 0;
        writer_complete = false;
        scratch.clear();
    }

    struct Control {
        bool hold_prefetch = false;
        bool hold_max = false;
        bool hold_product = false;
        bool hold_arithmetic = false;
        bool hold_scale = false;
        bool hold_quantized = false;
        bool read_error = false;
        bool write_error = false;
        bool abort = false;
    };

    void step() { step(Control{}); }

    void step(Control control) {
        dut.abort_request = control.abort;
        dut.read_request_error = control.read_error;
        dut.writer_error = control.write_error;
        require(!control.read_error || dma_pending, "read error had no accepted DMA request");
        dut.read_data_valid = dma_pending && !control.read_error &&
            !(control.hold_prefetch && dma_requests >= 3);
        dut.read_request_done = dut.read_data_valid;
        dut.read_data_tag = dma_tag;
        dut.read_data_last = 1;
        dut.read_data_span = 0;
        dut.read_byte_enable = 0xffff;
        for (unsigned i = 0; i < 4; ++i) dut.read_data[i] = 0x3f803f80;
        dut.arithmetic_rsp_valid = !control.hold_arithmetic && !arithmetic.empty();
        dut.arithmetic_rsp_tag = arithmetic.empty() ? 0 : arithmetic.front();
        dut.arithmetic_rsp_lane_mask = 0xff;
        for (unsigned i = 0; i < 32; ++i) dut.arithmetic_rsp_values[i] = 0x3f003f00;
        dut.max_rsp_valid = !control.hold_max && !maxima.empty();
        dut.max_rsp_tag = maxima.empty() ? 0 : maxima.front();
        dut.max_rsp_row_mask = 1;
        dut.max_rsp_values[0] = 0x3f00;
        dut.quant_scale_rsp_valid = !control.hold_scale && scale_pending;
        dut.quant_scale_rsp_values_bf16[0] = 0x3c00;
        dut.quant_values_rsp_valid = !control.hold_quantized && !quantized.empty();
        dut.quant_values_rsp_tag = quantized.empty() ? 0 : quantized.front();
        dut.quant_values_rsp_lane_mask = 0xff;
        for (unsigned i = 0; i < 16; ++i) dut.quant_values_rsp_values[i] = 0x40404040;
        dut.product_memory_rsp_valid = !control.hold_product && !products.empty();
        if (!products.empty())
            for (unsigned i = 0; i < 17; ++i) dut.product_memory_rsp[i] = products.front()[i];
        dut.writer_done_pulse = writer_complete;
        dut.eval();

        const bool read_req = dut.read_request_valid && dut.read_request_ready;
        const bool read_rsp = dut.read_data_valid && dut.read_data_ready;
        const unsigned read_tag = dut.read_request_tag;
        const bool ar_req = dut.arithmetic_req_valid && dut.arithmetic_req_ready;
        const bool ar_rsp = dut.arithmetic_rsp_valid && dut.arithmetic_rsp_ready;
        const unsigned ar_tag = dut.arithmetic_req_tag;
        const bool max_req = dut.max_req_valid && dut.max_req_ready;
        const bool max_rsp = dut.max_rsp_valid && dut.max_rsp_ready;
        const unsigned max_tag = dut.max_req_tag;
        const bool scale_req = dut.quant_scale_req_valid && dut.quant_scale_req_ready;
        const bool scale_rsp = dut.quant_scale_rsp_valid && dut.quant_scale_rsp_ready;
        const bool quant_req = dut.quant_values_req_valid && dut.quant_values_req_ready;
        const bool quant_rsp = dut.quant_values_rsp_valid && dut.quant_values_rsp_ready;
        const unsigned quant_tag = dut.quant_values_req_tag;
        const bool product_req = dut.product_memory_req_valid && dut.product_memory_req_ready;
        const bool product_rsp = dut.product_memory_rsp_valid && dut.product_memory_rsp_ready;
        const bool product_write = dut.product_memory_req[17] & 1;
        const unsigned word = (dut.product_memory_req[16] >> 17) & 0x7fff;
        Product product{};
        if (product_req) {
            if (product_write) {
                for (unsigned i = 0; i < 17; ++i) product[i] = dut.product_memory_req[i];
                product[16] &= 0x1ffff;
            } else {
                require(scratch.count(word), "product read preceded its write");
                product = scratch.at(word);
                product[0] = (product[0] & 0xfffe0000u) | (dut.product_memory_req[0] & 0x1ffff);
            }
        }
        const bool output = dut.writer_quantized_valid && dut.writer_quantized_ready;
        if (output && !control.write_error && !control.abort) {
            require(dut.writer_quantized_lane_mask == 0xff &&
                    dut.writer_quantized_element_base == output_tiles * 8 &&
                    dut.writer_quantized_values[0] == 0x40404040,
                    "quantized output forwarding mismatch");
        }
        if (read_req) require(!dma_pending, "overlapping DMA requests");
        tick();
        if (read_rsp || control.read_error) dma_pending = false;
        if (read_req) {
            require(!dut.read_second_span_valid, "fixture expects an unpaired one-row read");
            dma_pending = true; dma_tag = read_tag; ++dma_requests;
        }
        if (ar_rsp) arithmetic.pop_front();
        if (ar_req) arithmetic.push_back(ar_tag);
        if (max_rsp) maxima.pop_front();
        if (max_req) maxima.push_back(max_tag);
        if (scale_rsp) scale_pending = false;
        if (scale_req) scale_pending = true;
        if (quant_rsp) quantized.pop_front();
        if (quant_req) quantized.push_back(quant_tag);
        if (product_rsp) products.pop_front();
        if (product_req) {
            if (product_write) scratch[word] = product;
            else products.push_back(product);
        }
        writer_complete = false;
        if (output) { ++output_tiles; writer_complete = output_tiles == 2; }
        if (dut.done_valid || dut.abort_ack)
            require(drained(), "completion preceded accepted-response drain");
    }

    void finish(unsigned expected_error) {
        for (unsigned cycle = 0; !dut.done_valid && cycle < 160; ++cycle) step();
        require(dut.done_valid, "controller hung after accepted responses drained");
        require(dut.error == (expected_error != 0) && dut.error_id == expected_error,
                "completion error status mismatch");
        if (!expected_error)
            require(dma_requests == 4 && output_tiles == 2,
                    "normal restart accepted-event counts mismatch");
        step();
    }

public:
    Regression() {
        dut.rst = 1;
        dut.done_ready = dut.read_request_ready = dut.arithmetic_req_ready = 1;
        dut.max_req_ready = dut.quant_scale_req_ready = dut.quant_values_req_ready = 1;
        dut.writer_cfg_ready = dut.writer_scale_ready = dut.writer_quantized_ready = 1;
        dut.product_memory_req_ready = dut.trace_sample_ready = 1;
        dut.start_active_rows = dut.start_segment_count = dut.start_segment_row_count = 1;
        dut.start_segment_mode = 1;
        dut.start_activation_limit_byte_offset = 1024;
        dut.start_workspace_base = 0xa0000000ull;
        dut.start_workspace_limit = 0xa0001000ull;
        tick(); tick(); dut.rst = 0; tick();
    }

    void run(unsigned scenario) {
        if (scenario >= 6) {
            if (scenario == 6) dut.read_request_ready = 0;
            else dut.writer_quantized_ready = 0;
            start();
            bool stalled = false;
            for (unsigned cycle = 0; cycle < 160; ++cycle) {
                step();
                stalled = scenario == 6 ? dut.read_request_valid : dut.writer_quantized_valid;
                if (stalled) break;
            }
            require(stalled, "did not reach the directed output stall");
            step();
            Control abort;
            abort.abort = true;
            bool acknowledged = false;
            for (unsigned cycle = 0; cycle < 160; ++cycle) {
                step(abort);
                if (dut.abort_ack) { acknowledged = true; break; }
            }
            require(acknowledged, "abort did not drain accepted responses");
            for (unsigned cycle = 0; cycle < 3; ++cycle) {
                step(abort);
                require(!dut.start_ready && !dut.abort_ack, "held abort accepted restart or repeated acknowledgment");
            }
            dut.read_request_ready = dut.writer_quantized_ready = 1;
            step();
            start();
            finish(0);
            dut.final();
            std::cout << "PASS elementwise_engine: scenario=" << scenario
                      << " stalled_abort=1 restart=1 runtime_threads=" << dut.threads() << '\n';
            return;
        }
        start();
        Control hold;
        if (scenario == 0 || scenario == 2) {
            hold.hold_prefetch = true;
            hold.hold_max = scenario == 0;
            hold.hold_arithmetic = scenario == 2;
        } else {
            hold.hold_product = scenario == 1;
            hold.hold_scale = scenario == 3;
            hold.hold_quantized = scenario == 4;
        }
        bool reached = false;
        for (unsigned cycle = 0; cycle < 160; ++cycle) {
            step(hold);
            reached = scenario == 0 ? dma_pending && !maxima.empty() :
                scenario == 1 ? products.size() >= 2 :
                scenario == 2 ? dma_pending && !arithmetic.empty() :
                scenario == 3 ? scale_pending :
                scenario == 4 ? !quantized.empty() :
                dut.product_memory_req_valid && !(dut.product_memory_req[17] & 1);
            if (reached) break;
        }
        require(reached, "did not reach the directed response/error overlap");
        Control inject;
        inject.read_error = scenario == 0 || scenario == 2;
        inject.write_error = !inject.read_error;
        if (scenario == 1) dut.product_memory_req_ready = 0;
        if (scenario == 5) inject.hold_product = true;
        step(inject);
        dut.product_memory_req_ready = 1;
        if (scenario == 5) {
            Control delay;
            delay.hold_product = true;
            for (unsigned cycle = 0; cycle < 3; ++cycle) step(delay);
            require(!dut.done_valid, "error completed while its accepted SRAM read was pending");
        }
        finish(inject.read_error ? 2 : 6);
        start();
        finish(0);
        dut.final();
        std::cout << "PASS elementwise_engine: scenario=" << scenario
                  << " error_drain=1 restart=1 rows=1 features=16 runtime_threads="
                  << dut.threads() << '\n';
    }
};
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        if (argc > 1) { Regression test; test.run(std::stoul(argv[1])); }
        else for (unsigned scenario = 0; scenario < 8; ++scenario) {
            Regression test; test.run(scenario);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL elementwise_engine: " << error.what() << '\n';
        return 1;
    }
}
