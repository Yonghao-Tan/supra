#include "Vmemory_stream_width_adapter.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Beat {
    std::array<uint32_t, 8> data{};
    uint32_t mask = 0xffffffffu;
    bool last = false;
    uint8_t tag = 0x5a;
    bool span = false;
};

static void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<Beat> beats(const std::vector<uint32_t>& masks) {
    std::vector<Beat> result;
    for (size_t i = 0; i < masks.size(); ++i) {
        Beat b;
        for (unsigned w = 0; w < 8; ++w)
            b.data[w] = 0x12345678u ^ uint32_t(i * 0x1020101u + w * 0x18181818u);
        b.mask = masks[i];
        b.last = i + 1 == masks.size();
        result.push_back(b);
    }
    return result;
}

class Test {
public:
    VerilatedContext context;
    Vmemory_stream_width_adapter dut{&context};

    Test() {
        dut.clk = 0;
        dut.rst = 1;
        dut.read_request_valid = 0;
        dut.read_request_address = 0x100000ff0ull;
        dut.read_request_bytes = 64;
        dut.read_request_tag = 0x5a;
        dut.read_request_wide = 0;
        dut.read_second_span_valid = 0;
        dut.read_second_span_address = 0x200001ff0ull;
        dut.read_pair_stride = 0;
        dut.read_pair_count = 1;
        dut.read_data_ready = 0;
        dut.read_wide_data_ready = 0;
        dut.wide_read_request_ready = 0;
        dut.wide_read_request_done = 0;
        dut.wide_read_request_error = 0;
        dut.wide_read_data_valid = 0;
        dut.wide_read_byte_enable = 0;
        dut.wide_read_data_last = 0;
        dut.wide_read_data_tag = 0;
        dut.wide_read_data_span = 0;
        for (unsigned w = 0; w < 8; ++w) dut.wide_read_data[w] = 0;
        dut.write_request_valid = 0;
        dut.write_request_address = 0x300003ff1ull;
        dut.write_request_bytes = 65;
        dut.write_request_tag = 0x39;
        dut.write_data_valid = 0;
        dut.write_byte_enable = 0;
        dut.write_data_last = 0;
        for (unsigned w = 0; w < 4; ++w) dut.write_data[w] = 0;
        dut.wide_write_request_ready = 1;
        dut.wide_write_request_done = 0;
        dut.wide_write_request_error = 0;
        dut.wide_write_data_ready = 0;
        tick();
        tick();
        dut.rst = 0;
        dut.eval();
        check(dut.idle, "reset did not empty adapter");
    }

    void tick() {
        dut.clk = 0;
        dut.eval();
        context.timeInc(1);
        dut.clk = 1;
        dut.eval();
        context.timeInc(1);
        dut.clk = 0;
        dut.eval();
    }

    void request(bool wide, bool paired, uint32_t bytes = 64, unsigned pairs = 3) {
        dut.read_request_bytes = bytes;
        dut.read_request_wide = wide;
        dut.read_second_span_valid = paired;
        dut.read_pair_count = paired ? pairs : 1;
        dut.read_pair_stride = paired ? 0x1000 : 0;
        dut.read_request_valid = 1;
        for (unsigned i = 0; i < 3; ++i) {
            dut.eval();
            check(dut.wide_read_request_valid && !dut.read_request_ready,
                  "request did not remain valid under backpressure");
            check(dut.wide_read_request_address == dut.read_request_address &&
                  dut.wide_read_request_bytes == dut.read_request_bytes &&
                  dut.wide_read_request_tag == dut.read_request_tag &&
                  dut.wide_read_second_span_valid == paired &&
                  dut.wide_read_second_span_address == dut.read_second_span_address &&
                  dut.wide_read_pair_stride == dut.read_pair_stride &&
                  dut.wide_read_pair_count == dut.read_pair_count,
                  "request address/tag/span/pair metadata mismatch");
            tick();
        }
        dut.wide_read_request_ready = 1;
        dut.eval();
        check(dut.read_request_ready, "request did not become ready");
        tick();
        dut.read_request_valid = 0;
        dut.eval();
        check(!dut.read_request_ready && !dut.idle, "active read accepted new request");
    }
};

static void run_read(const std::string& name, std::vector<Beat> input,
                     bool wide, bool stalls, bool error, bool writes = false,
                     unsigned expected_input_ii = 0, bool paired = false) {
    Test t;
    auto& d = t.dut;
    uint32_t byte_count = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        byte_count += __builtin_popcount(input[i].mask);
        input[i].span = paired && i % 2;
    }
    t.request(wide, paired, paired ? __builtin_popcount(input[0].mask) : byte_count,
              input.size() / 2);
    std::vector<Beat> expected;
    for (const auto& b : input) {
        expected.push_back(b);
        if (!wide) {
            expected.back().mask &= 0xffffu;
            const bool high = (b.mask >> 16) != 0;
            expected.back().last = b.last && !high;
            if (high) {
                expected.push_back(b);
                for (unsigned w = 0; w < 4; ++w)
                    expected.back().data[w] = b.data[w + 4];
                expected.back().mask >>= 16;
            }
        }
    }
    size_t sent = 0, received = 0, write_sent = 0, write_received = 0;
    bool present = false, done_sent = false, done_seen = false;
    int previous_input = -1, previous_output = -1;
    unsigned replacements = 0;
    for (unsigned cycle = 0; cycle < 1000; ++cycle) {
        if (!present && sent < input.size() && (!stalls || cycle % 5 != 0))
            present = true;
        d.wide_read_data_valid = present;
        if (present) {
            const auto& b = input[sent];
            for (unsigned w = 0; w < 8; ++w) d.wide_read_data[w] = b.data[w];
            d.wide_read_byte_enable = b.mask;
            d.wide_read_data_last = b.last;
            d.wide_read_data_tag = b.tag;
            d.wide_read_data_span = b.span;
        }
        // Error completion arrives while the final buffered output is stalled.
        const bool ready = !(error && sent == input.size() && cycle % 11 < 7) &&
                           (!stalls || (cycle % 7 != 2 && cycle % 7 != 3));
        d.read_data_ready = ready;
        d.read_wide_data_ready = ready;
        d.wide_read_request_done = sent == input.size() && !done_sent;
        d.wide_read_request_error = error && d.wide_read_request_done;
        done_sent |= d.wide_read_request_done;
        d.write_request_valid = writes && cycle == 0;
        d.write_data_valid = writes && write_sent < 5;
        d.write_byte_enable = write_sent == 4 ? 1 : 0xffff;
        d.write_data_last = write_sent == 4;
        for (unsigned w = 0; w < 4; ++w) d.write_data[w] = 0xdead0000u + write_sent * 16 + w;
        d.wide_write_data_ready = cycle % 6 > 1;
        d.eval();
        check(!d.read_request_done || received == expected.size(),
              name + ": completion preceded buffered data drain");
        if (d.read_request_done) {
            check(!done_seen && bool(d.read_request_error) == error,
                  name + ": completion count/error mismatch");
            done_seen = true;
        }
        const bool output = wide ? d.read_wide_data_valid && d.read_wide_data_ready :
                                   d.read_data_valid && d.read_data_ready;
        const bool ingress = d.wide_read_data_valid && d.wide_read_data_ready;
        if (output) {
            check(received < expected.size(), name + ": duplicate output");
            const auto& b = expected[received++];
            for (unsigned w = 0; w < (wide ? 8u : 4u); ++w)
                check((wide ? d.read_wide_data[w] : d.read_data[w]) == b.data[w],
                      name + ": raw payload mismatch at output " + std::to_string(received - 1));
            check((wide ? d.read_wide_byte_enable : d.read_byte_enable) == b.mask &&
                  bool(wide ? d.read_wide_data_last : d.read_data_last) == b.last &&
                  (wide ? d.read_wide_data_tag : d.read_data_tag) == b.tag &&
                  (wide || bool(d.read_data_span) == b.span), name + ": output metadata mismatch");
            if (expected_input_ii && previous_output >= 0)
                check(int(cycle) - previous_output == 1, name + ": output bubble");
            previous_output = cycle;
        }
        if (ingress) {
            if (expected_input_ii && previous_input >= 0)
                check(int(cycle) - previous_input == int(expected_input_ii),
                      name + ": input initiation interval mismatch");
            previous_input = cycle;
            replacements += output && !wide;
            ++sent;
            present = false;
        }
        check(!(wide ? d.read_data_valid : d.read_wide_data_valid),
              name + ": wrong output channel asserted");
        if (d.write_data_valid && d.write_data_ready) ++write_sent;
        if (d.wide_write_data_valid && d.wide_write_data_ready) {
            check(writes && write_received < 3, name + ": unexpected packed write");
            for (unsigned w = 0; w < 8; ++w) {
                const unsigned logical = write_received * 2 + w / 4;
                const uint32_t value = logical < 5 ? 0xdead0000u + logical * 16 + w % 4 : 0;
                check(d.wide_write_data[w] == value, name + ": packed write mismatch");
            }
            check(d.wide_write_byte_enable == (write_received == 2 ? 1u : 0xffffffffu) &&
                  bool(d.wide_write_data_last) == (write_received == 2),
                  name + ": packed write mask/last mismatch");
            ++write_received;
        }
        if (done_seen && (!writes || write_received == 3) && d.idle) {
            if (expected_input_ii && !wide && input.size() > 1)
                check(replacements == input.size() - 1, name + ": missing consume/refill events");
            std::cout << "PASS " << name << " input_beats=" << sent
                      << " output_beats=" << received << " replacements=" << replacements
                      << " input_ii=" << expected_input_ii << " cycles=" << cycle
                      << " threads=" << d.threads() << '\n';
            return;
        }
        t.tick();
    }
    throw std::runtime_error(name + ": bounded completion timeout");
}

static void reset_and_error() {
    Test t;
    auto& d = t.dut;
    t.request(false, false);
    d.wide_read_data_valid = 1;
    d.wide_read_byte_enable = 0xffffffffu;
    t.tick();
    d.wide_read_data_valid = 0;
    d.rst = 1;
    t.tick();
    d.rst = 0;
    d.wide_read_request_ready = 0;
    d.eval();
    check(d.idle && !d.read_data_valid && !d.read_request_done,
          "reset retained buffered read state");
    t.request(false, false);
    d.wide_read_request_done = 1;
    d.wide_read_request_error = 1;
    t.tick();
    d.wide_read_request_done = 0;
    d.wide_read_request_error = 0;
    t.tick();
    check(d.read_request_done && d.read_request_error && d.idle,
          "empty read error did not complete");
    t.tick();
    check(!d.read_request_done && !d.read_request_error, "completion did not pulse");
    std::cout << "PASS reset_buffered_read_and_empty_error\n";
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        run_read("narrow_full_rate", beats(std::vector<uint32_t>(16, 0xffffffffu)),
                 false, false, false, false, 2);
        run_read("narrow_single_half_rate", beats(std::vector<uint32_t>(8, 0xffffu)),
                 false, false, false, false, 1, true);
        run_read("paired_tail_span_backpressure", beats(std::vector<uint32_t>(6, 0x1ffffu)),
                 false, true, false, true, 0, true);
        run_read("narrow_tail_one", beats({0xffffffffu, 1u}), false, true, false);
        run_read("narrow_tail_31", beats({0xffffffffu, 0x7fffffffu}), false, true, false);
        run_read("narrow_error_drain", beats({0xffffffffu, 0x1ffffu}),
                 false, true, true);
        run_read("wide_full_rate", beats(std::vector<uint32_t>(8, 0xffffffffu)),
                 true, false, false, false, 1);
        run_read("wide_tail_backpressure", beats({0xffffffffu, 0x1ffffu}),
                 true, true, false);
        run_read("wide_error_drain", beats({0xffffffffu, 1u}), true, true, true);
        reset_and_error();
        std::cout << "PASS memory_stream_width_adapter_regression\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
