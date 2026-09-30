#include "Vattention_dependency_update.h"
#include "verilated.h"
#include "generated/attention_dependency_job_packer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" {
#include "../cmodel/rtl_numeric.h"
}

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
constexpr std::array<std::uint16_t, 8> probabilities = {0, 0x3e80, 0x3f00, 0x3f40, 0x3f80, 0x3e00, 0x3ec0, 0x3f60};
constexpr std::array<unsigned, 8> scores = {0, 64, 128, 191, 255, 32, 96, 223};
enum Fault { None, Reserved, SourceLimit, OutputAlias, BadProbability, BadP8Probability, ReadError, WriteError, AbortRead, AbortWrite, DequantError, AbortDequant };
class Regression {
    Vattention_dependency_update dut;
    std::vector<std::uint8_t> memory = std::vector<std::uint8_t>(0x40000, 0xa5), expected;
    std::uint64_t cycle = 0;
    unsigned jobs = 0;
    void edge() { dut.clk = 1; dut.eval(); ++cycle; dut.clk = 0; dut.eval(); }
    void put16(unsigned address, unsigned value) { memory.at(address) = value; memory.at(address+1) = value >> 8; }
    void job(unsigned operation, unsigned mask, unsigned destination) {
        attention_dependency_job value{};
        value.source_base = 0x10000 + jobs*1024;
        value.head_stride = jobs%2 ? 32 : 16;
        value.source_limit = value.source_base + 31*value.head_stride+16;
        value.lane_mask = mask; value.operation = operation;
        value.output_base = destination; value.output_limit = destination + ((operation&3) == 2 ? 64 : (operation&3) == 3 ? 8 : 16);
        const auto packed = pack_attention_dependency_job(value);
        std::copy(packed.begin(), packed.end(), memory.begin()+0x1000+jobs*64);
        for (unsigned head = 0; head < 32; ++head) {
            const auto address = value.source_base+head*value.head_stride;
            if (operation & 8) {
                std::fill_n(memory.begin()+address, 16, 0);
                for (unsigned lane = 0; lane < 8; ++lane) memory[address+lane] = (head*97+lane*13)%128;
                put16(address+8, 0x3a80+head%7);
            } else for (unsigned lane = 0; lane < 8; ++lane) put16(address+lane*2, probabilities[lane]);
        }
        ++jobs;
    }
public:
    void run() {
        dut.rst = 1; edge(); dut.rst = 0;
        prepare(); execute(None);
        for (auto fault : {Reserved, SourceLimit, OutputAlias, BadProbability, BadP8Probability, ReadError, WriteError, AbortRead, AbortWrite, DequantError, AbortDequant}) {
            prepare(); execute(fault);
            prepare(); execute(None);
        }
        for (bool merge_initial : {false, true}) for (bool first : {true, false}) {
            std::fill(memory.begin(), memory.end(), 0xa5); jobs = 0;
            job(9 | (merge_initial ? 16 : 0), 0x7f, 0x20400);
            for (unsigned lane = 0; lane < 8; ++lane)
                put16(0x20400+lane*2, lane%2 ? 0x3f00 : 0);
            expected = memory;
            for (unsigned lane = 0; lane < 7; ++lane) {
                std::array<std::uint16_t, 32> values{};
                for (unsigned head = 0; head < 32; ++head)
                    values[head] = rtl_bf16_mul(rtl_f32_to_bf16(float((head*97+lane*13)%128)), 0x3a80+head%7);
                float mean = 0;
                require(rtl_probability_head_mean32(values.data(), 1, &mean) == 0, "invalid layer reference");
                auto raw = rtl_f32_to_bf16(mean);
                if (!first || merge_initial) raw = std::max<std::uint16_t>(raw, lane%2 ? 0x3f00 : 0);
                expected[0x20400+lane*2] = raw;
                expected[0x20401+lane*2] = raw >> 8;
            }
            execute(None, true, first);
        }
        std::fill(memory.begin(), memory.end(), 0xa5); jobs = 0;
        for (unsigned offset = 0; offset < 16; offset += 2)
            job(0, 0x3c, 0x20ff0+offset*16+offset);
        expected = memory;
        for (unsigned offset = 0; offset < 16; offset += 2)
            for (unsigned lane = 2; lane < 6; ++lane) {
                const unsigned address = 0x20ff0+offset*16+offset+lane*2;
                expected[address] = probabilities[lane]; expected[address+1] = probabilities[lane]>>8;
            }
        execute(None, false, false, true);
        dut.final();
        std::cout << "PASS relation_dma BF16_replace/layer_max/scout_vector/scout_row/masked_preservation/init/error/abort/restart cycles="
                  << cycle << " threads=" << dut.threads() << '\n';
    }
    void prepare() {
        std::fill(memory.begin(), memory.end(), 0xa5); jobs = 0;
        for (unsigned lane = 0; lane < 8; ++lane) {
            put16(0x20000+lane*2, 0x3f00);
            put16(0x20020+lane*2, 0x3f00);
        }
        job(0, 0x85, 0x20000);
        job(1, 0xff, 0x20020);
        job(6, 0x85, 0x20100);
        job(2, 0xff, 0x20100);
        job(7, 0x85, 0x20208);
        job(3, 0xff, 0x20208);
        job(6, 0, 0x20300);
        job(8, 0x7f, 0x20400);
        job(9, 0xff, 0x20420);
        for (unsigned lane = 0; lane < 8; ++lane) put16(0x20420+lane*2, 0x3a80);
        expected = memory;
        for (unsigned lane = 0; lane < 8; ++lane) {
            if ((0x85 >> lane) & 1) {
                expected[0x20000+lane*2] = probabilities[lane];
                expected[0x20001+lane*2] = probabilities[lane] >> 8;
            }
            const auto maximum = std::max<unsigned>(probabilities[lane], 0x3f00);
            expected[0x20020+lane*2] = maximum;
            expected[0x20021+lane*2] = maximum >> 8;
            expected[0x20100+lane*8] = scores[lane];
            expected[0x20300+lane*8] = 0;
        }
        expected[0x20208] = 255;
        for (unsigned lane = 0; lane < 8; ++lane) {
            std::array<std::uint16_t, 32> values{};
            for (unsigned head = 0; head < 32; ++head)
                values[head] = rtl_bf16_mul(rtl_f32_to_bf16(float((head*97+lane*13)%128)), 0x3a80+head%7);
            float mean = 0;
            require(rtl_probability_head_mean32(values.data(), 1, &mean) == 0, "invalid P8 reference");
            const auto raw = rtl_f32_to_bf16(mean);
            if (lane < 7) {
                expected[0x20400+lane*2] = raw;
                expected[0x20401+lane*2] = raw >> 8;
            }
            const auto maximum = std::max<unsigned>(raw, 0x3a80);
            expected[0x20420+lane*2] = maximum;
            expected[0x20421+lane*2] = maximum >> 8;
        }
    }
    void execute(Fault fault, bool layer_update = false, bool first_layer = false, bool aligned_matrix = false) {
        const bool aborted = fault == AbortRead || fault == AbortWrite || fault == AbortDequant;
        if (fault == Reserved) memory[0x103f] = 1;
        if (fault == SourceLimit) std::fill_n(memory.begin()+0x1018, 8, 0);
        if (fault == OutputAlias) { std::fill_n(memory.begin()+0x1010, 8, 0); memory[0x1011] = 0x10; }
        if (fault == BadProbability) put16(0x10000, 0x7fc0);
        if (fault == BadP8Probability) memory.at(0x10000+7*1024) = 0xff;
        dut.abort_request = 0; dut.read_error = 0; dut.read_abort_ack = 0; dut.read_valid = 0;
        dut.write_done = 0; dut.write_error = 0; dut.done_ready = 0;
        dut.start_valid = 1; dut.start_job_base = 0x1000; dut.start_job_limit = 0x1000+jobs*64; dut.start_job_count = jobs;
        dut.start_layer_update = layer_update; dut.start_first_layer = first_layer;
        dut.eval(); require(dut.start_ready, "relation controller not ready after previous job list");
        edge(); dut.start_valid = 0;
        bool reading = false, writing = false, injected = false;
        unsigned read_address = 0, read_bytes = 0, read_offset = 0, write_address = 0, write_bytes = 0, write_offset = 0;
        std::uint64_t read_due = 0, write_due = 0;
        unsigned reads = 0, writes = 0;
        struct Product { std::uint64_t due; std::array<std::uint32_t, 4> words; };
        std::deque<Product> products;
        unsigned dequant_requests = 0;
        dut.dequant_ready = 0; dut.dequant_result_valid = 0; dut.dequant_error = 0; dut.dequant_abort_ack = 0;
        for (unsigned local = 0; local < 30000; ++local) {
            dut.dequant_ready = 0;
            dut.dequant_result_valid = !products.empty() && cycle >= products.front().due;
            dut.dequant_error = fault == DequantError && dut.dequant_result_valid;
            if (dut.dequant_result_valid)
                for (unsigned word = 0; word < 4; ++word) dut.dequant_result_values[word] = products.front().words[word];
            dut.read_request_ready = local % 5 != 0;
            dut.write_request_ready = local % 7 != 0;
            dut.write_ready = local % 9 >= 3;
            dut.read_valid = reading && cycle >= read_due;
            dut.read_error = fault == ReadError && reading && reads == 5 && cycle >= read_due;
            if (dut.read_error) dut.read_valid = 0;
            dut.read_tag = 0xb7;
            const unsigned rbytes = std::min(16u, read_bytes-read_offset);
            dut.read_byte_enable = (1u << rbytes)-1;
            dut.read_last = read_offset+rbytes == read_bytes;
            for (unsigned word = 0; word < 4; ++word) {
                dut.read_data[word] = 0;
                for (unsigned byte = 0; byte < 4; ++byte) {
                    const unsigned offset = word*4+byte;
                    if (reading && offset < rbytes)
                        dut.read_data[word] |= unsigned(memory.at(read_address+read_offset+offset)) << (byte*8);
                }
            }
            dut.write_done = write_due && cycle == write_due;
            dut.write_error = dut.write_done && fault == WriteError;
            dut.eval();
            // The shared arithmetic arbiter grants ready only to a valid request.
            dut.dequant_ready = dut.dequant_valid && products.size() < 4 && local%7 != 0;
            dut.eval();
            if (!injected && ((fault == AbortRead && reading && reads == 3) ||
                              (fault == AbortDequant && !products.empty()) ||
                              (fault == AbortWrite && dut.write_valid && !dut.write_ready))) {
                injected = true; dut.abort_request = 1; dut.eval();
            }
            dut.dequant_abort_ack = dut.dequant_abort;
            if (dut.dequant_abort) { products.clear(); dut.dequant_result_valid = 0; dut.dequant_error = 0; dut.eval(); }
            if (dut.dequant_result_valid && dut.dequant_result_ready) products.pop_front();
            if (dut.dequant_valid && dut.dequant_ready) {
                Product product{cycle+4, {}};
                for (unsigned lane = 0; lane < 8; ++lane) {
                    const auto code = std::uint16_t(dut.dequant_values[lane/2] >> ((lane%2)*16));
                    const auto raw = (dut.dequant_lane_mask>>lane)&1 ? rtl_bf16_mul(code, dut.dequant_scale) : 0;
                    product.words[lane/2] |= std::uint32_t(raw) << ((lane%2)*16);
                }
                products.push_back(product); ++dequant_requests;
            }
            if (dut.read_error) reading = false;
            if (dut.read_valid && dut.read_ready) {
                read_offset += rbytes;
                if (read_offset == read_bytes) reading = false;
            }
            if (dut.read_request_valid && dut.read_request_ready) {
                require(!reading && dut.read_request_tag == 0xb7, "relation read request overlap/tag mismatch");
                reading = true; read_address = dut.read_request_address; read_bytes = dut.read_request_bytes;
                read_offset = 0; read_due = cycle+2; ++reads;
                require(read_address+read_bytes <= memory.size(), "relation read outside DDR model");
            }
            if (dut.write_request_valid && dut.write_request_ready) {
                require(!writing && dut.write_request_tag == 0xb8, "relation write request overlap/tag mismatch");
                writing = true; write_address = dut.write_request_address; write_bytes = dut.write_request_bytes;
                write_offset = 0; ++writes;
                require(write_address+write_bytes <= memory.size(), "relation write outside DDR model");
            }
            if (dut.write_valid && dut.write_ready) {
                require(writing, "relation data has no write request");
                const unsigned bytes = std::min(16u, write_bytes-write_offset);
                require(dut.write_byte_enable == (1u << bytes)-1 && bool(dut.write_last) == (write_offset+bytes == write_bytes),
                        "relation output strobe/last mismatch");
                for (unsigned byte = 0; byte < bytes; ++byte)
                    memory.at(write_address+write_offset+byte) = dut.write_data[byte/4] >> ((byte%4)*8);
                write_offset += bytes;
                if (write_offset == write_bytes) { writing = false; write_due = cycle+3; }
            }
            if (dut.done_valid || dut.abort_ack) {
                require(!reading && !writing && products.empty() && (!write_due || cycle > write_due), "relation completed before DMA/arithmetic drain");
                require(aborted ? dut.abort_ack : dut.done_valid, "relation completion kind mismatch");
                require(bool(dut.error) == (fault != None && !aborted), "relation error result mismatch");
                if (fault == None) {
                    require(memory == expected, "relation output values or retained metadata differ");
                    require(writes == jobs && reads == (aligned_matrix ? 148u : layer_update ? 3u : 150u) &&
                            dequant_requests == (aligned_matrix ? 0u : layer_update ? 32u : 64u), "relation burst/P8 request count mismatch");
                }
                dut.done_ready = 1; dut.abort_request = 0;
                dut.read_valid = 0; dut.read_error = 0; dut.write_done = 0; dut.write_error = 0;
                edge(); edge(); return;
            }
            edge();
        }
        throw std::runtime_error("relation DMA did not finish fault=" + std::to_string(fault));
    }
};
}
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try { Regression().run(); return 0; }
    catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
