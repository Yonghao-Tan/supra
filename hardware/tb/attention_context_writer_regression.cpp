#include "Vattention_context_writer.h"
#include "verilated.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

double sc_time_stamp() { return 0.0; }

namespace {

uint8_t context_byte(uint16_t local_word, unsigned byte) {
    return static_cast<uint8_t>((local_word * 29U + byte * 17U + 0x43U) & 0xffU);
}

uint16_t fragment_tag(uint16_t base_tag, uint8_t head, uint8_t query) {
    const uint16_t address_tag = static_cast<uint16_t>(
        ((static_cast<uint16_t>(head) & 0x1fU) << 11) |
        ((static_cast<uint16_t>(query) & 0x3fU) << 5));
    return static_cast<uint16_t>(base_tag ^ address_tag);
}

template <typename Wide>
void set_wide_byte(Wide &words, unsigned byte, uint8_t value) {
    const unsigned word = byte >> 2;
    const unsigned shift = (byte & 3U) * 8U;
    words[word] = (words[word] & ~(0xffU << shift)) |
                  (static_cast<uint32_t>(value) << shift);
}

template <typename Wide>
uint8_t get_wide_byte(const Wide &words, unsigned byte) {
    return static_cast<uint8_t>((words[byte >> 2] >> ((byte & 3U) * 8U)) & 0xffU);
}

struct Sim {
    Vattention_context_writer dut;
    uint64_t cycles = 0;

    void tick() {
        dut.clk = 1;
        dut.eval();
        ++cycles;
        dut.clk = 0;
        dut.eval();
    }

    void clear_inputs() {
        dut.abort_request = 0;
        dut.start_valid = 0;
        dut.start_query_count = 0;
        dut.start_head = 0;
        dut.start_context_base = 0;
        dut.start_context_query_stride = 0;
        dut.start_tag = 0;
        dut.context_read_req_ready = 0;
        dut.context_read_rsp_valid = 0;
        for (unsigned word = 0; word < 4; ++word)
            dut.context_read_rsp_data[word] = 0;
        dut.context_read_rsp_tag = 0;
        dut.write_request_ready = 0;
        dut.write_request_done = 0;
        dut.write_request_error = 0;
        dut.write_data_ready = 0;
    }

    void reset() {
        clear_inputs();
        dut.clk = 0;
        dut.rst = 1;
        for (unsigned cycle = 0; cycle < 4; ++cycle)
            tick();
        dut.rst = 0;
        tick();
    }
};

struct ReadResponse {
    bool pending = false;
    uint16_t address = 0;
    uint16_t tag = 0;
};

struct WriteFragment {
    bool active = false;
    uint64_t address = 0;
    uint16_t tag = 0;
    unsigned beats = 0;
};

struct WriteResponse {
    unsigned response_delay = 0;
};

void run_normal_case(uint8_t query_count, uint8_t head) {
    constexpr uint64_t kContextBase = 0x60000000ULL;
    constexpr uint32_t kQueryStride = 8192;
    constexpr uint16_t kBaseTag = 0x5c39;
    Sim sim;
    sim.reset();
    sim.dut.start_query_count = query_count;
    sim.dut.start_head = head;
    sim.dut.start_context_base = kContextBase;
    sim.dut.start_context_query_stride = kQueryStride;
    sim.dut.start_tag = kBaseTag;
    sim.dut.start_valid = 1;
    sim.dut.eval();
    if (!sim.dut.start_ready)
        throw std::runtime_error("valid context writer start was not ready");
    sim.tick();
    sim.dut.start_valid = 0;

    ReadResponse read;
    WriteFragment write;
    std::deque<WriteResponse> responses;
    std::vector<uint8_t> payload(static_cast<size_t>(query_count) * 256U, 0);
    uint64_t fragment_requests = 0;
    uint64_t read_requests = 0;
    uint64_t read_responses = 0;
    uint64_t write_beats = 0;
    uint64_t source_release_cycle = 0;
    uint64_t final_response_cycle = 0;
    size_t max_pending_responses = 0;
    std::array<bool, 48U * 16U> read_address_seen{};
    bool saw_done = false;

    for (unsigned guard = 0; guard < 20000 && !saw_done; ++guard) {
        sim.dut.context_read_req_ready = (sim.cycles & 3U) != 0;
        sim.dut.write_request_ready = !write.active && responses.size() < 8 &&
            (sim.cycles & 3U) != 0;
        sim.dut.write_data_ready = (sim.cycles & 7U) != 0;
        sim.dut.write_request_done = 0;
        sim.dut.write_request_error = 0;

        sim.dut.context_read_rsp_valid = read.pending;
        if (read.pending) {
            sim.dut.context_read_rsp_tag = read.tag;
            for (unsigned word = 0; word < 4; ++word)
                sim.dut.context_read_rsp_data[word] = 0;
            for (unsigned byte = 0; byte < 16; ++byte)
                set_wide_byte(sim.dut.context_read_rsp_data, byte,
                              context_byte(read.address, byte));
        }

        if (!responses.empty()) {
            if (responses.front().response_delay == 0) {
                sim.dut.write_request_done = 1;
            } else {
                --responses.front().response_delay;
            }
        }

        sim.dut.eval();
        const bool read_req_fire = sim.dut.context_read_req_valid &&
                                   sim.dut.context_read_req_ready;
        const bool read_rsp_fire = sim.dut.context_read_rsp_valid &&
                                   sim.dut.context_read_rsp_ready;
        const bool write_req_fire = sim.dut.write_request_valid &&
                                    sim.dut.write_request_ready;
        const bool write_data_fire = sim.dut.write_data_valid &&
                                     sim.dut.write_data_ready;

        if (read_req_fire) {
            if (read.pending)
                throw std::runtime_error("overlapping context SRAM reads");
            const unsigned expected_query = static_cast<unsigned>(read_requests / 16U);
            const unsigned expected_beat = static_cast<unsigned>(read_requests % 16U);
            const uint16_t expected_address = static_cast<uint16_t>(
                expected_query * 16U + expected_beat);
            if (sim.dut.context_read_req_address != expected_address)
                throw std::runtime_error("context SRAM read address mismatch: query=" +
                    std::to_string(expected_query) + " beat=" +
                    std::to_string(expected_beat) + " actual=" +
                    std::to_string(sim.dut.context_read_req_address));
            if (sim.dut.context_read_req_tag !=
                fragment_tag(kBaseTag, head, static_cast<uint8_t>(expected_query)))
                throw std::runtime_error("context SRAM read tag mismatch at query " +
                    std::to_string(expected_query));
            if (expected_address >= read_address_seen.size() ||
                read_address_seen[expected_address])
                throw std::runtime_error("context SRAM read address repeated or out of range");
            read_address_seen[expected_address] = true;
            read.pending = true;
            read.address = sim.dut.context_read_req_address;
            read.tag = sim.dut.context_read_req_tag;
            ++read_requests;
        }
        if (read_rsp_fire) {
            read.pending = false;
            ++read_responses;
        }

        if (write_req_fire) {
            if (write.active)
                throw std::runtime_error("context writer interleaved fragment payloads");
            const uint64_t expected_address = kContextBase +
                fragment_requests * kQueryStride + static_cast<uint64_t>(head) * 256ULL;
            if (sim.dut.write_request_address != expected_address ||
                sim.dut.write_request_bytes != 256)
                throw std::runtime_error("context fragment address or byte count mismatch");
            if (sim.dut.write_request_tag !=
                fragment_tag(kBaseTag, head,
                    static_cast<uint8_t>(fragment_requests)))
                throw std::runtime_error("context fragment tag mismatch at query " +
                    std::to_string(fragment_requests));
            write.active = true;
            write.address = sim.dut.write_request_address;
            write.tag = sim.dut.write_request_tag;
            write.beats = 0;
            ++fragment_requests;
        }

        if (write_data_fire) {
            if (!write.active || write.beats >= 16)
                throw std::runtime_error("write data without an active fragment");
            if (sim.dut.write_byte_enable != 0xffffU ||
                bool(sim.dut.write_data_last) != (write.beats == 15))
                throw std::runtime_error("context write strobe or last mismatch");
            const unsigned query = static_cast<unsigned>(fragment_requests - 1U);
            const uint16_t local_word = static_cast<uint16_t>(query * 16U + write.beats);
            for (unsigned byte = 0; byte < 16; ++byte) {
                const uint8_t actual = get_wide_byte(sim.dut.write_data, byte);
                const uint8_t expected = context_byte(local_word, byte);
                if (actual != expected)
                    throw std::runtime_error("context payload mismatch");
                payload[query * 256U + write.beats * 16U + byte] = actual;
            }
            ++write.beats;
            ++write_beats;
            if (write.beats == 16) {
                write.active = false;
                responses.push_back({fragment_requests == 1 ? 512U : 2U});
                max_pending_responses = std::max(max_pending_responses,
                                                 responses.size());
            }
        }

        if (sim.dut.write_request_done && !responses.empty()) {
            responses.pop_front();
            if (sim.dut.completed_fragment_response_count + 1 == query_count)
                final_response_cycle = sim.cycles;
        }
        if (sim.dut.source_buffer_released)
            source_release_cycle = sim.cycles;
        saw_done = sim.dut.done_pulse;
        if (saw_done && sim.dut.error)
            throw std::runtime_error("context writer reported an error in a valid case");
        sim.tick();
    }

    if (!saw_done)
        throw std::runtime_error("context writer did not complete");
    if (fragment_requests != query_count ||
        read_requests != query_count * 16ULL ||
        read_responses != query_count * 16ULL ||
        write_beats != query_count * 16ULL ||
        sim.dut.accepted_read_request_count != query_count * 16ULL ||
        sim.dut.completed_read_response_count != query_count * 16ULL ||
        sim.dut.accepted_fragment_request_count != query_count ||
        sim.dut.completed_fragment_response_count != query_count ||
        sim.dut.accepted_write_data_count != query_count * 16ULL ||
        sim.dut.accepted_write_byte_count != query_count * 256ULL ||
        sim.dut.completed_command_count != 1)
        throw std::runtime_error("context writer event count mismatch");
    if (source_release_cycle == 0 || final_response_cycle == 0 ||
        source_release_cycle >= final_response_cycle)
        throw std::runtime_error("source buffer was not released before the final response");
    for (unsigned address = 0; address < query_count * 16U; ++address) {
        if (!read_address_seen[address])
            throw std::runtime_error("context SRAM read address coverage gap");
    }
    if (query_count == 48 && (!read_address_seen[32U * 16U] ||
                             !read_address_seen[47U * 16U + 15U]))
        throw std::runtime_error("query 32..47 context address coverage missing");
    if (query_count == 48 && max_pending_responses != 8)
        throw std::runtime_error("context writer did not use the eight-response window");
}

void run_configuration_error_case() {
    Sim sim;
    sim.reset();
    sim.dut.start_query_count = 49;
    sim.dut.start_head = 0;
    sim.dut.start_context_base = 0x70000000ULL;
    sim.dut.start_context_query_stride = 8192;
    sim.dut.start_tag = 0x1122;
    sim.dut.start_valid = 1;
    sim.dut.eval();
    if (!sim.dut.start_ready)
        throw std::runtime_error("configuration-error start was not ready");
    sim.tick();
    sim.dut.start_valid = 0;
    sim.dut.eval();
    if (!sim.dut.done_pulse || !sim.dut.error || sim.dut.error_id != 1)
        throw std::runtime_error("query_count=49 did not report configuration error");
    if (sim.dut.accepted_fragment_request_count != 0 ||
        sim.dut.accepted_read_request_count != 0 ||
        sim.dut.completed_command_count != 0)
        throw std::runtime_error("invalid configuration issued a memory request");
    sim.tick();
    if (!sim.dut.start_ready)
        throw std::runtime_error("context writer did not recover from configuration error");
}

void run_response_error_case(bool corrupt_read_tag, bool fail_write_response,
                             uint8_t expected_error_id, uint8_t query_count = 1) {
    constexpr uint16_t kBaseTag = 0x3a17;
    Sim sim;
    sim.reset();
    sim.dut.start_query_count = query_count;
    sim.dut.start_head = 3;
    sim.dut.start_context_base = 0x72000000ULL;
    sim.dut.start_context_query_stride = 8192;
    sim.dut.start_tag = kBaseTag;
    sim.dut.start_valid = 1;
    sim.dut.eval();
    if (!sim.dut.start_ready)
        throw std::runtime_error("response-error start was not ready");
    sim.tick();
    sim.dut.start_valid = 0;

    bool read_pending = false;
    bool corrupt_next_response = corrupt_read_tag;
    uint16_t read_address = 0;
    uint16_t read_tag = 0;
    bool write_response_pending = false;
    bool saw_done = false;
    unsigned read_requests = 0;
    unsigned write_beats = 0;
    unsigned fragment_requests = 0;

    for (unsigned guard = 0; guard < 1000 && !saw_done; ++guard) {
        sim.dut.write_request_ready = 1;
        sim.dut.context_read_req_ready = 1;
        sim.dut.context_read_rsp_valid = read_pending;
        sim.dut.context_read_rsp_tag = corrupt_next_response ?
            static_cast<uint16_t>(read_tag ^ 1U) : read_tag;
        if (read_pending) {
            for (unsigned word = 0; word < 4; ++word)
                sim.dut.context_read_rsp_data[word] = 0;
            for (unsigned byte = 0; byte < 16; ++byte)
                set_wide_byte(sim.dut.context_read_rsp_data, byte,
                              context_byte(read_address, byte));
        }
        sim.dut.write_data_ready = 1;
        sim.dut.write_request_done = write_response_pending;
        sim.dut.write_request_error = write_response_pending &&
            fail_write_response;
        sim.dut.eval();

        const bool read_req_fire = sim.dut.context_read_req_valid &&
                                   sim.dut.context_read_req_ready;
        const bool read_rsp_fire = sim.dut.context_read_rsp_valid &&
                                   sim.dut.context_read_rsp_ready;
        const bool write_req_fire = sim.dut.write_request_valid &&
                                    sim.dut.write_request_ready;
        const bool write_data_fire = sim.dut.write_data_valid &&
                                     sim.dut.write_data_ready;
        const bool write_response_fire = sim.dut.write_request_done;

        if (write_req_fire) {
            if (fragment_requests != 0 || sim.dut.write_request_tag !=
                fragment_tag(kBaseTag, 3, 0))
                throw std::runtime_error("response-error fragment request mismatch");
            ++fragment_requests;
        }
        if (read_req_fire) {
            if (read_pending || sim.dut.context_read_req_address != read_requests ||
                sim.dut.context_read_req_tag != fragment_tag(kBaseTag, 3, 0))
                throw std::runtime_error("response-error SRAM read request mismatch");
            read_pending = true;
            read_address = sim.dut.context_read_req_address;
            read_tag = sim.dut.context_read_req_tag;
            ++read_requests;
        }
        if (read_rsp_fire) {
            read_pending = false;
            corrupt_next_response = false;
        }
        if (write_data_fire) {
            ++write_beats;
            if (sim.dut.write_data_last)
                write_response_pending = true;
        }
        saw_done = sim.dut.done_pulse;
        if (saw_done && (!sim.dut.error ||
                         sim.dut.error_id != expected_error_id))
            throw std::runtime_error("response error reported the wrong error_id");
        sim.tick();
        if (write_response_fire)
            write_response_pending = false;
    }

    if (!saw_done || fragment_requests != 1 || read_requests != 16 ||
        write_beats != 16 || sim.dut.accepted_fragment_request_count != 1 ||
        sim.dut.completed_fragment_response_count != 1 ||
        sim.dut.completed_command_count != 0)
        throw std::runtime_error("response error did not drain one accepted fragment");
    sim.dut.write_request_done = 0;
    sim.dut.write_request_error = 0;
    sim.tick();
    if (!sim.dut.start_ready)
        throw std::runtime_error("context writer did not recover from response error");
}

void run_abort_case() {
    Sim sim;
    sim.reset();
    sim.dut.start_query_count = 2;
    sim.dut.start_head = 1;
    sim.dut.start_context_base = 0x70000000ULL;
    sim.dut.start_context_query_stride = 8192;
    sim.dut.start_tag = 0x1234;
    sim.dut.start_valid = 1;
    sim.dut.eval();
    if (!sim.dut.start_ready)
        throw std::runtime_error("abort case start was not ready");
    sim.tick();
    sim.dut.start_valid = 0;
    sim.dut.write_request_ready = 1;
    unsigned wait_cycles = 0;
    while (!sim.dut.write_request_valid && wait_cycles++ < 100)
        sim.tick();
    if (!sim.dut.write_request_valid)
        throw std::runtime_error("abort case did not issue a write fragment");
    sim.dut.eval();
    if (!sim.dut.write_request_valid || !sim.dut.write_request_ready)
        throw std::runtime_error("abort case write fragment was not accepted");
    sim.tick();
    if (sim.dut.accepted_fragment_request_count != 1)
        throw std::runtime_error("abort case write fragment was not accepted");
    sim.dut.write_request_ready = 0;
    sim.dut.abort_request = 1;

    bool read_pending = false;
    uint16_t read_address = 0;
    uint16_t read_tag = 0;
    unsigned write_beats = 0;
    bool write_done = false;
    bool acknowledged = false;
    for (unsigned guard = 0; guard < 1000 && !acknowledged; ++guard) {
        sim.dut.context_read_req_ready = 1;
        sim.dut.context_read_rsp_valid = read_pending;
        if (read_pending) {
            sim.dut.context_read_rsp_tag = read_tag;
            for (unsigned word = 0; word < 4; ++word)
                sim.dut.context_read_rsp_data[word] = 0;
            for (unsigned byte = 0; byte < 16; ++byte)
                set_wide_byte(sim.dut.context_read_rsp_data, byte,
                              context_byte(read_address, byte));
        }
        sim.dut.write_data_ready = 1;
        sim.dut.write_request_done = write_done;
        sim.dut.write_request_error = 0;
        sim.dut.eval();
        const bool drove_write_done = sim.dut.write_request_done;
        if (sim.dut.context_read_req_valid && sim.dut.context_read_req_ready) {
            read_pending = true;
            read_address = sim.dut.context_read_req_address;
            read_tag = sim.dut.context_read_req_tag;
        }
        if (sim.dut.context_read_rsp_valid && sim.dut.context_read_rsp_ready)
            read_pending = false;
        if (sim.dut.write_data_valid && sim.dut.write_data_ready) {
            ++write_beats;
            if (sim.dut.write_data_last)
                write_done = true;
        }
        acknowledged = sim.dut.abort_ack;
        sim.tick();
        if (drove_write_done)
            write_done = false;
    }
    if (!acknowledged || write_beats != 16)
        throw std::runtime_error("context writer abort did not drain accepted fragment: ack=" +
            std::to_string(acknowledged) + " beats=" + std::to_string(write_beats) +
            " accepted_reads=" +
            std::to_string(sim.dut.accepted_read_request_count) +
            " completed_responses=" +
            std::to_string(sim.dut.completed_fragment_response_count));
    sim.dut.abort_request = 0;
    sim.dut.write_request_done = 0;
    sim.tick();
    if (!sim.dut.start_ready)
        throw std::runtime_error("context writer did not recover after abort: busy=" +
            std::to_string(sim.dut.busy) + " fragment_requests=" +
            std::to_string(sim.dut.accepted_fragment_request_count) +
            " fragment_responses=" +
            std::to_string(sim.dut.completed_fragment_response_count));
}

}  // namespace

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        run_normal_case(1, 0);
        run_normal_case(48, 31);
        run_configuration_error_case();
        run_response_error_case(true, false, 2);
        run_response_error_case(false, true, 3);
        run_response_error_case(false, true, 3, 2);
        run_abort_case();
        std::cout << "PASS attention context writer:\n"
                  << "normal_cases=2\n"
                  << "fragments=49\n"
                  << "sram_reads=784\n"
                  << "payload_bytes=12544\n"
                  << "query_32_47=PASS\n"
                  << "read_address_range=0x000..0x2ff\n"
                  << "query_tag_bit5=PASS\n"
                  << "token_major_address_mismatch=0\n"
                  << "payload_mismatch=0\n"
                  << "source_release_before_final_response=1\n"
                  << "max_pending_responses=8\n"
                  << "configuration_error=PASS\n"
                  << "read_tag_error_drain=PASS\n"
                  << "write_response_error_drain=PASS\n"
                  << "write_response_error_next_fragment=PASS\n"
                  << "abort_recovery=1\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "FAIL attention context writer: "
                  << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
