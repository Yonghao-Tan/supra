#include "Vrefresh_score_update.h"
#include "verilated.h"
#include "json.hpp"
#include "generated/token_state_entry_packer.hpp"
extern "C" {
#include "../cmodel/rtl_numeric.h"
}
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using Json = nlohmann::json;
static void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
static void flatten(const Json& value, std::vector<unsigned>& result) {
    if (value.is_array()) for (const auto& item : value) flatten(item, result);
    else result.push_back(value.is_boolean() ? unsigned(value.get<bool>()) : value.get<unsigned>());
}
static std::vector<unsigned> raw(const Json& value) {
    std::vector<unsigned> result; flatten(value.at("raw"), result); return result;
}
class Regression {
    Vrefresh_score_update dut;
    std::uint64_t cycle = 0;
    std::vector<std::uint8_t> memory = std::vector<std::uint8_t>(0x100000);
    unsigned completed = 0;
    void edge() { dut.clk = 1; dut.eval(); ++cycle; dut.clk = 0; dut.eval(); }
    void put16(unsigned address, unsigned value) { memory.at(address) = value; memory.at(address+1) = value>>8; }
    unsigned get16(unsigned address) { return memory.at(address) | unsigned(memory.at(address+1))<<8; }
public:
    Regression() { dut.rst = 1; edge(); dut.rst = 0; }
    void run(const char* path, bool cross) {
        std::ifstream stream(path); require(stream.good(), "missing pending fixture");
        Json fixture; stream >> fixture;
        run_document(fixture, cross, false);
        if (cross) run_document(fixture, cross, true);
        if (fixture.contains("variants")) for (const auto& variant : fixture.at("variants")) {
            run_document(variant, cross, false);
            if (cross) run_document(variant, cross, true);
        }
    }
    void run_document(const Json& fixture, bool cross, bool state_entries) {
        const auto mode = fixture.value("pending_confidence_mode", std::string("all_changes"));
        const unsigned rows = cross ? fixture.at("rows").get<unsigned>() :
            fixture.at("region_end").get<unsigned>() - fixture.at("region_start").get<unsigned>();
        const unsigned initial_start = cross ? fixture.at("block_start").get<unsigned>() : fixture.at("region_start").get<unsigned>();
        const unsigned keys = cross ? 32 : rows;
        const unsigned stride = keys <= 32 ? 64 : keys <= 64 ? 128 : 256;
        for (const auto& record : fixture.at("records")) {
            const unsigned start = record.value("block_start", initial_start);
            const bool advance = record.value("advance_block", false);
            std::fill(memory.begin(), memory.end(), 0);
            const auto& before = record.at("before"); const auto& expected = record.at("expected"); const auto& input = record.at("inputs");
            const auto relation = raw(expected.at(cross ? "relation" : "dependency"));
            const auto changed = raw(input.at("changed_global"));
            const auto remasked = raw(input.at("changed_remask_global"));
            const auto confidence = raw(input.at("changed_confidence_global"));
            std::vector<unsigned> consumed(rows, 0);
            if (cross) for (auto row : raw(input.at("consumed_positions"))) consumed.at(row) = 1;
            else consumed = raw(before.at("refresh"));
            const std::vector<std::string> fields = cross ?
                std::vector<std::string>{"pending", "actual_remask_pending", "actual_remask_epoch_pending", "future_pending", "future_actual_remask_pending"} :
                std::vector<std::string>{"pending"};
            for (unsigned row = 0; row < rows; ++row) {
                for (unsigned key = 0; key < keys; ++key) put16(0x10000+row*stride+key*2, relation.at(row*keys+key));
                for (unsigned field = 0; field < fields.size(); ++field) put16(0x80000+row*16+field*2, raw(before.at(fields[field])).at(row));
                memory.at(0x80000+row*16+10) = consumed.at(row);
            }
            for (unsigned key = 0; key < keys; ++key) {
                put16(0x1000+key*4, confidence.at(start+key));
                memory.at(0x1000+key*4+2) = changed.at(start+key) | remasked.at(start+key)<<1;
            }
            dut.start_rows = rows; dut.start_keys = keys; dut.start_block_end = cross ? start+32 : rows;
            dut.start_all_changes = cross;
            dut.start_live_consumed = 0; dut.live_consumed = 0;
            dut.start_clear_relation = 0; dut.start_advance_block = advance;
            dut.start_confidence_mode = !cross || mode == "all_changes" ? 0 : mode == "stable_unmask" ? 1 : 2;
            dut.start_relation_row_shift = stride == 64 ? 6 : stride == 128 ? 7 : 8;
            dut.start_state_changes = 0; dut.start_change_position_begin = 0; dut.start_change_capture_index = 0;
            dut.start_relation_base = 0x10000; dut.start_relation_limit = 0x10000+rows*stride;
            dut.start_change_base = 0x1000; dut.start_change_limit = 0x1000+((keys+3)/4)*16;
            dut.start_pending_base = 0x80000; dut.start_pending_limit = 0x80000+rows*16;
            if (state_entries) {
                dut.start_state_changes = 1; dut.start_change_position_begin = start; dut.start_change_capture_index = 17;
                dut.start_change_limit = 0x1000 + keys * 32;
                for (unsigned key = 0; key < keys; ++key) {
                    token_state_entry entry{};
                    entry.token_position = start + key; entry.block_local_position = key;
                    entry.state = 0; entry.activation_bits = 4; entry.capture_index = 17;
                    entry.change_flags = changed.at(start+key) | (remasked.at(start+key)<<1);
                    entry.change_confidence_bf16 = remasked.at(start+key) ? confidence.at(start+key) : 0x3f80;
                    entry.last_action_confidence_bf16 = confidence.at(start+key);
                    entry.last_action_confidence_valid = 1;
                    const auto bytes = pack_token_state_entry(entry);
                    std::copy(bytes.begin(), bytes.end(), memory.begin()+0x1000+key*32);
                }
            }
            if (advance) {
                // PREPARE must clear every old dependency, including queries
                // absent from this forward, without consuming pending scores.
                const auto old_relation = raw(before.at("relation"));
                for (unsigned row = 0; row < rows; ++row)
                    for (unsigned key = 0; key < keys; ++key)
                        put16(0x10000+row*stride+key*2, old_relation.at(row*keys+key));
                const std::vector<std::uint8_t> pending_before(memory.begin()+0x80000, memory.begin()+0x80000+rows*16);
                dut.start_clear_relation = 1;
                execute(0);
                const auto cleared = raw(record.at("after_advance").at("relation"));
                for (unsigned row = 0; row < rows; ++row)
                    for (unsigned key = 0; key < keys; ++key)
                        require(get16(0x10000+row*stride+key*2) == cleared.at(row*keys+key),
                            "block advance retained stale dependency");
                require(std::equal(pending_before.begin(), pending_before.end(), memory.begin()+0x80000),
                    "PREPARE changed pending scores before boundary selection");
                dut.start_clear_relation = 0;
                const auto queries = raw(input.at("query_positions"));
                const auto observations = raw(input.at("relation"));
                for (unsigned query = 0; query < queries.size(); ++query)
                    for (unsigned key = 0; key < keys; ++key)
                        put16(0x10000+queries[query]*stride+key*2, observations.at(query*keys+key));
            }
            if (state_entries && mode != "remask_only" && record.at("step_index") == 0) {
                unsigned ordinary = 0;
                while (ordinary < keys && (!changed.at(start+ordinary) || remasked.at(start+ordinary))) ++ordinary;
                require(ordinary < keys, "missing confidence rejection input lacks an ordinary change");
                memory.at(0x1000+ordinary*32+30) = 0;
                execute(6);
                memory.at(0x1000+ordinary*32+30) = 1;
            }
            execute(0);
            for (unsigned row = 0; row < rows; ++row) {
                for (unsigned field = 0; field < fields.size(); ++field) {
                    const auto wanted = raw(expected.at(fields[field])).at(row);
                    const auto actual = get16(0x80000+row*16+field*2);
                    require(actual == wanted, fields[field]+" DMA raw mismatch row="+std::to_string(row)+
                        " actual="+std::to_string(actual)+" expected="+std::to_string(wanted));
                }
                require(memory.at(0x80000+row*16+10) == 0, "consumed flag not cleared");
                if (!cross) require(get16(0x80000+row*16+12) == raw(expected.at("new_invalidation")).at(row), "invalidation output raw mismatch");
            }
        }
        if (cross && !state_entries) {
            dut.start_advance_block = 0;
            execute(1); execute(2); execute(3); execute(4); execute(5); execute(0);
        }
    }
    void execute(unsigned fault) {
        dut.start_valid = 1; dut.done_ready = 0; dut.abort_request = 0;
        dut.read_valid = 0; dut.read_error = 0; dut.read_abort_ack = 0;
        dut.write_done = 0; dut.write_error = 0; dut.arithmetic_result_valid = 0; dut.arithmetic_abort_ack = 0;
        dut.eval(); require(dut.start_ready, "pending start not ready"); edge(); dut.start_valid = 0;
        bool reading = false, writing = false, scalar = false, injected = false, acknowledged = false;
        unsigned read_address = 0, read_bytes = 0, read_offset = 0, write_address = 0, write_offset = 0, write_bytes = 0;
        std::uint64_t read_due = 0, write_due = 0, scalar_due = 0;
        std::uint16_t scalar_result = 0;
        // Ensure arithmetic-abort cases reach a changed key even after the empty-change fixture.
        if (fault >= 3 && fault <= 5) { memory[0x1002] = 3; put16(0x1000, 0x3f00); }
        const auto begin = cycle;
        for (unsigned step = 0; step < 200000; ++step) {
            dut.read_request_ready = !reading && cycle%5 != 1;
            dut.write_request_ready = !writing && cycle%7 != 1;
            dut.write_ready = cycle%9 >= 3;
            dut.read_valid = reading && cycle >= read_due;
            dut.read_error = fault == 1 && reading && !injected && cycle >= read_due;
            dut.read_last = read_offset+16 == read_bytes; dut.read_tag = 0xba; dut.read_byte_enable = 0xffff;
            for (unsigned word = 0; word < 4; ++word) {
                unsigned value = 0;
                if (reading) for (unsigned byte = 0; byte < 4; ++byte) value |= unsigned(memory.at(read_address+read_offset+word*4+byte))<<(byte*8);
                dut.read_data[word] = value;
            }
            dut.write_done = writing && write_offset == write_bytes && cycle >= write_due;
            dut.write_error = fault == 2 && dut.write_done && !injected;
            dut.arithmetic_ready = !scalar && cycle%3 != 1;
            dut.arithmetic_abort_ack = fault == 4 && scalar && dut.arithmetic_abort_request;
            dut.arithmetic_result_valid = scalar && !dut.arithmetic_abort_ack && cycle >= scalar_due;
            dut.arithmetic_result = scalar_result;
            dut.eval();
            const bool read_issue = dut.read_request_valid && dut.read_request_ready;
            const bool read_fire = dut.read_valid && dut.read_ready;
            const bool write_issue = dut.write_request_valid && dut.write_request_ready;
            const bool write_fire = dut.write_valid && dut.write_ready;
            const bool scalar_issue = dut.arithmetic_valid && dut.arithmetic_ready;
            const auto next_result = dut.arithmetic_multiply ? rtl_bf16_mul(dut.arithmetic_lhs, dut.arithmetic_rhs) : rtl_bf16_add(dut.arithmetic_lhs, dut.arithmetic_rhs);
            if (read_issue) {
                require(!reading, "overlapping reads"); reading = true;
                read_address = dut.read_request_address; read_bytes = dut.read_request_bytes; read_offset = 0; read_due = cycle+3;
                require(read_address+read_bytes <= memory.size() && read_bytes%16 == 0, "invalid read range");
            }
            if (read_fire) { read_offset += 16; if (dut.read_last) reading = false; }
            if (dut.read_error) { injected = true; reading = false; }
            if (write_issue) {
                require(!writing && dut.write_request_bytes != 0 && dut.write_request_bytes%16 == 0, "invalid update write"); writing = true;
                write_address = dut.write_request_address; write_offset = 0; write_bytes = dut.write_request_bytes;
                if (dut.start_clear_relation)
                    require(write_address >= dut.start_relation_base && write_address+write_bytes <= dut.start_relation_limit,
                        "relation clear write out of range");
                else require(write_address >= 0x80000 && write_address+write_bytes <= dut.start_pending_limit && write_bytes == 16,
                    "pending write out of range");
            }
            if (write_fire) {
                require(writing && bool(dut.write_last) == (write_offset+16 == write_bytes) && dut.write_byte_enable == 0xffff,
                    "invalid update write beat");
                for (unsigned byte = 0; byte < 16; ++byte) memory.at(write_address+write_offset+byte) = dut.write_data[byte/4]>>((byte%4)*8);
                write_offset += 16; write_due = cycle+4;
            }
            if (dut.write_done || dut.write_error) writing = false;
            if (dut.write_error) injected = true;
            if ((dut.arithmetic_result_valid && dut.arithmetic_result_ready) || dut.arithmetic_abort_ack) scalar = false;
            if (scalar_issue) { require(!scalar, "overlapping scalar operations"); scalar = true; scalar_result = next_result; scalar_due = cycle+5; }
            edge();
            if (((fault == 3 || fault == 4) && scalar) || (fault == 5 && reading)) { dut.abort_request = 1; injected = true; }
            acknowledged |= dut.abort_ack;
            if (fault >= 3 && fault <= 5 && acknowledged) break;
            if (dut.done_valid) break;
        }
        if (fault >= 3 && fault <= 5) {
            require(injected && acknowledged && !reading && !writing && !scalar, "pending abort did not drain");
            dut.abort_request = 0; edge();
        } else {
            require(dut.done_valid && bool(dut.error) == (fault != 0), "pending completion/error mismatch id="+std::to_string(dut.error_id));
            if (fault == 6) require(dut.error_id == 6, "missing action confidence must be a state format error");
            require(!reading && !writing && !scalar, "pending completion before drain");
            dut.done_ready = 1; edge(); dut.done_ready = 0;
        }
        ++completed;
        std::cout << "pending fault=" << fault << " rows=" << dut.start_rows << " cycles=" << cycle-begin << '\n';
    }
    void report() { std::cout << "PASS attention_pending DMA_fixture_sequences=" << completed << " cycles=" << cycle
        << " threads=" << dut.threads() << " raw/consumed/empty-change/read-write-error/abort-restart=checked\n"; }
};
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: attention_pending TRI_FIXTURE CROSS_FIXTURE");
        Regression regression; regression.run(argv[1], false); regression.run(argv[2], true); regression.report(); return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL attention_pending: " << error.what() << '\n'; return 1; }
}
