#include "Vattention_invalidation.h"
#include "verilated.h"
#include "json.hpp"
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
static void require(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error(message);
}
static void flatten(const Json& value, std::vector<unsigned>& result) {
    if (value.is_array()) for (const auto& item : value) flatten(item, result);
    else result.push_back(value.is_boolean() ? unsigned(value.get<bool>()) : value.get<unsigned>());
}
static std::vector<unsigned> raw(const Json& value) {
    std::vector<unsigned> result;
    flatten(value.at("raw"), result);
    return result;
}

class Regression {
    Vattention_invalidation dut;
    std::uint64_t cycles = 0, due = 0;
    unsigned rows_checked = 0;
    bool pending = false, cancel = false;
    std::uint16_t response = 0;
    void tick() {
        dut.arithmetic_ready = !pending && cycles % 5 != 1;
        dut.arithmetic_abort_ack = pending && cancel && dut.arithmetic_abort_request;
        dut.arithmetic_result_valid = pending && !dut.arithmetic_abort_ack && cycles >= due;
        dut.arithmetic_result = response;
        dut.eval();
        const bool issue = dut.arithmetic_valid && dut.arithmetic_ready;
        const auto next = dut.arithmetic_multiply ?
            rtl_bf16_mul(dut.arithmetic_lhs, dut.arithmetic_rhs) :
            rtl_bf16_add(dut.arithmetic_lhs, dut.arithmetic_rhs);
        if ((dut.arithmetic_result_valid && dut.arithmetic_result_ready) || dut.arithmetic_abort_ack)
            pending = false;
        if (issue) {
            require(!pending, "overlapping scalar operations");
            response = next; pending = true; due = cycles + 4;
        }
        dut.clk = 1; dut.eval(); ++cycles; dut.clk = 0; dut.eval();
    }
public:
    Regression() { dut.rst = 1; tick(); dut.rst = 0; }
    void run(const char* path, bool cross) {
        std::ifstream stream(path);
        require(stream.good(), "missing dependency fixture");
        Json fixture; stream >> fixture;
        run_document(fixture, cross);
        if (fixture.contains("variants"))
            for (const auto& variant : fixture.at("variants")) run_document(variant, cross);
    }
    void run_document(const Json& fixture, bool cross) {
        const auto mode = fixture.value("pending_confidence_mode", std::string("all_changes"));
        const unsigned rows = cross ? fixture.at("rows").get<unsigned>() :
            fixture.at("region_end").get<unsigned>() - fixture.at("region_start").get<unsigned>();
        const unsigned initial_start = cross ? fixture.at("block_start").get<unsigned>() :
            fixture.at("region_start").get<unsigned>();
        const unsigned keys = cross ? 32 : rows;
        for (const auto& record : fixture.at("records")) {
            const unsigned start = record.value("block_start", initial_start);
            const bool advance = cross && record.value("advance_block", false);
            const auto& input = record.at("inputs");
            const auto& before = record.at("before");
            const auto& expected = record.at("expected");
            const auto relation = raw(expected.at(cross ? "relation" : "dependency"));
            const auto changed = raw(input.at("changed_global"));
            const auto remasked = raw(input.at("changed_remask_global"));
            const auto confidence = raw(input.at("changed_confidence_global"));
            std::vector<unsigned> consumed(rows, 0), active;
            if (cross) for (auto row : raw(input.at("consumed_positions"))) consumed.at(row) = 1;
            else consumed = raw(before.at("refresh"));
            for (unsigned key = 0; key < keys; ++key) if (changed.at(start + key)) active.push_back(key);
            for (unsigned row = 0; row < rows; ++row) {
                const bool future = cross && row >= start + 32;
                const auto field = future ? "future_pending" : "pending";
                auto old_pending = raw(before.at(field)).at(row);
                // The scalar reducer receives the state after advance_block
                // merges future scores. Its parent is checked separately.
                if (advance && !future)
                    old_pending = std::max(old_pending, raw(before.at("future_pending")).at(row));
                const auto wanted = raw(expected.at(field)).at(row);
                dut.start_all_changes = cross && mode == "all_changes";
                dut.start_stable_unmask = cross && mode == "stable_unmask"; dut.start_pending = old_pending;
                dut.start_consumed = consumed.at(row); dut.start_changed_count = active.size();
                dut.start_valid = 1; dut.result_ready = 0; dut.value_valid = 0;
                dut.eval(); require(dut.start_ready, "row start not ready");
                tick(); dut.start_valid = 0;
                unsigned column = 0;
                for (unsigned wait = 0; wait < 20000 && !dut.result_valid; ++wait) {
                    dut.value_valid = column < active.size() && cycles % 3 != 0;
                    if (column < active.size()) {
                        const auto key = active[column];
                        dut.value_dependency = relation.at(row * keys + key);
                        dut.value_confidence = confidence.at(start + key);
                        dut.value_remasked = remasked.at(start + key);
                    }
                    dut.eval(); const bool accepted = dut.value_valid && dut.value_ready;
                    tick(); if (accepted) ++column;
                }
                dut.value_valid = 0;
                require(dut.result_valid && !dut.error && column == active.size(), "invalidation did not complete");
                const auto context = " row=" + std::to_string(row) + " step=" + record.at("step_index").dump() +
                    " block_start=" + std::to_string(start) + " mode=" + mode +
                    " actual=" + std::to_string(dut.result_pending) + " expected=" + std::to_string(wanted);
                require(dut.result_pending == wanted, "pending raw mismatch" + context);
                if (!cross) require(dut.result_invalidation == raw(expected.at("new_invalidation")).at(row),
                    "invalidation raw mismatch" + context);
                else {
                    const auto remask_field = future ? "future_actual_remask_pending" : "actual_remask_pending";
                    auto old_remask = raw(before.at(remask_field)).at(row);
                    if (advance && !future)
                        old_remask = std::max(old_remask, raw(before.at("future_actual_remask_pending")).at(row));
                    if (consumed[row]) old_remask = 0;
                    require(std::max<unsigned>(old_remask, dut.result_remask) == raw(expected.at(remask_field)).at(row),
                        "remask raw mismatch" + context);
                }
                const auto held_pending = dut.result_pending;
                for (unsigned hold = 0; hold < 3; ++hold) {
                    tick(); require(dut.result_valid && dut.result_pending == held_pending, "result changed under backpressure");
                }
                dut.result_ready = 1; tick(); dut.result_ready = 0; ++rows_checked;
            }
        }
    }
    void abort_test(bool cancelled) {
        cancel = cancelled;
        dut.start_changed_count = 1; dut.start_all_changes = 0; dut.start_stable_unmask = 0;
        dut.start_valid = 1; tick(); dut.start_valid = 0;
        dut.value_dependency = 0x3e80; dut.value_confidence = 0x3f00;
        dut.value_remasked = 1; dut.value_valid = 1;
        tick(); dut.value_valid = 0;
        for (unsigned wait = 0; wait < 30 && !pending; ++wait) tick();
        require(pending, "abort test did not reach in-flight arithmetic");
        dut.abort_request = 1;
        bool acknowledged = false;
        for (unsigned wait = 0; wait < 20; ++wait) {
            tick(); acknowledged |= dut.abort_ack;
            require(!dut.result_valid && !dut.arithmetic_valid, "abort issued work");
        }
        require(acknowledged && !pending, "arithmetic abort did not drain");
        dut.abort_request = 0; tick(); cancel = false;
        require(dut.start_ready, "restart not ready");
    }
    void report() { std::cout << "PASS attention_invalidation rows=" << rows_checked
        << " cycles=" << cycles << " threads=" << dut.threads()
        << " regular/cross-block/remask-order/backpressure/abort=checked\n"; }
};

int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: attention_invalidation TRI_FIXTURE CROSS_FIXTURE");
        Regression regression;
        regression.run(argv[1], false);
        regression.abort_test(false);
        regression.abort_test(true);
        regression.run(argv[2], true);
        regression.report();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL attention_invalidation: " << error.what() << '\n'; return 1;
    }
}
