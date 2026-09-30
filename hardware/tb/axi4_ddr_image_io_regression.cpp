#include "Vaxi4_ddr_model.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr std::size_t kBeatBytes = 16;
vluint64_t simulation_time = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "FAIL axi4_ddr_image_io: %s\n", message);
    std::exit(1);
}

void tick(Vaxi4_ddr_model& dut) {
    dut.clk = 0;
    dut.eval();
    ++simulation_time;
    dut.clk = 1;
    dut.eval();
    ++simulation_time;
}

void issue_read_address(Vaxi4_ddr_model& dut, std::uint64_t address,
                        std::uint8_t beats, std::uint8_t id) {
    dut.s_axi_arid = id;
    dut.s_axi_araddr = address;
    dut.s_axi_arlen = beats - 1;
    dut.s_axi_arvalid = 1;
    int cycles = 0;
    while (!dut.s_axi_arready && cycles++ < 256) tick(dut);
    if (!dut.s_axi_arready) fail("ARREADY did not assert");
    tick(dut);
    dut.s_axi_arvalid = 0;
}

void wait_read_response(Vaxi4_ddr_model& dut, std::uint8_t expected_id,
                        std::uint8_t expected_response, bool expected_last) {
    int cycles = 0;
    while (!dut.s_axi_rvalid && cycles++ < 256) tick(dut);
    if (!dut.s_axi_rvalid) fail("RVALID did not assert");
    if (dut.s_axi_rid != expected_id || dut.s_axi_rresp != expected_response ||
        static_cast<bool>(dut.s_axi_rlast) != expected_last)
        fail("read response metadata is incorrect");
    tick(dut);
}

void issue_write_burst(Vaxi4_ddr_model& dut, std::uint64_t address,
                       std::uint8_t beats, std::uint8_t id,
                       std::uint8_t expected_id,
                       std::uint8_t expected_response) {
    dut.s_axi_awid = id;
    dut.s_axi_awaddr = address;
    dut.s_axi_awlen = beats - 1;
    dut.s_axi_awvalid = 1;
    int cycles = 0;
    while (!dut.s_axi_awready && cycles++ < 256) tick(dut);
    if (!dut.s_axi_awready) fail("AWREADY did not assert");
    tick(dut);
    dut.s_axi_awvalid = 0;
    dut.s_axi_wstrb = 0xffff;
    dut.s_axi_wvalid = 1;
    for (std::uint8_t beat = 0; beat < beats; ++beat) {
        dut.s_axi_wlast = beat + 1 == beats;
        cycles = 0;
        while (!dut.s_axi_wready && cycles++ < 256) tick(dut);
        if (!dut.s_axi_wready) fail("WREADY did not assert");
        tick(dut);
    }
    dut.s_axi_wvalid = 0;
    cycles = 0;
    while (!dut.s_axi_bvalid && cycles++ < 256) tick(dut);
    if (!dut.s_axi_bvalid) fail("BVALID did not assert");
    if (dut.s_axi_bid != expected_id || dut.s_axi_bresp != expected_response)
        fail("write response metadata is incorrect");
    tick(dut);
}

void initialize_inputs(Vaxi4_ddr_model& dut) {
    dut.clk = 0;
    dut.rst = 1;
    dut.random_stall = 0;
    dut.performance_config = 0;
    dut.inject_read_id_error = 0;
    dut.inject_read_last_error = 0;
    dut.inject_read_response_error = 0;
    dut.inject_write_id_error = 0;
    dut.inject_write_response_error = 0;
    dut.s_axi_awid = 0;
    dut.s_axi_awaddr = 0;
    dut.s_axi_awlen = 0;
    dut.s_axi_awsize = 4;
    dut.s_axi_awburst = 1;
    dut.s_axi_awvalid = 0;
    for (int word = 0; word < 4; ++word) dut.s_axi_wdata[word] = 0;
    dut.s_axi_wstrb = 0;
    dut.s_axi_wlast = 0;
    dut.s_axi_wvalid = 0;
    dut.s_axi_bready = 1;
    dut.s_axi_arid = 0;
    dut.s_axi_araddr = 0;
    dut.s_axi_arlen = 0;
    dut.s_axi_arsize = 4;
    dut.s_axi_arburst = 1;
    dut.s_axi_arvalid = 0;
    dut.s_axi_rready = 1;
    dut.image_check_request = 0;
}

void wait_for_read(Vaxi4_ddr_model& dut,
                   const std::array<std::uint8_t, kBeatBytes>& expected) {
    issue_read_address(dut, 0, 1, 3);
    int cycles = 0;
    while (!dut.s_axi_rvalid && cycles++ < 64) tick(dut);
    if (!dut.s_axi_rvalid) fail("RVALID did not assert");
    if (dut.s_axi_rid != 3 || dut.s_axi_rresp != 0 || !dut.s_axi_rlast)
        fail("single-beat read response metadata is incorrect");
    for (std::size_t byte = 0; byte < expected.size(); ++byte) {
        const std::uint8_t actual =
            static_cast<std::uint8_t>((dut.s_axi_rdata[byte / 4] >>
                                      ((byte % 4) * 8)) & 0xffu);
        if (actual != expected[byte]) fail("preloaded AXI read payload mismatch");
    }
    tick(dut);
}

void check_preserved_protocol_behavior(Vaxi4_ddr_model& dut) {
    dut.s_axi_rready = 0;
    dut.random_stall = 1;
    for (std::uint8_t entry = 0; entry < 8; ++entry)
        issue_read_address(dut, 0, 1, entry);
    if (dut.max_read_queued != 8 || dut.s_axi_arready)
        fail("eight-entry read queue did not reach its bounded capacity");
    dut.random_stall = 0;
    dut.s_axi_rready = 1;
    for (std::uint8_t entry = 0; entry < 8; ++entry)
        wait_read_response(dut, entry, 0, true);

    dut.inject_read_id_error = 1;
    issue_read_address(dut, 0, 1, 4);
    wait_read_response(dut, 5, 0, true);
    dut.inject_read_id_error = 0;

    dut.inject_read_last_error = 1;
    issue_read_address(dut, 0, 1, 4);
    wait_read_response(dut, 4, 0, false);
    dut.inject_read_last_error = 0;

    dut.inject_read_response_error = 1;
    issue_read_address(dut, 0, 2, 4);
    wait_read_response(dut, 4, 2, false);
    wait_read_response(dut, 4, 0, true);
    dut.inject_read_response_error = 0;

    dut.inject_write_id_error = 1;
    issue_write_burst(dut, 64, 1, 6, 7, 0);
    dut.inject_write_id_error = 0;
    dut.inject_write_response_error = 1;
    issue_write_burst(dut, 80, 1, 6, 6, 2);
    dut.inject_write_response_error = 0;
    issue_write_burst(dut, 0xff0, 2, 6, 6, 2);
}

void check_performance_service_window(Vaxi4_ddr_model& dut) {
    dut.performance_config = 1;
    issue_read_address(dut, 0, 16, 2);
    int wait_cycles = 0;
    while (!dut.s_axi_rvalid && wait_cycles++ < 64) tick(dut);
    if (!dut.s_axi_rvalid) fail("performance read did not start");

    int accepted = 0;
    for (int cycle = 0; cycle < 10; ++cycle) {
        if (dut.s_axi_rvalid && dut.s_axi_rready) ++accepted;
        tick(dut);
    }
    if (accepted != 8)
        fail("performance service did not accept four beats per five cycles");

    int total_accepted = accepted;
    int drain_cycles = 0;
    while (total_accepted < 16 && drain_cycles++ < 64) {
        if (dut.s_axi_rvalid && dut.s_axi_rready) ++total_accepted;
        tick(dut);
    }
    if (total_accepted != 16) fail("performance read did not drain");
    dut.performance_config = 0;
}

void check_write_while_read_stalled(Vaxi4_ddr_model& dut) {
    // Exercise both scheduling settings. The master can hold RREADY low
    // while waiting for a previous read beat to complete its writeback.
    for (int performance = 0; performance < 2; ++performance) {
        dut.performance_config = performance;
        dut.s_axi_rready = 0;
        issue_read_address(dut, 0, 1, 3);
        int cycles = 0;
        while (!dut.s_axi_rvalid && cycles++ < 64) tick(dut);
        if (!dut.s_axi_rvalid) fail("held read did not become valid");
        std::array<std::uint32_t, 4> held{};
        for (int word = 0; word < 4; ++word) held[word] = dut.s_axi_rdata[word];
        issue_write_burst(dut, 96, 1, 6, 6, 0);
        if (!dut.s_axi_rvalid || dut.s_axi_rid != 3 ||
            dut.s_axi_rresp != 0 || !dut.s_axi_rlast)
            fail("write changed the stalled read response");
        for (int word = 0; word < 4; ++word)
            if (dut.s_axi_rdata[word] != held[word])
                fail("write changed the stalled read payload");
        dut.s_axi_rready = 1;
        tick(dut);
    }
    dut.performance_config = 0;
}

void request_image_check(Vaxi4_ddr_model& dut,
                         std::uint32_t expected_mismatches,
                         bool expected_pass) {
    dut.image_check_request = 1;
    tick(dut);
    dut.image_check_request = 0;
    if (!dut.image_check_done) fail("image check did not complete on request");
    if (dut.image_check_mismatch_count != expected_mismatches)
        fail("image mismatch count is incorrect");
    if (static_cast<bool>(dut.image_check_pass) != expected_pass)
        fail("image pass flag is incorrect");
    if (dut.expected_image_bytes != 64) fail("expected image byte count is incorrect");
    tick(dut);
    if (dut.image_check_done) fail("image check done must be a one-cycle pulse");
}

void write_first_beat(Vaxi4_ddr_model& dut,
                      const std::array<std::uint8_t, kBeatBytes>& payload) {
    dut.s_axi_awid = 5;
    dut.s_axi_awaddr = 0;
    dut.s_axi_awlen = 0;
    dut.s_axi_awvalid = 1;
    int cycles = 0;
    while (!dut.s_axi_awready && cycles++ < 32) tick(dut);
    if (!dut.s_axi_awready) fail("AWREADY did not assert");
    tick(dut);
    dut.s_axi_awvalid = 0;

    for (std::size_t byte = 0; byte < payload.size(); ++byte)
        dut.s_axi_wdata[byte / 4] |=
            static_cast<std::uint32_t>(payload[byte]) << ((byte % 4) * 8);
    dut.s_axi_wstrb = 0xffff;
    dut.s_axi_wlast = 1;
    dut.s_axi_wvalid = 1;
    cycles = 0;
    while (!dut.s_axi_wready && cycles++ < 32) tick(dut);
    if (!dut.s_axi_wready) fail("WREADY did not assert");
    tick(dut);
    dut.s_axi_wvalid = 0;

    cycles = 0;
    while (!dut.s_axi_bvalid && cycles++ < 64) tick(dut);
    if (!dut.s_axi_bvalid) fail("BVALID did not assert");
    if (dut.s_axi_bid != 5 || dut.s_axi_bresp != 0)
        fail("single-beat write response metadata is incorrect");
    tick(dut);
}

}  // namespace

double sc_time_stamp() {
    return static_cast<double>(simulation_time);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vaxi4_ddr_model dut;
    initialize_inputs(dut);
    dut.eval();
    if (!dut.initial_image_loaded || dut.initial_image_bytes != 64)
        fail("64-byte initial image was not loaded");

    for (int cycle = 0; cycle < 3; ++cycle) tick(dut);
    dut.rst = 0;
    tick(dut);

    std::array<std::uint8_t, kBeatBytes> initial{};
    std::array<std::uint8_t, kBeatBytes> expected{};
    for (std::size_t byte = 0; byte < kBeatBytes; ++byte) {
        initial[byte] = static_cast<std::uint8_t>(byte);
        expected[byte] = static_cast<std::uint8_t>(0xa0u + byte);
    }

    wait_for_read(dut, initial);
    request_image_check(dut, 16, false);
    write_first_beat(dut, expected);
    request_image_check(dut, 0, true);
    wait_for_read(dut, expected);
    check_preserved_protocol_behavior(dut);
    check_performance_service_window(dut);
    check_write_while_read_stalled(dut);

    std::printf(
        "PASS axi4_ddr_image_io initial_bytes=64 first_mismatches=16 "
        "final_mismatches=0 final_pass=1 read_queue_high_water=8 "
        "stall_error_4k_checks=1 performance_read_beats_per_10_cycles=8 "
        "write_during_held_read=1\n");
    dut.final();
    return 0;
}
