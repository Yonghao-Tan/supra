#include "Vaxi_master.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <cstdlib>

#ifdef AXI_PARAMETER_CHECK
int main(int argc, char** argv) {
    VerilatedContext context;
    context.commandArgs(argc, argv);
    context.threads(1);
    context.fatalOnError(false);
    Vaxi_master dut{&context};
    dut.eval();
    if (!context.gotError()) return 1;
    std::puts("PASS rejected AXI burst exceeding 256 beats");
    return 0;
}
#else
// Keep one read burst outstanding while its final R beat and the next AR
// handshake together. The peak must remain one across that exchange.
static void test_read_high_water() {
    VerilatedContext context;
    context.threads(1);
    Vaxi_master dut{&context};
    constexpr unsigned bytes = sizeof(dut.m_axi_rdata);
    constexpr unsigned burst_beats = 256 / bytes;
    unsigned cycles = 0, addresses = 0, beats = 0, simultaneous = 0;
    unsigned outstanding = 0, peak = 0;
    auto tick = [&]() {
        if (++cycles > 200) throw std::runtime_error("read high-water timeout");
        dut.clk = 0; dut.eval();
        const bool ar = dut.m_axi_arvalid && dut.m_axi_arready;
        const bool r = dut.m_axi_rvalid && dut.m_axi_rready;
        const bool last = r && dut.m_axi_rlast;
        dut.clk = 1; dut.eval();
        if (!dut.rst) {
            addresses += ar; beats += r; simultaneous += ar && last;
            outstanding += unsigned(ar); outstanding -= unsigned(last);
            if (outstanding > peak) peak = outstanding;
            if (dut.read_outstanding != outstanding ||
                dut.read_outstanding_high_water != peak)
                throw std::runtime_error("read outstanding high-water mismatch");
        }
        dut.clk = 0; dut.eval(); context.timeInc(1);
    };
    dut.rst = 1; dut.read_pair_count = 1; dut.read_data_ready = 1;
    tick(); tick(); dut.rst = 0;
    dut.read_request_address = 0x1000; dut.read_request_bytes = 512;
    dut.read_request_valid = 1; tick(); dut.read_request_valid = 0;
    dut.m_axi_arready = 1; tick(); dut.m_axi_arready = 0;
    for (unsigned burst = 0; burst < 2; ++burst) {
        dut.m_axi_rvalid = 1;
        for (unsigned beat = 0; beat < burst_beats; ++beat) {
            dut.m_axi_rlast = beat + 1 == burst_beats;
            dut.m_axi_arready = burst == 0 && dut.m_axi_rlast;
            tick();
        }
        dut.m_axi_rvalid = 0; dut.m_axi_arready = 0;
    }
    while (!dut.read_request_done) tick();
    if (dut.read_request_error || addresses != 2 || beats != 2 * burst_beats ||
        simultaneous != 1 || peak != 1 || !dut.read_idle)
        throw std::runtime_error("read high-water completion mismatch");
    std::printf("PASS simultaneous AR/RLAST: width=%u peak=%u\n", bytes * 8, peak);
    dut.final();
}

// Hold burst 1's AW while burst 0 returns B, then verify drain and restart.
static void test_stalled_aw(Vaxi_master& dut, VerilatedContext& context) {
    const bool error = context.commandArgsPlusMatch("response_error")[0] != '\0';
    const bool abort = context.commandArgsPlusMatch("abort_request")[0] != '\0';
    const char* delay_arg = context.commandArgsPlusMatch("aw_delay=");
    const unsigned delay = delay_arg[0] ? std::strtoul(delay_arg + 10, nullptr, 10) : 3;
    constexpr unsigned words = sizeof(dut.write_data) / sizeof(dut.write_data[0]);
    constexpr unsigned bytes = words * 4;
    constexpr unsigned burst_beats = 256 / bytes;
    constexpr uint32_t strobes = uint32_t((uint64_t{1} << bytes) - 1);
    unsigned cycles = 0, aw_count = 0, w_count = 0, b_count = 0, done_count = 0;
    unsigned outstanding = 0, peak = 0;
    bool restart = false;
    auto tick = [&]() {
        if (++cycles > 250) throw std::runtime_error("stalled AW drain/restart timeout");
        dut.clk = 0;
        dut.write_data_last = restart ? w_count == 3*burst_beats-1 : w_count == 2*burst_beats-1;
        dut.eval();
        const bool aw = dut.m_axi_awvalid && dut.m_axi_awready;
        const bool w = dut.m_axi_wvalid && dut.m_axi_wready;
        const bool b = dut.m_axi_bvalid && dut.m_axi_bready;
        if (!dut.rst && aw) {
            const uint64_t address = aw_count == 0 ? 0x1000 : aw_count == 1 ? 0x1100 : 0x2000;
            if (dut.m_axi_awaddr != address || dut.m_axi_awlen != burst_beats-1)
                throw std::runtime_error("AW payload mismatch");
        }
        if (!dut.rst && w) {
            const bool payload = restart || w_count < burst_beats || !(error || abort);
            if (dut.m_axi_wstrb != (payload ? strobes : 0) ||
                bool(dut.m_axi_wlast) != ((w_count+1)%burst_beats == 0))
                throw std::runtime_error("AW error drain W strobe/last mismatch");
            for (unsigned i=0; i<words; ++i)
                if (dut.m_axi_wdata[i] != (payload ? 0x12345678u : 0u))
                    throw std::runtime_error("AW error drain W data mismatch");
        }
        dut.clk = 1; dut.eval();
        if (!dut.rst) {
            aw_count += aw; w_count += w; b_count += b;
            outstanding += unsigned(aw); outstanding -= unsigned(b);
            if (outstanding > peak) peak = outstanding;
            if (dut.write_outstanding != outstanding ||
                dut.write_outstanding_high_water != peak)
                throw std::runtime_error("write outstanding high-water mismatch");
            if (dut.write_request_done) {
                ++done_count;
                if (bool(dut.write_request_error) != (!restart && (error || abort)))
                    throw std::runtime_error("AW error/restart completion status mismatch");
            }
        }
        dut.clk = 0; dut.eval(); context.timeInc(1);
    };
    dut.rst=1; dut.read_pair_count=1; dut.m_axi_arready=1; dut.read_data_ready=1;
    dut.m_axi_awready=1; dut.m_axi_wready=1;
    tick(); tick(); dut.rst=0;
    dut.write_request_address=0x1000; dut.write_request_bytes=512;
    dut.write_request_valid=1; dut.write_data_valid=1; dut.write_byte_enable=strobes;
    for (unsigned i=0;i<words;++i) dut.write_data[i]=0x12345678;
    dut.eval();
    if (!dut.write_request_ready) throw std::runtime_error("initial AW test request not ready");
    tick(); dut.write_request_valid=0;
    while (aw_count < 1) tick();
    dut.m_axi_awready=0;
    while (w_count < burst_beats) tick();
    tick(); // AWVALID has been presented with READY low.
    if (!dut.m_axi_awvalid || !dut.m_axi_bready || aw_count != 1)
        throw std::runtime_error("invalid stalled AW setup");
    dut.m_axi_bvalid=1; dut.m_axi_bresp=error ? 2:0; dut.abort_request=abort;
    dut.m_axi_awready=delay == 0;
    tick(); dut.m_axi_bvalid=0; dut.abort_request=0;
    for (unsigned i=1;i<delay;++i) tick();
    dut.m_axi_awready=1;
    while (w_count < 2*burst_beats) tick();
    if (done_count != 0) throw std::runtime_error("completed before final B");
    dut.m_axi_bvalid=1; dut.m_axi_bresp=0;
    while (b_count < 2) tick();
    dut.m_axi_bvalid=0;
    while (!dut.write_idle || done_count < 1) tick();
    if (dut.write_outstanding || aw_count != 2 || done_count != 1)
        throw std::runtime_error("first AW test request not drained");
    restart=true; dut.write_request_address=0x2000; dut.write_request_bytes=256;
    dut.write_request_valid=1; dut.eval();
    if (!dut.write_request_ready) throw std::runtime_error("restart not ready");
    tick(); dut.write_request_valid=0;
    while (w_count < 3*burst_beats) tick();
    dut.m_axi_bvalid=1;
    while (b_count < 3) tick();
    dut.m_axi_bvalid=0;
    while (!dut.write_idle || done_count < 2) tick();
    if (aw_count != 3 || done_count != 2 || dut.write_outstanding ||
        dut.actual_write_bytes != 768 || dut.useful_write_bytes != ((error || abort) ? 512u:768u))
        throw std::runtime_error("AW restart counters mismatch");
    std::printf("PASS stalled AW and restart: width=%u error=%u abort=%u delay=%u cycles=%u physical=%llu useful=%llu\n",
        bytes*8,error,abort,delay,cycles,
        static_cast<unsigned long long>(dut.actual_write_bytes),
        static_cast<unsigned long long>(dut.useful_write_bytes));
}

// Delay burst 0's B response until burst 1 holds a nonzero W beat.
// +response_error exercises SLVERR; +abort_request exercises abort.
// +stall_last holds the final beat instead of the first beat of burst 1.
int main(int argc, char** argv) {
    if (argc == 1) test_read_high_water();
    VerilatedContext context;
    context.commandArgs(argc, argv);
    context.threads(1);
    Vaxi_master dut{&context};
    std::printf("VERILATOR_MODEL_THREADS %u\n", dut.threads());
    if (context.commandArgsPlusMatch("stall_aw")[0] != '\0') {
        test_stalled_aw(dut, context); dut.final(); return 0;
    }
    const bool error = context.commandArgsPlusMatch("response_error")[0] != '\0';
    const bool abort = context.commandArgsPlusMatch("abort_request")[0] != '\0';
    const bool last = context.commandArgsPlusMatch("stall_last")[0] != '\0';
    constexpr unsigned words = sizeof(dut.write_data) / sizeof(dut.write_data[0]);
    constexpr unsigned beat_bytes = words * 4;
    constexpr unsigned total_beats = 512 / beat_bytes;
    constexpr uint32_t strobes = uint32_t((uint64_t{1} << beat_bytes) - 1);
    const unsigned held_after = last ? total_beats - 1 : total_beats / 2;
    unsigned cycles = 0, addresses = 0, beats = 0, source_beats = 0, responses = 0;
    unsigned enabled_bytes = 0;
    uint64_t accepted_data = 0;
    auto tick = [&]() {
        if (++cycles > 200) throw std::runtime_error("bounded AXI reproduction timeout");
        dut.clk = 0;
        dut.write_data_last = source_beats == total_beats - 1;
        dut.eval();
        const bool aw = dut.m_axi_awvalid && dut.m_axi_awready;
        const bool w = dut.m_axi_wvalid && dut.m_axi_wready;
        const bool src = dut.write_data_valid && dut.write_data_ready;
        const bool b = dut.m_axi_bvalid && dut.m_axi_bready;
        if (w && !dut.rst) {
            const bool payload = !(error || abort) || beats <= held_after;
            if (dut.m_axi_wstrb != (payload ? strobes : 0) ||
                bool(dut.m_axi_wlast) != ((beats + 1) % (total_beats / 2) == 0))
                throw std::runtime_error("accepted W strobe/last mismatch");
            for (unsigned i = 0; i < words; ++i) {
                if (dut.m_axi_wdata[i] != (payload ? 0x12345678u : 0u))
                    throw std::runtime_error("accepted W data mismatch");
                accepted_data += dut.m_axi_wdata[i];
            }
            enabled_bytes += __builtin_popcount(dut.m_axi_wstrb);
        }
        dut.clk = 1;
        dut.eval();
        if (!dut.rst) { addresses += aw; beats += w; source_beats += src; responses += b; }
        dut.clk = 0;
        dut.eval();
        context.timeInc(1);
    };
    auto show = [&](const char* label) {
        std::printf("%s cycle=%u aw=%u w=%u valid=%u ready=%u data=%08x%08x%08x%08x strb=%04x bresp=%u\n",
            label, cycles, addresses, beats, dut.m_axi_wvalid, dut.m_axi_wready,
            dut.m_axi_wdata[3], dut.m_axi_wdata[2], dut.m_axi_wdata[1], dut.m_axi_wdata[0],
            dut.m_axi_wstrb, dut.m_axi_bresp);
        std::fflush(stdout);
    };
    dut.rst = 1;
    dut.read_pair_count = 1;
    dut.m_axi_awready = 1;
    dut.m_axi_arready = 1;
    dut.read_data_ready = 1;
    tick(); tick();
    dut.rst = 0;
    dut.write_request_address = 0x1000;
    dut.write_request_bytes = 512;
    dut.write_request_valid = 1;
    dut.write_data_valid = 1;
    dut.write_byte_enable = strobes;
    for (unsigned i = 0; i < words; ++i) dut.write_data[i] = 0x12345678;
    dut.m_axi_wready = 1;
    dut.eval();
    if (!dut.write_request_ready) throw std::runtime_error("request not ready");
    tick();
    dut.write_request_valid = 0;
    while (beats < held_after) tick();
    dut.m_axi_wready = 0;
    while (addresses < 2) tick();
    tick();  // Present and hold second burst's first W beat before the response.
    if (!dut.m_axi_wvalid || !dut.m_axi_bready || dut.m_axi_wstrb != strobes || beats != held_after)
        throw std::runtime_error("invalid reproduction setup");
    dut.m_axi_bresp = error ? 2 : 0;
    dut.m_axi_bvalid = 1;
    dut.abort_request = abort;
    show("BEFORE_B");
    tick();
    dut.m_axi_bvalid = 0;
    dut.abort_request = 0;
    show("AFTER_B");
    tick(); tick();  // Existing RTL assertion must preserve the held W beat.
    dut.m_axi_wready = 1;
    while (beats < total_beats) tick();
    dut.m_axi_bresp = 0;
    dut.m_axi_bvalid = 1;
    while (responses < 2) tick();
    dut.m_axi_bvalid = 0;
    while (!dut.write_request_done) tick();
    if (bool(dut.write_request_error) != (error || abort)) throw std::runtime_error("completion mismatch");
    while (!dut.write_idle) tick();
    if (dut.write_outstanding || addresses != 2 || responses != 2 ||
        dut.actual_write_bytes != 512 || dut.useful_write_bytes != enabled_bytes ||
        enabled_bytes != ((error || abort) ? (held_after + 1) * beat_bytes : 512))
        throw std::runtime_error("write drain/counter mismatch");
    std::printf("PASS delayed B response with stalled later W: width=%u error=%u abort=%u last=%u cycles=%u physical=%llu useful=%u data_sum=%llu\n",
        beat_bytes * 8, error, abort, last, cycles,
        static_cast<unsigned long long>(dut.actual_write_bytes), enabled_bytes,
        static_cast<unsigned long long>(accepted_data));
    dut.final();
}
#endif
