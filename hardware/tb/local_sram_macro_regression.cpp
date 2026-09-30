#include "Vlocal_sram_macro.h"
#include "verilated.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

double sc_time_stamp() { return 0.0; }

namespace {

class Regression {
public:
    explicit Regression(unsigned depth) : depth_(depth) {
        dut_.clk = 0;
        dut_.rst = 1;
        clear_requests();
        tick();
        tick();
        dut_.rst = 0;
        tick();
    }

    void run() {
        const unsigned maximum_address = depth_ - 1u;
        const uint32_t high_data[4] = {
            0x55667788u, 0x11223344u, 0xddeeff00u, 0x99aabbccu};
        const uint32_t low_data[4] = {
            0x89abcdefu, 0x01234567u, 0x76543210u, 0xfedcba98u};

        drive_write(maximum_address, high_data, 0u, low_data);
        tick();
        clear_requests();
        tick();

        drive_read(maximum_address, 0u);
        tick();
        clear_requests();
        wait_and_check_reads(high_data, low_data);

        uint32_t partial_data[4] = {};
        partial_data[0] = 0xaau;
        dut_.a_req_valid = 1;
        dut_.a_write = 1;
        dut_.a_address = maximum_address;
        set_wide(dut_.a_write_data, partial_data);
        set_mask(dut_.a_write_mask_n, true);
        dut_.a_write_mask_n[0] &= 0xffffff00u;
        tick();
        clear_requests();
        tick();

        uint32_t partial_expected[4] = {
            (high_data[0] & 0xffffff00u) | 0xaau,
            high_data[1], high_data[2], high_data[3]};
        drive_read(maximum_address, 0u);
        tick();
        clear_requests();
        wait_and_check_reads(partial_expected, low_data);

        dut_.rst = 1;
        tick();
        if (dut_.a_read_valid || dut_.b_read_valid || dut_.a_req_ready || dut_.b_req_ready)
            throw std::runtime_error("SRAM interface was active during reset");
        dut_.rst = 0;
        tick();
        drive_read(maximum_address, 0u);
        tick();
        clear_requests();
        wait_and_check_reads(partial_expected, low_data);

        dut_.a_req_valid = 1;
        dut_.a_write = 1;
        dut_.a_address = 1;
        dut_.b_req_valid = 1;
        dut_.b_write = 0;
        dut_.b_address = 1;
        dut_.eval();
        if (!dut_.same_address_conflict)
            throw std::runtime_error("same-address write collision was not reported");
        clear_requests();
        dut_.eval();

        dut_.final();
        std::cout << "PASS local_sram_macro: depth=" << depth_
                  << " ports=2 maximum_address=" << maximum_address
                  << " partial_write=bit-mask read_latency=2"
                  << " reset_retention=PASS collision_check=PASS\n";
    }

private:
    Vlocal_sram_macro dut_;
    unsigned depth_;

    void tick() {
        dut_.clk = 0;
        dut_.eval();
        dut_.clk = 1;
        dut_.eval();
        dut_.clk = 0;
        dut_.eval();
    }

    void clear_requests() {
        dut_.a_req_valid = 0;
        dut_.a_write = 0;
        dut_.a_address = 0;
        set_mask(dut_.a_write_mask_n, true);
        dut_.b_req_valid = 0;
        dut_.b_write = 0;
        dut_.b_address = 0;
        set_mask(dut_.b_write_mask_n, true);
    }

    template <typename Wide>
    static void set_wide(Wide &destination, const uint32_t words[4]) {
        for (unsigned word = 0; word < 4; ++word)
            destination[word] = words[word];
    }

    template <typename Wide>
    static void set_mask(Wide &destination, bool masked) {
        for (unsigned word = 0; word < 4; ++word)
            destination[word] = masked ? 0xffffffffu : 0u;
    }

    template <typename Wide>
    static void check_wide(const Wide &actual, const uint32_t expected[4],
                           const char *port) {
        for (unsigned word = 0; word < 4; ++word) {
            if (actual[word] != expected[word]) {
                std::cerr << port << " read mismatch word=" << word
                          << " expected=0x" << std::hex << expected[word]
                          << " actual=0x" << actual[word] << std::dec << '\n';
                throw std::runtime_error("SRAM read data mismatch");
            }
        }
    }

    void drive_write(unsigned a_address, const uint32_t a_data[4],
                     unsigned b_address, const uint32_t b_data[4]) {
        dut_.a_req_valid = 1;
        dut_.a_write = 1;
        dut_.a_address = a_address;
        set_wide(dut_.a_write_data, a_data);
        set_mask(dut_.a_write_mask_n, false);
        dut_.b_req_valid = 1;
        dut_.b_write = 1;
        dut_.b_address = b_address;
        set_wide(dut_.b_write_data, b_data);
        set_mask(dut_.b_write_mask_n, false);
    }

    void drive_read(unsigned a_address, unsigned b_address) {
        dut_.a_req_valid = 1;
        dut_.a_write = 0;
        dut_.a_address = a_address;
        dut_.b_req_valid = 1;
        dut_.b_write = 0;
        dut_.b_address = b_address;
    }

    void wait_and_check_reads(const uint32_t a_expected[4],
                              const uint32_t b_expected[4]) {
        if (dut_.a_read_valid || dut_.b_read_valid)
            throw std::runtime_error("SRAM read response arrived before two cycles");
        tick();
        if (!dut_.a_read_valid || !dut_.b_read_valid)
            throw std::runtime_error("SRAM read response was not exactly two cycles");
        check_wide(dut_.a_read_data, a_expected, "port A");
        check_wide(dut_.b_read_data, b_expected, "port B");
        tick();
        if (dut_.a_read_valid || dut_.b_read_valid)
            throw std::runtime_error("SRAM read response was duplicated");
    }
};

}  // namespace

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " DEPTH\n";
        return 2;
    }
    try {
        Regression regression(static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10)));
        regression.run();
    } catch (const std::exception &error) {
        std::cerr << "FAIL local_sram_macro: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
