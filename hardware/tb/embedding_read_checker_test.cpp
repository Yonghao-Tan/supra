#include "embedding_read_checker.hpp"
#include <cassert>
struct Dut {
    bool rst = false, debug_memory_read_request_valid = true, debug_memory_read_request_ready = true;
    std::uint64_t debug_read_address = 8192, debug_read_bytes = 8192;
};
int main() {
    EmbeddingReadChecker check; Dut dut;
    check.start(true, 8192, 32768, {8192, 16384});
    dut.debug_memory_read_request_ready = false; check.observe(dut);
    dut.debug_memory_read_request_ready = true; check.observe(dut);
    dut.debug_read_address = 16384; check.observe(dut);
    assert(check.finish().at("mismatches") == 0);
    check.start(true, 8192, 32768, {8192}); check.observe(dut);
    assert(check.finish().at("mismatches") == 1);
    check.start(true, 8192, 32768, {8192});
    assert(check.finish().at("missing") == 1);
    std::cout << "PASS testcase accepted embedding address/order/backpressure/missing checks\n";
}
