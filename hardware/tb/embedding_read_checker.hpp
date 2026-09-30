#ifndef SUPRA_TESTCASE_EMBEDDING_CHECKER_HPP
#define SUPRA_TESTCASE_EMBEDDING_CHECKER_HPP
#include <json.hpp>
#include <cstdint>
#include <iostream>
#include <vector>

class EmbeddingReadChecker {
    bool enabled_ = false;
    std::uint64_t base_ = 0, limit_ = 0, events_ = 0, mismatches_ = 0;
    std::vector<std::uint64_t> addresses_;
public:
    void start(bool enabled, std::uint64_t base, std::uint64_t limit,
               std::vector<std::uint64_t> addresses) {
        enabled_ = enabled; base_ = base; limit_ = limit;
        addresses_ = std::move(addresses); events_ = mismatches_ = 0;
    }
    template<class Dut>
    void observe(const Dut& dut) {
        if (!enabled_ || dut.rst || !dut.debug_memory_read_request_valid ||
            !dut.debug_memory_read_request_ready || dut.debug_read_address < base_ ||
            dut.debug_read_address >= limit_) return;
        if (events_ >= addresses_.size() || dut.debug_read_bytes != 8192 ||
            dut.debug_read_address != addresses_[events_]) {
            if (mismatches_ < 8) std::cerr << "TESTCASE_EMBEDDING_MISMATCH event=" << events_
                << " address=" << dut.debug_read_address << " bytes=" << dut.debug_read_bytes << '\n';
            ++mismatches_;
        }
        ++events_;
    }
    nlohmann::json finish() {
        const auto missing = addresses_.size() > events_ ? addresses_.size() - events_ : 0;
        enabled_ = false;
        return {{"events", events_}, {"expected", addresses_.size()}, {"missing", missing},
                {"mismatches", mismatches_ + missing}};
    }
};
#endif
