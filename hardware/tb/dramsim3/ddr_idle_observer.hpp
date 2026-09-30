#pragma once

#include <cstdint>
#include <ostream>
#include <stdexcept>

// Records request-free intervals after both DMA/AXI paths and the DDR adapter
// drain. Refresh, bank timing and power-down entry/exit remain model inputs.
class DdrIdleObserver {
  public:
    explicit DdrIdleObserver(std::ostream& output) : output_(output) {
        output_ << "start_cycle\tend_cycle\tcycles\tnext_request\tboundary\n";
    }

    void observe(std::uint64_t cycle, bool drained,
                 bool read_requested, bool write_requested) {
        if (finished_ || (seen_ && cycle != last_cycle_ + 1))
            throw std::runtime_error("DDR idle samples must be consecutive and precede finish");
        const bool idle = drained && !read_requested && !write_requested;
        if (idle && !active_) {
            start_ = cycle;
            leading_ = !seen_;
            active_ = true;
        } else if (!idle && active_) {
            emit(cycle, read_requested, write_requested,
                 leading_ ? "leading" : "none");
        }
        seen_ = true;
        last_cycle_ = cycle;
    }

    void finish() {
        if (finished_)
            throw std::runtime_error("DDR idle observer finished twice");
        if (active_)
            emit(last_cycle_ + 1, false, false,
                 leading_ ? "leading_and_trailing" : "trailing");
        output_.flush();
        if (!output_)
            throw std::runtime_error("DDR idle trace write failed");
        finished_ = true;
    }

  private:
    void emit(std::uint64_t end, bool read, bool write, const char* boundary) {
        const char* direction = read ? (write ? "read_write" : "read") :
            (write ? "write" : "none");
        output_ << start_ << '\t' << end << '\t' << end - start_ << '\t'
                << direction << '\t' << boundary << '\n';
        active_ = false;
    }

    std::ostream& output_;
    std::uint64_t start_ = 0, last_cycle_ = 0;
    bool seen_ = false, active_ = false, leading_ = false, finished_ = false;
};
