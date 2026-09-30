#include "Vmatmul_output_controller_top.h"
#include "verilated.h"

extern "C" {
#include "../cmodel/rtl_numeric.h"
}

#include <array>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

double sc_time_stamp() { return 0.0; }

namespace {
#ifndef MATMUL_OUTPUT_ROWS
#define MATMUL_OUTPUT_ROWS 3
#endif
constexpr int kRows = MATMUL_OUTPUT_ROWS;
constexpr int kLayoutRows = kRows + (kRows & 1);
constexpr int kChannels = 16;

std::uint16_t input_value(int row, int channel) {
    return rtl_f32_to_bf16(float(row * 19 + channel - 11) / 8.0f);
}

std::uint16_t residual_value(int row, int channel) {
    return rtl_f32_to_bf16(float(row * 7 - channel + 5) / 16.0f);
}

template <typename Wide>
void clear_wide(Wide& words, int count) {
    for (int index = 0; index < count; ++index) words[index] = 0;
}

template <typename Wide>
void set16(Wide& words, int lane, std::uint16_t value) {
    const int word = lane / 2;
    const int shift = (lane % 2) * 16;
    words[word] = (words[word] & ~(0xffffu << shift)) |
                  (std::uint32_t(value) << shift);
}

template <typename Wide>
std::uint16_t get16(const Wide& words, int lane) {
    return std::uint16_t(words[lane / 2] >> ((lane % 2) * 16));
}

class Regression {
  public:
    Regression() { reset(); }

    void run() {
        for (int completion = 0; completion < 3; ++completion)
            abort_with_prefetched_read_pending(completion);
        std::cerr << "RUN matmul_output_controller local\n";
        run_mode(0, 0x10000, 64, false, false);
        run_two_local_batches(kRows);
        run_two_local_batches(1);
        std::cerr << "RUN matmul_output_controller paired_gate\n";
        run_mode(1, 0x20000, 0, false, true);
        std::cerr << "RUN matmul_output_controller paired_up\n";
        run_mode(2, 0x30000, 0, false, true);
        run_two_batches(1, 0x60000);
        run_two_batches(2, 0x70000);
        if constexpr (kRows == 48) {
            run_batches(1, 0x80000, {48, 48, 48, 48}, "four-full-gate");
            run_batches(2, 0x90000, {48, 48, 48, 48}, "four-full-up");
            run_batches(1, 0xd0000, {48, 48, 48, 48, 48, 48},
                        "six-full-gate");
            run_batches(2, 0xe0000, {48, 48, 48, 48, 44},
                        "five-tail-up");
            run_batches(1, 0xa0000, {48, 48, 44}, "three-tail-gate");
            run_batches(2, 0xb0000, {48, 28}, "two-tail-up");
            run_batches(1, 0xc0000, {48, 48, 5}, "single-phase-tail-gate");
        }
        std::cerr << "RUN matmul_output_controller residual_local\n";
        run_mode(3, 0x40000, 64, true, false);
        std::cerr << "RUN matmul_output_controller residual_ddr\n";
        run_mode(4, 0x50000, 0, true, true);
        if constexpr (kRows == 32) run_strided_residual();
        run_residual_batches({kRows, kRows}, 0);
        if constexpr (kRows > 1) {
            run_residual_batches({kRows, 1}, 0);
            run_residual_batches({kRows, kRows - 1}, 0);
            run_residual_batches({kRows, 1}, 0, true);
            run_residual_batches({kRows, 1, kRows, 2, 3, 1}, 0, true);
            for (int failure : {1, 2, 3, 4, 5, 6})
                run_residual_batches({kRows, 1}, failure);
        }
        std::cerr << "RUN matmul_output_controller read_error\n";
        read_error_and_restart();
        std::cerr << "RUN matmul_output_controller write_error_overlap\n";
        write_error_during_next_stripe();
        std::cerr << "RUN matmul_output_controller abort_read_pending\n";
        abort_with_read_pending();
        std::cerr << "RUN matmul_output_controller abort_write_pending\n";
        abort_with_write_pending();
        if constexpr (kRows == 48) {
            if (!std::all_of(staging_address_seen_.begin(),
                             staging_address_seen_.begin() + 96,
                             [](bool seen) { return seen; }))
                throw std::runtime_error(
                    "48-row staging did not cover addresses 0..95");
            if (std::any_of(staging_address_seen_.begin() + 96,
                            staging_address_seen_.end(),
                            [](bool seen) { return seen; }))
                throw std::runtime_error(
                    "48-row staging accessed an address above 95");
        }
        run_batches(5, 0x100000, {kRows, 1}, "fused-tail");
        run_batches(5, 0x120000, {kRows, 3, 1}, "fused-three-batches");
        if constexpr (kRows == 1)
            run_batches(6, 0x180000, {1, 1, 1, 1, 1, 1},
                        "ddr-six-single-token-batches");
        for (int state : {16, 18, 20}) abort_fused_at(state);
        if constexpr (kRows == 48) {
            run_batches(5, 0x140000, {48, 48, 48, 48, 48, 48},
                        "fused-288-tokens");
            require(std::all_of(staging_address_seen_.begin() + 96,
                                staging_address_seen_.begin() + 384,
                                [](bool seen) { return seen; }),
                    "Gate scratch did not cover all 288 tokens");
            run_batches(6, 0x200000, {48, 1}, "ddr-two-batch-tail");
            run_batches(6, 0x220000, {48, 48, 48, 48, 48, 48},
                        "ddr-six-batch-288-tokens");
        }
        dut_.final();
        std::cout << "PASS matmul_output_controller: modes="
                  << (kRows == 48 || kRows == 1 ? 7 : 6) << " rows=" << kRows
                  << " channels=16 "
                     "paired_slots=2 two_batches=2 four_batch_cases="
                  << (kRows == 48 ? 5 : 0)
                  << " residual_raw_mismatch=0 "
                     "write_payload_bytes=" << kRows * 16 << " "
                     "read_error=1 write_error_overlap=1 "
                     "abort_read_pending=1 abort_write_pending=1 abort_prefetch_cases=3 "
                     "restart=checked staging_address_range="
                  << (kRows == 48 ? "0..383" : "focused")
                  << " fused_abort_states=16,18,20"
                  << " residual_batch_cases=11 runtime_threads=" << dut_.threads()
                  << " cycles="
                  << cycles_ << "\n";
    }

  private:
    Vmatmul_output_controller_top dut_;
    std::uint64_t cycles_ = 0;
    std::map<std::uint64_t, std::uint16_t> observed_;
    int row_commits_ = 0;
    bool last_residual_accepted_ = false;
    bool last_residual_request_accepted_ = false;
    bool last_fragment_accepted_ = false;
    bool last_output_accepted_ = false;
    bool output_last_seen_ = false;
    bool output_response_pending_ = false;
    int output_overlap_fragments_ = 0;
    int output_accepted_beats_ = 0;
    int transfer_accepted_beats_ = 0;
    int active_transfer_rows_ = kRows;
    bool staging_read_pending_ = false;
    std::array<bool, 384> staging_address_seen_{};

    void require(bool condition, const std::string& message) {
        if (!condition) throw std::runtime_error(message);
    }

    void clear_inputs() {
        dut_.abort_request = 0;
        dut_.cfg_valid = 0;
        dut_.cfg_mode = 0;
        dut_.cfg_physical_row_base = 0;
        dut_.cfg_row_count = kRows;
        dut_.cfg_batch_count = 1;
        dut_.cfg_second_batch_rows = 0;
        dut_.cfg_third_batch_rows = 0;
        dut_.cfg_fourth_batch_rows = 0;
        dut_.cfg_fifth_batch_rows = 0;
        dut_.cfg_sixth_batch_rows = 0;
        dut_.cfg_output_channels = kChannels;
        dut_.cfg_output_base = 0;
        dut_.cfg_output_limit = 0;
        dut_.cfg_local_row_stride = 0;
        dut_.cfg_residual_from_ddr = 0;
        dut_.cfg_residual_base = 0;
        dut_.cfg_residual_limit = 0;
        dut_.cfg_residual_storage_rows = 0;
        dut_.in_valid = 0;
        dut_.in_batch_index = 0;
        dut_.in_physical_row = 0;
        dut_.in_output_channel = 0;
        dut_.in_row_byte_base = 0;
        clear_wide(dut_.in_data, 4);
        dut_.in_byte_enable = 0xffff;
        dut_.residual_read_request_ready = 0;
        dut_.residual_read_request_done = 0;
        dut_.residual_read_request_error = 0;
        dut_.residual_read_data_valid = 0;
        clear_wide(dut_.residual_read_data, 4);
        dut_.residual_read_byte_enable = 0xffff;
        dut_.residual_read_data_last = 0;
        dut_.residual_read_data_tag = 0;
        dut_.local_write_ready = 1;
        dut_.output_write_ready = 1;
        dut_.output_write_done = 0;
        dut_.output_write_error = 0;
        output_response_pending_ = false;
    }

    void reset() {
        clear_inputs();
        dut_.rst = 1;
        for (int cycle = 0; cycle < 4; ++cycle) tick(false);
        dut_.rst = 0;
        tick(false);
        require(dut_.cfg_ready, "matmul_output_controller did not reset to IDLE");
    }

    void tick(bool stalls) {
        dut_.local_write_ready = !stalls || cycles_ % 7 != 2;
        dut_.output_write_ready = !output_response_pending_ &&
            (!stalls || cycles_ % 11 != 3);
        dut_.residual_read_request_ready =
            !stalls || cycles_ % 5 != 1;
        dut_.clk = 0;
        dut_.eval();
        last_residual_accepted_ =
            dut_.residual_read_data_valid && dut_.residual_read_data_ready;
        last_residual_request_accepted_ =
            dut_.residual_read_request_valid &&
            dut_.residual_read_request_ready;
        last_fragment_accepted_ = dut_.in_valid && dut_.in_ready;
        last_output_accepted_ =
            (dut_.local_write_valid && dut_.local_write_ready) ||
            (dut_.output_write_valid && dut_.output_write_ready);
        if (dut_.output_write_valid && dut_.output_write_ready) {
            require(dut_.output_write_transaction_bytes == active_transfer_rows_ * 16,
                    "DDR transaction byte count mismatch");
            require(bool(dut_.output_write_first) ==
                        (transfer_accepted_beats_ == 0),
                    "DDR first-beat marker mismatch");
            require(bool(dut_.output_write_last) ==
                        (transfer_accepted_beats_ == active_transfer_rows_ - 1),
                    "DDR last-beat marker mismatch");
            ++output_accepted_beats_;
            transfer_accepted_beats_ = dut_.output_write_last ? 0 : transfer_accepted_beats_ + 1;
        }
        if (dut_.trace_staging_write_valid && dut_.trace_staging_write_ready)
            staging_address_seen_.at(dut_.trace_staging_write_address) = true;
        if (dut_.trace_staging_read_valid && dut_.trace_staging_read_ready) {
            staging_address_seen_.at(dut_.trace_staging_read_address) = true;
            staging_read_pending_ = true;
        }
        if (dut_.trace_staging_read_response_valid)
            staging_read_pending_ = false;
        require(!dut_.abort_ack || !staging_read_pending_,
                "abort acknowledged before accepted SRAM read response");
        if (dut_.output_write_valid && dut_.output_write_ready &&
            dut_.output_write_last) {
            output_last_seen_ = true;
            output_response_pending_ = true;
        }
        capture_output();
        if (dut_.row_commit) ++row_commits_;
        dut_.clk = 1;
        dut_.eval();
        dut_.clk = 0;
        dut_.eval();
        ++cycles_;
    }

    void configure(int mode, std::uint64_t base, int stride,
                   const std::vector<int>& batch_rows = {}, std::uint64_t exact_limit = 0) {
        require(dut_.cfg_ready, "configuration issued while busy");
        const std::vector<int> rows = batch_rows.empty() ?
            std::vector<int>{kRows} : batch_rows;
        require(!rows.empty() && rows.size() <= 6,
                "test requested an invalid batch count");
        require(rows.front() == kRows,
                "first output batch must match MATMUL_OUTPUT_ROWS");
        dut_.cfg_mode = mode;
        dut_.cfg_batch_count = rows.size();
        dut_.cfg_second_batch_rows = rows.size() > 1 ? rows[1] : 0;
        dut_.cfg_third_batch_rows = rows.size() > 2 ? rows[2] : 0;
        dut_.cfg_fourth_batch_rows = rows.size() > 3 ? rows[3] : 0;
        dut_.cfg_fifth_batch_rows = rows.size() > 4 ? rows[4] : 0;
        dut_.cfg_sixth_batch_rows = rows.size() > 5 ? rows[5] : 0;
        dut_.in_batch_index = 0;
        active_transfer_rows_ = kRows;
        dut_.cfg_output_base = base;
        dut_.cfg_local_row_stride = stride;
        std::uint64_t bytes = 0;
        unsigned total_token_count = 0;
        for (const int row_count : rows)
            bytes += std::uint64_t(row_count) * kChannels * 2;
        for (const int row_count : rows) total_token_count += row_count;
        std::uint64_t layout_bytes = 0;
        for (const int row_count : rows)
            layout_bytes += std::uint64_t(row_count + (row_count & 1)) *
                kChannels * 2;
        dut_.cfg_output_limit = base +
            ((mode == 1 || mode == 2) ? layout_bytes * 2 :
             mode == 5 ? layout_bytes :
             (mode == 0 || mode == 3) ? std::uint64_t(total_token_count) * stride :
                                        bytes);
        if (exact_limit) dut_.cfg_output_limit = exact_limit;
        dut_.cfg_valid = 1;
        tick(false);
        dut_.cfg_valid = 0;
        require(!dut_.error, "valid output configuration was rejected");
    }

    void serve_residual_stripe(int mode, int stripe, std::uint64_t base,
                               int storage_rows = 0) {
        for (int timeout = 0;
             timeout < 2000 && !dut_.residual_read_request_valid; ++timeout)
            tick(true);
        require(dut_.residual_read_request_valid,
                "residual request did not appear done=" +
                std::to_string(dut_.done) + " error=" +
                std::to_string(dut_.error) + " error_id=" +
                std::to_string(dut_.error_id) + " busy=" +
                std::to_string(dut_.busy) + " state=" +
                std::to_string(dut_.trace_output_state));
        const auto expected_address = base +
            (mode == 3 ? stripe * 16 :
             stripe * (storage_rows ? storage_rows : kRows) * 16);
        if (dut_.residual_read_request_address != expected_address ||
            dut_.residual_read_request_bytes != kRows * 16 ||
            dut_.residual_read_request_tag != stripe ||
            bool(dut_.residual_read_local) != (mode == 3))
            throw std::runtime_error(
                "residual request payload mismatch address=" +
                std::to_string(dut_.residual_read_request_address) + "/" +
                std::to_string(expected_address) + " bytes=" +
                std::to_string(dut_.residual_read_request_bytes) + "/" +
                std::to_string(kRows * 16) + " tag=" +
                std::to_string(dut_.residual_read_request_tag) + "/" +
                std::to_string(stripe) + " local=" +
                std::to_string(dut_.residual_read_local));
        last_residual_request_accepted_ = false;
        while (!last_residual_request_accepted_)
            tick(true);
        for (int row = 0; row < kRows; ++row) {
            clear_wide(dut_.residual_read_data, 4);
            for (int lane = 0; lane < 8; ++lane)
                set16(dut_.residual_read_data, lane,
                      residual_value(row, stripe * 8 + lane));
            dut_.residual_read_data_valid = 1;
            dut_.residual_read_data_tag = stripe;
            dut_.residual_read_data_last = row + 1 == kRows;
            last_residual_accepted_ = false;
            for (int timeout = 0;
                 timeout < 2000 && !last_residual_accepted_; ++timeout)
                tick(true);
            require(last_residual_accepted_,
                    "residual data stalled without progress");
            dut_.residual_read_data_valid = 0;
        }
        dut_.residual_read_data_last = 0;
        dut_.residual_read_request_done = 1;
        tick(false);
        dut_.residual_read_request_done = 0;
    }

    void drive_fragment(int row, int stripe, int value_row = -1) {
        dut_.in_physical_row = row;
        dut_.in_output_channel = stripe * 8;
        dut_.in_row_byte_base = 0;
        clear_wide(dut_.in_data, 4);
        for (int lane = 0; lane < 8; ++lane)
            set16(dut_.in_data, lane,
                  input_value(value_row < 0 ? row : value_row, stripe * 8 + lane));
        dut_.in_valid = 1;
        last_fragment_accepted_ = false;
        for (int timeout = 0;
             timeout < 20000 && !last_fragment_accepted_; ++timeout)
            tick(true);
        require(last_fragment_accepted_, "output fragment was not accepted");
        dut_.in_valid = 0;
    }

    void wait_for_output_payload() {
        output_last_seen_ = false;
        for (int timeout = 0;
             timeout < 20000 && !output_last_seen_; ++timeout)
            tick(true);
        require(output_last_seen_,
                "DDR output did not reach final beat");
    }

    void complete_output_write() {
        require(output_response_pending_,
                "DDR completion issued without a pending payload");
        dut_.output_write_done = 1;
        tick(false);
        dut_.output_write_done = 0;
        output_response_pending_ = false;
    }

    void run_mode(int mode, std::uint64_t base, int stride,
                  bool residual, bool ddr) {
        const auto start_cycle = cycles_;
        observed_.clear();
        row_commits_ = 0;
        output_overlap_fragments_ = 0;
        output_accepted_beats_ = 0;
        configure(mode, base, stride);
        for (int stripe = 0; stripe < kChannels / 8; ++stripe) {
            if (residual && stripe == 0)
                serve_residual_stripe(mode, stripe, base);
            if (residual && stripe + 1 < kChannels / 8)
                serve_residual_stripe(mode, stripe + 1, base);
            for (int row = 0; row < kRows; ++row) {
                drive_fragment(row, stripe);
                if (ddr && stripe != 0 && row == 0) {
                    require(output_response_pending_,
                            "next stripe started without a pending prior response");
                    ++output_overlap_fragments_;
                    for (int delay = 0; delay < 8; ++delay)
                        tick(true);
                    complete_output_write();
                }
            }
            if (ddr) {
                wait_for_output_payload();
                if (stripe + 1 == kChannels / 8)
                    complete_output_write();
            }
        }
        for (int timeout = 0; timeout < 20000 && !dut_.done; ++timeout)
            tick(true);
        require(dut_.done && !dut_.error,
                "valid matmul output command failed");
        require(row_commits_ == kRows,
                "final-stripe row commit count mismatch");
        require(!ddr || output_overlap_fragments_ == kChannels / 8 - 1,
                "prior B response did not overlap the next stripe");
        require(!ddr || output_accepted_beats_ ==
                    kChannels / 8 * kRows,
                "DDR accepted beat count mismatch");
        require(!output_response_pending_,
                "output response remained pending after command completion");
        verify_image(mode, base, stride, residual);
        std::cout << "RESULT matmul_output_controller_mode mode=" << mode
                  << " cycles=" << (cycles_ - start_cycle) << '\n';
        tick(false);
    }

    void run_strided_residual() {
        constexpr int kStorageRows = 48;
        const std::uint64_t output_base = 0x1a0000;
        const std::uint64_t residual_base = 0x1c0000;
        const std::uint64_t residual_end = residual_base +
            (kChannels / 8 - 1) * kStorageRows * 16 + kRows * 16;
        observed_.clear();
        row_commits_ = output_overlap_fragments_ = output_accepted_beats_ = 0;
        dut_.cfg_residual_from_ddr = 1;
        dut_.cfg_residual_base = residual_base;
        dut_.cfg_residual_limit = residual_end;
        dut_.cfg_residual_storage_rows = kStorageRows;
        configure(4, output_base, 0);
        for (int stripe = 0; stripe < kChannels / 8; ++stripe) {
            if (stripe == 0)
                serve_residual_stripe(4, stripe, residual_base, kStorageRows);
            if (stripe + 1 < kChannels / 8)
                serve_residual_stripe(4, stripe + 1, residual_base,
                                      kStorageRows);
            for (int row = 0; row < kRows; ++row) {
                drive_fragment(row, stripe);
                if (stripe != 0 && row == 0) complete_output_write();
            }
            wait_for_output_payload();
            if (stripe + 1 == kChannels / 8) complete_output_write();
        }
        for (int timeout = 0; timeout < 20000 && !dut_.done; ++timeout)
            tick(true);
        require(dut_.done && !dut_.error,
                "strided residual command did not complete");
        verify_image(4, output_base, 0, true);
        tick(false);

        dut_.cfg_residual_from_ddr = 1;
        dut_.cfg_residual_base = residual_base;
        dut_.cfg_residual_limit = residual_end - 1;
        dut_.cfg_residual_storage_rows = kStorageRows;
        dut_.cfg_output_base = output_base;
        dut_.cfg_output_limit = output_base + kRows * kChannels * 2;
        dut_.cfg_mode = 4;
        dut_.cfg_batch_count = 1;
        dut_.cfg_valid = 1;
        tick(false);
        dut_.cfg_valid = 0;
        for (int timeout = 0; timeout < 20 && !dut_.done; ++timeout)
            tick(false);
        require(dut_.done && dut_.error,
                "undersized strided residual region was accepted");
        tick(false);
        require(dut_.cfg_ready,
                "output controller did not restart after strided limit error");
        dut_.cfg_residual_storage_rows = 0;
        dut_.cfg_residual_from_ddr = 0;
        std::cout << "STRIDED_RESIDUAL rows=32 storage_rows=48 stripe1_offset=768 "
                     "limit_minus_one_rejected=1 raw_mismatch=0\n";
    }

    void run_residual_batches(const std::vector<int>& rows, int failure,
                              bool separate_residual = false) {
        // Failure: abort, read error, write error, bad input batch,
        // bad residual tag, and read error before the first returned beat.
        require(rows.size() >= 2 && rows.size() <= 6 && rows.front() == kRows,
                "residual batch test shape is invalid");
        const int batch_count = rows.size();
        const std::uint64_t base = 0x200000;
        const std::uint64_t residual_base = separate_residual ? 0x100000 : base;
        const auto start = cycles_;
        observed_.clear();
        row_commits_ = output_accepted_beats_ = transfer_accepted_beats_ = 0;
        dut_.cfg_residual_from_ddr = separate_residual;
        dut_.cfg_residual_base = residual_base;
        int total_token_count = 0;
        for (const int row_count : rows) total_token_count += row_count;
        dut_.cfg_residual_limit = residual_base + total_token_count * kChannels * 2;
        configure(4, base, 0, rows);
        int input_batch = 0, input_row = 0, input_stripe = 0;
        int requests = 0, read_row = 0, read_delay = 0, read_done_delay = -1;
        int write_delay = -1, responses = 0, early_fragments = 0, residual_wait_cycles = 0;
        bool read_active = false, injected = false, completed = false;
        for (int step = 0; step < 30000 && !completed; ++step) {
            const int read_batch = requests ? (requests - 1) % batch_count : 0;
            const int read_stripe = requests ? (requests - 1) / batch_count : 0;
            int input_row_base = 0;
            for (int batch = 0; batch < input_batch; ++batch)
                input_row_base += rows[batch];
            int read_row_base = 0;
            for (int batch = 0; batch < read_batch; ++batch)
                read_row_base += rows[batch];
            dut_.in_valid = input_stripe < kChannels / 8 && !injected;
            dut_.in_batch_index = input_batch;
            dut_.in_physical_row = input_row;
            dut_.in_output_channel = input_stripe * 8;
            for (int lane = 0; lane < 8; ++lane)
                set16(dut_.in_data, lane,
                      input_value(input_row + input_row_base,
                                  input_stripe * 8 + lane));
            if (!injected && requests == 2 && failure == 1) {
                injected = true;
                dut_.abort_request = 1;
                dut_.in_valid = 0;
            }
            if (!injected && requests == 1 && failure == 4) {
                injected = true;
                dut_.in_valid = 1;
                dut_.in_batch_index = input_batch ^ 1;
            }
            dut_.residual_read_data_valid = read_active && read_delay == 0 &&
                read_row < rows[read_batch] && step % 4 != 1;
            dut_.residual_read_data_tag = read_stripe;
            dut_.residual_read_data_last = read_row + 1 == rows[read_batch];
            for (int lane = 0; lane < 8; ++lane)
                set16(dut_.residual_read_data, lane,
                      residual_value(read_row + read_row_base,
                                     read_stripe * 8 + lane));
            dut_.residual_read_request_done = read_done_delay == 0;
            dut_.residual_read_request_error = 0;
            if (!injected && requests == 2 && failure == 5 &&
                dut_.residual_read_data_valid) {
                injected = true;
                dut_.residual_read_data_tag ^= 1;
                dut_.in_valid = 0;
            }
            if (!injected && requests == 2 && failure == 6) {
                injected = true;
                dut_.residual_read_request_error = 1;
                dut_.residual_read_data_valid = 0;
                dut_.in_valid = 0;
            }
            if (read_done_delay == 0 && requests == 2 && failure == 2) {
                injected = true;
                dut_.residual_read_request_done = 0;
                dut_.residual_read_request_error = 1;
                dut_.in_valid = 0;
            }
            if (output_response_pending_ && write_delay < 0)
                write_delay = failure ? 23 : 13;
            dut_.output_write_done = write_delay == 0;
            dut_.output_write_error = 0;
            if (write_delay == 0 && responses == 0 && failure == 3) {
                injected = true;
                dut_.output_write_done = 0;
                dut_.output_write_error = 1;
                dut_.in_valid = 0;
            }
            int accepted_row = output_accepted_beats_ % total_token_count;
            active_transfer_rows_ = rows.back();
            for (const int row_count : rows) {
                if (accepted_row < row_count) {
                    active_transfer_rows_ = row_count;
                    break;
                }
                accepted_row -= row_count;
            }
            dut_.clk = 0;
            dut_.eval();
            if (dut_.residual_read_request_valid) {
                const int batch = requests % batch_count;
                const int stripe = requests / batch_count;
                int row_base = 0;
                for (int prior = 0; prior < batch; ++prior)
                    row_base += rows[prior];
                require(!read_active, "dual residual issued overlapping reads");
                require(dut_.residual_read_request_address == residual_base +
                            row_base * kChannels * 2 +
                            stripe * rows[batch] * 16 &&
                        dut_.residual_read_request_bytes == rows[batch] * 16 &&
                        dut_.residual_read_request_tag == stripe &&
                        !dut_.residual_read_local,
                        "dual residual batch/stripe address or byte count mismatch");
            }
            if (dut_.trace_output_state >= 1 && dut_.trace_output_state <= 3)
                ++residual_wait_cycles;
            tick(true);
            if (dut_.residual_read_request_done || dut_.residual_read_request_error) {
                read_active = false;
                read_done_delay = -1;
            } else if (read_done_delay > 0) {
                --read_done_delay;
            }
            if (last_residual_request_accepted_) {
                ++requests;
                read_active = true;
                read_row = 0;
                read_delay = 7;
            } else if (read_delay > 0) {
                --read_delay;
            }
            if (last_residual_accepted_ && ++read_row == rows[read_batch])
                read_done_delay = 4;
            if (last_fragment_accepted_) {
                if (input_stripe * batch_count + input_batch >= requests)
                    ++early_fragments;
                if (++input_row == rows[input_batch]) {
                    input_row = 0;
                    if (++input_batch == batch_count) {
                        input_batch = 0;
                        ++input_stripe;
                    }
                }
            }
            if (dut_.output_write_done || dut_.output_write_error) {
                output_response_pending_ = false;
                write_delay = -1;
                ++responses;
            } else if (write_delay > 0) {
                --write_delay;
            }
            if (dut_.done || dut_.abort_ack) {
                require(!read_active && !output_response_pending_ &&
                        !staging_read_pending_, "dual residual completed before drain");
                completed = true;
            }
        }
        require(completed, "dual residual made no bounded progress");
        if (failure) {
            require(injected && (failure == 1 ? bool(dut_.abort_ack) : bool(dut_.error)),
                    "dual residual failure was not reported");
            if (failure != 1)
                require(dut_.error_id == (failure == 3 ? 4 : failure == 4 ? 1 : 2),
                        "dual residual error classification mismatch");
        } else {
            require(!dut_.error && requests == batch_count * (kChannels / 8) &&
                    responses == batch_count * (kChannels / 8) &&
                    row_commits_ == total_token_count && early_fragments > 0 &&
                    output_accepted_beats_ == total_token_count * (kChannels / 8) &&
                    observed_.size() == std::size_t(total_token_count * kChannels),
                    "dual residual lost data, wrote padding, or missed FIFO lookahead");
            int row_base = 0;
            for (int batch = 0; batch < batch_count; ++batch) {
                for (int stripe = 0; stripe < kChannels / 8; ++stripe)
                    for (int row = 0; row < rows[batch]; ++row)
                        for (int lane = 0; lane < 8; ++lane) {
                            const auto address = base +
                                row_base * kChannels * 2 +
                                (stripe * rows[batch] + row) * 16 + lane * 2;
                            const int value_row = row + row_base;
                            require(observed_.at(address) == rtl_bf16_add(
                                input_value(value_row, stripe * 8 + lane),
                                residual_value(value_row, stripe * 8 + lane)),
                                "dual residual BF16 raw mismatch");
                        }
                row_base += rows[batch];
            }
        }
        std::cout << "RESIDUAL_BATCHES count=" << batch_count
                  << " rows=" << total_token_count
                  << " failure=" << failure << " cycles=" << cycles_ - start
                  << " separate_residual=" << separate_residual
                  << " residual_wait_cycles=" << residual_wait_cycles
                  << " requests=" << requests << " early_fragments=" << early_fragments
                  << " raw_values=" << observed_.size() << '\n';
        clear_inputs();
        for (int cycle = 0; cycle < 3; ++cycle) tick(false);
        require(dut_.cfg_ready, "residual batches did not permit restart");
    }

    void run_two_local_batches(int second_rows) {
        observed_.clear();
        row_commits_ = output_accepted_beats_ = 0;
        const std::uint64_t base = 0x1a0000;
        const unsigned stride = 256;
        configure(0, base, stride, {kRows, second_rows},
                  base + (kRows + second_rows - 1) * stride + kChannels * 2 - 1);
        for (int timeout = 0; timeout < 1000 && !dut_.done; ++timeout) tick(true);
        require(dut_.done && dut_.error && observed_.empty(),
                "dual local output accepted an undersized second-batch region");
        tick(false);
        configure(0, base, stride, {kRows, second_rows});
        for (int stripe = 0; stripe < kChannels / 8; ++stripe)
            for (unsigned batch = 0; batch < 2; ++batch) {
                dut_.in_batch_index = batch;
                for (int row = 0; row < (batch ? second_rows : kRows); ++row)
                    drive_fragment(row, stripe, row + batch * kRows);
            }
        for (int timeout = 0; timeout < 20000 && !dut_.done; ++timeout) tick(true);
        require(dut_.done && !dut_.error, "dual local output did not finish");
        require(row_commits_ == kRows + second_rows && output_accepted_beats_ == 0,
                "dual local output lost commits or issued DDR writes");
        require(observed_.size() == (kRows + second_rows) * kChannels,
                "dual local output lost values or wrote padding");
        for (int row = 0; row < kRows + second_rows; ++row)
            for (int channel = 0; channel < kChannels; ++channel)
                require(observed_.at(base + row * stride + channel * 2) == input_value(row, channel),
                        "dual local output batch address/raw mismatch");
        std::cout << "PASS two local output batches rows=" << kRows << '+' << second_rows
                  << " stride=" << stride << " stalls=checked raw_values=" << observed_.size() << '\n';
        tick(false);
    }

    void run_two_batches(int mode, std::uint64_t base) {
        observed_.clear();
        row_commits_ = 0;
        output_accepted_beats_ = 0;
        configure(mode,base,0,{kRows,1});
        const auto start = cycles_;
        for (int stripe = 0; stripe < kChannels/8; ++stripe) {
            for (int batch = 0; batch < 2; ++batch) {
                const int rows = batch ? 1 : kRows;
                active_transfer_rows_ = rows;
                dut_.in_batch_index = batch;
                for (int row = 0; row < rows; ++row) {
                    drive_fragment(row,stripe,row+batch*kRows);
                    if (row == 0 && (batch || stripe)) {
                        require(output_response_pending_,"next batch did not overlap the prior B response");
                        for (int delay = 0; delay < 8; ++delay) tick(true);
                        complete_output_write();
                    }
                }
                wait_for_output_payload();
                if (stripe+1 == kChannels/8 && batch) complete_output_write();
            }
        }
        for (int timeout = 0; timeout < 20000 && !dut_.done; ++timeout) tick(true);
        require(dut_.done && !dut_.error,"two-batch output did not complete");
        require(row_commits_ == kRows+1,"two-batch final row count mismatch");
        require(output_accepted_beats_ == (kRows+1)*kChannels/8,"two-batch payload beat count mismatch");
        require(observed_.size() == (kRows+1)*kChannels,"two-batch output added or lost a value");
        for (int batch = 0; batch < 2; ++batch) {
            const int rows = batch ? 1 : kRows, stride_rows = batch ? 2 : kLayoutRows;
            const auto batch_base = base + (batch ? kLayoutRows*kChannels*4 : 0);
            for (int stripe = 0; stripe < kChannels/8; ++stripe)
                for (int row = 0; row < rows; ++row)
                    for (int lane = 0; lane < 8; ++lane) {
                        const auto address = batch_base + stripe*stride_rows*32 +
                            (mode == 2 ? stride_rows*16 : 0) + row*16 + lane*2;
                        require(observed_.at(address) == input_value(row+batch*kRows,stripe*8+lane),
                                "two-batch Gate/Up DDR raw mismatch");
                    }
        }
        std::cout << "TWO_BATCH_OUTPUT mode=" << mode << " rows=" << kRows << "+1 cycles="
                  << cycles_-start << " prior_response_overlap=1 raw_values=" << observed_.size() << '\n';
        tick(false);
    }

    void run_batches(int mode, std::uint64_t base,
                     const std::vector<int>& batch_rows,
                     const std::string& name) {
        require(mode == 1 || mode == 2 || mode == 5 || mode == 6,
                "multi-batch case requires a supported grouped DDR mode");
        observed_.clear();
        row_commits_ = 0;
        output_accepted_beats_ = 0;
        transfer_accepted_beats_ = 0;
        configure(mode, base, 0, batch_rows);
        const auto start = cycles_;
        int prior_transactions = 0;
        int value_row_base = 0;
        for (int virtual_stripe = 0;
             virtual_stripe < kChannels / 8 * (mode == 5 ? 2 : 1);
             ++virtual_stripe) {
            const int stripe = mode == 5 ? virtual_stripe / 2 : virtual_stripe;
            const bool gate_only = mode == 5 && virtual_stripe % 2 == 0;
            value_row_base = 0;
            for (std::size_t batch = 0; batch < batch_rows.size(); ++batch) {
                const int rows = batch_rows[batch];
                active_transfer_rows_ = rows;
                dut_.in_batch_index = batch;
                for (int row = 0; row < rows; ++row) {
                    drive_fragment(row, stripe, value_row_base + row +
                        (mode == 5 && !gate_only ? 1000 : 0));
                    if (!gate_only && row == 0 && prior_transactions != 0) {
                        require(output_response_pending_,
                                name + ": next batch/stripe did not overlap prior B response");
                        require(!dut_.done,
                                name + ": command completed before final batch/stripe");
                        for (int delay = 0; delay < 8; ++delay) tick(true);
                        complete_output_write();
                    }
                }
                if (!gate_only) {
                    wait_for_output_payload();
                    ++prior_transactions;
                    if (stripe + 1 == kChannels / 8 &&
                        batch + 1 == batch_rows.size())
                        complete_output_write();
                }
                value_row_base += rows;
            }
        }
        for (int timeout = 0; timeout < 40000 && !dut_.done; ++timeout)
            tick(true);
        require(dut_.done && !dut_.error,
                name + ": multi-batch output did not complete");

        int total_token_count = 0;
        for (const int rows : batch_rows) total_token_count += rows;
        require(row_commits_ == total_token_count,
                name + ": final-stripe row commit count mismatch");
        require(output_accepted_beats_ == total_token_count * kChannels / 8,
                name + ": accepted DDR beat count mismatch");
        require(observed_.size() == std::size_t(total_token_count * kChannels),
                name + ": output added, lost, or overwrote a value");
        require(!output_response_pending_,
                name + ": response remained pending at completion");

        std::uint64_t batch_base = base;
        value_row_base = 0;
        for (std::size_t batch = 0; batch < batch_rows.size(); ++batch) {
            const int rows = batch_rows[batch];
            const int padded_rows = rows + (rows & 1);
            for (int stripe = 0; stripe < kChannels / 8; ++stripe) {
                for (int row = 0; row < rows; ++row) {
                    for (int lane = 0; lane < 8; ++lane) {
                        const auto address = batch_base +
                            stripe * (mode == 6 ? rows : padded_rows) *
                                (mode == 5 || mode == 6 ? 16 : 32) +
                            (mode == 2 ? padded_rows * 16 : 0) +
                            row * 16 + lane * 2;
                        const auto it = observed_.find(address);
                        const auto gate = input_value(value_row_base + row,
                                                      stripe * 8 + lane);
                        const auto expected = mode == 5 ? rtl_bf16_mul(
                            rtl_bf16_silu_pwl(gate),
                            input_value(value_row_base + row + 1000,
                                        stripe * 8 + lane)) : gate;
                        require(it != observed_.end() && it->second == expected,
                                name + ": grouped DDR raw/address mismatch");
                    }
                }
            }
            batch_base += std::uint64_t(mode == 6 ? rows : padded_rows) *
                kChannels * (mode == 5 || mode == 6 ? 2 : 4);
            value_row_base += rows;
        }
        std::cout << "MULTI_BATCH_OUTPUT name=" << name
                  << " mode=" << mode
                  << " batches=" << batch_rows.size()
                  << " rows=" << total_token_count
                  << " transactions="
                  << batch_rows.size() * (kChannels / 8)
                  << " raw_values=" << observed_.size()
                  << " prior_response_overlap=1 cycles="
                  << cycles_ - start << '\n';
        tick(false);
    }

    void capture_output() {
        if (!last_output_accepted_) return;
        const auto address = dut_.local_write_valid ?
            dut_.local_write_byte_address : dut_.output_write_byte_address;
        const auto& data = dut_.local_write_valid ?
            dut_.local_write_data : dut_.output_write_data;
        for (int lane = 0; lane < 8; ++lane)
            observed_[address + lane * 2] = get16(data, lane);
    }

    void verify_image(int mode, std::uint64_t base, int stride,
                      bool residual) {
        for (int stripe = 0; stripe < kChannels / 8; ++stripe) {
            for (int row = 0; row < kRows; ++row) {
                std::uint64_t address;
                if (mode == 0 || mode == 3)
                    address = base + row * stride + stripe * 16;
                else if (mode == 1 || mode == 2)
                    address = base + stripe * 2 * kLayoutRows * 16 +
                        (mode == 2 ? kLayoutRows * 16 : 0) + row * 16;
                else
                    address = base + stripe * kRows * 16 + row * 16;
                for (int lane = 0; lane < 8; ++lane) {
                    const int channel = stripe * 8 + lane;
                    const auto expected = residual ?
                        rtl_bf16_add(input_value(row, channel),
                                    residual_value(row, channel)) :
                        input_value(row, channel);
                    const auto it = observed_.find(address + lane * 2);
                    if (it == observed_.end() || it->second != expected) {
                        const auto actual = it == observed_.end() ?
                            0xffffffffu : std::uint32_t(it->second);
                        throw std::runtime_error(
                            "matmul output mismatch mode=" +
                            std::to_string(mode) + " stripe=" +
                            std::to_string(stripe) + " row=" +
                            std::to_string(row) + " lane=" +
                            std::to_string(lane) + " expected=" +
                            std::to_string(expected) + " actual=" +
                            std::to_string(actual));
                    }
                }
            }
        }
        require(observed_.size() == std::size_t(kRows * kChannels),
                "matmul output wrote padding or duplicate addresses");
    }

    void read_error_and_restart() {
        configure(4, 0x60000, 0);
        while (!dut_.residual_read_request_valid) tick(false);
        last_residual_request_accepted_ = false;
        while (!last_residual_request_accepted_)
            tick(true);
        for (int row = 0; row < kRows; ++row) {
            dut_.residual_read_data_valid = 1;
            dut_.residual_read_data_tag = 0;
            dut_.residual_read_data_last = row + 1 == kRows;
            last_residual_accepted_ = false;
            while (!last_residual_accepted_) tick(false);
            dut_.residual_read_data_valid = 0;
        }
        dut_.residual_read_data_last = 0;
        dut_.residual_read_request_error = 1;
        tick(false);
        dut_.residual_read_request_error = 0;
        for (int timeout = 0; timeout < 100 && !dut_.done; ++timeout)
            tick(false);
        require(dut_.done && dut_.error && dut_.error_id == 2,
                "residual read error did not terminate");
        tick(false);
        require(dut_.cfg_ready, "read error did not release output controller");
    }

    void write_error_during_next_stripe() {
        configure(1, 0x68000, 0);
        output_accepted_beats_ = 0;
        for (int row = 0; row < kRows; ++row)
            drive_fragment(row, 0);
        wait_for_output_payload();
        require(output_response_pending_,
                "write-error setup did not leave a response pending");

        drive_fragment(0, 1);
        require(output_response_pending_,
                "next stripe did not overlap the pending write response");
        dut_.output_write_error = 1;
        tick(false);
        dut_.output_write_error = 0;
        output_response_pending_ = false;
        for (int row = 1; row < kRows; ++row)
            drive_fragment(row, 1);
        for (int timeout = 0; timeout < 20000 && !dut_.done; ++timeout)
            tick(true);
        require(dut_.done && dut_.error && dut_.error_id == 4,
                "pending write error did not stop the next stripe");
        require(output_accepted_beats_ == kRows,
                "write error exposed the next stripe on the DDR interface");
        tick(false);
        require(dut_.cfg_ready,
                "write error did not release output controller");
    }

    void abort_with_prefetched_read_pending(int completion) {
        constexpr std::uint64_t base = 0x72000;
        configure(4, base, 0);
        serve_residual_stripe(4, 0, base);
        for (int timeout = 0;
             timeout < 100 && !dut_.residual_read_request_valid; ++timeout)
            tick(false);
        require(dut_.residual_read_request_valid &&
                    dut_.residual_read_request_tag == 1,
                "prefetch abort did not reach the next residual request");
        tick(false);
        require(last_residual_request_accepted_, "residual prefetch was not accepted");
        // A completed stripe must not satisfy the next stripe's drain condition.
        if (completion == 2) {
            for (int row = 0; row < kRows; ++row) {
                dut_.residual_read_data_valid = 1;
                dut_.residual_read_data_tag = 1;
                dut_.residual_read_data_last = row + 1 == kRows;
                tick(false);
                require(last_residual_accepted_, "prefetched data was not accepted");
            }
            dut_.residual_read_data_valid = 0;
            dut_.residual_read_data_last = 0;
            dut_.residual_read_request_done = 1;
        }
        dut_.abort_request = 1;
        tick(false);
        dut_.residual_read_request_done = 0;
        if (completion != 2) {
            for (int delay = 0; delay < 4; ++delay) {
                tick(false);
                require(!dut_.abort_ack, "prefetched read acknowledged abort before current completion");
            }
            if (completion == 0) {
                for (int row = 0; row < kRows; ++row) {
                    dut_.residual_read_data_valid = 1;
                    dut_.residual_read_data_tag = 1;
                    dut_.residual_read_data_last = row + 1 == kRows;
                    tick(false);
                    require(last_residual_accepted_ && !dut_.abort_ack,
                            "prefetched read did not drain before its terminal response");
                }
                dut_.residual_read_data_valid = 0;
                dut_.residual_read_data_last = 0;
                dut_.residual_read_request_done = 1;
            } else {
                dut_.residual_read_request_error = 1;
            }
            tick(false);
            dut_.residual_read_request_done = 0;
            dut_.residual_read_request_error = 0;
        }
        for (int timeout = 0; timeout < 100 && !dut_.abort_ack; ++timeout)
            tick(false);
        require(dut_.abort_ack, "prefetched read abort did not finish after current completion");
        tick(false);
        require(!dut_.abort_ack && !dut_.cfg_ready, "held prefetch abort was not retained");
        dut_.abort_request = 0;
        tick(false);
        require(dut_.cfg_ready, "prefetch abort did not permit restart");
        run_mode(4, base, 0, true, true);
    }

    void abort_with_read_pending() {
        configure(4, 0x70000, 0);
        for (int timeout = 0;
             timeout < 1000 && !dut_.residual_read_request_valid; ++timeout)
            tick(false);
        require(dut_.residual_read_request_valid,
                "abort setup request did not appear");
        last_residual_request_accepted_ = false;
        for (int timeout = 0;
             timeout < 1000 && !last_residual_request_accepted_; ++timeout)
            tick(true);
        require(last_residual_request_accepted_,
                "abort setup request was not accepted");
        dut_.abort_request = 1;
        tick(false);
        require(!dut_.abort_ack,
                "accepted residual read acknowledged abort before drain");
        for (int row = 0; row < kRows; ++row) {
            dut_.residual_read_data_valid = 1;
            dut_.residual_read_data_tag = 0;
            dut_.residual_read_data_last = row + 1 == kRows;
            last_residual_accepted_ = false;
            for (int timeout = 0;
                 timeout < 1000 && !last_residual_accepted_; ++timeout)
                tick(false);
            require(last_residual_accepted_,
                    "abort residual drain made no progress");
            dut_.residual_read_data_valid = 0;
        }
        dut_.residual_read_data_last = 0;
        dut_.residual_read_request_done = 1;
        tick(false);
        dut_.residual_read_request_done = 0;
        for (int timeout = 0; timeout < 100 && !dut_.abort_ack; ++timeout)
            tick(false);
        require(dut_.abort_ack, "abort acknowledgement missing after drain");
        tick(false);
        require(!dut_.abort_ack,
                "held abort produced duplicate acknowledgement");
        dut_.abort_request = 0;
        tick(false);
        require(dut_.cfg_ready, "output controller did not restart after abort");
    }

    void abort_with_write_pending() {
        configure(1, 0x78000, 0);
        output_accepted_beats_ = 0;
        for (int row = 0; row < kRows; ++row)
            drive_fragment(row, 0);
        wait_for_output_payload();
        require(output_response_pending_,
                "write-abort setup did not leave a response pending");

        dut_.abort_request = 1;
        tick(false);
        require(!dut_.abort_ack,
                "pending write acknowledged abort before B response");
        require(output_accepted_beats_ == kRows,
                "abort accepted an extra DDR write beat");
        complete_output_write();
        for (int timeout = 0; timeout < 100 && !dut_.abort_ack; ++timeout)
            tick(false);
        require(dut_.abort_ack,
                "abort acknowledgement missing after B response");
        tick(false);
        require(!dut_.abort_ack,
                "held write abort produced duplicate acknowledgement");
        dut_.abort_request = 0;
        tick(false);
        require(dut_.cfg_ready,
                "output controller did not restart after write abort");
    }

    void abort_fused_at(int target_state) {
        configure(5, 0x180000, 0, {kRows, 1});
        for (int batch = 0; batch < 2; ++batch) {
            dut_.in_batch_index = batch;
            for (int row = 0; row < (batch ? 1 : kRows); ++row)
                drive_fragment(row, 0, batch * kRows + row);
        }
        dut_.in_batch_index = 0;
        drive_fragment(0, 0, 1000);
        for (int timeout = 0;
             timeout < 200 && dut_.trace_output_state != target_state; ++timeout)
            tick(false);
        require(dut_.trace_output_state == target_state,
                "fused abort did not reach requested outstanding operation");
        const auto writes = output_accepted_beats_;
        dut_.abort_request = 1;
        tick(false);
        for (int timeout = 0; timeout < 200 && !dut_.abort_ack; ++timeout)
            tick(false);
        require(dut_.abort_ack, "fused abort did not drain SRAM/arithmetic response");
        require(output_accepted_beats_ == writes,
                "fused abort exposed an unfinished product");
        tick(false);
        require(!dut_.abort_ack, "fused abort acknowledged twice");
        dut_.abort_request = 0;
        tick(false);
        require(dut_.cfg_ready, "fused abort did not permit restart");
        run_batches(5, 0x190000, {kRows, 1}, "fused-abort-restart");
    }
};
}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Regression regression;
        regression.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL matmul_output_controller: " << error.what() << "\n";
        return 1;
    }
}
