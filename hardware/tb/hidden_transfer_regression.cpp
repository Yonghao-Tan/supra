#include "Vhidden_transfer.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

double sc_time_stamp() { return 0.0; }

namespace {
constexpr std::uint64_t kBase = 0x90000000ull;
constexpr std::uint64_t kLimit = kBase + 0x1000000ull;
constexpr std::uint64_t kEmbeddingBase = 0x200000000ull;
constexpr std::uint64_t kEmbeddingBytes = 126464ull * 8192ull;
constexpr std::uint8_t kTag = 0x8f;
constexpr int kChunksPerRow = 512;

struct Location {
    std::uint8_t physical_row;
    std::uint16_t channel_word;
};

template <typename WideWords>
void clear_wide(WideWords& words) {
    for (std::size_t index = 0; index < sizeof(words) / sizeof(words[0]); ++index)
        words[index] = 0;
}

template <typename WideWords>
void set_bits(WideWords& words, unsigned bit, unsigned width,
              std::uint32_t value) {
    for (unsigned index = 0; index < width; ++index) {
        const unsigned target = bit + index;
        const std::uint32_t mask = 1u << (target % 32);
        if ((value >> index) & 1u)
            words[target / 32] |= mask;
        else
            words[target / 32] &= ~mask;
    }
}

template <typename WideWords>
void set_beat(WideWords& words, std::uint8_t bank, std::uint16_t word) {
    for (int index = 0; index < 4; ++index) words[index] = 0;
    for (int byte = 0; byte < 16; ++byte) {
        const std::uint8_t value = std::uint8_t(
            (std::uint32_t(bank) * 37u + std::uint32_t(word) * 13u +
             std::uint32_t(byte) * 7u + 3u) & 0xffu);
        words[byte / 4] |= std::uint32_t(value) << (8 * (byte % 4));
    }
}

template <typename WideWords>
std::uint8_t get_byte(const WideWords& words, int byte) {
    return std::uint8_t(words[byte / 4] >> (8 * (byte % 4)));
}

class Regression {
  public:
    explicit Regression(const char* embedding_path) {
        std::cout << "VERILATOR_MODEL_THREADS " << dut_.threads() << '\n';
        if (embedding_path != nullptr) {
            embedding_file_.open(embedding_path, std::ios::binary);
            require(embedding_file_.is_open(),
                    "cannot open BF16 embedding payload");
            embedding_file_.seekg(0, std::ios::end);
            require(embedding_file_.tellg() >= std::streamoff(kEmbeddingBytes),
                    "BF16 embedding payload is shorter than 126464 rows");
            embedding_file_.seekg(0);
            embedding_payload_loaded_ = true;
        }
    }

    void run() {
        reset();
        run_read(0, false, false, false);
        run_write(1, false, false);
        run_write(2, false, false);
        run_read(3, false, false, false);
        invalid_configuration_and_permutation();
        completion_backpressure();
        run_read(3, true, false, false);
        run_write(2, true, false);
        run_read(0, false, true, true);
        run_write(1, false, true);
        abort_before_request();
        early_write_errors();
        direct_read_indices();
        direct_write_indices();
        direct_range_error();
        direct_dma_error_stops_rows();
        embedding_read_rows(16);
        embedding_read_rows(32);
        embedding_read_rows(48);
        embedding_configuration_errors();
        embedding_abort_drain();
        dut_.final();
        std::cout << "PASS hidden_transfer: operations=4 permuted_rows=3 "
                     "row_major_permutation=2 stripe_major=2 "
                     "direct_read_rows=5 direct_write_rows=4 "
                     "continuous_sparse_reverse_cross_block=PASS "
                     "direct_range_error=1 direct_middle_dma_error=1 "
                     "embedding_rows=16/32/48 token_first_middle_last=PASS "
                     "embedding_repeat_noncontiguous=PASS "
                     "embedding_payload="
                  << (embedding_payload_loaded_ ? "PASS " : "not_run ")
                  << "embedding_payload_rows="
                  << embedding_payload_rows_ << " "
                  << "embedding_values="
                  << embedding_payload_rows_ * 4096 << " "
                  << "embedding_range_error=2 "
                     "embedding_abort_drain=1 "
                     "read_error=1 write_error=1 protocol_error=1 "
                     "abort_before_accept=2 early_write_error_drain=4 read_abort_drain=1 "
                     "write_abort_drain=1 completion_backpressure=1 "
                     "local_mapping_mismatch=0 "
                     "payload_mismatch=0 cycles=" << cycles_ << "\n";
    }

    void run_control_errors() {
        reset();
        abort_before_request();
        run_read(3, true, false, false);
        early_write_errors();
        run_write(2, false, false);
        dut_.final();
        std::cout << "PASS hidden_transfer abort/read drain/early write error/restart\n";
    }

  private:
    Vhidden_transfer dut_;
    std::uint64_t cycles_ = 0;
    const std::array<std::uint8_t, 3> physical_to_logical_{2, 0, 1};
    std::ifstream embedding_file_;
    bool embedding_payload_loaded_ = false;
    unsigned embedding_payload_rows_ = 0;

    void tick() {
        dut_.clk = 0;
        dut_.eval();
        dut_.clk = 1;
        dut_.eval();
        dut_.clk = 0;
        dut_.eval();
        ++cycles_;
    }

    void eval_low() {
        dut_.clk = 0;
        dut_.eval();
    }

    void clear_inputs() {
        dut_.start_valid = 0;
        dut_.start_operation = 0;
        dut_.start_active_rows = 3;
        clear_wide(dut_.start_physical_to_token_ordinal);
        for (unsigned row = 0; row < physical_to_logical_.size(); ++row)
            set_bits(dut_.start_physical_to_token_ordinal, row * 6, 6,
                     physical_to_logical_[row]);
        dut_.start_direct_row_index_enable = 0;
        clear_wide(dut_.start_direct_row_index);
        dut_.start_source_descriptor_enable = 0;
        clear_wide(dut_.start_source_index);
        dut_.start_source_embedding = 0;
        dut_.start_embedding_base = 0;
        dut_.start_embedding_limit = 0;
        dut_.start_ddr_base = kBase;
        dut_.start_ddr_limit = kLimit;
        dut_.start_for_ffn = 0;
        dut_.done_ready = 1;
        dut_.abort_request = 0;
        dut_.read_request_ready = 0;
        dut_.read_request_done = 0;
        dut_.read_request_error = 0;
        dut_.read_data_valid = 0;
        clear_wide(dut_.read_data);
        dut_.read_byte_enable = 0xffff;
        dut_.read_data_last = 0;
        dut_.read_data_tag = kTag;
        dut_.write_request_ready = 0;
        dut_.write_request_done = 0;
        dut_.write_request_error = 0;
        dut_.write_data_ready = 0;
        dut_.local_write_ready = 1;
        dut_.local_read_ready = 0;
        dut_.local_read_response_valid = 0;
        clear_wide(dut_.local_read_data);
    }

    void reset() {
        clear_inputs();
        dut_.rst = 1;
        for (int cycle = 0; cycle < 4; ++cycle) tick();
        dut_.rst = 0;
        tick();
        require(dut_.start_ready && !dut_.busy && !dut_.done_valid,
                "reset did not reach idle");
    }

    void require(bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    }

    std::array<std::uint8_t, 3> inverse() const {
        std::array<std::uint8_t, 3> result{};
        for (std::uint8_t physical = 0; physical < 3; ++physical)
            result[physical_to_logical_[physical]] = physical;
        return result;
    }

    Location location(int operation, std::uint32_t beat) const {
        if (operation == 0 || operation == 1) {
            const auto logical_to_physical = inverse();
            const std::uint32_t token_ordinal = beat / kChunksPerRow;
            const std::uint32_t chunk = beat % kChunksPerRow;
            const std::uint32_t physical_row =
                logical_to_physical[token_ordinal];
            return {std::uint8_t(physical_row), std::uint16_t(chunk)};
        }
        const std::uint32_t stripe = beat / 3;
        const std::uint32_t physical_row = beat % 3;
        return {std::uint8_t(physical_row), std::uint16_t(stripe)};
    }

    void start(int operation, bool for_ffn = false, int rows = 3) {
        require(dut_.start_ready, "start issued while hidden transfer was busy");
        dut_.start_operation = operation;
        dut_.start_for_ffn = for_ffn;
        dut_.start_active_rows = rows;
        dut_.start_valid = 1;
        tick();
        dut_.start_valid = 0;
    }

    void accept_request(bool read) {
        for (int timeout = 0; timeout < 80; ++timeout) {
            if ((read && dut_.read_request_valid) ||
                (!read && dut_.write_request_valid))
                break;
            tick();
        }
        require(read ? dut_.read_request_valid : dut_.write_request_valid,
                "logical DMA request was not issued");
        require((read ? dut_.read_request_address : dut_.write_request_address) ==
                    kBase &&
                (read ? dut_.read_request_bytes : dut_.write_request_bytes) ==
                    3u * 8192u &&
                (read ? dut_.read_request_tag : dut_.write_request_tag) == kTag,
                "logical DMA request payload mismatch");
        if (read)
            dut_.read_request_ready = 1;
        else
            dut_.write_request_ready = 1;
        tick();
        dut_.read_request_ready = 0;
        dut_.write_request_ready = 0;
    }

    void run_read(int operation, bool abort_after_accept, bool dma_error,
                  bool bad_protocol) {
        clear_inputs();
        start(operation);
        accept_request(true);
        if (abort_after_accept) {
            dut_.abort_request = 1;
            tick();
            require(!dut_.abort_ack,
                    "accepted read acknowledged abort before DMA drain");
        }
        constexpr std::uint32_t beats = 3u * kChunksPerRow;
        for (std::uint32_t beat = 0; beat < beats; ++beat) {
            const Location expected = location(operation, beat);
            set_beat(dut_.read_data, expected.physical_row,
                     expected.channel_word);
            dut_.read_data_valid = 1;
            dut_.read_data_last = beat + 1 == beats;
            dut_.read_data_tag = bad_protocol && beat == 7 ? kTag ^ 1 : kTag;
            dut_.local_write_ready = abort_after_accept ? 0 : 1;
            if (!abort_after_accept && beat % 11 == 3) {
                dut_.local_write_ready = 0;
                tick();
                tick();
                dut_.local_write_ready = 1;
            }
            eval_low();
            for (int timeout = 0; timeout < 40 && !dut_.read_data_ready;
                 ++timeout)
                tick();
            if (!dut_.read_data_ready)
                throw std::runtime_error("read data made no progress operation=" +
                                         std::to_string(operation) + " beat=" +
                                         std::to_string(beat));
            if (!abort_after_accept) {
                if (!dut_.local_write_valid ||
                    dut_.local_write_physical_row != expected.physical_row ||
                    dut_.local_write_channel_word != expected.channel_word ||
                    dut_.local_write_byte_enable != 0xffff)
                    throw std::runtime_error(
                        "read-to-local mapping mismatch operation=" +
                        std::to_string(operation) + " beat=" +
                        std::to_string(beat) + " expected_row=" +
                        std::to_string(expected.physical_row) + " actual_row=" +
                        std::to_string(dut_.local_write_physical_row) +
                        " expected_channel_word=" +
                        std::to_string(expected.channel_word) +
                        " actual_channel_word=" +
                        std::to_string(dut_.local_write_channel_word) + " valid=" +
                        std::to_string(dut_.local_write_valid) + " be=" +
                        std::to_string(dut_.local_write_byte_enable));
                for (int byte = 0; byte < 16; ++byte)
                    require(get_byte(dut_.local_write_data, byte) ==
                                get_byte(dut_.read_data, byte),
                            "read-to-local payload mismatch");
            } else {
                require(!dut_.local_write_valid,
                        "aborted read modified hidden SRAM");
            }
            tick();
            dut_.read_data_valid = 0;
        }
        dut_.read_data_last = 0;
        dut_.read_data_tag = kTag;
        dut_.local_write_ready = 1;
        if (dma_error)
            dut_.read_request_error = 1;
        else
            dut_.read_request_done = 1;
        tick();
        dut_.read_request_done = 0;
        dut_.read_request_error = 0;
        wait_terminal(abort_after_accept, dma_error || bad_protocol);
        dut_.abort_request = 0;
        tick();
    }

    void run_write(int operation, bool abort_after_accept, bool dma_error) {
        clear_inputs();
        start(operation);
        accept_request(false);
        if (abort_after_accept) {
            dut_.abort_request = 1;
            tick();
            require(!dut_.abort_ack,
                    "accepted write acknowledged abort before W and B drain");
        }
        constexpr std::uint32_t beats = 3u * kChunksPerRow;
        for (std::uint32_t beat = 0; beat < beats; ++beat) {
            const Location expected = location(operation, beat);
            dut_.local_read_ready = 1;
            if (beat % 13 == 5) {
                dut_.local_read_ready = 0;
                tick();
                tick();
                dut_.local_read_ready = 1;
            }
            eval_low();
            for (int timeout = 0;
                 timeout < 40 &&
                 (!dut_.local_read_valid || !dut_.local_read_ready);
                 ++timeout)
                tick();
            if (!dut_.local_read_valid || !dut_.local_read_ready)
                throw std::runtime_error("local read made no progress operation=" +
                                         std::to_string(operation) + " beat=" +
                                         std::to_string(beat) + " busy=" +
                                         std::to_string(dut_.busy) + " done_valid=" +
                                         std::to_string(dut_.done_valid) + " error=" +
                                         std::to_string(dut_.error) + " write_req=" +
                                         std::to_string(dut_.write_request_valid));
            require(dut_.local_read_physical_row == expected.physical_row &&
                        dut_.local_read_channel_word == expected.channel_word,
                    "local-to-write mapping mismatch");
            tick();
            dut_.local_read_ready = 0;
            set_beat(dut_.local_read_data, expected.physical_row,
                     expected.channel_word);
            dut_.local_read_response_valid = 1;
            tick();
            dut_.local_read_response_valid = 0;
            for (int timeout = 0; timeout < 20 && !dut_.write_data_valid;
                 ++timeout)
                tick();
            require(dut_.write_data_valid &&
                        dut_.write_data_last == (beat + 1 == beats) &&
                        dut_.write_byte_enable == 0xffff,
                    "write data metadata mismatch");
            for (int byte = 0; byte < 16; ++byte)
                require(get_byte(dut_.write_data, byte) ==
                            get_byte(dut_.local_read_data, byte),
                        "local-to-write payload mismatch");
            dut_.write_data_ready = (beat % 17) != 9;
            eval_low();
            if (!dut_.write_data_ready) {
                const std::uint8_t held = get_byte(dut_.write_data, 0);
                tick();
                require(dut_.write_data_valid &&
                            get_byte(dut_.write_data, 0) == held,
                        "stalled write payload changed");
                dut_.write_data_ready = 1;
            }
            tick();
            dut_.write_data_ready = 0;
        }
        if (dma_error)
            dut_.write_request_error = 1;
        else
            dut_.write_request_done = 1;
        tick();
        dut_.write_request_done = 0;
        dut_.write_request_error = 0;
        wait_terminal(abort_after_accept, dma_error);
        dut_.abort_request = 0;
        tick();
    }

    void wait_terminal(bool aborted, bool expect_error) {
        for (int timeout = 0; timeout < 20; ++timeout) {
            if ((aborted && dut_.abort_ack) || (!aborted && dut_.done_valid))
                break;
            tick();
        }
        require(aborted ? dut_.abort_ack : dut_.done_valid,
                "terminal event was not reported");
        if (!aborted)
            require(bool(dut_.error) == expect_error,
                    "terminal error status mismatch");
    }

    void set_direct_indices(const std::vector<std::uint16_t>& indices) {
        require(!indices.empty() && indices.size() <= 48,
                "direct row test has an invalid resident row count");
        dut_.start_direct_row_index_enable = 1;
        clear_wide(dut_.start_direct_row_index);
        for (std::size_t row = 0; row < indices.size(); ++row) {
            require(indices[row] < 2048,
                    "direct row test index exceeds the 11-bit field");
            set_bits(dut_.start_direct_row_index, unsigned(row * 11), 11,
                     indices[row]);
        }
    }

    std::uint32_t embedding_token(unsigned row) const {
        if (row == 0) return 0;
        if (row == 1) return 126463;
        if (row == 2) return 63232;
        if (row == 3 || row == 4) return 17;
        return (row * 7919u + 123u) % 126464u;
    }

    std::array<std::uint8_t, 8192> read_embedding_row(
            std::uint32_t token) {
        std::array<std::uint8_t, 8192> row{};
        require(embedding_payload_loaded_,
                "embedding row requested without a payload");
        embedding_file_.clear();
        embedding_file_.seekg(std::uint64_t(token) * row.size());
        embedding_file_.read(reinterpret_cast<char*>(row.data()), row.size());
        require(embedding_file_.good(),
                "cannot read complete BF16 embedding row");
        return row;
    }

    template <typename WideWords>
    void set_embedding_beat(WideWords& words,
                            const std::array<std::uint8_t, 8192>& row,
                            unsigned beat) {
        clear_wide(words);
        for (unsigned byte = 0; byte < 16; ++byte)
            words[byte / 4] |= std::uint32_t(row[beat * 16 + byte]) <<
                (8 * (byte % 4));
    }

    void set_embedding_descriptors(unsigned rows) {
        require(rows >= 1 && rows <= 48,
                "embedding test row count is out of range");
        dut_.start_direct_row_index_enable = 1;
        dut_.start_source_descriptor_enable = 1;
        clear_wide(dut_.start_source_index);
        dut_.start_source_embedding = 0;
        for (unsigned row = 0; row < rows; ++row) {
            set_bits(dut_.start_source_index, row * 17, 17,
                     embedding_token(row));
            dut_.start_source_embedding |= std::uint64_t{1} << row;
        }
        dut_.start_embedding_base = kEmbeddingBase;
        dut_.start_embedding_limit = kEmbeddingBase + kEmbeddingBytes;
    }

    void embedding_read_rows(unsigned rows) {
        clear_inputs();
        set_embedding_descriptors(rows);
        if (embedding_payload_loaded_) embedding_payload_rows_ += rows;
        const std::uint64_t request_count = dut_.accepted_dma_request_count;
        const std::uint64_t data_count = dut_.accepted_dma_data_count;
        const std::uint64_t local_count = dut_.accepted_local_request_count;
        start(0, false, int(rows));
        for (unsigned row = 0; row < rows; ++row) {
            const auto token = embedding_token(row);
            const auto payload_row = embedding_payload_loaded_ ?
                read_embedding_row(token) : std::array<std::uint8_t, 8192>{};
            wait_direct_request(true,
                kEmbeddingBase + std::uint64_t(token) * 8192ull,
                row == 0 ? 2 : 0);
            for (int beat = 0; beat < kChunksPerRow; ++beat) {
                if (embedding_payload_loaded_)
                    set_embedding_beat(dut_.read_data, payload_row, beat);
                else
                    set_beat(dut_.read_data, std::uint8_t(token),
                             std::uint16_t(beat));
                dut_.read_data_valid = 1;
                dut_.read_data_last = beat + 1 == kChunksPerRow;
                dut_.local_write_ready = (beat % 97) != 23;
                if (!dut_.local_write_ready) {
                    eval_low();
                    require(!dut_.read_data_ready,
                            "embedding read ignored local SRAM backpressure");
                    tick();
                    dut_.local_write_ready = 1;
                }
                eval_low();
                require(dut_.read_data_ready && dut_.local_write_valid,
                        "embedding read data beat was not accepted");
                require(dut_.local_write_physical_row == row &&
                            dut_.local_write_channel_word == beat,
                        "embedding read wrote the wrong resident row");
                for (int byte = 0; byte < 16; ++byte)
                    require(get_byte(dut_.local_write_data, byte) ==
                                get_byte(dut_.read_data, byte),
                            "embedding BF16 raw payload mismatch");
                tick();
                dut_.read_data_valid = 0;
            }
            dut_.read_data_last = 0;
            dut_.read_request_done = 1;
            tick();
            dut_.read_request_done = 0;
        }
        for (int cycle = 0; cycle < 20 && !dut_.done_valid; ++cycle) tick();
        require(dut_.done_valid && !dut_.error,
                "embedding indexed input did not complete");
        require(dut_.accepted_dma_request_count - request_count == rows &&
                    dut_.accepted_dma_data_count - data_count ==
                        std::uint64_t(rows) * kChunksPerRow &&
                    dut_.accepted_local_request_count - local_count ==
                        std::uint64_t(rows) * kChunksPerRow,
                "embedding accepted-event counts mismatch");
        tick();
    }

    void embedding_configuration_errors() {
        clear_inputs();
        set_embedding_descriptors(1);
        set_bits(dut_.start_source_index, 0, 17, 126464);
        const auto request_count = dut_.accepted_dma_request_count;
        start(0, false, 1);
        for (int cycle = 0; cycle < 10 && !dut_.done_valid; ++cycle) tick();
        require(dut_.done_valid && dut_.error && dut_.error_id == 1 &&
                    dut_.accepted_dma_request_count == request_count,
                "out-of-range embedding token was not rejected");
        tick();

        clear_inputs();
        set_embedding_descriptors(1);
        set_bits(dut_.start_source_index, 0, 17, 126463);
        dut_.start_embedding_limit =
            kEmbeddingBase + 126463ull * 8192ull + 4096ull;
        start(0, false, 1);
        for (int cycle = 0; cycle < 10 && !dut_.done_valid; ++cycle) tick();
        require(dut_.done_valid && dut_.error && dut_.error_id == 1 &&
                    dut_.accepted_dma_request_count == request_count,
                "embedding row beyond limit was not rejected");
        tick();
    }

    void embedding_abort_drain() {
        clear_inputs();
        set_embedding_descriptors(1);
        const auto token = embedding_token(0);
        start(0, false, 1);
        wait_direct_request(true,
            kEmbeddingBase + std::uint64_t(token) * 8192ull, 0);
        dut_.abort_request = 1;
        tick();
        require(!dut_.abort_ack,
                "embedding read acknowledged abort before response drain");
        for (int beat = 0; beat < kChunksPerRow; ++beat) {
            set_beat(dut_.read_data, std::uint8_t(token),
                     std::uint16_t(beat));
            dut_.read_data_valid = 1;
            dut_.read_data_last = beat + 1 == kChunksPerRow;
            dut_.local_write_ready = 0;
            eval_low();
            require(dut_.read_data_ready && !dut_.local_write_valid,
                    "aborted embedding response modified resident hidden");
            tick();
            dut_.read_data_valid = 0;
        }
        dut_.read_data_last = 0;
        dut_.read_request_done = 1;
        tick();
        dut_.read_request_done = 0;
        wait_terminal(true, false);
        dut_.abort_request = 0;
        tick();
        require(dut_.start_ready,
                "hidden transfer did not restart after embedding abort");
    }

    void wait_direct_request(bool read, std::uint64_t expected_address,
                             int stall_cycles) {
        for (int cycles = 0; cycles < 40; ++cycles) {
            if (read ? dut_.read_request_valid : dut_.write_request_valid)
                break;
            tick();
        }
        require(read ? dut_.read_request_valid : dut_.write_request_valid,
                "direct row DMA request was not issued");
        for (int cycle = 0; cycle < stall_cycles; ++cycle) {
            require((read ? dut_.read_request_address :
                            dut_.write_request_address) == expected_address &&
                        (read ? dut_.read_request_bytes :
                                dut_.write_request_bytes) == 8192u &&
                        (read ? dut_.read_request_tag :
                                dut_.write_request_tag) == kTag,
                    "stalled direct row DMA request changed");
            tick();
        }
        require((read ? dut_.read_request_address :
                        dut_.write_request_address) == expected_address &&
                    (read ? dut_.read_request_bytes :
                            dut_.write_request_bytes) == 8192u,
                "direct row DMA request payload mismatch");
        if (read)
            dut_.read_request_ready = 1;
        else
            dut_.write_request_ready = 1;
        tick();
        dut_.read_request_ready = 0;
        dut_.write_request_ready = 0;
    }

    void direct_read_indices() {
        const std::vector<std::uint16_t> indices{10, 11, 733, 72, 1025};
        clear_inputs();
        set_direct_indices(indices);
        const std::uint64_t request_count = dut_.accepted_dma_request_count;
        const std::uint64_t data_count = dut_.accepted_dma_data_count;
        const std::uint64_t local_count = dut_.accepted_local_request_count;
        start(0, false, int(indices.size()));

        for (std::size_t row = 0; row < indices.size(); ++row) {
            const std::uint64_t address = kBase +
                std::uint64_t(indices[row]) * 8192ull;
            wait_direct_request(true, address, row == 0 ? 3 : 0);
            for (int beat = 0; beat < kChunksPerRow; ++beat) {
                set_beat(dut_.read_data, std::uint8_t(row),
                         std::uint16_t(beat));
                dut_.read_data_valid = 1;
                dut_.read_data_last = beat + 1 == kChunksPerRow;
                dut_.local_write_ready = 1;
                if ((beat % 113) == 17) {
                    dut_.local_write_ready = 0;
                    eval_low();
                    require(!dut_.read_data_ready,
                            "direct read ignored local SRAM backpressure");
                    tick();
                    tick();
                    dut_.local_write_ready = 1;
                }
                eval_low();
                require(dut_.read_data_ready && dut_.local_write_valid,
                        "direct read data beat was not accepted");
                require(dut_.local_write_physical_row == row &&
                            dut_.local_write_channel_word == beat,
                        "direct read used the wrong resident SRAM row");
                for (int byte = 0; byte < 16; ++byte)
                    require(get_byte(dut_.local_write_data, byte) ==
                                get_byte(dut_.read_data, byte),
                            "direct read payload mismatch");
                tick();
                dut_.read_data_valid = 0;
            }
            dut_.read_data_last = 0;
            for (int cycle = 0; cycle < 2; ++cycle) {
                require(!dut_.read_request_valid,
                        "next direct read started before the DMA terminal event");
                tick();
            }
            if (row + 1 == indices.size())
                dut_.done_ready = 0;
            dut_.read_request_done = 1;
            tick();
            dut_.read_request_done = 0;
        }

        for (int cycle = 0; cycle < 20 && !dut_.done_valid; ++cycle) tick();
        require(dut_.done_valid && !dut_.error,
                "direct indexed NFE input did not complete");
        for (int cycle = 0; cycle < 3; ++cycle) {
            tick();
            require(dut_.done_valid && !dut_.error,
                    "direct read completion changed while stalled");
        }
        require(dut_.accepted_dma_request_count - request_count ==
                    indices.size() &&
                    dut_.accepted_dma_data_count - data_count ==
                    indices.size() * kChunksPerRow &&
                    dut_.accepted_local_request_count - local_count ==
                    indices.size() * kChunksPerRow,
                "direct read accepted-event counts mismatch");
        dut_.done_ready = 1;
        tick();
    }

    void direct_write_indices() {
        const std::vector<std::uint16_t> indices{1500, 1499, 48, 333};
        clear_inputs();
        set_direct_indices(indices);
        const std::uint64_t request_count = dut_.accepted_dma_request_count;
        const std::uint64_t data_count = dut_.accepted_dma_data_count;
        const std::uint64_t local_count = dut_.accepted_local_request_count;
        start(1, false, int(indices.size()));

        for (std::size_t row = 0; row < indices.size(); ++row) {
            const std::uint64_t address = kBase +
                std::uint64_t(indices[row]) * 8192ull;
            wait_direct_request(false, address, row == 0 ? 2 : 0);
            for (int beat = 0; beat < kChunksPerRow; ++beat) {
                dut_.local_read_ready = 1;
                eval_low();
                require(dut_.local_read_valid,
                        "direct write local SRAM request was not issued");
                require(dut_.local_read_physical_row == row &&
                            dut_.local_read_channel_word == beat,
                        "direct write used the wrong resident SRAM row");
                tick();
                dut_.local_read_ready = 0;

                set_beat(dut_.local_read_data, std::uint8_t(row),
                         std::uint16_t(beat));
                dut_.local_read_response_valid = 1;
                tick();
                dut_.local_read_response_valid = 0;
                for (int cycle = 0; cycle < 20 && !dut_.write_data_valid;
                     ++cycle)
                    tick();
                require(dut_.write_data_valid &&
                            dut_.write_data_last ==
                                (beat + 1 == kChunksPerRow) &&
                            dut_.write_byte_enable == 0xffff,
                        "direct write data metadata mismatch");
                for (int byte = 0; byte < 16; ++byte)
                    require(get_byte(dut_.write_data, byte) ==
                                get_byte(dut_.local_read_data, byte),
                            "direct write payload mismatch");
                dut_.write_data_ready = (beat % 127) != 31;
                if (!dut_.write_data_ready) {
                    const std::uint8_t held = get_byte(dut_.write_data, 0);
                    tick();
                    tick();
                    require(dut_.write_data_valid &&
                                get_byte(dut_.write_data, 0) == held,
                            "direct write payload changed while stalled");
                    dut_.write_data_ready = 1;
                }
                tick();
                dut_.write_data_ready = 0;
            }
            for (int cycle = 0; cycle < 2; ++cycle) {
                require(!dut_.write_request_valid,
                        "next direct write started before the DMA terminal event");
                tick();
            }
            dut_.write_request_done = 1;
            tick();
            dut_.write_request_done = 0;
        }

        for (int cycle = 0; cycle < 20 && !dut_.done_valid; ++cycle) tick();
        require(dut_.done_valid && !dut_.error,
                "direct indexed NFE output did not complete");
        require(dut_.accepted_dma_request_count - request_count ==
                    indices.size() &&
                    dut_.accepted_dma_data_count - data_count ==
                    indices.size() * kChunksPerRow &&
                    dut_.accepted_local_request_count - local_count ==
                    indices.size() * kChunksPerRow,
                "direct write accepted-event counts mismatch");
        tick();
    }

    void direct_range_error() {
        clear_inputs();
        const std::vector<std::uint16_t> indices{2047};
        set_direct_indices(indices);
        dut_.start_ddr_limit = kBase + 2047ull * 8192ull + 4096ull;
        const std::uint64_t request_count = dut_.accepted_dma_request_count;
        start(0, false, 1);
        for (int cycle = 0; cycle < 10 && !dut_.done_valid; ++cycle) tick();
        require(dut_.done_valid && dut_.error && dut_.error_id == 1,
                "direct row outside the DDR region was not rejected");
        require(dut_.accepted_dma_request_count == request_count,
                "out-of-range direct row issued a DMA request");
        tick();
    }

    void direct_dma_error_stops_rows() {
        clear_inputs();
        const std::vector<std::uint16_t> indices{2, 900, 3};
        set_direct_indices(indices);
        const std::uint64_t request_count = dut_.accepted_dma_request_count;
        start(0, false, int(indices.size()));
        wait_direct_request(true, kBase + 2ull * 8192ull, 0);
        dut_.done_ready = 0;
        dut_.read_request_error = 1;
        tick();
        dut_.read_request_error = 0;
        for (int cycle = 0; cycle < 10 && !dut_.done_valid; ++cycle) tick();
        require(dut_.done_valid && dut_.error && dut_.error_id == 4,
                "direct row DMA error did not reach completion");
        require(dut_.accepted_dma_request_count - request_count == 1 &&
                    !dut_.read_request_valid,
                "direct row DMA error incorrectly started a later row");
        dut_.done_ready = 1;
        tick();
    }

    void invalid_configuration_and_permutation() {
        clear_inputs();
        dut_.start_ddr_limit = kBase + 4096;
        start(3);
        require(dut_.done_valid && dut_.error && dut_.error_id == 1,
                "undersized DDR range was not rejected");
        tick();

        clear_inputs();
        set_bits(dut_.start_physical_to_token_ordinal, 6, 6, 2);
        start(0);
        for (int timeout = 0; timeout < 10 && !dut_.done_valid; ++timeout) tick();
        require(dut_.done_valid && dut_.error && dut_.error_id == 2,
                "duplicate row permutation was not rejected");
        tick();
    }

    void completion_backpressure() {
        clear_inputs();
        dut_.start_ddr_limit = kBase + 4096;
        dut_.done_ready = 0;
        start(3, true);
        require(dut_.done_valid && dut_.done_for_ffn && dut_.error &&
                    dut_.error_id == 1 && !dut_.start_ready,
                "FFN completion was not held until accepted");
        for (int cycle = 0; cycle < 3; ++cycle) {
            tick();
            require(dut_.done_valid && dut_.done_for_ffn && dut_.error &&
                        dut_.error_id == 1 && !dut_.start_ready,
                    "stalled FFN completion changed");
        }
        dut_.done_ready = 1;
        tick();
        require(!dut_.done_valid && dut_.start_ready,
                "hidden transfer did not release an accepted completion");
    }

    void abort_before_request() {
      for (int operation : {2, 3}) {
        clear_inputs();
        const bool read = operation == 3;
        start(operation);
        for (int timeout = 0; timeout < 40 &&
             !(read ? dut_.read_request_valid : dut_.write_request_valid);
             ++timeout)
            tick();
        require(read ? dut_.read_request_valid : dut_.write_request_valid,
                "abort setup logical request was not issued");
        dut_.abort_request = 1;
        dut_.read_request_ready = 1;
        dut_.write_request_ready = 1;
        eval_low();
        require(!dut_.read_request_valid && !dut_.write_request_valid,
                "abort accepted an untracked DMA request");
        tick();
        require(dut_.abort_ack && !dut_.write_request_valid,
                "unaccepted logical request did not abort immediately");
        tick();
        require(!dut_.abort_ack,
                "held abort generated more than one acknowledgement");
        dut_.abort_request = 0;
        tick();
        require(dut_.start_ready, "hidden transfer did not restart after abort");
      }
    }

    void early_write_errors() {
      for (unsigned scenario = 0; scenario < 4; ++scenario) {
        clear_inputs();
        start(2);
        accept_request(false);
        const unsigned pending = scenario < 3 ? scenario : 1;
        if (scenario == 3) {
            dut_.local_read_ready = 1;
            tick(); dut_.local_read_ready = 0;
            dut_.local_read_response_valid = 1;
            tick(); dut_.local_read_response_valid = 0;
            tick();
            require(dut_.write_data_valid, "early-error stalled payload missing");
        }
        dut_.local_read_ready = 1;
        for (unsigned request = 0; request < pending; ++request) {
            eval_low();
            require(dut_.local_read_valid, "early-error SRAM request missing");
            tick();
        }
        dut_.write_request_error = 1;
        eval_low();
        require(!dut_.local_read_valid && !dut_.write_data_valid,
                "terminal DMA error continued SRAM or write issue");
        tick(); dut_.write_request_error = 0;
        for (unsigned response = 0; response < pending; ++response) {
            require(!dut_.done_valid, "completed before SRAM responses drained");
            dut_.local_read_response_valid = 1;
            tick();
        }
        dut_.local_read_response_valid = 0;
        wait_terminal(false, true);
        require(dut_.error_id == 4 && !dut_.local_read_valid &&
                !dut_.write_data_valid, "early write error completion mismatch");
        tick();
        require(dut_.start_ready, "early write error did not release transfer");
      }
    }
};
}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::threadContextp()->threads(1);
    try {
        const bool control_only = argc > 1 &&
            std::strcmp(argv[1], "--control-errors") == 0;
        Regression regression(argc > 1 && !control_only ? argv[1] : nullptr);
        if (control_only) regression.run_control_errors();
        else regression.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL hidden_transfer: " << error.what() << "\n";
        return 1;
    }
}
