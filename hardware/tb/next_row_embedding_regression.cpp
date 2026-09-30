#include "Vtoken_issue_embedding_top.h"
#include "generated/execution_config_packer.hpp"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint64_t kMetadataBase = 0x48000000ull;
constexpr std::uint64_t kEmbeddingBase = 0x200000000ull;
constexpr std::uint64_t kEmbeddingBytes = 126464ull * 8192ull;
constexpr unsigned kWordsPerRow = 512;

template <typename Wide>
void clear_wide(Wide& value, unsigned words) {
    for (unsigned word = 0; word < words; ++word) value[word] = 0;
}

template <typename Wide>
void set_byte(Wide& value, unsigned byte, std::uint8_t data) {
    const unsigned word = byte / 4;
    const unsigned shift = (byte % 4) * 8;
    value[word] = (value[word] & ~(0xffu << shift)) |
        (std::uint32_t(data) << shift);
}

template <typename Wide>
std::uint8_t get_byte(const Wide& value, unsigned byte) {
    return std::uint8_t(value[byte / 4] >> ((byte % 4) * 8));
}

std::uint32_t token_for(unsigned global_row) {
    if (global_row == 0) return 0;
    if (global_row == 1) return 126463;
    if (global_row == 2 || global_row == 3) return 17;
    return (global_row * 7919u + 123u) % 126464u;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class Regression {
  public:
    explicit Regression(unsigned total_token_count) : total_rows_(total_token_count) {
        require(total_rows_ >= 49 && total_rows_ <= 96,
                "total rows must be between 49 and 96");
        context_.threads(1);
        dut_ = new Vtoken_issue_embedding_top(&context_);
    }

    ~Regression() {
        dut_->final();
        delete dut_;
    }

    void run() {
        reset();
        pack_round(48, 0, 0);
        pack_round(total_rows_ - 48, 1, 48);
        for (unsigned round = 0; round < 2; ++round) {
            const auto offset = round_offsets_[round];
            for (unsigned physical = 0; physical < metadata_[offset]; ++physical) {
                const auto entry = offset+32+16*physical;
                const unsigned position = metadata_[entry+4] | unsigned(metadata_[entry+5])<<8;
                if (position % 3 == 1) metadata_[offset+24+physical/8] |= 1u << (physical%8);
            }
        }
        configure_metadata();
        unsigned first_row = 0;
        for (unsigned round = 0; round < 2; ++round) {
            start_metadata();
            wait_metadata(round);
            const unsigned rows = round == 0 ? 48 : total_rows_ - 48;
            run_hidden_round(round, first_row, rows);
            first_row += rows;
            dut_->metadata_config_release = 1;
            tick();
            dut_->metadata_config_release = 0;
            tick();
        }
        require(dut_->hidden_request_count == total_rows_ &&
                    dut_->hidden_data_count ==
                        std::uint64_t(total_rows_) * kWordsPerRow &&
                    dut_->hidden_local_write_count ==
                        std::uint64_t(total_rows_) * kWordsPerRow,
                "next-row embedding accepted-event totals mismatch");
        require(loaded_events_ == total_rows_, "metadata scan must not emit duplicate token events");
        for (unsigned position = 0; position < total_rows_; ++position)
            require(loaded_seen_.at(position), "metadata load omitted a token event");
        require(metadata_.size() == round_offsets_[1] + token_batch_bytes_[1],
                "metadata byte accounting mismatch");
        std::cout << "PASS next-row embedding handoff: rows=" << total_rows_
                  << " rounds=48+" << total_rows_ - 48 << ' '
                  << "packer_to_loader_to_hidden=raw_bit "
                     "token_first_middle_last="
                     "PASS repeat_noncontiguous=PASS bytes="
                  << std::uint64_t(total_rows_) * 8192ull
                  << " requests=" << dut_->hidden_request_count
                  << " cycles=" << cycles_ << " threads=" << dut_->threads()
                  << '\n';
    }

  private:
    VerilatedContext context_;
    Vtoken_issue_embedding_top* dut_ = nullptr;
    unsigned total_rows_ = 0;
    std::vector<std::uint8_t> metadata_;
    std::array<std::size_t, 2> round_offsets_{};
    std::array<std::size_t, 2> token_batch_bytes_{};
    std::uint64_t cycles_ = 0;
    unsigned loaded_events_ = 0;
    std::array<bool,96> loaded_seen_{};

    bool metadata_transfer_active_ = false;
    std::uint64_t metadata_address_ = 0;
    std::uint32_t metadata_bytes_ = 0;
    std::uint32_t metadata_offset_ = 0;

    void clear_inputs() {
        dut_->clk = 0;
        dut_->rst = 0;
        dut_->pack_start_valid = 0;
        dut_->pack_start_rows = 0;
        dut_->pack_start_round = 0;
        dut_->pack_start_first_row = 0;
        dut_->pack_row_valid = 0;
        dut_->pack_row_index = 0;
        dut_->pack_row_source_index = 0;
        dut_->pack_row_token_position = 0;
        dut_->pack_row_kv_index = 0;
        dut_->pack_activation_bits = 0;
        dut_->pack_row_query_group = 0;
        dut_->pack_row_cache_group = 0;
        dut_->pack_beat_ready = 0;
        dut_->pack_done_ready = 0;
        dut_->metadata_start_valid = 0;
        clear_wide(dut_->configuration_bits, 80);
        dut_->metadata_request_ready = 0;
        dut_->metadata_response_valid = 0;
        clear_wide(dut_->metadata_response_data, 4);
        dut_->metadata_response_byte_enable = 0;
        dut_->metadata_response_last = 0;
        dut_->metadata_request_done = 0;
        dut_->metadata_request_error = 0;
        dut_->metadata_done_ready = 0;
        dut_->metadata_config_release = 0;
        dut_->hidden_start_valid = 0;
        dut_->resident_base = 0x100000000ull;
        dut_->resident_limit = 0x100000000ull + 2048ull * 8192ull;
        dut_->embedding_base = kEmbeddingBase;
        dut_->embedding_limit = kEmbeddingBase + kEmbeddingBytes;
        dut_->hidden_request_ready = 0;
        dut_->hidden_request_done = 0;
        dut_->hidden_request_error = 0;
        dut_->hidden_data_valid = 0;
        clear_wide(dut_->hidden_data, 4);
        dut_->hidden_byte_enable = 0xffff;
        dut_->hidden_data_last = 0;
        dut_->hidden_local_write_ready = 1;
        dut_->hidden_done_ready = 0;
    }

    void reset() {
        clear_inputs();
        dut_->rst = 1;
        for (unsigned cycle = 0; cycle < 4; ++cycle) tick();
        dut_->rst = 0;
        tick();
        require(dut_->pack_start_ready && dut_->metadata_start_ready,
                "handoff top did not reset to idle");
    }

    void tick() {
        dut_->clk = 0;
        dut_->eval();
        dut_->clk = 1;
        dut_->eval();
        dut_->clk = 0;
        dut_->eval();
        ++cycles_;
    }

    void pack_round(unsigned rows, unsigned round, unsigned first) {
        round_offsets_[round] = metadata_.size();
        dut_->pack_start_rows = rows;
        dut_->pack_start_round = round;
        dut_->pack_start_first_row = first;
        dut_->pack_start_valid = 1;
        for (unsigned timeout = 0; timeout < 20; ++timeout) {
            dut_->eval();
            if (dut_->pack_start_ready) {
                tick();
                break;
            }
            tick();
        }
        dut_->pack_start_valid = 0;
        for (unsigned row = 0; row < rows; ++row) {
            const unsigned global = first + row;
            dut_->pack_row_index = row;
            dut_->pack_row_source_index = token_for(global);
            dut_->pack_row_token_position = global;
            dut_->pack_row_kv_index = global;
            dut_->pack_activation_bits = (global & 1u) ? 4 : 8;
            dut_->pack_row_query_group = global / 8;
            dut_->pack_row_cache_group = global / 8;
            dut_->pack_row_valid = 1;
            for (unsigned timeout = 0; timeout < 100; ++timeout) {
                dut_->eval();
                if (dut_->pack_row_ready) {
                    tick();
                    break;
                }
                tick();
            }
            dut_->pack_row_valid = 0;
        }
        dut_->pack_beat_ready = 1;
        dut_->pack_done_ready = 0;
        bool saw_last = false;
        for (unsigned timeout = 0; timeout < 20000 && !dut_->pack_done_valid;
             ++timeout) {
            dut_->pack_beat_ready = (cycles_ % 7) != 3;
            dut_->clk = 0;
            dut_->eval();
            const bool beat = dut_->pack_beat_valid && dut_->pack_beat_ready;
            if (beat) {
                for (unsigned byte = 0; byte < 16; ++byte)
                    metadata_.push_back(get_byte(dut_->pack_beat_data, byte));
                saw_last = dut_->pack_beat_last;
            }
            dut_->clk = 1;
            dut_->eval();
            dut_->clk = 0;
            dut_->eval();
            ++cycles_;
        }
        require(dut_->pack_done_valid && !dut_->pack_error && saw_last,
                "next-row packer did not complete");
        dut_->pack_done_ready = 1;
        tick();
        dut_->pack_done_ready = 0;
        token_batch_bytes_[round] = metadata_.size() - round_offsets_[round];
    }

    void configure_metadata() {
        execution_config configuration{};
        configuration.magic = 0x344e4c44u;
        configuration.version = 4;
        configuration.header_bytes = EXECUTION_CONFIG_BYTES;
        configuration.total_bytes = EXECUTION_CONFIG_BYTES;
        configuration.total_token_count = total_rows_;
        configuration.sequence_length = 512;
        configuration.token_metadata_format = 3;
        configuration.token_metadata_base = kMetadataBase;
        configuration.token_metadata_limit = kMetadataBase + metadata_.size();
        const auto packed = pack_execution_config(configuration);
        clear_wide(dut_->configuration_bits, 80);
        for (unsigned byte = 0; byte < packed.size(); ++byte)
            set_byte(dut_->configuration_bits, byte, packed[byte]);
    }

    void drive_metadata_cycle() {
        dut_->metadata_request_ready = !metadata_transfer_active_ &&
            cycles_ % 5 != 1;
        dut_->metadata_response_valid = 0;
        dut_->metadata_response_byte_enable = 0;
        dut_->metadata_response_last = 0;
        dut_->metadata_request_done = 0;
        clear_wide(dut_->metadata_response_data, 4);
        if (metadata_transfer_active_ && cycles_ % 7 != 2) {
            dut_->metadata_response_valid = 1;
            dut_->metadata_response_byte_enable = 0xffff;
            for (unsigned byte = 0; byte < 16; ++byte) {
                const auto index = metadata_address_ - kMetadataBase +
                    metadata_offset_ + byte;
                require(index < metadata_.size(),
                        "metadata request exceeded packed bytes");
                set_byte(dut_->metadata_response_data, byte, metadata_[index]);
            }
            dut_->metadata_response_last =
                metadata_offset_ + 16 == metadata_bytes_;
        }
        dut_->clk = 0;
        dut_->eval();
        const bool request = dut_->metadata_request_valid &&
            dut_->metadata_request_ready;
        const bool response = dut_->metadata_response_valid &&
            dut_->metadata_response_ready;
        const bool last = response && dut_->metadata_response_last;
        if (last) {
            dut_->metadata_request_done = 1;
            dut_->eval();
        }
        if (dut_->loaded_token_valid) {
            require(response, "loaded token event without accepted DMA response");
            const unsigned position = dut_->loaded_token_position;
            require(position < total_rows_ && !loaded_seen_.at(position), "loaded token event duplicated or out of range");
            require(bool(dut_->loaded_token_kv_write) == (position % 3 != 1), "KV-disable mask lost during physical token reorder");
            loaded_seen_[position] = true; ++loaded_events_;
        }
        const auto address = dut_->metadata_request_address;
        const auto bytes = dut_->metadata_request_bytes;
        dut_->clk = 1;
        dut_->eval();
        dut_->clk = 0;
        dut_->eval();
        ++cycles_;
        if (request) {
            require(!metadata_transfer_active_,
                    "metadata requests overlapped");
            metadata_transfer_active_ = true;
            metadata_address_ = address;
            metadata_bytes_ = bytes;
            metadata_offset_ = 0;
        }
        if (response) {
            if (last) {
                metadata_transfer_active_ = false;
                metadata_offset_ = 0;
            } else {
                metadata_offset_ += 16;
            }
        }
    }

    void start_metadata() {
        dut_->metadata_start_valid = 1;
        for (unsigned timeout = 0; timeout < 100; ++timeout) {
            dut_->eval();
            if (dut_->metadata_start_ready) {
                drive_metadata_cycle();
                dut_->metadata_start_valid = 0;
                return;
            }
            drive_metadata_cycle();
        }
        throw std::runtime_error("metadata start timeout");
    }

    void wait_metadata(unsigned round) {
        dut_->metadata_done_ready = 0;
        for (unsigned timeout = 0; timeout < 30000; ++timeout) {
            drive_metadata_cycle();
            if (dut_->metadata_done_valid) break;
        }
        require(dut_->metadata_done_valid && !dut_->metadata_done_error &&
                    dut_->metadata_round == round,
                "metadata loader did not publish expected round");
        dut_->metadata_done_ready = 1;
        drive_metadata_cycle();
        dut_->metadata_done_ready = 0;
        require(dut_->metadata_config_valid,
                "metadata loader did not retain the published round");
    }

    std::uint32_t physical_token(unsigned round, unsigned physical) const {
        const auto entry = round_offsets_[round] + 32 + physical * 16;
        return std::uint32_t(metadata_[entry]) |
            (std::uint32_t(metadata_[entry + 1]) << 8) |
            ((std::uint32_t(metadata_[entry + 2]) & 1u) << 16);
    }

    std::uint8_t payload_byte(std::uint32_t token, unsigned word,
                              unsigned byte) const {
        return std::uint8_t((token * 13u + word * 7u + byte * 3u) & 0xffu);
    }

    void run_hidden_round(unsigned round, unsigned first, unsigned rows) {
        require(dut_->metadata_active_rows == rows,
                "metadata active row count mismatch before hidden start");
        dut_->hidden_start_valid = 1;
        for (unsigned timeout = 0; timeout < 20; ++timeout) {
            dut_->eval();
            if (dut_->hidden_start_ready) {
                tick();
                break;
            }
            tick();
        }
        dut_->hidden_start_valid = 0;
        for (unsigned row = 0; row < rows; ++row) {
            const auto token = physical_token(round, row);
            for (unsigned timeout = 0;
                 timeout < 40 && !dut_->hidden_request_valid; ++timeout)
                tick();
            require(dut_->hidden_request_valid &&
                        dut_->hidden_request_address ==
                            kEmbeddingBase + std::uint64_t(token) * 8192ull &&
                        dut_->hidden_request_bytes == 8192,
                    "hidden embedding request did not match packed token");
            dut_->hidden_request_ready = 1;
            tick();
            dut_->hidden_request_ready = 0;
            for (unsigned word = 0; word < kWordsPerRow; ++word) {
                clear_wide(dut_->hidden_data, 4);
                for (unsigned byte = 0; byte < 16; ++byte)
                    set_byte(dut_->hidden_data, byte,
                             payload_byte(token, word, byte));
                dut_->hidden_data_valid = 1;
                dut_->hidden_data_last = word + 1 == kWordsPerRow;
                dut_->hidden_local_write_ready = (cycles_ % 11) != 4;
                if (!dut_->hidden_local_write_ready) {
                    dut_->eval();
                    require(!dut_->hidden_data_ready,
                            "hidden handoff ignored local backpressure");
                    tick();
                    dut_->hidden_local_write_ready = 1;
                }
                dut_->eval();
                require(dut_->hidden_data_ready &&
                            dut_->hidden_local_write_valid &&
                            dut_->hidden_local_write_row == row &&
                            dut_->hidden_local_write_word == word,
                        "hidden handoff local address mismatch");
                for (unsigned byte = 0; byte < 16; ++byte)
                    require(get_byte(dut_->hidden_local_write_data, byte) ==
                                payload_byte(token, word, byte),
                            "hidden handoff BF16 raw mismatch");
                tick();
                dut_->hidden_data_valid = 0;
            }
            dut_->hidden_data_last = 0;
            dut_->hidden_request_done = 1;
            tick();
            dut_->hidden_request_done = 0;
        }
        dut_->hidden_done_ready = 0;
        for (unsigned timeout = 0; timeout < 40 && !dut_->hidden_done_valid;
             ++timeout)
            tick();
        require(dut_->hidden_done_valid && !dut_->hidden_error,
                "hidden handoff did not complete");
        dut_->hidden_done_ready = 1;
        tick();
        dut_->hidden_done_ready = 0;
        (void)first;
    }
};
}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        if (argc > 2)
            throw std::runtime_error(
                "usage: next_row_embedding_regression [total_token_count]");
        const unsigned total_token_count = argc == 2 ?
            static_cast<unsigned>(std::stoul(argv[1])) : 49u;
        Regression regression(total_token_count);
        regression.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL next-row embedding handoff: " << error.what()
                  << '\n';
        return 1;
    }
}
