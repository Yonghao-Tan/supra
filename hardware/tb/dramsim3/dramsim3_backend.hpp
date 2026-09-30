#ifndef SUPRA_DRAMSIM3_BACKEND_HPP
#define SUPRA_DRAMSIM3_BACKEND_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace supra::ddr {

constexpr std::size_t kTransactionBytes = 32;

struct Completion {
    std::uint64_t token = 0;
    std::uint64_t address = 0;
    bool write = false;
    std::array<std::uint8_t, kTransactionBytes> data{};
};

struct BackendStats {
    struct Channel {
        std::uint64_t reads_submitted = 0;
        std::uint64_t writes_submitted = 0;
        std::uint64_t reads_completed = 0;
        std::uint64_t writes_completed = 0;
        std::uint64_t requested_burst_read_bytes = 0;
        std::uint64_t requested_burst_write_bytes = 0;
        std::uint64_t can_accept_stalls = 0;
    };

    std::uint64_t core_cycles = 0;
    std::uint64_t dram_ticks = 0;
    std::uint64_t reads_submitted = 0;
    std::uint64_t writes_submitted = 0;
    std::uint64_t reads_completed = 0;
    std::uint64_t writes_completed = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;  // Enabled WSTRB bytes; full DQ uses writes_submitted * 32.
    std::uint64_t can_accept_stalls = 0;
    std::uint64_t maximum_outstanding = 0;
    std::vector<Channel> channels;
};

struct NamedRegion {
    std::string name;
    std::uint64_t base = 0;
    std::uint64_t limit = 0;
    bool readable = false;
    bool writable = false;
};

class Dramsim3Backend {
public:
    Dramsim3Backend(const std::string &config_path,
                    const std::string &region_manifest_path,
                    const std::string &image_path,
                    const std::string &output_directory,
                    unsigned core_frequency_mhz);
    ~Dramsim3Backend();

    Dramsim3Backend(const Dramsim3Backend &) = delete;
    Dramsim3Backend &operator=(const Dramsim3Backend &) = delete;

    bool can_accept(std::uint64_t address, bool write);
    bool submit_read(std::uint64_t address, std::uint64_t token);
    bool submit_write(std::uint64_t address, std::uint64_t token,
                      const std::array<std::uint8_t, kTransactionBytes> &data,
                      std::uint32_t byte_enable);
    void tick_core();
    bool poll_read(Completion &completion);
    bool poll_write(Completion &completion);
    bool idle() const;
    void drain(std::uint64_t maximum_core_cycles);
    void reset();

    std::uint8_t inspect_byte(std::uint64_t address) const;
    void initialize_byte(std::uint64_t address, std::uint8_t value);
    const BackendStats &stats() const { return stats_; }
    std::string stats_json() const;
    unsigned effective_channels() const;
    unsigned effective_ranks_per_channel() const;
    std::uint64_t effective_channel_bytes() const;
    const std::vector<NamedRegion> &named_regions() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    BackendStats stats_;
};

}  // namespace supra::ddr

extern "C" {
int supra_dramsim3_create(const char *config_path, const char *region_manifest_path,
                         const char *image_path, const char *output_directory,
                         int core_frequency_mhz);
void supra_dramsim3_destroy();
int supra_dramsim3_reset();
int supra_dramsim3_can_accept(unsigned long long address, int write);
int supra_dramsim3_submit_read(unsigned long long address,
                              unsigned long long token);
int supra_dramsim3_submit_write(unsigned long long address,
                               unsigned long long token,
                               const unsigned int *data_words,
                               unsigned int byte_enable);
void supra_dramsim3_tick_core();
int supra_dramsim3_poll_read(unsigned long long *token,
                            unsigned int *data_words);
int supra_dramsim3_poll_write(unsigned long long *token);
int supra_dramsim3_idle();
int supra_dramsim3_inspect_byte(unsigned long long address);
int supra_dramsim3_initialize_byte(unsigned long long address, unsigned char value);
unsigned long long supra_dramsim3_physical_read_bytes();
unsigned long long supra_dramsim3_physical_write_bytes();
const char *supra_dramsim3_stats_json();
}

#endif
