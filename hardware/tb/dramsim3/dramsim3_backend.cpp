#include "dramsim3_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "json.hpp"
#include "memory_system.h"

namespace supra::ddr {

namespace {

constexpr std::size_t kPageBytes = 4096;
struct Region {
    std::string name;
    std::uint64_t base = 0;
    std::uint64_t limit = 0;
    bool readable = false;
    bool writable = false;
};

struct PendingWrite {
    std::uint64_t token = 0;
    std::uint64_t address = 0;
    std::array<std::uint8_t, kTransactionBytes> data{};
    std::uint32_t byte_enable = 0;
    unsigned channel = 0;
};

struct PendingRead {
    std::uint64_t token = 0;
    unsigned channel = 0;
};

std::string errno_string(const std::string &operation) {
    return operation + ": " + std::strerror(errno);
}

}  // namespace

struct Dramsim3Backend::Impl {
    std::string config_path;
    std::string output_directory;
    std::uint64_t aperture_base = 0;
    std::uint64_t aperture_limit = 0;
    std::uint64_t dram_tick_numerator = 0;
    std::uint64_t dram_tick_denominator = 0;
    std::uint64_t dram_tick_phase = 0;
    std::vector<Region> regions;
    std::vector<NamedRegion> named_regions;
    std::unique_ptr<dramsim3::MemorySystem> memory_system;
    int image_fd = -1;
    const std::uint8_t *image = nullptr;
    std::size_t image_bytes = 0;
    std::unordered_map<std::uint64_t,
                       std::array<std::uint8_t, kPageBytes>> writable_pages;
    std::map<std::uint64_t, std::deque<PendingRead>> pending_reads;
    std::map<std::uint64_t, PendingWrite> pending_writes;
    std::deque<Completion> read_completions;
    std::deque<Completion> write_completions;
    BackendStats *stats = nullptr;
    mutable bool native_stats_written = false;

    ~Impl() {
        memory_system.reset();
        if (image != nullptr)
            munmap(const_cast<std::uint8_t *>(image), image_bytes);
        if (image_fd >= 0) close(image_fd);
    }

    const Region *find_region(std::uint64_t address, bool write) const {
        if (address > std::numeric_limits<std::uint64_t>::max() -
                          kTransactionBytes)
            return nullptr;
        const auto limit = address + kTransactionBytes;
        for (const auto &region : regions) {
            if (address >= region.base && limit <= region.limit &&
                (write ? region.writable : region.readable))
                return &region;
        }
        return nullptr;
    }

    std::uint64_t normalized(std::uint64_t address) const {
        if (address < aperture_base || address >= aperture_limit)
            throw std::out_of_range("DDR address outside logical aperture");
        return address - aperture_base;
    }

    std::uint8_t read_byte(std::uint64_t address) const {
        const auto page_base = address & ~(std::uint64_t{kPageBytes - 1});
        const auto page = writable_pages.find(page_base);
        if (page != writable_pages.end())
            return page->second[address - page_base];
        const auto offset = address - aperture_base;
        return offset < image_bytes ? image[offset] : 0;
    }

    void write_byte(std::uint64_t address, std::uint8_t value) {
        const auto page_base = address & ~(std::uint64_t{kPageBytes - 1});
        auto page = writable_pages.find(page_base);
        if (page == writable_pages.end()) {
            std::array<std::uint8_t, kPageBytes> initial{};
            for (std::size_t index = 0; index < kPageBytes; ++index) {
                const auto offset = page_base + index - aperture_base;
                initial[index] = offset < image_bytes ? image[offset] : 0;
            }
            page = writable_pages.emplace(page_base, std::move(initial)).first;
        }
        page->second[address - page_base] = value;
    }

    void create_memory_system() {
        memory_system = std::make_unique<dramsim3::MemorySystem>(
            config_path, output_directory,
            [this](std::uint64_t address) { read_callback(address); },
            [this](std::uint64_t address) { write_callback(address); });
    }

    void read_callback(std::uint64_t normalized_address) {
        auto pending = pending_reads.find(normalized_address);
        if (pending == pending_reads.end() || pending->second.empty())
            throw std::runtime_error("DRAMSim3 returned an unknown read address");
        Completion completion;
        completion.token = pending->second.front().token;
        completion.address = aperture_base + normalized_address;
        const auto channel = pending->second.front().channel;
        pending->second.pop_front();
        if (pending->second.empty()) pending_reads.erase(pending);
        for (std::size_t index = 0; index < kTransactionBytes; ++index)
            completion.data[index] = read_byte(completion.address + index);
        read_completions.push_back(completion);
        ++stats->reads_completed;
        ++stats->channels[channel].reads_completed;
    }

    void write_callback(std::uint64_t normalized_address) {
        auto pending = pending_writes.find(normalized_address);
        if (pending == pending_writes.end())
            throw std::runtime_error("DRAMSim3 returned an unknown write address");
        for (std::size_t index = 0; index < kTransactionBytes; ++index) {
            if ((pending->second.byte_enable >> index) & 1U)
                write_byte(pending->second.address + index,
                           pending->second.data[index]);
        }
        Completion completion;
        completion.token = pending->second.token;
        completion.address = pending->second.address;
        completion.write = true;
        write_completions.push_back(completion);
        const auto channel = pending->second.channel;
        pending_writes.erase(pending);
        ++stats->writes_completed;
        ++stats->channels[channel].writes_completed;
    }

    std::size_t outstanding() const {
        std::size_t count = pending_writes.size();
        for (const auto &entry : pending_reads) count += entry.second.size();
        return count;
    }

    unsigned channel(std::uint64_t normalized_address) const {
        const auto decoded = memory_system->GetChannel(normalized_address);
        if (decoded < 0 ||
            decoded >= static_cast<int>(memory_system->GetChannels()))
            throw std::runtime_error("DRAMSim3 decoded channel outside configured range");
        return static_cast<unsigned>(decoded);
    }

    void write_native_stats_once() const {
        if (!native_stats_written) {
            memory_system->PrintStats();
            native_stats_written = true;
        }
    }
};

Dramsim3Backend::Dramsim3Backend(const std::string &config_path,
                                 const std::string &region_manifest_path,
                                 const std::string &image_path,
                                 const std::string &output_directory,
                                 unsigned core_frequency_mhz)
    : impl_(std::make_unique<Impl>()) {
    if (core_frequency_mhz == 0)
        throw std::invalid_argument("core frequency must be positive");
    struct stat output_stat {};
    if (output_directory.empty() || stat(output_directory.c_str(), &output_stat) != 0 ||
        !S_ISDIR(output_stat.st_mode) || access(output_directory.c_str(), W_OK) != 0)
        throw std::invalid_argument("DRAMSim3 output directory must exist and be writable: " +
                                    output_directory);
    impl_->config_path = config_path;
    impl_->output_directory = output_directory;
    impl_->stats = &stats_;

    std::ifstream manifest_stream(region_manifest_path);
    if (!manifest_stream) throw std::runtime_error("DDR region manifest open failed");
    nlohmann::json manifest;
    manifest_stream >> manifest;
    impl_->aperture_base = manifest.at("base_address").get<std::uint64_t>();
    impl_->aperture_limit = manifest.at("limit_address").get<std::uint64_t>();
    if (impl_->aperture_limit <= impl_->aperture_base)
        throw std::runtime_error("DDR logical aperture is empty");
    const bool has_memory_map = manifest.contains("memory_map");
    const bool has_regions = manifest.contains("regions");
    if (has_memory_map == has_regions)
        throw std::runtime_error(
            "DDR region manifest must define exactly one of memory_map or regions");
    const auto &region_entries = has_memory_map ?
        manifest.at("memory_map") : manifest.at("regions");
    for (const auto &item : region_entries) {
        Region region;
        region.name = item.at("name").get<std::string>();
        region.base = item.at("base").get<std::uint64_t>();
        region.limit = item.at("limit").get<std::uint64_t>();
        const auto access = item.at("access").get<std::string>();
        region.readable = access == "read_only" || access == "read_write";
        region.writable = access == "write_only" || access == "read_write";
        impl_->named_regions.push_back({region.name, region.base, region.limit,
                                        region.readable, region.writable});
        impl_->regions.push_back(region);
    }

    impl_->image_fd = open(image_path.c_str(), O_RDONLY);
    if (impl_->image_fd < 0) throw std::runtime_error(errno_string("DDR image open"));
    struct stat image_stat {};
    if (fstat(impl_->image_fd, &image_stat) != 0)
        throw std::runtime_error(errno_string("DDR image stat"));
    if (image_stat.st_size <= 0 ||
        static_cast<std::uint64_t>(image_stat.st_size) >
            impl_->aperture_limit - impl_->aperture_base)
        throw std::runtime_error("DDR image size is invalid");
    impl_->image_bytes = static_cast<std::size_t>(image_stat.st_size);
    impl_->image = static_cast<const std::uint8_t *>(mmap(
        nullptr, impl_->image_bytes, PROT_READ, MAP_PRIVATE, impl_->image_fd, 0));
    if (impl_->image == MAP_FAILED) {
        impl_->image = nullptr;
        throw std::runtime_error(errno_string("DDR image mmap"));
    }

    impl_->create_memory_system();
    const auto ranks = impl_->memory_system->GetRanks();
    const auto tck_ns = impl_->memory_system->GetTCK();
    if (!(tck_ns > 0.0))
        throw std::runtime_error("DRAMSim3 tCK must be positive");
    const auto tck_as = std::llround(tck_ns * 1'000'000'000.0);
    const auto channel_bytes = static_cast<std::uint64_t>(
        impl_->memory_system->GetChannelSizeMB()) * 1024 * 1024;
    const auto aperture_bytes = impl_->aperture_limit - impl_->aperture_base;
    const auto channels = impl_->memory_system->GetChannels();
    const auto bus_bits = impl_->memory_system->GetBusBits();
    const auto burst_length = impl_->memory_system->GetBurstLength();
    const bool supported_bus =
        (channels == 2 && bus_bits == 16 && burst_length == 16) ||
        (channels == 1 && bus_bits == 32 && burst_length == 8);
    if (tck_as <= 0 ||
        !supported_bus ||
        impl_->memory_system->GetRequestSizeBytes() != 32 ||
        channel_bytes * channels != aperture_bytes ||
        (ranks != 1 && ranks != 2 && ranks != 4))
        throw std::runtime_error("DRAMSim3 effective configuration mismatch");
    stats_.channels.resize(channels);
    impl_->dram_tick_numerator = 1'000'000'000'000ULL;
    impl_->dram_tick_denominator =
        core_frequency_mhz * static_cast<std::uint64_t>(tck_as);
}

Dramsim3Backend::~Dramsim3Backend() = default;

bool Dramsim3Backend::can_accept(std::uint64_t address, bool write) {
    if ((address & (kTransactionBytes - 1)) != 0 ||
        impl_->find_region(address, write) == nullptr) {
        std::ostringstream message;
        message << "DRAMSim3 " << (write ? "write" : "read")
                << " address=0x" << std::hex << address << std::dec
                << " bytes=" << kTransactionBytes
                << " is unaligned, unmapped, or lacks the requested access permission";
        throw std::runtime_error(message.str());
    }
    const auto normalized_address = impl_->normalized(address);
    const auto channel = impl_->channel(normalized_address);
    const bool accepted =
        impl_->memory_system->WillAcceptTransaction(normalized_address, write);
    if (!accepted) {
        ++stats_.can_accept_stalls;
        ++stats_.channels[channel].can_accept_stalls;
    }
    return accepted;
}

bool Dramsim3Backend::submit_read(std::uint64_t address, std::uint64_t token) {
    if (!can_accept(address, false)) return false;
    const auto normalized_address = impl_->normalized(address);
    const auto channel = impl_->channel(normalized_address);
    impl_->pending_reads[normalized_address].push_back(PendingRead{token, channel});
    if (!impl_->memory_system->AddTransaction(normalized_address, false)) {
        impl_->pending_reads[normalized_address].pop_back();
        if (impl_->pending_reads[normalized_address].empty())
            impl_->pending_reads.erase(normalized_address);
        return false;
    }
    ++stats_.reads_submitted;
    stats_.read_bytes += kTransactionBytes;
    ++stats_.channels[channel].reads_submitted;
    stats_.channels[channel].requested_burst_read_bytes += kTransactionBytes;
    stats_.maximum_outstanding = std::max<std::uint64_t>(
        stats_.maximum_outstanding, impl_->outstanding());
    return true;
}

bool Dramsim3Backend::submit_write(
    std::uint64_t address, std::uint64_t token,
    const std::array<std::uint8_t, kTransactionBytes> &data,
    std::uint32_t byte_enable) {
    if (!can_accept(address, true)) return false;
    const auto normalized_address = impl_->normalized(address);
    const auto channel = impl_->channel(normalized_address);
    impl_->pending_writes.emplace(normalized_address,
        PendingWrite{token, address, data, byte_enable, channel});
    if (!impl_->memory_system->AddTransaction(normalized_address, true)) {
        impl_->pending_writes.erase(normalized_address);
        return false;
    }
    ++stats_.writes_submitted;
    stats_.write_bytes += static_cast<std::uint64_t>(__builtin_popcount(byte_enable));
    ++stats_.channels[channel].writes_submitted;
    stats_.channels[channel].requested_burst_write_bytes += kTransactionBytes;
    stats_.maximum_outstanding = std::max<std::uint64_t>(
        stats_.maximum_outstanding, impl_->outstanding());
    return true;
}

void Dramsim3Backend::tick_core() {
    impl_->dram_tick_phase += impl_->dram_tick_numerator;
    while (impl_->dram_tick_phase >= impl_->dram_tick_denominator) {
        impl_->memory_system->ClockTick();
        impl_->dram_tick_phase -= impl_->dram_tick_denominator;
        ++stats_.dram_ticks;
    }
    ++stats_.core_cycles;
}

bool Dramsim3Backend::poll_read(Completion &completion) {
    if (impl_->read_completions.empty()) return false;
    completion = impl_->read_completions.front();
    impl_->read_completions.pop_front();
    return true;
}

bool Dramsim3Backend::poll_write(Completion &completion) {
    if (impl_->write_completions.empty()) return false;
    completion = impl_->write_completions.front();
    impl_->write_completions.pop_front();
    return true;
}

bool Dramsim3Backend::idle() const {
    return impl_->outstanding() == 0 && impl_->read_completions.empty() &&
        impl_->write_completions.empty();
}

void Dramsim3Backend::drain(std::uint64_t maximum_core_cycles) {
    for (std::uint64_t cycle = 0; !idle() && cycle < maximum_core_cycles; ++cycle) {
        tick_core();
        Completion completion;
        while (poll_read(completion)) {}
        while (poll_write(completion)) {}
    }
    if (!idle()) throw std::runtime_error("DRAMSim3 backend drain timed out");
}

void Dramsim3Backend::reset() {
    if (!idle()) throw std::runtime_error("DRAMSim3 reset requires an idle backend");
    impl_->writable_pages.clear();
    impl_->dram_tick_phase = 0;
    impl_->memory_system.reset();
    impl_->create_memory_system();
    impl_->native_stats_written = false;
    const auto channels = stats_.channels.size();
    stats_ = BackendStats{};
    stats_.channels.resize(channels);
}

std::uint8_t Dramsim3Backend::inspect_byte(std::uint64_t address) const {
    (void)impl_->normalized(address);
    return impl_->read_byte(address);
}

void Dramsim3Backend::initialize_byte(std::uint64_t address,
                                      std::uint8_t value) {
    (void)impl_->normalized(address);
    impl_->write_byte(address, value);
}

std::string Dramsim3Backend::stats_json() const {
    impl_->write_native_stats_once();
    std::ostringstream stream;
    stream << '{'
           << "\"core_cycles\":" << stats_.core_cycles << ','
           << "\"dram_ticks\":" << stats_.dram_ticks << ','
           << "\"reads_submitted\":" << stats_.reads_submitted << ','
           << "\"writes_submitted\":" << stats_.writes_submitted << ','
           << "\"reads_completed\":" << stats_.reads_completed << ','
           << "\"writes_completed\":" << stats_.writes_completed << ','
           << "\"read_bytes\":" << stats_.read_bytes << ','
           << "\"write_bytes\":" << stats_.write_bytes << ','
           << "\"physical_read_bytes\":" << stats_.reads_submitted * kTransactionBytes << ','
           << "\"physical_write_bytes\":" << stats_.writes_submitted * kTransactionBytes << ','
           << "\"can_accept_stalls\":" << stats_.can_accept_stalls << ','
           << "\"maximum_outstanding\":" << stats_.maximum_outstanding << ','
           << "\"effective_ranks_per_channel\":"
           << effective_ranks_per_channel() << ','
           << "\"effective_channel_bytes\":" << effective_channel_bytes() << ','
           << "\"channels\":[";
    for (std::size_t channel = 0; channel < stats_.channels.size(); ++channel) {
        if (channel != 0) stream << ',';
        const auto &entry = stats_.channels[channel];
        stream << '{'
               << "\"channel\":" << channel << ','
               << "\"reads_submitted\":" << entry.reads_submitted << ','
               << "\"writes_submitted\":" << entry.writes_submitted << ','
               << "\"reads_completed\":" << entry.reads_completed << ','
               << "\"writes_completed\":" << entry.writes_completed << ','
               << "\"requested_burst_read_bytes\":"
               << entry.requested_burst_read_bytes << ','
               << "\"requested_burst_write_bytes\":"
               << entry.requested_burst_write_bytes << ','
               << "\"can_accept_stalls\":" << entry.can_accept_stalls
               << '}';
    }
    stream << "]}";
    return stream.str();
}

unsigned Dramsim3Backend::effective_channels() const {
    return impl_->memory_system->GetChannels();
}

unsigned Dramsim3Backend::effective_ranks_per_channel() const {
    return impl_->memory_system->GetRanks();
}

std::uint64_t Dramsim3Backend::effective_channel_bytes() const {
    return static_cast<std::uint64_t>(impl_->memory_system->GetChannelSizeMB()) *
        1024 * 1024;
}

const std::vector<NamedRegion> &Dramsim3Backend::named_regions() const {
    return impl_->named_regions;
}

}  // namespace supra::ddr

namespace {
std::unique_ptr<supra::ddr::Dramsim3Backend> dpi_backend;
}

extern "C" int supra_dramsim3_create(
    const char *config_path, const char *region_manifest_path,
    const char *image_path, const char *output_directory,
    int core_frequency_mhz) {
    try {
        if (core_frequency_mhz <= 0)
            throw std::invalid_argument("core frequency must be positive");
        dpi_backend = std::make_unique<supra::ddr::Dramsim3Backend>(
            config_path, region_manifest_path, image_path, output_directory,
            static_cast<unsigned>(core_frequency_mhz));
        return 1;
    } catch (const std::exception &error) {
        std::cerr << "DRAMSim3 backend create failed: " << error.what() << '\n';
        dpi_backend.reset();
        return 0;
    }
}

extern "C" void supra_dramsim3_destroy() { dpi_backend.reset(); }

extern "C" int supra_dramsim3_reset() {
    try {
        if (!dpi_backend) return 0;
        dpi_backend->reset();
        return 1;
    } catch (const std::exception &error) {
        std::cerr << "DRAMSim3 backend reset failed: " << error.what() << '\n';
        return 0;
    }
}

extern "C" int supra_dramsim3_can_accept(unsigned long long address, int write) {
    return dpi_backend && dpi_backend->can_accept(address, write != 0);
}

extern "C" int supra_dramsim3_submit_read(unsigned long long address,
                                           unsigned long long token) {
    return dpi_backend && dpi_backend->submit_read(address, token);
}

extern "C" int supra_dramsim3_submit_write(
    unsigned long long address, unsigned long long token,
    const unsigned int *data_words, unsigned int byte_enable) {
    if (!dpi_backend || data_words == nullptr) return 0;
    std::array<std::uint8_t, supra::ddr::kTransactionBytes> data{};
    std::memcpy(data.data(), data_words, data.size());
    return dpi_backend->submit_write(address, token, data, byte_enable);
}

extern "C" void supra_dramsim3_tick_core() {
    if (dpi_backend) dpi_backend->tick_core();
}

extern "C" int supra_dramsim3_poll_read(unsigned long long *token,
                                         unsigned int *data_words) {
    if (!dpi_backend || token == nullptr || data_words == nullptr) return 0;
    supra::ddr::Completion completion;
    if (!dpi_backend->poll_read(completion)) return 0;
    *token = completion.token;
    std::memcpy(data_words, completion.data.data(), completion.data.size());
    return 1;
}

extern "C" int supra_dramsim3_poll_write(unsigned long long *token) {
    if (!dpi_backend || token == nullptr) return 0;
    supra::ddr::Completion completion;
    if (!dpi_backend->poll_write(completion)) return 0;
    *token = completion.token;
    return 1;
}

extern "C" int supra_dramsim3_idle() {
    return dpi_backend && dpi_backend->idle();
}

extern "C" int supra_dramsim3_inspect_byte(unsigned long long address) {
    if (!dpi_backend) return -1;
    try {
        return dpi_backend->inspect_byte(address);
    } catch (const std::exception &) {
        return -1;
    }
}

extern "C" int supra_dramsim3_initialize_byte(unsigned long long address,
                                             unsigned char value) {
    if (!dpi_backend || !dpi_backend->idle()) return -1;
    try {
        dpi_backend->initialize_byte(address, value);
        return 0;
    } catch (const std::exception &) {
        return -1;
    }
}

extern "C" unsigned long long supra_dramsim3_physical_read_bytes() {
    return dpi_backend ? dpi_backend->stats().reads_submitted *
        supra::ddr::kTransactionBytes : 0;
}

extern "C" unsigned long long supra_dramsim3_physical_write_bytes() {
    return dpi_backend ? dpi_backend->stats().writes_submitted *
        supra::ddr::kTransactionBytes : 0;
}

extern "C" const char *supra_dramsim3_stats_json() {
    static std::string stats;
    stats = dpi_backend ? dpi_backend->stats_json() : "{}";
    return stats.c_str();
}
