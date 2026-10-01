#include "Vlayer_address_loader.h"
#include "verilated.h"

#include "generated/layer_address_table_packer.hpp"
#include "generated/execution_config_packer.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

vluint64_t simulation_time = 0;
double sc_time_stamp() { return static_cast<double>(simulation_time); }

namespace {

constexpr std::uint64_t kTableBase = 0x0000000000010000ULL;
constexpr std::uint32_t kEntryStride = 880;

template <std::size_t N>
void set_u(std::array<std::uint8_t, N>& image, std::size_t offset,
           std::uint64_t value, std::size_t bytes) {
    for (std::size_t byte = 0; byte < bytes; ++byte)
        image.at(offset + byte) = static_cast<std::uint8_t>(value >> (8U * byte));
}

template <std::size_t N>
void set_region(std::array<std::uint8_t, N>& image, std::size_t base_offset,
                std::uint64_t base, std::uint64_t limit) {
    set_u(image, base_offset, base, 8);
    set_u(image, base_offset + 8, limit, 8);
}

std::array<std::uint8_t, EXECUTION_CONFIG_BYTES> make_configuration(
    std::uint16_t active_token_count = 32) {
    std::array<std::uint8_t, EXECUTION_CONFIG_BYTES> image{};
    set_u(image, EXECUTION_CONFIG_MAGIC_OFFSET, 0x344e4c44U, 4);
    set_u(image, EXECUTION_CONFIG_VERSION_OFFSET, 4, 2);
    set_u(image, EXECUTION_CONFIG_FLAGS_OFFSET, 1, 4);
    set_u(image, EXECUTION_CONFIG_LAYER_COUNT_OFFSET, 4, 2);
    set_u(image, EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET, active_token_count, 2);
    set_region(image, EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET,
               kTableBase, kTableBase + 4ULL * kEntryStride);
    set_region(image, EXECUTION_CONFIG_BF16_TEMPORARY_BASE_OFFSET,
               0x00a00000ULL, 0x00d00000ULL);
    set_region(image, EXECUTION_CONFIG_CURRENT_K_CACHE_BASE_OFFSET,
               0x00100000ULL, 0x00200000ULL);
    set_region(image, EXECUTION_CONFIG_CURRENT_V_CACHE_BASE_OFFSET,
               0x00200000ULL, 0x00300000ULL);
    set_region(image, EXECUTION_CONFIG_RETAINED_K_CACHE_BASE_OFFSET,
               0x00300000ULL, 0x00400000ULL);
    set_region(image, EXECUTION_CONFIG_RETAINED_V_CACHE_BASE_OFFSET,
               0x00400000ULL, 0x00500000ULL);
    set_region(image, EXECUTION_CONFIG_K_SCALE_BASE_OFFSET,
               0x00500000ULL, 0x00600000ULL);
    set_region(image, EXECUTION_CONFIG_V_SCALE_BASE_OFFSET,
               0x00600000ULL, 0x00700000ULL);
    set_region(image, EXECUTION_CONFIG_ATTENTION_WORKSPACE_BASE_OFFSET,
               0x00700000ULL, 0x00a00000ULL);
    set_u(image, EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET, kEntryStride, 4);
    return image;
}

std::array<std::uint8_t, LAYER_ADDRESS_TABLE_BYTES> make_entry() {
    std::array<std::uint8_t, LAYER_ADDRESS_TABLE_BYTES> image{};
    set_u(image, LAYER_ADDRESS_TABLE_MAGIC_OFFSET, 0x3141544cU, 4);
    set_u(image, LAYER_ADDRESS_TABLE_VERSION_OFFSET, 2, 2);
    set_u(image, LAYER_ADDRESS_TABLE_HEADER_BYTES_OFFSET, 64, 2);
    set_u(image, LAYER_ADDRESS_TABLE_ENTRY_BYTES_OFFSET, LAYER_ADDRESS_TABLE_BYTES, 4);
    set_u(image, LAYER_ADDRESS_TABLE_HIDDEN_SIZE_OFFSET, 4096, 4);
    set_u(image, LAYER_ADDRESS_TABLE_FFN_SIZE_OFFSET, 12288, 4);
    set_u(image, LAYER_ADDRESS_TABLE_HEAD_COUNT_OFFSET, 32, 2);
    set_u(image, LAYER_ADDRESS_TABLE_HEAD_DIMENSION_OFFSET, 128, 2);
    set_u(image, LAYER_ADDRESS_TABLE_MAXIMUM_SEQUENCE_LENGTH_OFFSET, 2048, 2);
    set_u(image, LAYER_ADDRESS_TABLE_KV_HEAD_STRIDE_BYTES_OFFSET, 262144, 4);
    set_u(image, LAYER_ADDRESS_TABLE_KV_TOKEN_STRIDE_BYTES_OFFSET, 128, 4);
    set_u(image, LAYER_ADDRESS_TABLE_KV_CHUNK_STRIDE_BYTES_OFFSET, 8, 4);
    set_u(image, LAYER_ADDRESS_TABLE_K_SCALE_HEAD_STRIDE_BYTES_OFFSET, 4096, 4);
    set_u(image, LAYER_ADDRESS_TABLE_K_SCALE_TOKEN_STRIDE_BYTES_OFFSET, 2, 4);
    set_u(image, LAYER_ADDRESS_TABLE_CONTEXT_ROW_STRIDE_BYTES_OFFSET, 8192, 4);

    const std::array<std::size_t, 34> base_offsets = {
        LAYER_ADDRESS_TABLE_QUERY_BASE_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_QUERY_ENHANCEMENT_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_QUERY_WEIGHT_SCALE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_KEY_BASE_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_KEY_ENHANCEMENT_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_KEY_WEIGHT_SCALE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_VALUE_BASE_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_VALUE_ENHANCEMENT_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_VALUE_WEIGHT_SCALE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_BASE_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_ENHANCEMENT_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_WEIGHT_SCALE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_GATE_BASE_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_GATE_ENHANCEMENT_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_GATE_WEIGHT_SCALE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_UP_BASE_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_UP_ENHANCEMENT_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_UP_WEIGHT_SCALE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_DOWN_BASE_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_DOWN_ENHANCEMENT_WEIGHT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_DOWN_WEIGHT_SCALE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_ATTENTION_RMS_GAMMA_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_RMS_GAMMA_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_ROPE_COS_LUT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_ROPE_SIN_LUT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_CURRENT_K_CACHE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_CURRENT_V_CACHE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_K_SCALE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_V_SCALE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_CONTEXT_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET,
    };

    std::uint64_t model_address = 0x00d00000ULL;
    for (std::size_t region = 0; region < base_offsets.size(); ++region) {
        std::uint64_t base = model_address;
        std::uint64_t limit = base + 0x100;
        if (region == 25) { base = 0x00100100; limit = 0x00100200; }
        else if (region == 26) { base = 0x00200100; limit = 0x00200200; }
        else if (region == 27) { base = 0x00300100; limit = 0x00300200; }
        else if (region == 28) { base = 0x00400100; limit = 0x00400200; }
        else if (region == 29) { base = 0x00500100; limit = 0x00500200; }
        else if (region == 30) { base = 0x00600100; limit = 0x00600200; }
        else if (region == 31) { base = 0x00700000; limit = 0x00740000; }
        else if (region == 32) { base = 0x00a00000; limit = 0x00c40000; }
        else if (region == 33) { base = 0x00c40000; limit = 0x00ca0000; }
        set_region(image, base_offsets[region], base, limit);
        model_address += 0x200;
    }
    return image;
}

std::array<std::uint8_t, EXECUTION_CONFIG_BYTES> make_32_layer_configuration() {
    auto image = make_configuration();
    set_u(image, EXECUTION_CONFIG_MAGIC_OFFSET, 0x344e4c44U, 4);
    set_u(image, EXECUTION_CONFIG_VERSION_OFFSET, 4, 2);
    set_u(image, EXECUTION_CONFIG_LAYER_COUNT_OFFSET, 32, 2);
    set_u(image, EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET, 48, 2);
    set_region(image, EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET,
               kTableBase, kTableBase + 32ULL * kEntryStride);
    set_region(image, EXECUTION_CONFIG_V_SCALE_BASE_OFFSET,
               0x00600000ULL, 0x00600800ULL);
    set_region(image, EXECUTION_CONFIG_BF16_TEMPORARY_BASE_OFFSET,
               0x00a00000ULL, 0x00d00000ULL);
    return image;
}

std::array<std::uint8_t, LAYER_ADDRESS_TABLE_BYTES> make_48_token_entry() {
    auto image = make_entry();
    set_region(image, LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET,
               0x00a00000ULL, 0x00c40000ULL);
    set_region(image, LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET,
               0x00c40000ULL, 0x00ca0000ULL);
    return image;
}

class TestCase {
  public:
    TestCase() { reset(); }
    ~TestCase() {
        dut_.final();
        simulation_time = 0;
    }

    void reset() {
        dut_.clk = 0;
        dut_.rst = 1;
        dut_.done_ready = 1;
        dut_.dma_request_ready = 0;
        set_configuration(make_configuration());
        for (int cycle = 0; cycle < 3; ++cycle) tick();
        dut_.rst = 0;
        tick();
    }

    void set_configuration(const std::array<std::uint8_t, EXECUTION_CONFIG_BYTES>& image) {
        for (std::size_t word = 0; word < EXECUTION_CONFIG_BYTES / 4; ++word) {
            dut_.execution_configuration_bits[word] =
                static_cast<std::uint32_t>(image[word * 4]) |
                (static_cast<std::uint32_t>(image[word * 4 + 1]) << 8U) |
                (static_cast<std::uint32_t>(image[word * 4 + 2]) << 16U) |
                (static_cast<std::uint32_t>(image[word * 4 + 3]) << 24U);
        }
    }

    void tick() {
        dut_.clk = 0; dut_.eval();
        ++simulation_time;
        dut_.clk = 1; dut_.eval();
        ++simulation_time;
        ++cycles_;
        if (cycles_ > 200000) throw std::runtime_error("no-progress watchdog");
    }

    void start(std::uint8_t layer) {
        dut_.load_layer_index = layer;
        dut_.load_valid = 1;
        dut_.clk = 0;
        dut_.eval();
        while (!dut_.load_ready) tick();
        tick();
        dut_.load_valid = 0;
    }

    std::uint64_t accept_request() {
        for (int cycle = 0; cycle < 100; ++cycle) {
            if (dut_.dma_request_valid) {
                if (dut_.dma_request_bytes != LAYER_ADDRESS_TABLE_BYTES)
                    throw std::runtime_error("wrong request byte count");
                const auto address = dut_.dma_request_address;
                dut_.dma_request_ready = 1;
                tick();
                dut_.dma_request_ready = 0;
                ++request_count_;
                return address;
            }
            tick();
        }
        throw std::runtime_error("DMA request not issued");
    }

    void send_entry(const std::array<std::uint8_t, LAYER_ADDRESS_TABLE_BYTES>& image,
                    int bad_byte_enable_beat = -1, bool request_error = false,
                    int abort_after_beat = -1) {
        for (int beat = 0; beat < 39; ++beat) {
            for (int word = 0; word < 4; ++word) {
                const std::size_t offset = beat * 16 + word * 4;
                dut_.dma_response_data[word] =
                    static_cast<std::uint32_t>(image[offset]) |
                    (static_cast<std::uint32_t>(image[offset + 1]) << 8U) |
                    (static_cast<std::uint32_t>(image[offset + 2]) << 16U) |
                    (static_cast<std::uint32_t>(image[offset + 3]) << 24U);
            }
            dut_.dma_response_valid = 1;
            dut_.dma_response_byte_enable = beat == bad_byte_enable_beat ? 0x7fff : 0xffff;
            dut_.dma_response_last = beat == 38;
            dut_.dma_request_done = beat == 38;
            dut_.dma_request_error = beat == 38 && request_error;
            dut_.clk = 0;
            dut_.eval();
            while (!dut_.dma_response_ready) tick();
            tick();
            if (beat == abort_after_beat) dut_.abort_request = 1;
        }
        dut_.dma_response_valid = 0;
        dut_.dma_response_last = 0;
        dut_.dma_request_done = 0;
        dut_.dma_request_error = 0;
    }

    std::uint16_t wait_done(bool expect_error) {
        for (int cycle = 0; cycle < 100; ++cycle) {
            if (dut_.done_valid) {
                const bool error = dut_.done_error;
                const auto quantized = dut_.done_error_id;
                tick();
                if (error != expect_error)
                    throw std::runtime_error("completion error flag mismatch");
                return quantized;
            }
            tick();
        }
        throw std::runtime_error("completion not observed");
    }

    void wait_abort_ack() {
        for (int cycle = 0; cycle < 100; ++cycle) {
            if (dut_.abort_ack) return;
            tick();
        }
        throw std::runtime_error("abort ack not observed");
    }

    Vlayer_address_loader dut_;
    int request_count() const { return request_count_; }

  private:
    std::uint64_t cycles_ = 0;
    int request_count_ = 0;
};

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <std::size_t N>
std::uint64_t packed_u64(const VlWide<N>& words, unsigned lsb) {
    if ((lsb % 32U) != 0U)
        throw std::runtime_error("unaligned packed field in descriptor test");
    const unsigned word = lsb / 32U;
    return static_cast<std::uint64_t>(words[word]) |
           (static_cast<std::uint64_t>(words[word + 1]) << 32U);
}

void expect_descriptor_outputs(const TestCase& test_case) {
    for (unsigned field = 0; field < 21; ++field) {
        const auto actual = packed_u64(
            test_case.dut_.matmul_weight_config, (20U - field) * 64U);
        const auto expected = 0x00d00000ULL + field * 0x200ULL;
        expect(actual == expected,
               "Matmul weight descriptor field mismatch at index " +
                   std::to_string(field));
    }

    expect(packed_u64(test_case.dut_.rmsnorm_config, 64) == 0x00d02a00ULL,
           "Attention RMS gamma descriptor mismatch");
    expect(packed_u64(test_case.dut_.rmsnorm_config, 0) == 0x00d02c00ULL,
           "FFN RMS gamma descriptor mismatch");
    expect(test_case.dut_.rmsnorm_config[4] == 0,
           "RMS epsilon descriptor mismatch");

    expect(packed_u64(test_case.dut_.qkv_config, 128) == 0x00d02e00ULL,
           "RoPE cosine descriptor mismatch");
    expect(packed_u64(test_case.dut_.qkv_config, 64) == 0x00d03000ULL,
           "RoPE sine descriptor mismatch");
    expect(packed_u64(test_case.dut_.qkv_config, 0) == 0x00600100ULL,
           "V scale descriptor mismatch");

    expect(packed_u64(test_case.dut_.attention_cache_config, 352) ==
               0x00100100ULL,
           "current K cache descriptor mismatch");
    expect(packed_u64(test_case.dut_.attention_cache_config, 288) ==
               0x00200100ULL,
           "current V cache descriptor mismatch");
    expect(packed_u64(test_case.dut_.attention_cache_config, 224) ==
               0x00300100ULL,
           "retained K cache descriptor mismatch");
    expect(packed_u64(test_case.dut_.attention_cache_config, 160) ==
               0x00400100ULL,
           "retained V cache descriptor mismatch");
    expect(packed_u64(test_case.dut_.attention_cache_config, 96) ==
               0x00500100ULL,
           "K scale descriptor mismatch");
    expect(test_case.dut_.attention_cache_config[2] == 262144U &&
               test_case.dut_.attention_cache_config[1] == 128U &&
               test_case.dut_.attention_cache_config[0] == 4096U,
           "cache stride descriptor mismatch");

    expect(packed_u64(test_case.dut_.attention_context_config, 32) ==
               0x00700000ULL &&
               test_case.dut_.attention_context_config[0] == 8192U,
           "Attention context descriptor mismatch");
    expect(packed_u64(test_case.dut_.ffn_workspace_config, 192) ==
               0x00a00000ULL &&
               packed_u64(test_case.dut_.ffn_workspace_config, 128) ==
                   0x00c40000ULL &&
               packed_u64(test_case.dut_.ffn_workspace_config, 64) ==
                   0x00c40000ULL &&
               packed_u64(test_case.dut_.ffn_workspace_config, 0) ==
                   0x00ca0000ULL,
           "FFN workspace descriptor mismatch");
}

void test_normal_and_reuse() {
    TestCase test_case;
    const auto entry = make_entry();
    test_case.start(2);
    expect(test_case.accept_request() == kTableBase + 2ULL * kEntryStride,
           "layer address calculation mismatch");
    test_case.send_entry(entry);
    expect(test_case.wait_done(false) == 0, "normal completion quantized mismatch");
    expect(test_case.dut_.entry_valid && test_case.dut_.entry_layer_index == 2,
           "validated layer entry not retained");
    expect_descriptor_outputs(test_case);
    test_case.start(2);
    expect(test_case.wait_done(false) == 0, "same-layer reuse failed");
    expect(test_case.request_count() == 1, "same layer issued a second DMA request");
    test_case.start(3);
    expect(test_case.accept_request() == kTableBase + 3ULL * kEntryStride,
           "next layer address calculation mismatch");
}

void test_w4_empty_enhancement_planes() {
    TestCase test_case;
    auto entry = make_entry();
    for (unsigned linear = 0; linear < 7; ++linear)
        set_region(entry, 80 + linear * 48, 0, 0);
    test_case.start(0);
    test_case.accept_request();
    test_case.send_entry(entry);
    expect(test_case.wait_done(false) == 0,
           "W4 weights must not require an enhancement allocation");
}

void test_per_layer_v_scale_addresses() {
    TestCase test_case;
    test_case.set_configuration(make_32_layer_configuration());
    for (unsigned layer = 0; layer < 32; ++layer) {
        auto entry = make_48_token_entry();
        const std::uint64_t scale_base = 0x00600000ULL + layer * 64ULL;
        set_region(entry, LAYER_ADDRESS_TABLE_V_SCALE_BASE_OFFSET,
                   scale_base, scale_base + 64ULL);
        test_case.start(layer);
        expect(test_case.accept_request() ==
                   kTableBase + std::uint64_t(layer) * kEntryStride,
               "layer table address mismatch at layer " +
                   std::to_string(layer));
        test_case.send_entry(entry);
        expect(test_case.wait_done(false) == 0,
               "layer entry failed at layer " + std::to_string(layer));
        expect(packed_u64(test_case.dut_.qkv_config, 0) == scale_base,
               "V scale base mismatch at layer " +
                   std::to_string(layer));
        expect((test_case.dut_.qkv_config[6] & 1U) != 0,
               "per-head V scale flag missing at layer " +
                   std::to_string(layer));
    }
}

void test_region_errors() {
    const auto valid = make_entry();
    const std::array<std::size_t, 34> offsets = {
        64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,
        336,352,368,384,400,416,440,456,472,488,504,520,536,552,568,584,600};
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        TestCase test_case;
        auto entry = valid;
        set_u(entry, offsets[index], 0x123, 8);
        test_case.start(0);
        test_case.accept_request();
        test_case.send_entry(entry);
        expect(test_case.wait_done(true) == 5,
               "misaligned region was not rejected at index " + std::to_string(index));
    }
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        TestCase test_case;
        auto entry = valid;
        set_region(entry, offsets[index], 0x00d01000, 0x00d00000);
        test_case.start(0);
        test_case.accept_request();
        test_case.send_entry(entry);
        expect(test_case.wait_done(true) == 5,
               "reversed region was not rejected at index " + std::to_string(index));
    }
}

void test_retained_k_scale_region() {
    constexpr std::uint64_t base = 0x01000000;
    const std::array<std::uint64_t, 6> table_bases = {
        base, base, base + 1, base - 16, UINT64_C(0xfffffffffffffff0), base};
    const std::array<std::uint64_t, 6> region_limits = {
        base + 131072, base + 131056, base + 131088,
        base + 131072, UINT64_C(0xfffffffffffffff0), 0};
    for (unsigned test = 0; test < table_bases.size(); ++test) {
        TestCase test_case;
        auto config = make_32_layer_configuration();
        set_region(config, EXECUTION_CONFIG_RETAINED_K_SCALE_BASE_OFFSET,
                   test == 5 ? 0 : base, region_limits[test]);
        test_case.set_configuration(config);
        auto entry = make_48_token_entry();
        set_u(entry, LAYER_ADDRESS_TABLE_RETAINED_K_SCALE_BASE_OFFSET,
              table_bases[test], 8);
        test_case.start(0);
        test_case.accept_request();
        test_case.send_entry(entry);
        expect(test_case.wait_done(test != 0) == (test ? 6 : 0),
               "retained K scale containment mismatch in case " + std::to_string(test));
        if (test == 0)
            expect(packed_u64(test_case.dut_.attention_cache_config, 416) == base,
                   "retained K scale descriptor lost its separate base");
    }
}

void test_cache_alias_ranges() {
    for (unsigned test = 0; test < 4; ++test) {
        TestCase test_case;
        auto config = make_32_layer_configuration();
        set_region(config, EXECUTION_CONFIG_RETAINED_K_CACHE_BASE_OFFSET, 0x00100000, 0x00200000);
        set_region(config, EXECUTION_CONFIG_RETAINED_V_CACHE_BASE_OFFSET, 0x00200000, 0x00300000);
        test_case.set_configuration(config);
        auto entry = make_48_token_entry();
        const uint64_t k_offset = test == 1 ? 0x80 : test == 3 ? 0x100 : 0;
        const uint64_t v_offset = test == 2 ? 0x80 : test == 3 ? 0x100 : 0;
        set_region(entry, LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_BASE_OFFSET,
                   0x00100100 + k_offset, 0x00100200 + k_offset);
        set_region(entry, LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_BASE_OFFSET,
                   0x00200100 + v_offset, 0x00200200 + v_offset);
        test_case.start(0);
        test_case.accept_request();
        test_case.send_entry(entry);
        const bool error = test == 1 || test == 2;
        expect(test_case.wait_done(error) == (error ? 6 : 0),
               "per-layer cache partial overlap was not distinguished from exact alias/disjoint ranges");
    }
}

void test_header_stream_dma_and_address_errors() {
    for (unsigned version = 1; version <= 3; ++version) {
        TestCase test_case;
        auto configuration = make_configuration();
        set_u(configuration, EXECUTION_CONFIG_MAGIC_OFFSET,
              0x304e4c44u + (version << 24), 4);
        set_u(configuration, EXECUTION_CONFIG_VERSION_OFFSET, version, 2);
        test_case.set_configuration(configuration);
        test_case.start(0);
        expect(test_case.wait_done(true) == 4, "unsupported execution version was accepted");
        expect(test_case.request_count() == 0, "unsupported execution header issued a DMA request");
    }
    const std::array<std::pair<std::size_t, std::uint64_t>, 5> header_errors = {{
        {LAYER_ADDRESS_TABLE_MAGIC_OFFSET, 0},
        {LAYER_ADDRESS_TABLE_VERSION_OFFSET, 1},
        {LAYER_ADDRESS_TABLE_HEADER_BYTES_OFFSET, 48},
        {LAYER_ADDRESS_TABLE_ENTRY_BYTES_OFFSET, 880},
        {LAYER_ADDRESS_TABLE_FLAGS_OFFSET, 2},
    }};
    const std::array<std::size_t, 5> header_widths = {4,2,2,4,4};
    for (std::size_t index = 0; index < header_errors.size(); ++index) {
        TestCase test_case;
        auto entry = make_entry();
        set_u(entry, header_errors[index].first, header_errors[index].second,
              header_widths[index]);
        test_case.start(0); test_case.accept_request(); test_case.send_entry(entry);
        expect(test_case.wait_done(true) == 4,
               "bad header field was not rejected at index " + std::to_string(index));
    }
    const std::array<std::pair<std::size_t, std::uint64_t>, 6> shape_errors = {{
        {LAYER_ADDRESS_TABLE_HIDDEN_SIZE_OFFSET, 2048},
        {LAYER_ADDRESS_TABLE_FFN_SIZE_OFFSET, 4096},
        {LAYER_ADDRESS_TABLE_HEAD_COUNT_OFFSET, 16},
        {LAYER_ADDRESS_TABLE_HEAD_DIMENSION_OFFSET, 64},
        {LAYER_ADDRESS_TABLE_MAXIMUM_SEQUENCE_LENGTH_OFFSET, 1024},
        {LAYER_ADDRESS_TABLE_KV_TOKEN_STRIDE_BYTES_OFFSET, 64},
    }};
    const std::array<std::size_t, 6> shape_widths = {4, 4, 2, 2, 2, 4};
    for (std::size_t index = 0; index < shape_errors.size(); ++index) {
        TestCase test_case;
        auto entry = make_entry();
        set_u(entry, shape_errors[index].first, shape_errors[index].second,
              shape_widths[index]);
        test_case.start(0); test_case.accept_request(); test_case.send_entry(entry);
        expect(test_case.wait_done(true) == 4,
               "bad geometry field was not rejected at index " + std::to_string(index));
    }
    {
        TestCase test_case;
        test_case.start(0); test_case.accept_request();
        test_case.send_entry(make_entry(), 12);
        expect(test_case.wait_done(true) == 3, "bad byte enable was not rejected");
    }
    {
        TestCase test_case;
        test_case.start(0); test_case.accept_request();
        test_case.send_entry(make_entry(), -1, true);
        expect(test_case.wait_done(true) == 2, "terminal DMA error was not reported");
    }
    {
        TestCase test_case;
        auto configuration = make_configuration();
        set_u(configuration, EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_LIMIT_OFFSET,
              kTableBase + LAYER_ADDRESS_TABLE_BYTES, 8);
        test_case.set_configuration(configuration);
        test_case.start(1);
        expect(test_case.wait_done(true) == 1, "table containment error was not rejected");
        expect(test_case.request_count() == 0, "invalid table issued DMA request");
    }
    const std::array<std::uint32_t, 2> invalid_strides = {864, 881};
    for (const auto stride : invalid_strides) {
        TestCase test_case;
        auto configuration = make_configuration();
        set_u(configuration, EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET,
              stride, 4);
        test_case.set_configuration(configuration);
        test_case.start(0);
        expect(test_case.wait_done(true) == 1, "invalid entry stride was not rejected");
        expect(test_case.request_count() == 0, "invalid stride issued DMA request");
    }
    {
        TestCase test_case;
        test_case.start(4);
        expect(test_case.wait_done(true) == 1, "out-of-range layer was not rejected");
        expect(test_case.request_count() == 0, "out-of-range layer issued DMA request");
    }
    {
        TestCase test_case;
        auto configuration = make_configuration();
        set_region(configuration, EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET,
                   0xffffffffffffff00ULL, 0xfffffffffffffff0ULL);
        test_case.set_configuration(configuration);
        test_case.start(0);
        expect(test_case.wait_done(true) == 1, "65-bit address overflow was not rejected");
        expect(test_case.request_count() == 0, "overflowing address issued DMA request");
    }
    {
        const std::array<std::size_t, 9> contained_offsets = {
            LAYER_ADDRESS_TABLE_CURRENT_K_CACHE_BASE_OFFSET,
            LAYER_ADDRESS_TABLE_CURRENT_V_CACHE_BASE_OFFSET,
            LAYER_ADDRESS_TABLE_RETAINED_K_CACHE_BASE_OFFSET,
            LAYER_ADDRESS_TABLE_RETAINED_V_CACHE_BASE_OFFSET,
            LAYER_ADDRESS_TABLE_K_SCALE_BASE_OFFSET,
            LAYER_ADDRESS_TABLE_V_SCALE_BASE_OFFSET,
            LAYER_ADDRESS_TABLE_CONTEXT_BASE_OFFSET,
            LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET,
            LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET,
        };
        for (std::size_t index = 0; index < contained_offsets.size(); ++index) {
            TestCase test_case;
            auto entry = make_entry();
            set_region(entry, contained_offsets[index], 0x00010000, 0x00010100);
            test_case.start(0); test_case.accept_request(); test_case.send_entry(entry);
            expect(test_case.wait_done(true) == 6,
                   "global containment error was not rejected at class " +
                   std::to_string(index));
        }
    }
}

void test_workspace_errors() {
    {
        TestCase test_case;
        auto configuration = make_configuration();
        set_u(configuration, EXECUTION_CONFIG_FLAGS_OFFSET, 0, 4);
        test_case.set_configuration(configuration);
        test_case.start(0); test_case.accept_request(); test_case.send_entry(make_entry());
        expect(test_case.wait_done(true) == 4,
               "missing allow_bf16_temporary_spill flag was not rejected");
    }
    {
        TestCase test_case;
        auto entry = make_entry();
        set_region(entry, LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET,
                   0x00a00000, 0x00c3fff0);
        test_case.start(0); test_case.accept_request(); test_case.send_entry(entry);
        expect(test_case.wait_done(true) == 6,
               "undersized Gate/Up workspace was not rejected");
    }
    {
        TestCase test_case;
        auto entry = make_entry();
        set_region(entry, LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET,
                   0x00c40000, 0x00c9fff0);
        test_case.start(0); test_case.accept_request(); test_case.send_entry(entry);
        expect(test_case.wait_done(true) == 6,
               "undersized residual workspace was not rejected");
    }
    {
        TestCase test_case;
        auto entry = make_entry();
        set_region(entry, LAYER_ADDRESS_TABLE_FFN_RESIDUAL_WORKSPACE_BASE_OFFSET,
                   0x00c30000, 0x00c90000);
        test_case.start(0); test_case.accept_request(); test_case.send_entry(entry);
        expect(test_case.wait_done(true) == 6,
               "overlapping FFN workspaces were not rejected");
    }
}

void test_resident_workspace() {
    for (const unsigned total_token_count : {31U, 80U}) {
        for (const bool short_workspace : {false, true}) {
            TestCase test_case;
            test_case.set_configuration(make_configuration(total_token_count));
            auto entry = make_entry();
            if (short_workspace)
                set_region(entry, LAYER_ADDRESS_TABLE_FFN_GATE_UP_WORKSPACE_BASE_OFFSET,
                           0x00a00000, 0x00c3fff0);
            test_case.start(0);
            test_case.accept_request();
            test_case.send_entry(entry);
            expect(test_case.wait_done(short_workspace) == (short_workspace ? 6 : 0),
                   "workspace capacity must cover 48 resident rows independently of total rows");
        }
    }
}

void test_abort_drain() {
    {
        TestCase test_case;
        test_case.start(0);
        for (int cycle = 0; cycle < 10 && !test_case.dut_.dma_request_valid; ++cycle)
            test_case.tick();
        expect(test_case.dut_.dma_request_valid, "request did not reach stall point");
        test_case.dut_.abort_request = 1;
        test_case.tick();
        test_case.wait_abort_ack();
        expect(!test_case.dut_.dma_request_valid, "unaccepted request survived abort");
    }
    {
        TestCase test_case;
        test_case.start(0); test_case.accept_request();
        test_case.send_entry(make_entry(), -1, false, 7);
        test_case.wait_abort_ack();
        expect(!test_case.dut_.done_valid && !test_case.dut_.entry_valid,
               "aborted request committed an entry or completion");
    }
}

void test_clipping_configuration() {
    const std::array<std::size_t, 7> offsets = {
        LAYER_ADDRESS_TABLE_QUERY_CLIP_RATIO_BF16_OFFSET,
        LAYER_ADDRESS_TABLE_KEY_CLIP_RATIO_BF16_OFFSET,
        LAYER_ADDRESS_TABLE_VALUE_CLIP_RATIO_BF16_OFFSET,
        LAYER_ADDRESS_TABLE_ATTENTION_OUTPUT_CLIP_RATIO_BF16_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_GATE_CLIP_RATIO_BF16_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_UP_CLIP_RATIO_BF16_OFFSET,
        LAYER_ADDRESS_TABLE_FFN_DOWN_CLIP_RATIO_BF16_OFFSET};
    const std::array<std::uint16_t, 7> ratios{0x3f4d, 0x3f4d, 0x3f4d, 0x3f1a, 0x3f80, 0x3f80, 1};
    auto entry = make_entry();
    set_u(entry, LAYER_ADDRESS_TABLE_FLAGS_OFFSET, 1, 4);
    for (std::size_t i = 0; i < offsets.size(); ++i) set_u(entry, offsets[i], ratios[i], 2);
    {
        TestCase test_case;
        test_case.start(0); test_case.accept_request(); test_case.send_entry(entry);
        expect(test_case.wait_done(false) == 0, "valid clipping ratios were rejected");
        for (unsigned i = 0; i < ratios.size(); ++i) {
            const auto bit = 21*64 + (6-i)*16;
            const auto value = (test_case.dut_.matmul_weight_config[bit/32] >> (bit%32)) & 65535;
            expect(value == ratios[i], "clipping ratio descriptor order differs");
        }
        // Reload a deep-layer override and then normal/output ratios in the
        // same instance; a successful prior entry must not retain clip fields.
        const std::array<std::array<std::uint16_t, 7>, 2> stages{{
            {{0x3f60, 0x3f60, 0x3f60, 0x3f60, 0x3f60, 0x3f60, 0x3f60}},
            {{0x3f4d, 0x3f4d, 0x3f4d, 0x3f1a, 0x3f4d, 0x3f4d, 0x3f1a}}}};
        for (unsigned stage = 0; stage < stages.size(); ++stage) {
            auto changed = entry;
            for (unsigned i = 0; i < offsets.size(); ++i)
                set_u(changed, offsets[i], stages[stage][i], 2);
            test_case.start(stage == 0 ? 1 : 0);
            test_case.accept_request(); test_case.send_entry(changed);
            expect(test_case.wait_done(false) == 0, "clipping phase reload failed");
            for (unsigned i = 0; i < offsets.size(); ++i) {
                const auto bit = 21*64 + (6-i)*16;
                const auto value = (test_case.dut_.matmul_weight_config[bit/32] >> (bit%32)) & 65535;
                expect(value == stages[stage][i], "clipping ratio leaked from previous layer");
            }
        }
    }
    for (unsigned failure = 0; failure < 4; ++failure) {
        TestCase test_case;
        auto invalid = entry;
        if (failure == 0) set_u(invalid, LAYER_ADDRESS_TABLE_FLAGS_OFFSET, 0, 4);
        if (failure == 1) set_u(invalid, offsets[3], 0x3f81, 2);
        if (failure == 2) set_u(invalid, offsets[1], 0x3f4c, 2);
        if (failure == 3) set_u(invalid, offsets[5], 0, 2);
        test_case.start(0); test_case.accept_request(); test_case.send_entry(invalid);
        expect(test_case.wait_done(true) == 4, "invalid clipping header was accepted");
        test_case.start(0); test_case.accept_request(); test_case.send_entry(entry);
        expect(test_case.wait_done(false) == 0, "clipping header failure prevented restart");
    }
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        test_normal_and_reuse();
        test_w4_empty_enhancement_planes();
        test_per_layer_v_scale_addresses();
        test_retained_k_scale_region();
        test_cache_alias_ranges();
        test_region_errors();
        test_header_stream_dma_and_address_errors();
        test_workspace_errors();
        test_resident_workspace();
        test_abort_drain();
        test_clipping_configuration();
        std::cout << "PASS layer_address_loader regression: "
                  << "normal/reuse/layer-change, 34 alignment and 34 range errors, "
                  << "per-head V scale addresses for 32 layers, "
                  << "separate retained K scale/containment/overflow, "
                  << "per-layer K/V exact alias/disjoint/partial overlap, "
                  << "header/geometry/stride, stream, DMA, nine containment classes, "
                  << "workspace flag/capacity/overlap, table/global containment, "
                  << "fixed 48-row workspace capacity for total rows 31/80, "
                  << "clipping ratio/order/range/shared-input constraints/deep-normal-reload/restart, "
                  << "abort-before-accept and abort-drain\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL layer_address_loader regression: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
