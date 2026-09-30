#ifndef ATTENTION_PROBABILITY_CONFIG_PACKER_HPP
#define ATTENTION_PROBABILITY_CONFIG_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/attention_probability_config.h"
}

inline std::array<std::uint8_t, ATTENTION_PROBABILITY_CONFIG_BYTES>
pack_attention_probability_config(const attention_probability_config &value) {
    std::array<std::uint8_t, ATTENTION_PROBABILITY_CONFIG_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_OUTPUT_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.output_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_OUTPUT_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.output_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_BATCH_STRIDE_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.batch_stride_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_HEAD_STRIDE_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.head_stride_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_ROUND_STRIDE_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.round_stride_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_LAYER_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.layer_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS0_OFFSET + byte] = static_cast<std::uint8_t>(value.key_groups0 >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS1_OFFSET + byte] = static_cast<std::uint8_t>(value.key_groups1 >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS2_OFFSET + byte] = static_cast<std::uint8_t>(value.key_groups2 >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_KEY_GROUPS3_OFFSET + byte] = static_cast<std::uint8_t>(value.key_groups3 >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_QUERY_BEGIN_OFFSET + byte] = static_cast<std::uint8_t>(value.query_begin >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_QUERY_END_OFFSET + byte] = static_cast<std::uint8_t>(value.query_end >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_CAPTURE_P8_OFFSET + byte] = static_cast<std::uint8_t>(value.capture_p8 >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_PROBABILITY_CONFIG_SHORTLIST_CONFIGURATION_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.shortlist_configuration_base >> (8u * byte));
    return bytes;
}

#endif
