#ifndef CROSS_BLOCK_SHORTLIST_CONFIG_PACKER_HPP
#define CROSS_BLOCK_SHORTLIST_CONFIG_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/cross_block_shortlist_config.h"
}

inline std::array<std::uint8_t, CROSS_BLOCK_SHORTLIST_CONFIG_BYTES>
pack_cross_block_shortlist_config(const cross_block_shortlist_config &value) {
    std::array<std::uint8_t, CROSS_BLOCK_SHORTLIST_CONFIG_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[CROSS_BLOCK_SHORTLIST_CONFIG_PENDING_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.pending_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[CROSS_BLOCK_SHORTLIST_CONFIG_PENDING_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.pending_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[CROSS_BLOCK_SHORTLIST_CONFIG_SHORTLIST_TOKEN_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.shortlist_token_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[CROSS_BLOCK_SHORTLIST_CONFIG_FLAGS_OFFSET + byte] = static_cast<std::uint8_t>(value.flags >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[CROSS_BLOCK_SHORTLIST_CONFIG_RELATIVE_SCORE_FLOOR_BF16_OFFSET + byte] = static_cast<std::uint8_t>(value.relative_score_floor_bf16 >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[CROSS_BLOCK_SHORTLIST_CONFIG_PROTECTED_BEGIN_OFFSET + byte] = static_cast<std::uint8_t>(value.protected_begin >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[CROSS_BLOCK_SHORTLIST_CONFIG_PROTECTED_END_OFFSET + byte] = static_cast<std::uint8_t>(value.protected_end >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[CROSS_BLOCK_SHORTLIST_CONFIG_RESERVED0_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved0 >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[CROSS_BLOCK_SHORTLIST_CONFIG_RESERVED1_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved1 >> (8u * byte));
    return bytes;
}

#endif
