#ifndef REFRESH_SCORE_CONFIG_PACKER_HPP
#define REFRESH_SCORE_CONFIG_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/refresh_score_config.h"
}

inline std::array<std::uint8_t, REFRESH_SCORE_CONFIG_BYTES>
pack_refresh_score_config(const refresh_score_config &value) {
    std::array<std::uint8_t, REFRESH_SCORE_CONFIG_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[REFRESH_SCORE_CONFIG_MAGIC_OFFSET + byte] = static_cast<std::uint8_t>(value.magic >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[REFRESH_SCORE_CONFIG_VERSION_OFFSET + byte] = static_cast<std::uint8_t>(value.version >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[REFRESH_SCORE_CONFIG_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[REFRESH_SCORE_CONFIG_TOKEN_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.token_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[REFRESH_SCORE_CONFIG_KEYS_OFFSET + byte] = static_cast<std::uint8_t>(value.keys >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[REFRESH_SCORE_CONFIG_BLOCK_END_OFFSET + byte] = static_cast<std::uint8_t>(value.block_end >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[REFRESH_SCORE_CONFIG_RELATION_ROW_SHIFT_OFFSET + byte] = static_cast<std::uint8_t>(value.relation_row_shift >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[REFRESH_SCORE_CONFIG_FLAGS_OFFSET + byte] = static_cast<std::uint8_t>(value.flags >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[REFRESH_SCORE_CONFIG_RELATION_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.relation_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[REFRESH_SCORE_CONFIG_RELATION_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.relation_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[REFRESH_SCORE_CONFIG_CHANGE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.change_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[REFRESH_SCORE_CONFIG_CHANGE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.change_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[REFRESH_SCORE_CONFIG_PENDING_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.pending_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[REFRESH_SCORE_CONFIG_PENDING_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.pending_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[REFRESH_SCORE_CONFIG_REGULAR_BUDGET_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.regular_budget_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[REFRESH_SCORE_CONFIG_REGULAR_BUDGET_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.regular_budget_limit >> (8u * byte));
    return bytes;
}

#endif
