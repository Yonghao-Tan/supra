#ifndef UAPS_CONFIG_PACKER_HPP
#define UAPS_CONFIG_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/uaps_config.h"
}

inline std::array<std::uint8_t, UAPS_CONFIG_BYTES>
pack_uaps_config(const uaps_config &value) {
    std::array<std::uint8_t, UAPS_CONFIG_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[UAPS_CONFIG_MAGIC_OFFSET + byte] = static_cast<std::uint8_t>(value.magic >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[UAPS_CONFIG_VERSION_OFFSET + byte] = static_cast<std::uint8_t>(value.version >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[UAPS_CONFIG_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[UAPS_CONFIG_BASE_TOKEN_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.base_token_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[UAPS_CONFIG_FUTURE_TOKEN_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.future_token_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[UAPS_CONFIG_NEXT_BLOCK_START_OFFSET + byte] = static_cast<std::uint8_t>(value.next_block_start >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_CONFIG_MAX_NEXT_TOKENS_OFFSET + byte] = static_cast<std::uint8_t>(value.max_next_tokens >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_CONFIG_PRIORITY_CONTROL_OFFSET + byte] = static_cast<std::uint8_t>(value.priority_control >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[UAPS_CONFIG_BASE_TABLE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.base_table_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[UAPS_CONFIG_BASE_TABLE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.base_table_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[UAPS_CONFIG_FUTURE_TABLE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.future_table_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[UAPS_CONFIG_FUTURE_TABLE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.future_table_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[UAPS_CONFIG_RESULT_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.result_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[UAPS_CONFIG_RESULT_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.result_limit >> (8u * byte));
    return bytes;
}

#endif
