#ifndef TOKEN_STATE_ENTRY_PACKER_HPP
#define TOKEN_STATE_ENTRY_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/token_state_entry.h"
}

inline std::array<std::uint8_t, TOKEN_STATE_ENTRY_BYTES>
pack_token_state_entry(const token_state_entry &value) {
    std::array<std::uint8_t, TOKEN_STATE_ENTRY_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_STATE_ENTRY_TOKEN_POSITION_OFFSET + byte] = static_cast<std::uint8_t>(value.token_position >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_BLOCK_LOCAL_POSITION_OFFSET + byte] = static_cast<std::uint8_t>(value.block_local_position >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_BLOCK_SLOT_OFFSET + byte] = static_cast<std::uint8_t>(value.block_slot >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_GLOBAL_BLOCK_ID_OFFSET + byte] = static_cast<std::uint8_t>(value.global_block_id >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_STATE_OFFSET + byte] = static_cast<std::uint8_t>(value.state >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_ORIGIN_OFFSET + byte] = static_cast<std::uint8_t>(value.origin >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_ACTIVATION_BITS_OFFSET + byte] = static_cast<std::uint8_t>(value.activation_bits >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[TOKEN_STATE_ENTRY_TOKEN_ID_OFFSET + byte] = static_cast<std::uint8_t>(value.token_id >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[TOKEN_STATE_ENTRY_LAST_TOP1_OFFSET + byte] = static_cast<std::uint8_t>(value.last_top1 >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_STATE_ENTRY_PRECISION_AGE_OFFSET + byte] = static_cast<std::uint8_t>(static_cast<std::uint16_t>(value.precision_age) >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_CACHE_VALID_OFFSET + byte] = static_cast<std::uint8_t>(value.cache_valid >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_REFRESH_REQUIRED_OFFSET + byte] = static_cast<std::uint8_t>(value.refresh_required >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[TOKEN_STATE_ENTRY_CAPTURE_INDEX_OFFSET + byte] = static_cast<std::uint8_t>(value.capture_index >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_PREDICTION_FLAG_OFFSET + byte] = static_cast<std::uint8_t>(value.prediction_flag >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_CHANGE_FLAGS_OFFSET + byte] = static_cast<std::uint8_t>(value.change_flags >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_STATE_ENTRY_CHANGE_CONFIDENCE_BF16_OFFSET + byte] = static_cast<std::uint8_t>(value.change_confidence_bf16 >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_BF16_OFFSET + byte] = static_cast<std::uint8_t>(value.last_action_confidence_bf16 >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_VALID_OFFSET + byte] = static_cast<std::uint8_t>(value.last_action_confidence_valid >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[TOKEN_STATE_ENTRY_SOURCE_A_PENDING_OFFSET + byte] = static_cast<std::uint8_t>(value.source_a_pending >> (8u * byte));
    return bytes;
}

#endif
