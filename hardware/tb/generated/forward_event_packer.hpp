#ifndef FORWARD_EVENT_PACKER_HPP
#define FORWARD_EVENT_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/forward_event.h"
}

inline std::array<std::uint8_t, FORWARD_EVENT_BYTES>
pack_forward_event(const forward_event &value) {
    std::array<std::uint8_t, FORWARD_EVENT_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_CAPTURE_INDEX_OFFSET + byte] = static_cast<std::uint8_t>(value.capture_index >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[FORWARD_EVENT_BLOCK_SLOT_OFFSET + byte] = static_cast<std::uint8_t>(value.block_slot >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[FORWARD_EVENT_GLOBAL_BLOCK_ID_OFFSET + byte] = static_cast<std::uint8_t>(value.global_block_id >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[FORWARD_EVENT_POSITION_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.position_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_CONFIRMED_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.confirmed_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_REMASKED_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.remasked_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_SELECTED_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.selected_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_DIRECT_LOCKED_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.direct_locked_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_STABLE_TENTATIVE_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.stable_tentative_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_FALLBACK_TENTATIVE_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.fallback_tentative_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_MANDATORY_REFRESH_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.mandatory_refresh_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_CACHE_COMMIT_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.cache_commit_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_CACHE_INVALIDATE_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.cache_invalidate_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_CACHE_KEEP_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.cache_keep_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_TOKEN_CHANGED_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.token_changed_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_TAIL_CLOSED_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.tail_closed_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_RESERVED0_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved0 >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[FORWARD_EVENT_RESERVED1_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved1 >> (8u * byte));
    return bytes;
}

#endif
