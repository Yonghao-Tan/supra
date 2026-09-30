#ifndef UAPS_RESULT_PACKER_HPP
#define UAPS_RESULT_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/uaps_result.h"
}

inline std::array<std::uint8_t, UAPS_RESULT_BYTES>
pack_uaps_result(const uaps_result &value) {
    std::array<std::uint8_t, UAPS_RESULT_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[UAPS_RESULT_FUTURE_PREDICTION_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.future_prediction_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[UAPS_RESULT_ADDED_FUTURE_TOKEN_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.added_future_token_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_RESULT_FUTURE_PREDICTION_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.future_prediction_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_RESULT_ADDED_FUTURE_TOKEN_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.added_future_token_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_RESULT_BASE_ACTIVATION_SLOTS_OFFSET + byte] = static_cast<std::uint8_t>(value.base_activation_slots >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_RESULT_JOINT_ACTIVATION_SLOTS_OFFSET + byte] = static_cast<std::uint8_t>(value.joint_activation_slots >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_RESULT_NEXT_PASS_ACTIVATION_SLOTS_OFFSET + byte] = static_cast<std::uint8_t>(value.next_pass_activation_slots >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_RESULT_RESERVED0_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved0 >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[UAPS_RESULT_RESERVED1_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved1 >> (8u * byte));
    return bytes;
}

#endif
