#ifndef IN_BLOCK_REFRESH_BUDGET_PACKER_HPP
#define IN_BLOCK_REFRESH_BUDGET_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/in_block_refresh_budget.h"
}

inline std::array<std::uint8_t, IN_BLOCK_REFRESH_BUDGET_BYTES>
pack_in_block_refresh_budget(const in_block_refresh_budget &value) {
    std::array<std::uint8_t, IN_BLOCK_REFRESH_BUDGET_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[IN_BLOCK_REFRESH_BUDGET_TARGET_TOKEN_COUNT_X2_OFFSET + byte] = static_cast<std::uint8_t>(value.target_token_count_x2 >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[IN_BLOCK_REFRESH_BUDGET_SELECTION_CONTROL_OFFSET + byte] = static_cast<std::uint8_t>(value.selection_control >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[IN_BLOCK_REFRESH_BUDGET_REGION_START_OFFSET + byte] = static_cast<std::uint8_t>(value.region_start >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[IN_BLOCK_REFRESH_BUDGET_TOKEN_COUNT_X2_CREDIT_OFFSET + byte] = static_cast<std::uint8_t>(value.token_count_x2_credit >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[IN_BLOCK_REFRESH_BUDGET_REGULAR_STEPS_OFFSET + byte] = static_cast<std::uint8_t>(value.regular_steps >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[IN_BLOCK_REFRESH_BUDGET_CUMULATIVE_ACTIVE_TOKEN_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.cumulative_active_token_count >> (8u * byte));
    return bytes;
}

#endif
