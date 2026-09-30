#ifndef UAPS_ATTEMPT_CONFIG_PACKER_HPP
#define UAPS_ATTEMPT_CONFIG_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/uaps_attempt_config.h"
}

inline std::array<std::uint8_t, UAPS_ATTEMPT_CONFIG_BYTES>
pack_uaps_attempt_config(const uaps_attempt_config &value) {
    std::array<std::uint8_t, UAPS_ATTEMPT_CONFIG_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_STATE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.state_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_STATE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.state_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_MAX_ATTEMPTS_OFFSET + byte] = static_cast<std::uint8_t>(value.max_attempts >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_RETRY_MIN_CONFIDENCE_BF16_OFFSET + byte] = static_cast<std::uint8_t>(value.retry_min_confidence_bf16 >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_PREDICTION_TARGET_OFFSET + byte] = static_cast<std::uint8_t>(value.prediction_target >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_SOURCE_B_FLAGS_OFFSET + byte] = static_cast<std::uint8_t>(value.source_b_flags >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_MIN_REUSE_SCORE_BF16_OFFSET + byte] = static_cast<std::uint8_t>(value.min_reuse_score_bf16 >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_MAX_CURRENT_UNRESOLVED_OFFSET + byte] = static_cast<std::uint8_t>(value.max_current_unresolved >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_MAX_HANDOFF_TOKENS_OFFSET + byte] = static_cast<std::uint8_t>(value.max_handoff_tokens >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_ADMISSION_BUDGET_OFFSET + byte] = static_cast<std::uint8_t>(value.admission_budget >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_STATE_CONTROL_OFFSET + byte] = static_cast<std::uint8_t>(value.state_control >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[UAPS_ATTEMPT_CONFIG_BLOCK_STEP_INDEX_OFFSET + byte] = static_cast<std::uint8_t>(value.block_step_index >> (8u * byte));
    return bytes;
}

#endif
