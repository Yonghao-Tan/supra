#ifndef ATTENTION_DEPENDENCY_JOB_PACKER_HPP
#define ATTENTION_DEPENDENCY_JOB_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/attention_dependency_job.h"
}

inline std::array<std::uint8_t, ATTENTION_DEPENDENCY_JOB_BYTES>
pack_attention_dependency_job(const attention_dependency_job &value) {
    std::array<std::uint8_t, ATTENTION_DEPENDENCY_JOB_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_SOURCE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.source_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_HEAD_STRIDE_OFFSET + byte] = static_cast<std::uint8_t>(value.head_stride >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_LANE_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.lane_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_OPERATION_OFFSET + byte] = static_cast<std::uint8_t>(value.operation >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_RESERVED0_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved0 >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_OUTPUT_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.output_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_SOURCE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.source_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_OUTPUT_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.output_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_RESERVED1_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved1 >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_RESERVED2_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved2 >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[ATTENTION_DEPENDENCY_JOB_RESERVED3_OFFSET + byte] = static_cast<std::uint8_t>(value.reserved3 >> (8u * byte));
    return bytes;
}

#endif
