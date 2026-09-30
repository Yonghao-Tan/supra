#ifndef TOKEN_REFRESH_CONFIG_PACKER_HPP
#define TOKEN_REFRESH_CONFIG_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/token_refresh_config.h"
}

inline std::array<std::uint8_t, TOKEN_REFRESH_CONFIG_BYTES>
pack_token_refresh_config(const token_refresh_config &value) {
    std::array<std::uint8_t, TOKEN_REFRESH_CONFIG_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_MAGIC_OFFSET + byte] = static_cast<std::uint8_t>(value.magic >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_VERSION_OFFSET + byte] = static_cast<std::uint8_t>(value.version >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_FLAGS_OFFSET + byte] = static_cast<std::uint8_t>(value.flags >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_PENDING_CONFIGURATION_OFFSET_OFFSET + byte] = static_cast<std::uint8_t>(value.pending_configuration_offset >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_SEQUENCE_LENGTH_OFFSET + byte] = static_cast<std::uint8_t>(value.sequence_length >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_TARGET_TOKEN_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.target_token_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_REQUIRED_QUOTA_OFFSET + byte] = static_cast<std::uint8_t>(value.required_quota >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_PROBABILITY_CONFIGURATION_OFFSET_OFFSET + byte] = static_cast<std::uint8_t>(value.probability_configuration_offset >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_HEAD_STRIDE_OFFSET + byte] = static_cast<std::uint8_t>(value.head_stride >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_SCALE_HEAD_STRIDE_OFFSET + byte] = static_cast<std::uint8_t>(value.scale_head_stride >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_METADATA_VERSION_OFFSET + byte] = static_cast<std::uint8_t>(value.metadata_version >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_TABLE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.table_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_TABLE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.table_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_METADATA_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.metadata_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_METADATA_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.metadata_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_SOURCE_K_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.source_k_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_SOURCE_V_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.source_v_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_SOURCE_SCALE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.source_scale_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_DESTINATION_K_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.destination_k_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_DESTINATION_V_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.destination_v_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_DESTINATION_SCALE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.destination_scale_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_SOURCE_K_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.source_k_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_SOURCE_V_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.source_v_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_SOURCE_SCALE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.source_scale_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_DESTINATION_K_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.destination_k_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_DESTINATION_V_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.destination_v_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_DESTINATION_SCALE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.destination_scale_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_CAPTURE_INDEX_OFFSET + byte] = static_cast<std::uint8_t>(value.capture_index >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_RELATION_JOB_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.relation_job_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[TOKEN_REFRESH_CONFIG_RELATION_JOB_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.relation_job_base >> (8u * byte));
    return bytes;
}

#endif
