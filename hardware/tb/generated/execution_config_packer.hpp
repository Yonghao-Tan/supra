#ifndef EXECUTION_CONFIG_PACKER_HPP
#define EXECUTION_CONFIG_PACKER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
extern "C" {
#include "../../cmodel/generated/execution_config.h"
}

inline std::array<std::uint8_t, EXECUTION_CONFIG_BYTES>
pack_execution_config(const execution_config &value) {
    std::array<std::uint8_t, EXECUTION_CONFIG_BYTES> bytes{};
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[EXECUTION_CONFIG_MAGIC_OFFSET + byte] = static_cast<std::uint8_t>(value.magic >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_VERSION_OFFSET + byte] = static_cast<std::uint8_t>(value.version >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_HEADER_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.header_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[EXECUTION_CONFIG_TOTAL_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.total_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[EXECUTION_CONFIG_FLAGS_OFFSET + byte] = static_cast<std::uint8_t>(value.flags >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_LAYER_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.layer_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.total_token_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET + byte] = static_cast<std::uint8_t>(value.sequence_length >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_TOKEN_METADATA_FORMAT_OFFSET + byte] = static_cast<std::uint8_t>(value.token_metadata_format >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_START_LAYER_OFFSET + byte] = static_cast<std::uint8_t>(value.start_layer >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_REFRESH_CONFIGURATION_OFFSET_OFFSET + byte] = static_cast<std::uint8_t>(value.refresh_configuration_offset >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[EXECUTION_CONFIG_ROTATION_ENABLE_MASK_OFFSET + byte] = static_cast<std::uint8_t>(value.rotation_enable_mask >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[EXECUTION_CONFIG_FFN_PAIR_FIRST_BATCH_PLUS1_OFFSET + byte] = static_cast<std::uint8_t>(value.ffn_pair_first_batch_plus1 >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_JOINT_CONFIGURATION_OFFSET_OFFSET + byte] = static_cast<std::uint8_t>(value.joint_configuration_offset >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_NEXT_TOKEN_METADATA_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.next_token_metadata_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_NEXT_TOKEN_METADATA_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.next_token_metadata_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_OUTPUT_HIDDEN_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.output_hidden_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_OUTPUT_HIDDEN_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.output_hidden_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_BF16_TEMPORARY_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.bf16_temporary_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_BF16_TEMPORARY_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.bf16_temporary_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.layer_weight_table_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_LAYER_WEIGHT_TABLE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.layer_weight_table_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_CURRENT_K_CACHE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.current_k_cache_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_CURRENT_K_CACHE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.current_k_cache_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_CURRENT_V_CACHE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.current_v_cache_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_CURRENT_V_CACHE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.current_v_cache_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_RETAINED_K_CACHE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.retained_k_cache_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_RETAINED_K_CACHE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.retained_k_cache_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_RETAINED_V_CACHE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.retained_v_cache_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_RETAINED_V_CACHE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.retained_v_cache_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_K_SCALE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.k_scale_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_K_SCALE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.k_scale_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_V_SCALE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.v_scale_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_V_SCALE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.v_scale_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_ATTENTION_WORKSPACE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.attention_workspace_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_ATTENTION_WORKSPACE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.attention_workspace_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.token_metadata_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.token_metadata_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_EMBEDDING_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.embedding_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_EMBEDDING_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.embedding_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.forward_postprocess_configuration_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.forward_postprocess_configuration_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.prediction_table_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_PREDICTION_TABLE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.prediction_table_limit >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[EXECUTION_CONFIG_LAYER_WEIGHT_ENTRY_STRIDE_OFFSET + byte] = static_cast<std::uint8_t>(value.layer_weight_entry_stride >> (8u * byte));
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[EXECUTION_CONFIG_EMBEDDING_ROW_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.embedding_row_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_DEPLOYMENT_ARTIFACT_ID_OFFSET + byte] = static_cast<std::uint8_t>(value.deployment_artifact_id >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_TOKEN_METADATA_ENTRY_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.token_metadata_entry_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_TOKEN_METADATA_BATCH_HEADER_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.token_metadata_batch_header_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.forward_postprocess_configuration_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_PREDICTION_ENTRY_BYTES_OFFSET + byte] = static_cast<std::uint8_t>(value.prediction_entry_bytes >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_FORWARD_POSTPROCESS_CONFIGURATION_FORMAT_OFFSET + byte] = static_cast<std::uint8_t>(value.forward_postprocess_configuration_format >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_PREDICTION_TABLE_FORMAT_OFFSET + byte] = static_cast<std::uint8_t>(value.prediction_table_format >> (8u * byte));
    for (std::size_t byte = 0; byte < 2; ++byte)
        bytes[EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.prediction_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[EXECUTION_CONFIG_GENERATION_BLOCK_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.generation_block_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 1; ++byte)
        bytes[EXECUTION_CONFIG_SUPPRESSED_TOKEN_COUNT_OFFSET + byte] = static_cast<std::uint8_t>(value.suppressed_token_count >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_RETAINED_K_SCALE_BASE_OFFSET + byte] = static_cast<std::uint8_t>(value.retained_k_scale_base >> (8u * byte));
    for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[EXECUTION_CONFIG_RETAINED_K_SCALE_LIMIT_OFFSET + byte] = static_cast<std::uint8_t>(value.retained_k_scale_limit >> (8u * byte));
    return bytes;
}

#endif
