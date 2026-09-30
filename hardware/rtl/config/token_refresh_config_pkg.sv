`default_nettype none

package token_refresh_config_pkg;
    localparam int unsigned TOKEN_REFRESH_CONFIG_BYTES = 176;
    localparam int unsigned TOKEN_REFRESH_CONFIG_ALIGNMENT = 16;
    localparam int unsigned TOKEN_REFRESH_CONFIG_MAGIC_OFFSET = 0;
    localparam int unsigned TOKEN_REFRESH_CONFIG_VERSION_OFFSET = 4;
    localparam int unsigned TOKEN_REFRESH_CONFIG_BYTES_OFFSET = 6;
    localparam int unsigned TOKEN_REFRESH_CONFIG_FLAGS_OFFSET = 8;
    localparam int unsigned TOKEN_REFRESH_CONFIG_PENDING_CONFIGURATION_OFFSET_OFFSET = 10;
    localparam int unsigned TOKEN_REFRESH_CONFIG_SEQUENCE_LENGTH_OFFSET = 12;
    localparam int unsigned TOKEN_REFRESH_CONFIG_TARGET_TOKEN_COUNT_OFFSET = 14;
    localparam int unsigned TOKEN_REFRESH_CONFIG_REQUIRED_QUOTA_OFFSET = 16;
    localparam int unsigned TOKEN_REFRESH_CONFIG_PROBABILITY_CONFIGURATION_OFFSET_OFFSET = 18;
    localparam int unsigned TOKEN_REFRESH_CONFIG_HEAD_STRIDE_OFFSET = 20;
    localparam int unsigned TOKEN_REFRESH_CONFIG_SCALE_HEAD_STRIDE_OFFSET = 24;
    localparam int unsigned TOKEN_REFRESH_CONFIG_METADATA_VERSION_OFFSET = 28;
    localparam int unsigned TOKEN_REFRESH_CONFIG_TABLE_BASE_OFFSET = 32;
    localparam int unsigned TOKEN_REFRESH_CONFIG_TABLE_LIMIT_OFFSET = 40;
    localparam int unsigned TOKEN_REFRESH_CONFIG_METADATA_BASE_OFFSET = 48;
    localparam int unsigned TOKEN_REFRESH_CONFIG_METADATA_LIMIT_OFFSET = 56;
    localparam int unsigned TOKEN_REFRESH_CONFIG_SOURCE_K_BASE_OFFSET = 64;
    localparam int unsigned TOKEN_REFRESH_CONFIG_SOURCE_V_BASE_OFFSET = 72;
    localparam int unsigned TOKEN_REFRESH_CONFIG_SOURCE_SCALE_BASE_OFFSET = 80;
    localparam int unsigned TOKEN_REFRESH_CONFIG_DESTINATION_K_BASE_OFFSET = 88;
    localparam int unsigned TOKEN_REFRESH_CONFIG_DESTINATION_V_BASE_OFFSET = 96;
    localparam int unsigned TOKEN_REFRESH_CONFIG_DESTINATION_SCALE_BASE_OFFSET = 104;
    localparam int unsigned TOKEN_REFRESH_CONFIG_SOURCE_K_LIMIT_OFFSET = 112;
    localparam int unsigned TOKEN_REFRESH_CONFIG_SOURCE_V_LIMIT_OFFSET = 120;
    localparam int unsigned TOKEN_REFRESH_CONFIG_SOURCE_SCALE_LIMIT_OFFSET = 128;
    localparam int unsigned TOKEN_REFRESH_CONFIG_DESTINATION_K_LIMIT_OFFSET = 136;
    localparam int unsigned TOKEN_REFRESH_CONFIG_DESTINATION_V_LIMIT_OFFSET = 144;
    localparam int unsigned TOKEN_REFRESH_CONFIG_DESTINATION_SCALE_LIMIT_OFFSET = 152;
    localparam int unsigned TOKEN_REFRESH_CONFIG_CAPTURE_INDEX_OFFSET = 160;
    localparam int unsigned TOKEN_REFRESH_CONFIG_RELATION_JOB_COUNT_OFFSET = 164;
    localparam int unsigned TOKEN_REFRESH_CONFIG_RELATION_JOB_BASE_OFFSET = 168;
endpackage

`default_nettype wire
