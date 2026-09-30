`default_nettype none

package uaps_config_pkg;
    localparam int unsigned UAPS_CONFIG_BYTES = 64;
    localparam int unsigned UAPS_CONFIG_ALIGNMENT = 16;
    localparam int unsigned UAPS_CONFIG_MAGIC_OFFSET = 0;
    localparam int unsigned UAPS_CONFIG_VERSION_OFFSET = 4;
    localparam int unsigned UAPS_CONFIG_BYTES_OFFSET = 6;
    localparam int unsigned UAPS_CONFIG_BASE_TOKEN_COUNT_OFFSET = 8;
    localparam int unsigned UAPS_CONFIG_FUTURE_TOKEN_COUNT_OFFSET = 10;
    localparam int unsigned UAPS_CONFIG_NEXT_BLOCK_START_OFFSET = 12;
    localparam int unsigned UAPS_CONFIG_MAX_NEXT_TOKENS_OFFSET = 14;
    localparam int unsigned UAPS_CONFIG_PRIORITY_CONTROL_OFFSET = 15;
    localparam int unsigned UAPS_CONFIG_BASE_TABLE_BASE_OFFSET = 16;
    localparam int unsigned UAPS_CONFIG_BASE_TABLE_LIMIT_OFFSET = 24;
    localparam int unsigned UAPS_CONFIG_FUTURE_TABLE_BASE_OFFSET = 32;
    localparam int unsigned UAPS_CONFIG_FUTURE_TABLE_LIMIT_OFFSET = 40;
    localparam int unsigned UAPS_CONFIG_RESULT_BASE_OFFSET = 48;
    localparam int unsigned UAPS_CONFIG_RESULT_LIMIT_OFFSET = 56;
endpackage

`default_nettype wire
