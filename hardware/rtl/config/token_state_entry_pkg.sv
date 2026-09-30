`default_nettype none

package token_state_entry_pkg;
    localparam int unsigned TOKEN_STATE_ENTRY_BYTES = 32;
    localparam int unsigned TOKEN_STATE_ENTRY_ALIGNMENT = 16;
    localparam int unsigned TOKEN_STATE_ENTRY_TOKEN_POSITION_OFFSET = 0;
    localparam int unsigned TOKEN_STATE_ENTRY_BLOCK_LOCAL_POSITION_OFFSET = 2;
    localparam int unsigned TOKEN_STATE_ENTRY_BLOCK_SLOT_OFFSET = 3;
    localparam int unsigned TOKEN_STATE_ENTRY_GLOBAL_BLOCK_ID_OFFSET = 4;
    localparam int unsigned TOKEN_STATE_ENTRY_STATE_OFFSET = 5;
    localparam int unsigned TOKEN_STATE_ENTRY_ORIGIN_OFFSET = 6;
    localparam int unsigned TOKEN_STATE_ENTRY_ACTIVATION_BITS_OFFSET = 7;
    localparam int unsigned TOKEN_STATE_ENTRY_TOKEN_ID_OFFSET = 8;
    localparam int unsigned TOKEN_STATE_ENTRY_LAST_TOP1_OFFSET = 12;
    localparam int unsigned TOKEN_STATE_ENTRY_PRECISION_AGE_OFFSET = 16;
    localparam int unsigned TOKEN_STATE_ENTRY_CACHE_VALID_OFFSET = 18;
    localparam int unsigned TOKEN_STATE_ENTRY_REFRESH_REQUIRED_OFFSET = 19;
    localparam int unsigned TOKEN_STATE_ENTRY_CAPTURE_INDEX_OFFSET = 20;
    localparam int unsigned TOKEN_STATE_ENTRY_PREDICTION_FLAG_OFFSET = 24;
    localparam int unsigned TOKEN_STATE_ENTRY_CHANGE_FLAGS_OFFSET = 25;
    localparam int unsigned TOKEN_STATE_ENTRY_CHANGE_CONFIDENCE_BF16_OFFSET = 26;
    localparam int unsigned TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_BF16_OFFSET = 28;
    localparam int unsigned TOKEN_STATE_ENTRY_LAST_ACTION_CONFIDENCE_VALID_OFFSET = 30;
    localparam int unsigned TOKEN_STATE_ENTRY_SOURCE_A_PENDING_OFFSET = 31;
endpackage

`default_nettype wire
