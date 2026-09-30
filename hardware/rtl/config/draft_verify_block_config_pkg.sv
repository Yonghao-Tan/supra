`default_nettype none

package draft_verify_block_config_pkg;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_BYTES = 32;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_ALIGNMENT = 16;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_BLOCK_SLOT_OFFSET = 0;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_GLOBAL_BLOCK_ID_OFFSET = 1;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_FLAGS_OFFSET = 2;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_SCHEDULED_QUOTA_OFFSET = 4;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_REMAINING_FORWARDS_OFFSET = 6;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_STEP_INDEX_OFFSET = 8;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_TAIL_AFTER_STEP_OFFSET = 10;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_MATURITY_AGE_OFFSET = 12;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_POSITION_COUNT_OFFSET = 14;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_CAPTURE_INDEX_OFFSET = 16;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_CURRENT_STATE_ENTRY_OFFSET = 20;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_INPUT_SLOT_OVERRIDE_OFFSET = 21;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_NEXT_STATE_ENTRY_OFFSET = 22;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_INPUT_CAPTURE_INDEX_OFFSET = 24;
    localparam int unsigned DRAFT_VERIFY_BLOCK_CONFIG_OBSERVED_MASK_OFFSET = 28;
endpackage

`default_nettype wire
