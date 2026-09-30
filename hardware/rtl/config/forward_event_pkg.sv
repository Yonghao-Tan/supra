`default_nettype none

package forward_event_pkg;
    localparam int unsigned FORWARD_EVENT_BYTES = 64;
    localparam int unsigned FORWARD_EVENT_ALIGNMENT = 16;
    localparam int unsigned FORWARD_EVENT_CAPTURE_INDEX_OFFSET = 0;
    localparam int unsigned FORWARD_EVENT_BLOCK_SLOT_OFFSET = 4;
    localparam int unsigned FORWARD_EVENT_GLOBAL_BLOCK_ID_OFFSET = 5;
    localparam int unsigned FORWARD_EVENT_POSITION_COUNT_OFFSET = 6;
    localparam int unsigned FORWARD_EVENT_CONFIRMED_MASK_OFFSET = 8;
    localparam int unsigned FORWARD_EVENT_REMASKED_MASK_OFFSET = 12;
    localparam int unsigned FORWARD_EVENT_SELECTED_MASK_OFFSET = 16;
    localparam int unsigned FORWARD_EVENT_DIRECT_LOCKED_MASK_OFFSET = 20;
    localparam int unsigned FORWARD_EVENT_STABLE_TENTATIVE_MASK_OFFSET = 24;
    localparam int unsigned FORWARD_EVENT_FALLBACK_TENTATIVE_MASK_OFFSET = 28;
    localparam int unsigned FORWARD_EVENT_MANDATORY_REFRESH_MASK_OFFSET = 32;
    localparam int unsigned FORWARD_EVENT_CACHE_COMMIT_MASK_OFFSET = 36;
    localparam int unsigned FORWARD_EVENT_CACHE_INVALIDATE_MASK_OFFSET = 40;
    localparam int unsigned FORWARD_EVENT_CACHE_KEEP_MASK_OFFSET = 44;
    localparam int unsigned FORWARD_EVENT_TOKEN_CHANGED_MASK_OFFSET = 48;
    localparam int unsigned FORWARD_EVENT_TAIL_CLOSED_MASK_OFFSET = 52;
    localparam int unsigned FORWARD_EVENT_RESERVED0_OFFSET = 56;
    localparam int unsigned FORWARD_EVENT_RESERVED1_OFFSET = 60;
endpackage

`default_nettype wire
