`default_nettype none

package forward_postprocess_completion_pkg;
    localparam int unsigned FORWARD_POSTPROCESS_COMPLETION_BYTES = 16;
    localparam int unsigned FORWARD_POSTPROCESS_COMPLETION_ALIGNMENT = 16;
    localparam int unsigned FORWARD_POSTPROCESS_COMPLETION_MAGIC_OFFSET = 0;
    localparam int unsigned FORWARD_POSTPROCESS_COMPLETION_COMPLETION_VERSION_OFFSET = 4;
    localparam int unsigned FORWARD_POSTPROCESS_COMPLETION_CAPTURE_INDEX_OFFSET = 8;
    localparam int unsigned FORWARD_POSTPROCESS_COMPLETION_BLOCK_COUNT_OFFSET = 12;
    localparam int unsigned FORWARD_POSTPROCESS_COMPLETION_STATUS_OFFSET = 13;
    localparam int unsigned FORWARD_POSTPROCESS_COMPLETION_RESERVED_OFFSET = 14;
endpackage

`default_nettype wire
