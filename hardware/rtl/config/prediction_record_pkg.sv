`default_nettype none

package prediction_record_pkg;
    localparam int unsigned PREDICTION_RECORD_BYTES = 32;
    localparam int unsigned PREDICTION_RECORD_ALIGNMENT = 16;
    localparam int unsigned PREDICTION_RECORD_TOKEN_POSITION_OFFSET = 0;
    localparam int unsigned PREDICTION_RECORD_PREDICTION_INDEX_OFFSET = 2;
    localparam int unsigned PREDICTION_RECORD_BLOCK_SLOT_OFFSET = 3;
    localparam int unsigned PREDICTION_RECORD_GLOBAL_BLOCK_ID_OFFSET = 4;
    localparam int unsigned PREDICTION_RECORD_BLOCK_LOCAL_POSITION_OFFSET = 5;
    localparam int unsigned PREDICTION_RECORD_ACTIVATION_BITS_OFFSET = 6;
    localparam int unsigned PREDICTION_RECORD_FORWARD_START_STATE_OFFSET = 7;
    localparam int unsigned PREDICTION_RECORD_CURRENT_TOKEN_ID_OFFSET = 8;
    localparam int unsigned PREDICTION_RECORD_TENTATIVE_TOKEN_ID_OFFSET = 12;
    localparam int unsigned PREDICTION_RECORD_SOURCE_TOKEN_BATCH_INDEX_OFFSET = 16;
    localparam int unsigned PREDICTION_RECORD_SOURCE_ROW_INDEX_OFFSET = 17;
    localparam int unsigned PREDICTION_RECORD_FLAGS_OFFSET = 18;
    localparam int unsigned PREDICTION_RECORD_FINAL_HIDDEN_DDR_ROW_INDEX_OFFSET = 20;
    localparam int unsigned PREDICTION_RECORD_QUERY_GROUP_OFFSET = 22;
    localparam int unsigned PREDICTION_RECORD_CACHE_GROUP_OFFSET = 23;
    localparam int unsigned PREDICTION_RECORD_CURRENT_STATE_ENTRY_OFFSET = 24;
    localparam int unsigned PREDICTION_RECORD_NEXT_STATE_ENTRY_OFFSET = 28;
endpackage

`default_nettype wire
