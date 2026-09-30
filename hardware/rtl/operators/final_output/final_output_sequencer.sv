`default_nettype none

module final_output_sequencer #(
    parameter integer MAX_PREDICTIONS = 96,
    parameter integer HIDDEN_ELEMENTS = 4096
) (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    output logic          abort_ack,
    output logic [2:0]    abort_wait_status,

    input  logic          start_valid,
    output logic          start_ready,
    input  logic [63:0]   start_post_config_base,
    input  logic [63:0]   start_post_config_limit,
    input  logic [63:0]   start_prediction_base,
    input  logic [63:0]   start_prediction_limit,
    input  logic [7:0]    start_prediction_count,
    input  logic [63:0]   start_final_hidden_base,
    input  logic [63:0]   start_final_hidden_limit,

    output logic          dma_request_valid,
    input  logic          dma_request_ready,
    output logic [63:0]   dma_request_address,
    output logic [31:0]   dma_request_bytes,
    output logic [7:0]    dma_request_tag,
    input  logic          dma_response_valid,
    output logic          dma_response_ready,
    input  logic [127:0]  dma_response_data,
    input  logic [15:0]   dma_response_byte_enable,
    input  logic          dma_response_last,
    input  logic          dma_done_pulse,
    input  logic          dma_error,

    output logic          descriptor_valid,
    input  logic          descriptor_ready,
    output logic [255:0]  descriptor_data,

    output logic          descriptor_lookup_valid,
    input  logic          descriptor_lookup_ready,
    output logic [6:0]    descriptor_lookup_index,
    input  logic          descriptor_lookup_rsp_valid,
    output logic          descriptor_lookup_rsp_ready,
    input  logic [255:0]  descriptor_lookup_rsp_data,

    output logic          hidden_start_valid,
    input  logic          hidden_start_ready,
    output logic [63:0]   hidden_start_address,
    output logic [63:0]   hidden_start_limit,
    output logic [31:0]   hidden_start_bytes,
    output logic [5:0]    hidden_start_rows,
    output logic [527:0]  hidden_start_ddr_row_index,
    output logic [55:0]   hidden_start_source_round,
    output logic [47:0]   hidden_start_source_row,
    input  logic          hidden_done_valid,
    output logic          hidden_done_ready,
    input  logic          hidden_error,

    output logic          rms_start_valid,
    input  logic          rms_start_ready,
    output logic [5:0]    rms_start_rows,
    output logic [15:0]   rms_start_elements,
    output logic [15:0]   rms_start_epsilon_bf16,
    output logic          rms_start_gamma_bypass,
    input  logic          rms_done_valid,
    output logic          rms_done_ready,
    input  logic          rms_error,

    output logic          quant_start_valid,
    input  logic          quant_start_ready,
    output logic [6:0]    quant_start_prediction_base,
    output logic [3:0]    quant_start_group,
    output logic [3:0]    quant_start_rows,
    input  logic          quant_done_valid,
    output logic          quant_done_ready,
    input  logic          quant_error,

    output logic          done_valid,
    input  logic          done_ready,
    output logic          error,
    output logic [7:0]    error_id,
    output logic [7:0]    prediction_count,
    output logic [1535:0] post_config_bits
);
    import forward_postprocess_config_pkg::*;

    localparam integer POST_BEATS = FORWARD_POSTPROCESS_CONFIG_BYTES / 16;
    localparam integer HIDDEN_ROW_BYTES = HIDDEN_ELEMENTS * 2;

    localparam logic [7:0] ERROR_START = 8'h01;
    localparam logic [7:0] ERROR_POST_DMA = 8'h02;
    localparam logic [7:0] ERROR_POST_FORMAT = 8'h03;
    localparam logic [7:0] ERROR_PREDICTION_DMA = 8'h04;
    localparam logic [7:0] ERROR_PREDICTION_FORMAT = 8'h05;
    localparam logic [7:0] ERROR_DUPLICATE_SOURCE = 8'h06;
    localparam logic [7:0] ERROR_HIDDEN_READ = 8'h07;
    localparam logic [7:0] ERROR_RMSNORM = 8'h08;
    localparam logic [7:0] ERROR_QUANTIZER = 8'h09;

    typedef enum logic [4:0] {
        IDLE,
        POST_REQUEST,
        POST_STREAM,
        POST_VALIDATE,
        PREDICTION_REQUEST,
        PREDICTION_STREAM,
        DUPLICATE_ROW_REQUEST,
        DUPLICATE_ROW_WAIT,
        DUPLICATE_PRIOR_REQUEST,
        DUPLICATE_PRIOR_WAIT,
        GROUP_LOOKUP_REQUEST,
        GROUP_LOOKUP_WAIT,
        HIDDEN_START,
        HIDDEN_WAIT,
        RMS_START,
        RMS_WAIT,
        QUANT_START,
        QUANT_WAIT,
        COMPLETE,
        ABORT_WAIT_LOW
    } state_t;

    state_t state;
    logic [63:0] saved_post_config_base;
    logic [63:0] saved_prediction_base;
    logic [63:0] saved_final_hidden_base;
    logic [63:0] saved_final_hidden_limit;
    logic [7:0] saved_prediction_count;
    logic [4:0] post_beat;
    logic [3:0] post_check;
    logic [8:0] prediction_beat;
    logic [127:0] prediction_first_half;
    logic dma_data_done;
    logic dma_completion_seen;
    logic terminal_error;
    logic [7:0] terminal_error_id;
    logic [6:0] duplicate_row;
    logic [6:0] duplicate_prior;
    logic [10:0] duplicate_token_position;
    logic [10:0] duplicate_ddr_row;
    logic [3:0] group_index;
    logic [3:0] group_lookup_row;
    logic abort_pending;

    logic [87:0] group_ddr_rows;
    logic [55:0] group_source_rounds;
    logic [47:0] group_source_token_indices;

    logic response_fire;
    logic response_last_fire;
    logic dma_transaction_complete;
    logic post_check_error;
    logic [3:0] current_group_rows;
    logic [6:0] current_group_base;
    logic prediction_record_format_error;
    logic prediction_response_error;
    logic [31:0] prediction_vocabulary_size;
    logic [10:0] prediction_final_hidden_row;
    logic [64:0] prediction_final_hidden_end;

    localparam logic [2:0] ABORT_WAIT_NONE = 3'd0;
    localparam logic [2:0] ABORT_WAIT_POST_DMA = 3'd1;
    localparam logic [2:0] ABORT_WAIT_PREDICTION_DMA = 3'd2;
    localparam logic [2:0] ABORT_WAIT_HIDDEN = 3'd3;
    localparam logic [2:0] ABORT_WAIT_RMS = 3'd4;
    localparam logic [2:0] ABORT_WAIT_QUANT = 3'd5;
    localparam logic [2:0] ABORT_WAIT_DESCRIPTOR = 3'd6;

    function automatic logic bf16_finite(input logic [15:0] value);
        bf16_finite = value[14:7] != 8'hff;
    endfunction

    function automatic logic bf16_nonnegative_finite(input logic [15:0] value);
        bf16_nonnegative_finite = !value[15] && bf16_finite(value);
    endfunction

    function automatic logic bf16_positive_finite(input logic [15:0] value);
        bf16_positive_finite = bf16_nonnegative_finite(value) &&
            value[14:0] != 15'd0;
    endfunction

    function automatic logic [10:0] select_8x11(
        input logic [87:0] values, input logic [2:0] index);
        case (index)
            3'd0: select_8x11 = values[0*11 +: 11];
            3'd1: select_8x11 = values[1*11 +: 11];
            3'd2: select_8x11 = values[2*11 +: 11];
            3'd3: select_8x11 = values[3*11 +: 11];
            3'd4: select_8x11 = values[4*11 +: 11];
            3'd5: select_8x11 = values[5*11 +: 11];
            3'd6: select_8x11 = values[6*11 +: 11];
            default: select_8x11 = values[7*11 +: 11];
        endcase
    endfunction

    function automatic logic [6:0] select_8x7(
        input logic [55:0] values, input logic [2:0] index);
        case (index)
            3'd0: select_8x7 = values[0*7 +: 7];
            3'd1: select_8x7 = values[1*7 +: 7];
            3'd2: select_8x7 = values[2*7 +: 7];
            3'd3: select_8x7 = values[3*7 +: 7];
            3'd4: select_8x7 = values[4*7 +: 7];
            3'd5: select_8x7 = values[5*7 +: 7];
            3'd6: select_8x7 = values[6*7 +: 7];
            default: select_8x7 = values[7*7 +: 7];
        endcase
    endfunction

    function automatic logic [5:0] select_8x6(
        input logic [47:0] values, input logic [2:0] index);
        case (index)
            3'd0: select_8x6 = values[0*6 +: 6];
            3'd1: select_8x6 = values[1*6 +: 6];
            3'd2: select_8x6 = values[2*6 +: 6];
            3'd3: select_8x6 = values[3*6 +: 6];
            3'd4: select_8x6 = values[4*6 +: 6];
            3'd5: select_8x6 = values[5*6 +: 6];
            3'd6: select_8x6 = values[6*6 +: 6];
            default: select_8x6 = values[7*6 +: 6];
        endcase
    endfunction

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign prediction_count = saved_prediction_count;

    assign dma_request_valid = !abort_pending && !abort_request &&
        (state == POST_REQUEST || state == PREDICTION_REQUEST);
    assign dma_request_address = state == POST_REQUEST ?
        saved_post_config_base : saved_prediction_base;
    assign dma_request_bytes = state == POST_REQUEST ?
        FORWARD_POSTPROCESS_CONFIG_BYTES : 32'(saved_prediction_count) * 32;
    assign dma_request_tag = state == POST_REQUEST ? 8'hb0 : 8'hb1;
    assign descriptor_data = {dma_response_data, prediction_first_half};
    assign descriptor_valid = state == PREDICTION_STREAM &&
        prediction_beat[0] && dma_response_valid && !terminal_error &&
        !prediction_response_error;
    assign prediction_response_error =
        (prediction_beat[0] && prediction_record_format_error) ||
        dma_response_byte_enable != 16'hffff ||
        dma_response_last !=
            (prediction_beat == 9'(saved_prediction_count)*2-1);
    assign descriptor_lookup_valid =
        state == DUPLICATE_ROW_REQUEST ||
        state == DUPLICATE_PRIOR_REQUEST ||
        state == GROUP_LOOKUP_REQUEST;
    assign descriptor_lookup_index = state == DUPLICATE_ROW_REQUEST ?
        duplicate_row : state == DUPLICATE_PRIOR_REQUEST ?
            duplicate_prior : current_group_base + {3'd0, group_lookup_row};
    assign descriptor_lookup_rsp_ready =
        state == DUPLICATE_ROW_WAIT ||
        state == DUPLICATE_PRIOR_WAIT || state == GROUP_LOOKUP_WAIT;
    // Rejected records must drain even when the descriptor store cannot accept.
    assign dma_response_ready = state == POST_STREAM ||
        (state == PREDICTION_STREAM &&
         (!prediction_beat[0] || descriptor_ready || terminal_error ||
          prediction_response_error));
    assign response_fire = dma_response_valid && dma_response_ready;
    assign response_last_fire = response_fire && dma_response_last;
    assign dma_transaction_complete = dma_completion_seen || dma_done_pulse ?
        (terminal_error || dma_error || dma_data_done || response_last_fire) :
        1'b0;

    assign current_group_base = {group_index, 3'b000};
    assign current_group_rows =
        integer'(saved_prediction_count) - integer'(current_group_base) >= 8 ?
            4'd8 : 4'(saved_prediction_count - current_group_base);

    assign hidden_start_valid = state == HIDDEN_START &&
        !abort_pending && !abort_request;
    assign hidden_start_address = saved_final_hidden_base;
    assign hidden_start_limit = saved_final_hidden_limit;
    assign hidden_start_bytes = 32'(current_group_rows) * HIDDEN_ROW_BYTES;
    assign hidden_start_rows = {2'b00, current_group_rows};
    always_comb begin
        hidden_start_ddr_row_index = '0;
        hidden_start_source_round = '0;
        hidden_start_source_row = '0;
        for (integer row = 0; row < 8; row++) begin
            if (row < current_group_rows) begin
                hidden_start_ddr_row_index[row*11 +: 11] =
                    select_8x11(group_ddr_rows, 3'(row));
                hidden_start_source_round[row*7 +: 7] =
                    select_8x7(group_source_rounds, 3'(row));
                hidden_start_source_row[row*6 +: 6] =
                    select_8x6(group_source_token_indices, 3'(row));
            end
        end
    end
    assign hidden_done_ready = state == HIDDEN_WAIT;

    assign rms_start_valid = state == RMS_START &&
        !abort_pending && !abort_request;
    assign rms_start_rows = {2'b00, current_group_rows};
    assign rms_start_elements = 16'(HIDDEN_ELEMENTS);
    assign rms_start_epsilon_bf16 = post_config_bits[
        FORWARD_POSTPROCESS_CONFIG_FINAL_RMS_EPSILON_BF16_OFFSET*8 +: 16];
    assign rms_start_gamma_bypass = 1'b1;
    assign rms_done_ready = state == RMS_WAIT;

    assign quant_start_valid = state == QUANT_START &&
        !abort_pending && !abort_request;
    assign quant_start_prediction_base = current_group_base;
    assign quant_start_group = group_index;
    assign quant_start_rows = current_group_rows;
    assign quant_done_ready = state == QUANT_WAIT;

    always_comb begin
        prediction_vocabulary_size = post_config_bits[
            FORWARD_POSTPROCESS_CONFIG_VOCABULARY_SIZE_OFFSET*8 +: 32];
        prediction_final_hidden_row = dma_response_data[42:32];
        prediction_final_hidden_end = {1'b0, saved_final_hidden_base} +
            (65'(prediction_final_hidden_row) << 13) +
            65'(HIDDEN_ROW_BYTES);
        prediction_record_format_error =
            |prediction_first_half[15:11] ||
            prediction_first_half[23:16] != 8'(prediction_beat[7:1]) ||
            prediction_first_half[31:24] > 3 ||
            prediction_first_half[39:32] > 63 ||
            prediction_first_half[47:40] > 31 ||
            prediction_first_half[63:56] > 2 ||
            prediction_first_half[95:64] >= prediction_vocabulary_size ||
            prediction_first_half[127:96] >= prediction_vocabulary_size ||
            dma_response_data[7] ||
            dma_response_data[15:8] > 47 ||
            |dma_response_data[31:19] ||
            |dma_response_data[47:43] ||
            prediction_final_hidden_end[64] ||
            prediction_final_hidden_end > {1'b0, saved_final_hidden_limit};
    end

    always_comb begin
        abort_wait_status = ABORT_WAIT_NONE;
        if (abort_pending || abort_request) begin
            case (state)
                POST_STREAM:
                    abort_wait_status = ABORT_WAIT_POST_DMA;
                PREDICTION_STREAM:
                    abort_wait_status = descriptor_valid &&
                        !descriptor_ready ? ABORT_WAIT_DESCRIPTOR :
                        ABORT_WAIT_PREDICTION_DMA;
                HIDDEN_WAIT:
                    abort_wait_status = ABORT_WAIT_HIDDEN;
                RMS_WAIT:
                    abort_wait_status = ABORT_WAIT_RMS;
                QUANT_WAIT:
                    abort_wait_status = ABORT_WAIT_QUANT;
                default:
                    abort_wait_status = ABORT_WAIT_NONE;
            endcase
        end
    end

    always_comb begin
        post_check_error = 1'b0;
        case (post_check)
            4'd0: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_MAGIC_OFFSET*8 +: 32] !=
                    32'h31434250 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_VERSION_OFFSET*8 +: 16] !=
                    16'd1 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_HEADER_BYTES_OFFSET*8 +: 16] !=
                    16'(FORWARD_POSTPROCESS_CONFIG_BYTES) ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_TOTAL_BYTES_OFFSET*8 +: 32] !=
                    FORWARD_POSTPROCESS_CONFIG_BYTES ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_FLAGS_OFFSET*8+1 +: 15] != 0;
            4'd1: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_FORWARD_EVENT_ENTRY_BYTES_OFFSET*8 +: 16] != 64 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_VOCABULARY_SIZE_OFFSET*8 +: 32] < 8 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_VOCABULARY_SIZE_OFFSET*8 +: 32] > 126464 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_INPUT_FEATURES_OFFSET*8 +: 16] < 8 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_INPUT_FEATURES_OFFSET*8 +: 16] > 16'(HIDDEN_ELEMENTS) ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_CANDIDATE_LANES_OFFSET*8 +: 16] != 64;
            4'd2: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_RAW_W8_LAYOUT_ID_OFFSET*8 +: 32] != 1 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_STATE_FORMAT_OFFSET*8 +: 16] != 1 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_DESCRIPTOR_FORMAT_OFFSET*8 +: 16] != 1;
            4'd3: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_RAW_W8_BASE_OFFSET*8 +: 64] >=
                    post_config_bits[FORWARD_POSTPROCESS_CONFIG_RAW_W8_LIMIT_OFFSET*8 +: 64] ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_RAW_W8_BASE_OFFSET*8 +: 4] != 0;
            4'd4: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_WEIGHT_SCALE_BASE_OFFSET*8 +: 64] >=
                    post_config_bits[FORWARD_POSTPROCESS_CONFIG_WEIGHT_SCALE_LIMIT_OFFSET*8 +: 64] ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_WEIGHT_SCALE_BASE_OFFSET*8 +: 4] != 0;
            4'd5: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_CURRENT_STATE_BASE_OFFSET*8 +: 64] >=
                    post_config_bits[FORWARD_POSTPROCESS_CONFIG_CURRENT_STATE_LIMIT_OFFSET*8 +: 64] ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_NEXT_STATE_BASE_OFFSET*8 +: 64] >=
                    post_config_bits[FORWARD_POSTPROCESS_CONFIG_NEXT_STATE_LIMIT_OFFSET*8 +: 64];
            4'd6: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_FORWARD_EVENT_BASE_OFFSET*8 +: 64] >=
                    post_config_bits[FORWARD_POSTPROCESS_CONFIG_FORWARD_EVENT_LIMIT_OFFSET*8 +: 64] ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_COMPLETION_BASE_OFFSET*8 +: 64] >=
                    post_config_bits[FORWARD_POSTPROCESS_CONFIG_COMPLETION_LIMIT_OFFSET*8 +: 64];
            4'd7: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_MASK_TOKEN_ID_OFFSET*8 +: 32] >=
                    post_config_bits[FORWARD_POSTPROCESS_CONFIG_VOCABULARY_SIZE_OFFSET*8 +: 32];
            4'd8: post_check_error =
                !bf16_nonnegative_finite(post_config_bits[
                    FORWARD_POSTPROCESS_CONFIG_HIGH_CONFIDENCE_THRESHOLD_BF16_OFFSET*8 +: 16]) ||
                !bf16_nonnegative_finite(post_config_bits[
                    FORWARD_POSTPROCESS_CONFIG_TAIL_HIGH_CONFIDENCE_THRESHOLD_BF16_OFFSET*8 +: 16]) ||
                !bf16_nonnegative_finite(post_config_bits[
                    FORWARD_POSTPROCESS_CONFIG_LOW_CONFIDENCE_THRESHOLD_BF16_OFFSET*8 +: 16]) ||
                !bf16_nonnegative_finite(post_config_bits[
                    FORWARD_POSTPROCESS_CONFIG_VERIFY_THRESHOLD_BF16_OFFSET*8 +: 16]);
            4'd9: post_check_error =
                !bf16_nonnegative_finite(post_config_bits[
                    FORWARD_POSTPROCESS_CONFIG_STABILITY_BONUS_BF16_OFFSET*8 +: 16]) ||
                !bf16_positive_finite(post_config_bits[
                    FORWARD_POSTPROCESS_CONFIG_BUDGET_SCALE_BF16_OFFSET*8 +: 16]) ||
                !bf16_positive_finite(post_config_bits[
                    FORWARD_POSTPROCESS_CONFIG_FINAL_RMS_EPSILON_BF16_OFFSET*8 +: 16]);
            4'd10: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_SCALE_PREFETCH_PANEL_COUNT_OFFSET*8 +: 16] != 128 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_BLOCK_CONFIGURATION_ENTRY_BYTES_OFFSET*8 +: 16] != 32 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_STATE_ENTRY_BYTES_OFFSET*8 +: 16] != 32 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_SUPPRESSED_TOKEN_ENTRY_BYTES_OFFSET*8 +: 16] != 4 ||
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_COMPLETION_ENTRY_BYTES_OFFSET*8 +: 16] != 16;
            default: post_check_error =
                post_config_bits[FORWARD_POSTPROCESS_CONFIG_BLOCK_CONFIGURATION_BASE_OFFSET*8 +: 64] >=
                    post_config_bits[FORWARD_POSTPROCESS_CONFIG_BLOCK_CONFIGURATION_LIMIT_OFFSET*8 +: 64];
        endcase
    end

    always_ff @(posedge clk) begin
        logic record_format_error;

        if (rst) begin
            state <= IDLE;
            saved_post_config_base <= '0;
            saved_prediction_base <= '0;
            saved_final_hidden_base <= '0;
            saved_final_hidden_limit <= '0;
            saved_prediction_count <= '0;
            post_beat <= '0;
            post_check <= '0;
            prediction_beat <= '0;
            prediction_first_half <= '0;
            dma_data_done <= 1'b0;
            dma_completion_seen <= 1'b0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
            duplicate_row <= '0;
            duplicate_prior <= '0;
            duplicate_token_position <= '0;
            duplicate_ddr_row <= '0;
            group_index <= '0;
            group_lookup_row <= '0;
            group_ddr_rows <= '0;
            group_source_rounds <= '0;
            group_source_token_indices <= '0;
            abort_pending <= 1'b0;
            abort_ack <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            if (abort_request && state != IDLE && state != ABORT_WAIT_LOW)
                abort_pending <= 1'b1;
            case (state)
                IDLE: if (abort_request) begin
                    abort_ack <= 1'b1;
                    state <= ABORT_WAIT_LOW;
                end else if (start_valid && start_ready) begin
                    saved_post_config_base <= start_post_config_base;
                    saved_prediction_base <= start_prediction_base;
                    saved_final_hidden_base <= start_final_hidden_base;
                    saved_final_hidden_limit <= start_final_hidden_limit;
                    saved_prediction_count <= start_prediction_count;
                    terminal_error <= 1'b0;
                    terminal_error_id <= '0;
                    abort_pending <= 1'b0;
                    if (start_prediction_count == 0 ||
                        start_prediction_count > 8'(MAX_PREDICTIONS) ||
                        start_post_config_base[3:0] != 0 ||
                        start_prediction_base[3:0] != 0 ||
                        start_post_config_limit < start_post_config_base ||
                        start_post_config_limit - start_post_config_base <
                            64'(FORWARD_POSTPROCESS_CONFIG_BYTES) ||
                        start_prediction_limit < start_prediction_base ||
                        start_prediction_limit - start_prediction_base <
                            64'(start_prediction_count) * 64'd32 ||
                        start_final_hidden_limit <= start_final_hidden_base) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_START;
                        state <= COMPLETE;
                    end else begin
                        state <= POST_REQUEST;
                    end
                end

                POST_REQUEST: if (abort_pending || abort_request) begin
                    abort_ack <= 1'b1;
                    abort_pending <= 1'b0;
                    state <= ABORT_WAIT_LOW;
                end else if (dma_request_valid && dma_request_ready) begin
                    post_beat <= '0;
                    dma_data_done <= 1'b0;
                    dma_completion_seen <= 1'b0;
                    state <= POST_STREAM;
                end

                POST_STREAM: begin
                    if (dma_done_pulse) begin
                        dma_completion_seen <= 1'b1;
                        if (dma_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_POST_DMA;
                        end
                    end
                    if (response_fire) begin
                        post_config_bits[post_beat*128 +: 128] <= dma_response_data;
                        if (dma_response_byte_enable != 16'hffff ||
                            dma_response_last !=
                                (post_beat == 5'(POST_BEATS-1))) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_POST_DMA;
                        end
                        post_beat <= post_beat + 5'd1;
                        if (dma_response_last)
                            dma_data_done <= 1'b1;
                    end
                    if (dma_transaction_complete) begin
                        if (abort_pending || abort_request) begin
                            abort_ack <= 1'b1;
                            abort_pending <= 1'b0;
                            state <= ABORT_WAIT_LOW;
                        end else if (terminal_error || dma_error) begin
                            terminal_error <= 1'b1;
                            if (!terminal_error)
                                terminal_error_id <= ERROR_POST_DMA;
                            state <= COMPLETE;
                        end else begin
                            post_check <= '0;
                            state <= POST_VALIDATE;
                        end
                    end
                end

                POST_VALIDATE: begin
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (terminal_error) begin
                        state <= COMPLETE;
                    end else if (post_check_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_POST_FORMAT;
                        state <= COMPLETE;
                    end else if (post_check == 4'd11) begin
                        state <= PREDICTION_REQUEST;
                    end else begin
                        post_check <= post_check + 4'd1;
                    end
                end

                PREDICTION_REQUEST: if (abort_pending || abort_request) begin
                    abort_ack <= 1'b1;
                    abort_pending <= 1'b0;
                    state <= ABORT_WAIT_LOW;
                end else if (dma_request_valid && dma_request_ready) begin
                    prediction_beat <= '0;
                    dma_data_done <= 1'b0;
                    dma_completion_seen <= 1'b0;
                    state <= PREDICTION_STREAM;
                end

                PREDICTION_STREAM: begin
                    if (dma_done_pulse) begin
                        dma_completion_seen <= 1'b1;
                        if (dma_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_PREDICTION_DMA;
                        end
                    end
                    if (response_fire) begin
                        if (!prediction_beat[0]) begin
                            prediction_first_half <= dma_response_data;
                        end else begin
                            record_format_error =
                                prediction_record_format_error;
                            if (record_format_error) begin
                                terminal_error <= 1'b1;
                                terminal_error_id <= ERROR_PREDICTION_FORMAT;
                            end
                        end
                        if (dma_response_byte_enable != 16'hffff ||
                            dma_response_last !=
                                (prediction_beat ==
                                 9'(saved_prediction_count)*2-1)) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_PREDICTION_DMA;
                        end
                        prediction_beat <= prediction_beat + 9'd1;
                        if (dma_response_last)
                            dma_data_done <= 1'b1;
                    end
                    if (dma_transaction_complete) begin
                        if (abort_pending || abort_request) begin
                            abort_ack <= 1'b1;
                            abort_pending <= 1'b0;
                            state <= ABORT_WAIT_LOW;
                        end else if (terminal_error || dma_error ||
                                     (response_fire && prediction_response_error)) begin
                            terminal_error <= 1'b1;
                            if (!terminal_error &&
                                !(response_fire && prediction_response_error))
                                terminal_error_id <= ERROR_PREDICTION_DMA;
                            state <= COMPLETE;
                        end else begin
                            duplicate_row <= 7'd1;
                            duplicate_prior <= 7'd0;
                            group_index <= '0;
                            group_lookup_row <= '0;
                            group_ddr_rows <= '0;
                            group_source_rounds <= '0;
                            group_source_token_indices <= '0;
                            state <= saved_prediction_count == 1 ?
                                GROUP_LOOKUP_REQUEST : DUPLICATE_ROW_REQUEST;
                        end
                    end
                end

                DUPLICATE_ROW_REQUEST: begin
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (descriptor_lookup_valid &&
                                 descriptor_lookup_ready) begin
                        state <= DUPLICATE_ROW_WAIT;
                    end
                end

                DUPLICATE_ROW_WAIT:
                    if (descriptor_lookup_rsp_valid &&
                        descriptor_lookup_rsp_ready) begin
                        if (abort_pending || abort_request) begin
                            abort_ack <= 1'b1;
                            abort_pending <= 1'b0;
                            state <= ABORT_WAIT_LOW;
                        end else begin
                            duplicate_token_position <=
                                descriptor_lookup_rsp_data[10:0];
                            duplicate_ddr_row <=
                                descriptor_lookup_rsp_data[170:160];
                            state <= DUPLICATE_PRIOR_REQUEST;
                        end
                    end

                DUPLICATE_PRIOR_REQUEST: begin
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (descriptor_lookup_valid &&
                                 descriptor_lookup_ready) begin
                        state <= DUPLICATE_PRIOR_WAIT;
                    end
                end

                DUPLICATE_PRIOR_WAIT:
                    if (descriptor_lookup_rsp_valid &&
                        descriptor_lookup_rsp_ready) begin
                        if (duplicate_ddr_row ==
                                descriptor_lookup_rsp_data[170:160] ||
                            duplicate_token_position ==
                                descriptor_lookup_rsp_data[10:0]) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_DUPLICATE_SOURCE;
                            state <= COMPLETE;
                        end else if (duplicate_prior + 7'd1 < duplicate_row) begin
                            duplicate_prior <= duplicate_prior + 7'd1;
                            state <= DUPLICATE_PRIOR_REQUEST;
                        end else if (duplicate_row + 7'd1 <
                                     saved_prediction_count) begin
                            duplicate_row <= duplicate_row + 7'd1;
                            duplicate_prior <= 7'd0;
                            state <= DUPLICATE_ROW_REQUEST;
                        end else begin
                            group_index <= '0;
                            group_lookup_row <= '0;
                            group_ddr_rows <= '0;
                            group_source_rounds <= '0;
                            group_source_token_indices <= '0;
                            state <= GROUP_LOOKUP_REQUEST;
                        end
                    end

                GROUP_LOOKUP_REQUEST: begin
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (descriptor_lookup_valid &&
                                 descriptor_lookup_ready) begin
                        state <= GROUP_LOOKUP_WAIT;
                    end
                end

                GROUP_LOOKUP_WAIT:
                    if (descriptor_lookup_rsp_valid &&
                        descriptor_lookup_rsp_ready) begin
                        group_ddr_rows[group_lookup_row*11 +: 11] <=
                            descriptor_lookup_rsp_data[170:160];
                        group_source_rounds[group_lookup_row*7 +: 7] <=
                            descriptor_lookup_rsp_data[134:128];
                        group_source_token_indices[group_lookup_row*6 +: 6] <=
                            descriptor_lookup_rsp_data[141:136];
                        if (group_lookup_row + 4'd1 >= current_group_rows) begin
                            state <= HIDDEN_START;
                        end else begin
                            group_lookup_row <= group_lookup_row + 4'd1;
                            state <= GROUP_LOOKUP_REQUEST;
                        end
                    end

                HIDDEN_START: if (abort_pending || abort_request) begin
                    abort_ack <= 1'b1;
                    abort_pending <= 1'b0;
                    state <= ABORT_WAIT_LOW;
                end else if (hidden_start_valid && hidden_start_ready) begin
                    state <= HIDDEN_WAIT;
                end

                HIDDEN_WAIT: if (hidden_done_valid && hidden_done_ready) begin
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (hidden_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_HIDDEN_READ;
                        state <= COMPLETE;
                    end else begin
                        state <= RMS_START;
                    end
                end

                RMS_START: if (abort_pending || abort_request) begin
                    abort_ack <= 1'b1;
                    abort_pending <= 1'b0;
                    state <= ABORT_WAIT_LOW;
                end else if (rms_start_valid && rms_start_ready) begin
                    state <= RMS_WAIT;
                end

                RMS_WAIT: if (rms_done_valid && rms_done_ready) begin
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (rms_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_RMSNORM;
                        state <= COMPLETE;
                    end else begin
                        state <= QUANT_START;
                    end
                end

                QUANT_START: if (abort_pending || abort_request) begin
                    abort_ack <= 1'b1;
                    abort_pending <= 1'b0;
                    state <= ABORT_WAIT_LOW;
                end else if (quant_start_valid && quant_start_ready) begin
                    state <= QUANT_WAIT;
                end

                QUANT_WAIT: if (quant_done_valid && quant_done_ready) begin
                    if (abort_pending || abort_request) begin
                        abort_ack <= 1'b1;
                        abort_pending <= 1'b0;
                        state <= ABORT_WAIT_LOW;
                    end else if (quant_error) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_QUANTIZER;
                        state <= COMPLETE;
                    end else if (8'(current_group_base) +
                                 8'(current_group_rows) >=
                                 saved_prediction_count) begin
                        state <= COMPLETE;
                    end else begin
                        group_index <= group_index + 4'd1;
                        group_lookup_row <= '0;
                        group_ddr_rows <= '0;
                        group_source_rounds <= '0;
                        group_source_token_indices <= '0;
                        state <= GROUP_LOOKUP_REQUEST;
                    end
                end

                COMPLETE: if (abort_pending || abort_request) begin
                    abort_ack <= 1'b1;
                    abort_pending <= 1'b0;
                    state <= ABORT_WAIT_LOW;
                end else if (done_valid && done_ready) begin
                    state <= IDLE;
                end

                ABORT_WAIT_LOW: if (!abort_request)
                    state <= IDLE;

                default: state <= IDLE;
            endcase
        end
    end

    initial begin
        if (MAX_PREDICTIONS < 16 || MAX_PREDICTIONS > 96 ||
            HIDDEN_ELEMENTS != 4096)
            $error("final_output_sequencer parameter configuration is invalid");
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (hidden_start_valid)
                assert (hidden_start_rows != 0 && hidden_start_rows <= 8)
                    else $error("final-output hidden group row count is invalid");
            if (rms_start_valid)
                assert (rms_start_gamma_bypass)
                    else $error("final-output RMSNorm did not bypass gamma");
            if (quant_start_valid)
                assert (quant_start_rows != 0 && quant_start_rows <= 8)
                    else $error("final-output quantizer group row count is invalid");
            if (abort_pending)
                assert (!dma_request_valid && !hidden_start_valid &&
                        !rms_start_valid && !quant_start_valid)
                    else $error("final-output started new work during abort");
        end
    end
    assert property (@(posedge clk) disable iff (rst)
        descriptor_valid && !descriptor_ready |=>
            descriptor_valid && $stable(descriptor_data))
        else $error("final-output descriptor changed while stalled");
`endif
endmodule

`default_nettype wire
