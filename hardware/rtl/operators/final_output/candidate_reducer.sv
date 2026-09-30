`default_nettype none

// Reduces an increasing-token BF16 logit stream without storing full logits.
// One accepted beat contains eight prediction tokens by eight adjacent vocabulary entries.
module candidate_reducer (
    input  logic clk,
    input  logic rst,
    input  logic abort_request,
    output logic abort_ack,

    input  logic start_valid,
    output logic start_ready,
    input  logic [7:0] start_row_count,
    input  logic [16:0] start_vocabulary_size,
    input  logic [7:0] start_suppressed_count,

    input  logic logit_valid,
    output logic logit_ready,
    input  logic [6:0] logit_row_base,
    input  logic [16:0] logit_vocab_base,
    input  logic [1023:0] logit_values,
    input  logic [63:0] logit_lane_mask,
    input  logic [135:0] logit_selected_token_ids,

    output logic candidate_state_read_valid,
    input  logic candidate_state_read_ready,
    output logic [3:0] candidate_state_read_row_group,
    output logic [2:0] candidate_state_read_lane_block,
    output logic candidate_state_read_row_half,
    output logic [1:0] candidate_state_read_word,
    output logic [3:0] candidate_state_read_row_mask,
    output logic [15:0] candidate_state_read_tag,
    input  logic candidate_state_read_rsp_valid,
    output logic candidate_state_read_rsp_ready,
    input  logic [511:0] candidate_state_read_rsp_data,
    input  logic [3:0] candidate_state_read_rsp_row_mask,
    input  logic [15:0] candidate_state_read_rsp_tag,

    output logic candidate_state_write_valid,
    input  logic candidate_state_write_ready,
    output logic [3:0] candidate_state_write_row_group,
    output logic [2:0] candidate_state_write_lane_block,
    output logic candidate_state_write_row_half,
    output logic [1:0] candidate_state_write_word,
    output logic [3:0] candidate_state_write_row_mask,
    output logic [511:0] candidate_state_write_data,

    output logic bf16_abort_request,
    input  logic bf16_abort_ack,
    output logic bf16_req_valid,
    input  logic bf16_req_ready,
    output hardware_types_pkg::bf16_request_t bf16_req,
    input  logic bf16_rsp_valid,
    output logic bf16_rsp_ready,
    input  hardware_types_pkg::bf16_response_t bf16_rsp,

    output logic max_req_valid,
    input  logic max_req_ready,
    output hardware_types_pkg::maximum_request_t max_req,
    input  logic max_rsp_valid,
    output logic max_rsp_ready,
    input  hardware_types_pkg::maximum_response_t max_rsp,

    output logic reduction_req_valid,
    input  logic reduction_req_ready,
    output hardware_types_pkg::reduction_request_t reduction_req,
    input  logic reduction_rsp_valid,
    output logic reduction_rsp_ready,
    input  hardware_types_pkg::reduction_response_t reduction_rsp,

    output logic exp_req_valid,
    input  logic exp_req_ready,
    output hardware_types_pkg::softmax_exp_request_t exp_req,
    input  logic exp_rsp_valid,
    output logic exp_rsp_ready,
    input  hardware_types_pkg::softmax_exp_response_t exp_rsp,

    output logic reciprocal_req_valid,
    input  logic reciprocal_req_ready,
    output hardware_types_pkg::softmax_reciprocal_request_t reciprocal_req,
    input  logic reciprocal_rsp_valid,
    output logic reciprocal_rsp_ready,
    input  hardware_types_pkg::softmax_reciprocal_response_t reciprocal_rsp,

    output logic suppressed_read_valid,
    input  logic suppressed_read_ready,
    output logic [7:0] suppressed_read_index,
    input  logic suppressed_rsp_valid,
    output logic suppressed_rsp_ready,
    input  logic [16:0] suppressed_rsp_token_id,

    output logic result_valid,
    input  logic result_ready,
    output logic [6:0] result_row_index,
    output logic [16:0] result_top1_token_id,
    output logic [15:0] result_top_logit_bf16,
    output logic [15:0] result_raw_confidence_bf16,
    output logic [15:0] result_selected_token_logit_bf16,
    output logic [15:0] result_selected_probability_bf16,
    output logic result_suppressed_winner,
    output logic [15:0] result_action_confidence_bf16,

    output logic done_valid,
    input  logic done_ready,
    output logic error,
    output logic [7:0] error_id,
    output logic [63:0] accepted_logit_count,
    output logic [63:0] logit_wait_cycle_count
);
    import hardware_types_pkg::*;

    localparam logic [7:0] ERROR_CONFIGURATION = 8'h01;
    localparam logic [7:0] ERROR_LOGIT_ORDER = 8'h02;
    localparam logic [7:0] ERROR_LOGIT_VALUE = 8'h03;
    localparam logic [7:0] ERROR_SELECTED_TOKEN = 8'h04;
    localparam logic [7:0] ERROR_SRAM_RESPONSE = 8'h05;
    localparam logic [7:0] ERROR_SHARED_RESPONSE = 8'h06;

    typedef enum logic [5:0] {
        IDLE,
        RECEIVE_LOGIT,
        READ_STATE_REQUEST,
        READ_STATE_RESPONSE,
        INITIALIZE_LANE,
        UPDATE_MAX_REQUEST,
        UPDATE_MAX_RESPONSE,
        UPDATE_OLD_DELTA_REQUEST,
        UPDATE_OLD_DELTA_RESPONSE,
        UPDATE_NEW_DELTA_REQUEST,
        UPDATE_NEW_DELTA_RESPONSE,
        UPDATE_OLD_EXP_REQUEST,
        UPDATE_OLD_EXP_RESPONSE,
        UPDATE_NEW_EXP_REQUEST,
        UPDATE_NEW_EXP_RESPONSE,
        UPDATE_SUM_REQUEST,
        UPDATE_SUM_RESPONSE,
        WRITE_STATE,
        MERGE_BLOCK_MAX_REQUEST,
        MERGE_BLOCK_MAX_RESPONSE,
        MERGE_GLOBAL_MAX_REQUEST,
        MERGE_GLOBAL_MAX_RESPONSE,
        MERGE_WINNER_SCAN,
        MERGE_DELTA_REQUEST,
        MERGE_DELTA_RESPONSE,
        MERGE_EXP_REQUEST,
        MERGE_EXP_RESPONSE,
        MERGE_MULTIPLY_REQUEST,
        MERGE_MULTIPLY_RESPONSE,
        MERGE_BLOCK_SUM_REQUEST,
        MERGE_BLOCK_SUM_RESPONSE,
        MERGE_FINAL_SUM_REQUEST,
        MERGE_FINAL_SUM_RESPONSE,
        MERGE_RECIPROCAL_REQUEST,
        MERGE_RECIPROCAL_RESPONSE,
        SELECTED_DELTA_REQUEST,
        SELECTED_DELTA_RESPONSE,
        SELECTED_EXP_REQUEST,
        SELECTED_EXP_RESPONSE,
        SELECTED_MULTIPLY_REQUEST,
        SELECTED_MULTIPLY_RESPONSE,
        SUPPRESSED_REQUEST,
        SUPPRESSED_RESPONSE,
        SEND_RESULT,
        COMPLETE,
        ERROR_DRAIN,
        ABORT_DRAIN,
        ABORT_WAIT_LOW
    } state_t;

    state_t state;
    logic [7:0] saved_row_count;
    logic [16:0] saved_vocabulary_size;
    logic [7:0] saved_suppressed_count;
    logic [4:0] row_group_count;
    logic [13:0] panel_count;
    logic [13:0] expected_panel;
    logic [3:0] expected_row_group;
    logic [1023:0] saved_logit_values;
    logic [63:0] saved_logit_lane_mask;
    logic [7:0] saved_logit_row_mask;
    logic [7:0] saved_logit_column_mask;
    logic [16:0] selected_metadata [0:7];
    logic [2:0] active_lane_block;
    logic [10:0] active_quotient;
    logic [2:0] lane_index;
    logic [2:0] state_slice_index;
    logic state_read_outstanding;
    logic [383:0] working_row_state [0:7];
    logic [47:0] current_lane_state [0:7];
    logic lane_update_valid;
    logic [47:0] lane_update_value [0:7];
    logic [15:0] lane_new_max [0:7];
    logic [15:0] lane_old_delta [0:7];
    logic [15:0] lane_new_delta [0:7];
    logic [15:0] lane_old_exp [0:7];
    logic [15:0] lane_new_exp [0:7];

    logic merge_sum_pass;
    logic merge_collect_max;
    logic [3:0] merge_row_group;
    logic [2:0] merge_lane_block;
    logic [2:0] merge_lane_index;
    logic [15:0] block_value [0:7][0:7];
    logic [15:0] global_maximum [0:7];
    logic [16:0] winner_token [0:7];
    logic [15:0] raw_confidence [0:7];
    logic [15:0] selected_probability [0:7];
    logic [15:0] merge_value [0:7][0:7];
    logic [63:0] merge_product_mask;
    logic [2:0] result_row;
    logic [7:0] suppressed_index;
    logic suppressed_match;

    logic bf16_outstanding;
    logic max_outstanding;
    logic reduction_outstanding;
    logic exp_outstanding;
    logic reciprocal_outstanding;
    logic suppressed_outstanding;
    logic bf16_abort_pending;

    logic [7:0] input_row_mask;
    logic [7:0] input_column_mask;
    logic [2:0] last_valid_lane_block;
    logic [63:0] expected_logit_lane_mask;
    logic [63:0] input_finite_mask;
    logic [7:0] input_selected_id_mask;
    logic input_values_finite;
    logic input_selected_ids_valid;
    logic logit_fire;
    logic state_read_fire;
    logic state_read_response_fire;
    logic state_write_fire;
    logic bf16_request_fire;
    logic bf16_response_fire;
    logic max_request_fire;
    logic max_response_fire;
    logic reduction_request_fire;
    logic reduction_response_fire;
    logic exp_request_fire;
    logic exp_response_fire;
    logic reciprocal_request_fire;
    logic reciprocal_response_fire;
    logic suppressed_request_fire;
    logic suppressed_response_fire;

    function automatic logic bf16_finite(input logic [15:0] value);
        bf16_finite = value[14:7] != 8'hff;
    endfunction

    function automatic logic bf16_equal(input logic [15:0] lhs,
                                          input logic [15:0] rhs);
        bf16_equal = (lhs[14:0] == 0 && rhs[14:0] == 0) || lhs == rhs;
    endfunction

    function automatic logic bf16_greater(input logic [15:0] lhs,
                                            input logic [15:0] rhs);
        if (bf16_equal(lhs, rhs))
            bf16_greater = 1'b0;
        else if (lhs[15] != rhs[15])
            bf16_greater = rhs[15];
        else if (!lhs[15])
            bf16_greater = lhs[14:0] > rhs[14:0];
        else
            bf16_greater = lhs[14:0] < rhs[14:0];
    endfunction

    function automatic logic [47:0] select_lane_state(
        input logic [383:0] row_state,
        input logic [2:0] lane
    );
        case (lane)
            3'd0: select_lane_state = row_state[0 +: 48];
            3'd1: select_lane_state = row_state[48 +: 48];
            3'd2: select_lane_state = row_state[96 +: 48];
            3'd3: select_lane_state = row_state[144 +: 48];
            3'd4: select_lane_state = row_state[192 +: 48];
            3'd5: select_lane_state = row_state[240 +: 48];
            3'd6: select_lane_state = row_state[288 +: 48];
            default: select_lane_state = row_state[336 +: 48];
        endcase
    endfunction

    function automatic logic [15:0] select_lane_maximum(
        input logic [383:0] row_state,
        input logic [2:0] lane
    );
        logic [47:0] lane_state;
        begin
            lane_state = select_lane_state(row_state, lane);
            select_lane_maximum = lane_state[47:32];
        end
    endfunction

    function automatic logic [15:0] select_lane_sum(
        input logic [383:0] row_state,
        input logic [2:0] lane
    );
        logic [47:0] lane_state;
        begin
            lane_state = select_lane_state(row_state, lane);
            select_lane_sum = lane_state[31:16];
        end
    endfunction

    function automatic logic [10:0] select_lane_quotient(
        input logic [383:0] row_state,
        input logic [2:0] lane
    );
        logic [47:0] lane_state;
        begin
            lane_state = select_lane_state(row_state, lane);
            select_lane_quotient = lane_state[10:0];
        end
    endfunction

    function automatic logic [127:0] select_state_word(
        input logic [383:0] row_state,
        input logic [1:0] word_index
    );
        case (word_index)
            2'd0: select_state_word = row_state[0 +: 128];
            2'd1: select_state_word = row_state[128 +: 128];
            default: select_state_word = row_state[256 +: 128];
        endcase
    endfunction

    function automatic logic [16:0] select_selected_metadata(
        input logic [383:0] row_state
    );
        select_selected_metadata = {
            row_state[155 +: 2], row_state[107 +: 5],
            row_state[59 +: 5], row_state[11 +: 5]};
    endfunction

    function automatic logic selected_metadata_valid(
        input logic [383:0] row_state
    );
        selected_metadata_valid = row_state[156];
    endfunction

    function automatic logic [4:0] selected_metadata_chunk(
        input logic [16:0] metadata,
        input logic [2:0] lane
    );
        case (lane)
            3'd0: selected_metadata_chunk = metadata[4:0];
            3'd1: selected_metadata_chunk = metadata[9:5];
            3'd2: selected_metadata_chunk = metadata[14:10];
            3'd3: selected_metadata_chunk = {3'd0, metadata[16:15]};
            default: selected_metadata_chunk = '0;
        endcase
    endfunction

    function automatic logic [15:0] negate_bf16(input logic [15:0] value);
        negate_bf16 = {~value[15], value[14:0]};
    endfunction

    always_comb begin
        input_row_mask = '0;
        input_column_mask = '0;
        expected_logit_lane_mask = '0;
        input_finite_mask = '0;
        input_selected_id_mask = '0;
        for (integer row = 0; row < 8; row = row + 1)
            input_row_mask[row] = {1'b0, logit_row_base} + 8'(row) <
                saved_row_count;
        for (integer column = 0; column < 8; column = column + 1)
            input_column_mask[column] = logit_vocab_base + 17'(column) <
                saved_vocabulary_size;
        for (integer row = 0; row < 8; row = row + 1) begin
            input_selected_id_mask[row] = !input_row_mask[row] ||
                logit_selected_token_ids[row*17 +: 17] <
                    saved_vocabulary_size;
            for (integer column = 0; column < 8; column = column + 1) begin
                expected_logit_lane_mask[row*8+column] =
                    input_row_mask[row] && input_column_mask[column];
                input_finite_mask[row*8+column] =
                    !expected_logit_lane_mask[row*8+column] ||
                    bf16_finite(logit_values[(row*8+column)*16 +: 16]);
            end
        end
        input_values_finite = &input_finite_mask;
        input_selected_ids_valid = &input_selected_id_mask;
    end

    for (genvar row = 0; row < 8; row = row + 1) begin : g_state_rows
        always_comb begin
            current_lane_state[row] = select_lane_state(
                working_row_state[row], lane_index);
        end
    end

    always_comb begin
        lane_update_valid = 1'b0;
        for (integer row = 0; row < 8; row = row + 1)
            lane_update_value[row] = current_lane_state[row];
        if (state == INITIALIZE_LANE) begin
            lane_update_valid = 1'b1;
            for (integer row = 0; row < 8; row = row + 1)
                if (saved_logit_row_mask[row] &&
                    saved_logit_column_mask[lane_index])
                    lane_update_value[row] = {
                        saved_logit_values[
                            (row*8+integer'(lane_index))*16 +: 16],
                        16'h3f80,
                        selected_metadata[row][16] ?
                            selected_metadata_chunk(
                            selected_metadata[row], lane_index) :
                            5'd0,
                        active_quotient};
                else if (active_quotient == 0) begin
                    lane_update_value[row] = '0;
                    if (saved_logit_row_mask[row] &&
                        selected_metadata[row][16] && lane_index < 4)
                        lane_update_value[row][15:11] =
                            selected_metadata_chunk(
                                selected_metadata[row], lane_index);
                end
        end else if (state == UPDATE_SUM_RESPONSE && bf16_response_fire) begin
            lane_update_valid = 1'b1;
            for (integer row = 0; row < 8; row = row + 1)
                if (saved_logit_row_mask[row] &&
                    saved_logit_column_mask[lane_index])
                    lane_update_value[row] = {
                        lane_new_max[row],
                        bf16_rsp.values[row*8*16 +: 16],
                        selected_metadata[row][16] ?
                            selected_metadata_chunk(
                                selected_metadata[row], lane_index) :
                            current_lane_state[row][15:11],
                        bf16_greater(
                            saved_logit_values[
                                (row*8+integer'(lane_index))*16 +: 16],
                            current_lane_state[row][47:32]) ?
                            active_quotient :
                                current_lane_state[row][10:0]};
        end else if (state == UPDATE_MAX_REQUEST &&
                     !saved_logit_column_mask[lane_index]) begin
            lane_update_valid = 1'b1;
            for (integer row = 0; row < 8; row = row + 1)
                if (saved_logit_row_mask[row] &&
                    selected_metadata[row][16] && lane_index < 4)
                    lane_update_value[row][15:11] =
                        selected_metadata_chunk(
                            selected_metadata[row], lane_index);
        end
    end

    assign start_ready = state == IDLE && !abort_request;
    assign last_valid_lane_block = saved_vocabulary_size >= 17'd64 ? 3'd7 :
        3'((saved_vocabulary_size - 1'b1) >> 3);
    assign logit_ready = state == RECEIVE_LOGIT && !abort_request;
    assign logit_fire = logit_valid && logit_ready;

    always_comb begin
        candidate_state_read_valid = 1'b0;
        candidate_state_read_row_group = merge_sum_pass ? merge_row_group :
            expected_row_group;
        candidate_state_read_lane_block = merge_sum_pass ? merge_lane_block :
            active_lane_block;
        candidate_state_read_row_half = state_slice_index >= 3;
        candidate_state_read_word = state_slice_index >= 3 ?
            2'(state_slice_index - 3) : 2'(state_slice_index);
        candidate_state_read_row_mask = candidate_state_read_row_half ?
            saved_logit_row_mask[7:4] : saved_logit_row_mask[3:0];
        candidate_state_read_tag = {13'd0, state_slice_index};
        candidate_state_read_rsp_ready = state == READ_STATE_RESPONSE ||
            state == ERROR_DRAIN || state == ABORT_DRAIN;
        if (state == READ_STATE_REQUEST && !abort_request &&
            (|candidate_state_read_row_mask))
            candidate_state_read_valid = 1'b1;

        candidate_state_write_row_group = expected_row_group;
        candidate_state_write_lane_block = active_lane_block;
        candidate_state_write_row_half = state_slice_index >= 3;
        candidate_state_write_word = state_slice_index >= 3 ?
            2'(state_slice_index - 3) : 2'(state_slice_index);
        candidate_state_write_row_mask = candidate_state_write_row_half ?
            saved_logit_row_mask[7:4] : saved_logit_row_mask[3:0];
        candidate_state_write_valid = state == WRITE_STATE &&
            !abort_request && (|candidate_state_write_row_mask);
        candidate_state_write_data = '0;
        for (integer stripe = 0; stripe < 4; stripe = stripe + 1)
            candidate_state_write_data[stripe*128 +: 128] = select_state_word(
                working_row_state[(candidate_state_write_row_half ? 4 : 0) +
                    stripe], candidate_state_write_word);
    end
    assign state_read_fire = candidate_state_read_valid &&
        candidate_state_read_ready;
    assign state_read_response_fire = candidate_state_read_rsp_valid &&
        candidate_state_read_rsp_ready;
    assign state_write_fire = candidate_state_write_valid &&
        candidate_state_write_ready;

    always_comb begin
        bf16_req_valid = 1'b0;
        bf16_req = '0;
        bf16_rsp_ready = state == ERROR_DRAIN || state == ABORT_DRAIN;
        case (state)
            UPDATE_OLD_DELTA_REQUEST: begin
                bf16_req_valid = !abort_request;
                bf16_req.operation = BF16_VECTOR_ADD;
                for (integer row = 0; row < 8; row = row + 1) begin
                    bf16_req.values[row*8*16 +: 16] =
                        current_lane_state[row][47:32];
                    bf16_req.paired_values[row*8*16 +: 16] =
                        negate_bf16(lane_new_max[row]);
                    bf16_req.lane_mask[row*8] = saved_logit_row_mask[row] &&
                        saved_logit_column_mask[lane_index];
                end
                bf16_req.tag = 16'h0100 | {13'd0, lane_index};
            end
            UPDATE_NEW_DELTA_REQUEST: begin
                bf16_req_valid = !abort_request;
                bf16_req.operation = BF16_VECTOR_ADD;
                for (integer row = 0; row < 8; row = row + 1) begin
                    bf16_req.values[row*8*16 +: 16] =
                        saved_logit_values[(row*8+integer'(lane_index))*16 +: 16];
                    bf16_req.paired_values[row*8*16 +: 16] =
                        negate_bf16(lane_new_max[row]);
                    bf16_req.lane_mask[row*8] = saved_logit_row_mask[row] &&
                        saved_logit_column_mask[lane_index];
                end
                bf16_req.tag = 16'h0200 | {13'd0, lane_index};
            end
            UPDATE_SUM_REQUEST: begin
                bf16_req_valid = !abort_request;
                bf16_req.operation = BF16_VECTOR_MULTIPLY_ADD;
                for (integer row = 0; row < 8; row = row + 1) begin
                    bf16_req.values[row*8*16 +: 16] =
                        current_lane_state[row][31:16];
                    bf16_req.factor0_values[row*8*16 +: 16] =
                        lane_old_exp[row];
                    bf16_req.factor1_values[row*8*16 +: 16] =
                        lane_new_exp[row];
                    bf16_req.lane_mask[row*8] = saved_logit_row_mask[row] &&
                        saved_logit_column_mask[lane_index];
                end
                bf16_req.tag = 16'h0300 | {13'd0, lane_index};
            end
            MERGE_DELTA_REQUEST: begin
                bf16_req_valid = !abort_request;
                bf16_req.operation = BF16_VECTOR_ADD;
                for (integer row = 0; row < 8; row = row + 1)
                    for (integer lane = 0; lane < 8; lane = lane + 1) begin
                        bf16_req.values[(row*8+lane)*16 +: 16] =
                            select_lane_maximum(
                                working_row_state[row], 3'(lane));
                        bf16_req.paired_values[(row*8+lane)*16 +: 16] =
                            negate_bf16(global_maximum[row]);
                        bf16_req.lane_mask[row*8+lane] =
                            saved_logit_row_mask[row] &&
                            ({11'd0, merge_lane_block, 3'(lane)} <
                                saved_vocabulary_size);
                    end
                bf16_req.tag = 16'h0400 | {13'd0, merge_lane_block};
            end
            MERGE_MULTIPLY_REQUEST: begin
                bf16_req_valid = !abort_request;
                bf16_req.operation = BF16_VECTOR_MULTIPLY;
                for (integer row = 0; row < 8; row = row + 1)
                    for (integer lane = 0; lane < 8; lane = lane + 1) begin
                        bf16_req.values[(row*8+lane)*16 +: 16] =
                            select_lane_sum(
                                working_row_state[row], 3'(lane));
                        bf16_req.factor0_values[(row*8+lane)*16 +: 16] =
                            merge_value[row][lane];
                        bf16_req.lane_mask[row*8+lane] =
                            saved_logit_row_mask[row] &&
                            ({11'd0, merge_lane_block, 3'(lane)} <
                                saved_vocabulary_size);
                    end
                bf16_req.tag = 16'h0500 | {13'd0, merge_lane_block};
            end
            SELECTED_DELTA_REQUEST: begin
                bf16_req_valid = !abort_request;
                bf16_req.operation = BF16_VECTOR_ADD;
                for (integer row = 0; row < 8; row = row + 1) begin
                    bf16_req.values[row*8*16 +: 16] =
                        selected_metadata[row][15:0];
                    bf16_req.paired_values[row*8*16 +: 16] =
                        negate_bf16(global_maximum[row]);
                    bf16_req.lane_mask[row*8] = saved_logit_row_mask[row];
                end
                bf16_req.tag = 16'h0600 | {12'd0, merge_row_group};
            end
            SELECTED_MULTIPLY_REQUEST: begin
                bf16_req_valid = !abort_request;
                bf16_req.operation = BF16_VECTOR_MULTIPLY;
                for (integer row = 0; row < 8; row = row + 1) begin
                    bf16_req.values[row*8*16 +: 16] = raw_confidence[row];
                    bf16_req.factor0_values[row*8*16 +: 16] =
                        merge_value[row][0];
                    bf16_req.lane_mask[row*8] = saved_logit_row_mask[row];
                end
                bf16_req.tag = 16'h0700 | {12'd0, merge_row_group};
            end
            UPDATE_OLD_DELTA_RESPONSE,
            UPDATE_NEW_DELTA_RESPONSE,
            UPDATE_SUM_RESPONSE,
            MERGE_DELTA_RESPONSE,
            MERGE_MULTIPLY_RESPONSE,
            SELECTED_DELTA_RESPONSE,
            SELECTED_MULTIPLY_RESPONSE:
                bf16_rsp_ready = 1'b1;
            default: begin end
        endcase
    end
    assign bf16_request_fire = bf16_req_valid && bf16_req_ready;
    assign bf16_response_fire = bf16_rsp_valid && bf16_rsp_ready;

    always_comb begin
        max_req_valid = 1'b0;
        max_req = '0;
        max_rsp_ready = state == UPDATE_MAX_RESPONSE ||
            state == MERGE_BLOCK_MAX_RESPONSE ||
            state == MERGE_GLOBAL_MAX_RESPONSE ||
            state == ERROR_DRAIN || state == ABORT_DRAIN;
        if (state == UPDATE_MAX_REQUEST) begin
            max_req_valid = !abort_request &&
                saved_logit_column_mask[lane_index];
            for (integer row = 0; row < 8; row = row + 1) begin
                max_req.values[(row*8)*16 +: 16] =
                    current_lane_state[row][47:32];
                max_req.values[(row*8+1)*16 +: 16] =
                    saved_logit_values[(row*8+integer'(lane_index))*16 +: 16];
                max_req.lane_mask[row*8 +: 2] =
                    {2{saved_logit_row_mask[row] &&
                        saved_logit_column_mask[lane_index]}};
            end
            max_req.tag = 16'h1000 | {13'd0, lane_index};
        end else if (state == MERGE_BLOCK_MAX_REQUEST) begin
            max_req_valid = !abort_request;
            for (integer row = 0; row < 8; row = row + 1)
                for (integer lane = 0; lane < 8; lane = lane + 1) begin
                    max_req.values[(row*8+lane)*16 +: 16] =
                        select_lane_maximum(
                            working_row_state[row], 3'(lane));
                    max_req.lane_mask[row*8+lane] = saved_logit_row_mask[row] &&
                        ({11'd0, merge_lane_block, 3'(lane)} <
                            saved_vocabulary_size);
                end
            max_req.tag = 16'h1100 | {13'd0, merge_lane_block};
        end else if (state == MERGE_GLOBAL_MAX_REQUEST) begin
            max_req_valid = !abort_request;
            for (integer row = 0; row < 8; row = row + 1)
                for (integer block = 0; block < 8; block = block + 1) begin
                    max_req.values[(row*8+block)*16 +: 16] =
                        block_value[row][block];
                    max_req.lane_mask[row*8+block] = saved_logit_row_mask[row] &&
                        (17'(block*8) < saved_vocabulary_size);
                end
            max_req.tag = 16'h1200 | {12'd0, merge_row_group};
        end
    end
    assign max_request_fire = max_req_valid && max_req_ready;
    assign max_response_fire = max_rsp_valid && max_rsp_ready;

    always_comb begin
        exp_req_valid = 1'b0;
        exp_req = '0;
        exp_rsp_ready = state == UPDATE_OLD_EXP_RESPONSE ||
            state == UPDATE_NEW_EXP_RESPONSE ||
            state == MERGE_EXP_RESPONSE ||
            state == SELECTED_EXP_RESPONSE ||
            state == ERROR_DRAIN || state == ABORT_DRAIN;
        if (state == UPDATE_OLD_EXP_REQUEST ||
            state == UPDATE_NEW_EXP_REQUEST) begin
            exp_req_valid = !abort_request;
            for (integer row = 0; row < 8; row = row + 1) begin
                exp_req.delta_values[row*8*16 +: 16] =
                    state == UPDATE_OLD_EXP_REQUEST ?
                        lane_old_delta[row] : lane_new_delta[row];
                exp_req.lane_mask[row*8] = saved_logit_row_mask[row] &&
                    saved_logit_column_mask[lane_index];
            end
            exp_req.tag = state == UPDATE_OLD_EXP_REQUEST ?
                (16'h2000 | {13'd0, lane_index}) :
                (16'h2100 | {13'd0, lane_index});
        end else if (state == MERGE_EXP_REQUEST) begin
            exp_req_valid = !abort_request;
            for (integer row = 0; row < 8; row = row + 1)
                for (integer lane = 0; lane < 8; lane = lane + 1) begin
                    exp_req.delta_values[(row*8+lane)*16 +: 16] =
                        merge_value[row][lane];
                    exp_req.lane_mask[row*8+lane] = saved_logit_row_mask[row] &&
                        ({11'd0, merge_lane_block, 3'(lane)} <
                            saved_vocabulary_size);
                end
            exp_req.tag = 16'h2200 | {13'd0, merge_lane_block};
        end else if (state == SELECTED_EXP_REQUEST) begin
            exp_req_valid = !abort_request;
            for (integer row = 0; row < 8; row = row + 1) begin
                exp_req.delta_values[row*8*16 +: 16] =
                    merge_value[row][0];
                exp_req.lane_mask[row*8] = saved_logit_row_mask[row];
            end
            exp_req.tag = 16'h2300 | {12'd0, merge_row_group};
        end
    end
    assign exp_request_fire = exp_req_valid && exp_req_ready;
    assign exp_response_fire = exp_rsp_valid && exp_rsp_ready;

    always_comb begin
        reduction_req_valid = 1'b0;
        reduction_req = '0;
        reduction_rsp_ready = state == MERGE_BLOCK_SUM_RESPONSE ||
            state == MERGE_FINAL_SUM_RESPONSE || state == ERROR_DRAIN ||
            state == ABORT_DRAIN;
        if (state == MERGE_BLOCK_SUM_REQUEST) begin
            reduction_req_valid = !abort_request;
            for (integer row = 0; row < 8; row = row + 1)
                for (integer lane = 0; lane < 8; lane = lane + 1)
                    reduction_req.values[(row*8+lane)*16 +: 16] =
                        merge_value[row][lane];
            reduction_req.lane_mask = merge_product_mask;
            reduction_req.tag = 16'h3000 | {13'd0, merge_lane_block};
        end else if (state == MERGE_FINAL_SUM_REQUEST) begin
            reduction_req_valid = !abort_request;
            for (integer row = 0; row < 8; row = row + 1)
                for (integer block = 0; block < 8; block = block + 1) begin
                    reduction_req.values[(row*8+block)*16 +: 16] =
                        block_value[row][block];
                    reduction_req.lane_mask[row*8+block] =
                        saved_logit_row_mask[row] &&
                        (17'(block*8) < saved_vocabulary_size);
                end
            reduction_req.tag = 16'h3100 | {12'd0, merge_row_group};
        end
    end
    assign reduction_request_fire = reduction_req_valid && reduction_req_ready;
    assign reduction_response_fire = reduction_rsp_valid &&
        reduction_rsp_ready;

    always_comb begin
        reciprocal_req_valid = state == MERGE_RECIPROCAL_REQUEST &&
            !abort_request;
        reciprocal_req = '0;
        reciprocal_req.row_mask = saved_logit_row_mask;
        reciprocal_req.tag = 16'h4000 | {12'd0, merge_row_group};
        for (integer row = 0; row < 8; row = row + 1)
            reciprocal_req.sum_values[row*16 +: 16] = block_value[row][0];
        reciprocal_rsp_ready = state == MERGE_RECIPROCAL_RESPONSE ||
            state == ERROR_DRAIN || state == ABORT_DRAIN;
    end
    assign reciprocal_request_fire = reciprocal_req_valid &&
        reciprocal_req_ready;
    assign reciprocal_response_fire = reciprocal_rsp_valid &&
        reciprocal_rsp_ready;

    assign suppressed_read_valid = state == SUPPRESSED_REQUEST &&
        !abort_request;
    assign suppressed_read_index = suppressed_index;
    assign suppressed_rsp_ready = state == SUPPRESSED_RESPONSE ||
        state == ERROR_DRAIN || state == ABORT_DRAIN;
    assign suppressed_request_fire = suppressed_read_valid &&
        suppressed_read_ready;
    assign suppressed_response_fire = suppressed_rsp_valid &&
        suppressed_rsp_ready;

    assign result_valid = state == SEND_RESULT;
    assign result_row_index = {merge_row_group, 3'b000} + 7'(result_row);
    assign result_top1_token_id = winner_token[result_row];
    assign result_top_logit_bf16 = global_maximum[result_row];
    assign result_raw_confidence_bf16 = raw_confidence[result_row];
    assign result_selected_token_logit_bf16 =
        selected_metadata[result_row][15:0];
    assign result_selected_probability_bf16 = selected_probability[result_row];
    assign result_suppressed_winner = suppressed_match;
    assign result_action_confidence_bf16 = suppressed_match ? 16'h0000 :
        raw_confidence[result_row];
    assign done_valid = state == COMPLETE;
    assign bf16_abort_request = bf16_abort_pending;

`ifdef SYNTHESIS
    always_comb begin
        accepted_logit_count = '0;
        logit_wait_cycle_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            saved_row_count <= '0;
            saved_vocabulary_size <= '0;
            saved_suppressed_count <= '0;
            row_group_count <= '0;
            panel_count <= '0;
            expected_panel <= '0;
            expected_row_group <= '0;
            saved_logit_values <= '0;
            saved_logit_lane_mask <= '0;
            saved_logit_row_mask <= '0;
            saved_logit_column_mask <= '0;
            active_lane_block <= '0;
            active_quotient <= '0;
            lane_index <= '0;
            state_slice_index <= '0;
            state_read_outstanding <= 1'b0;
            merge_sum_pass <= 1'b0;
            merge_collect_max <= 1'b0;
            merge_row_group <= '0;
            merge_lane_block <= '0;
            merge_lane_index <= '0;
            merge_product_mask <= '0;
            result_row <= '0;
            suppressed_index <= '0;
            suppressed_match <= 1'b0;
            bf16_outstanding <= 1'b0;
            max_outstanding <= 1'b0;
            reduction_outstanding <= 1'b0;
            exp_outstanding <= 1'b0;
            reciprocal_outstanding <= 1'b0;
            suppressed_outstanding <= 1'b0;
            bf16_abort_pending <= 1'b0;
            abort_ack <= 1'b0;
            error <= 1'b0;
            error_id <= '0;
`ifndef SYNTHESIS
            accepted_logit_count <= '0;
            logit_wait_cycle_count <= '0;
`endif
            for (integer row = 0; row < 8; row = row + 1)
                selected_metadata[row] <= '0;
        end else begin
            abort_ack <= 1'b0;
`ifndef SYNTHESIS
            if (logit_valid && !logit_ready && state != COMPLETE)
                logit_wait_cycle_count <= logit_wait_cycle_count + 1'b1;
`endif

            if (bf16_request_fire) bf16_outstanding <= 1'b1;
            if (bf16_response_fire || bf16_abort_ack)
                bf16_outstanding <= 1'b0;
            if (max_request_fire) max_outstanding <= 1'b1;
            if (max_response_fire) max_outstanding <= 1'b0;
            if (reduction_request_fire) reduction_outstanding <= 1'b1;
            if (reduction_response_fire) reduction_outstanding <= 1'b0;
            if (exp_request_fire) exp_outstanding <= 1'b1;
            if (exp_response_fire) exp_outstanding <= 1'b0;
            if (reciprocal_request_fire) reciprocal_outstanding <= 1'b1;
            if (reciprocal_response_fire) reciprocal_outstanding <= 1'b0;
            if (suppressed_request_fire) suppressed_outstanding <= 1'b1;
            if (suppressed_response_fire) suppressed_outstanding <= 1'b0;
            if (state_read_fire) state_read_outstanding <= 1'b1;
            if (state_read_response_fire) state_read_outstanding <= 1'b0;

            if (abort_request && state != IDLE && state != ABORT_DRAIN &&
                state != ABORT_WAIT_LOW) begin
                state <= ABORT_DRAIN;
                bf16_abort_pending <= 1'b1;
            end else begin
                case (state)
                    IDLE: if (start_valid && start_ready) begin
                        error <= 1'b0;
                        error_id <= '0;
`ifndef SYNTHESIS
                        accepted_logit_count <= '0;
                        logit_wait_cycle_count <= '0;
`endif
                        if (start_row_count == 0 || start_row_count > 128 ||
                            start_vocabulary_size == 0 ||
                            start_vocabulary_size > 17'd126464) begin
                            error <= 1'b1;
                            error_id <= ERROR_CONFIGURATION;
                            state <= COMPLETE;
                        end else begin
                            saved_row_count <= start_row_count;
                            saved_vocabulary_size <= start_vocabulary_size;
                            saved_suppressed_count <= start_suppressed_count;
                            row_group_count <= 5'((start_row_count + 8'd7) >> 3);
                            panel_count <= 14'((start_vocabulary_size + 17'd7) >> 3);
                            expected_panel <= '0;
                            expected_row_group <= '0;
                            state <= RECEIVE_LOGIT;
                        end
                    end
                    RECEIVE_LOGIT: if (logit_fire) begin
                        if (logit_row_base != {expected_row_group, 3'b000} ||
                            logit_vocab_base != {expected_panel, 3'b000} ||
                            logit_lane_mask != expected_logit_lane_mask) begin
                            error <= 1'b1;
                            error_id <= ERROR_LOGIT_ORDER;
                            state <= ERROR_DRAIN;
                        end else if (!input_values_finite) begin
                            error <= 1'b1;
                            error_id <= ERROR_LOGIT_VALUE;
                            state <= ERROR_DRAIN;
                        end else if (!input_selected_ids_valid) begin
                            error <= 1'b1;
                            error_id <= ERROR_SELECTED_TOKEN;
                            state <= ERROR_DRAIN;
                        end else begin
                            saved_logit_values <= logit_values;
                            saved_logit_lane_mask <= logit_lane_mask;
                            saved_logit_row_mask <= input_row_mask;
                            saved_logit_column_mask <= input_column_mask;
                            active_lane_block <= expected_panel[2:0];
                            active_quotient <= expected_panel[13:3];
`ifndef SYNTHESIS
                            accepted_logit_count <= accepted_logit_count + 1'b1;
`endif
                            for (integer row = 0; row < 8; row = row + 1) begin
                                selected_metadata[row] <= '0;
                                if (input_row_mask[row]) begin
                                    for (integer column = 0; column < 8;
                                         column = column + 1)
                                        if (input_column_mask[column] &&
                                            logit_selected_token_ids[row*17 +: 17] ==
                                                logit_vocab_base + 17'(column))
                                            selected_metadata[row] <= {
                                                1'b1,
                                                logit_values[
                                                    (row*8+column)*16 +: 16]};
                                end
                            end
                            lane_index <= '0;
                            state_slice_index <= '0;
                            // Each of the first eight panels starts a distinct
                            // lane block, so there is no prior SRAM state to
                            // read until the panel quotient becomes nonzero.
                            if (expected_panel[13:3] == 0) begin
                                state <= INITIALIZE_LANE;
                            end else begin
                                merge_sum_pass <= 1'b0;
                                state <= READ_STATE_REQUEST;
                            end
                        end
                    end
                    READ_STATE_REQUEST: begin
                        if (!(|candidate_state_read_row_mask)) begin
                            if (state_slice_index == 5) begin
                                lane_index <= '0;
                                merge_lane_index <= '0;
                                if (merge_sum_pass)
                                    state <= merge_collect_max ?
                                        MERGE_BLOCK_MAX_REQUEST :
                                        MERGE_WINNER_SCAN;
                                else if (active_quotient == 0)
                                    state <= INITIALIZE_LANE;
                                else
                                    state <= UPDATE_MAX_REQUEST;
                            end else
                                state_slice_index <= state_slice_index + 1'b1;
                        end else if (state_read_fire)
                            state <= READ_STATE_RESPONSE;
                    end
                    READ_STATE_RESPONSE: if (state_read_response_fire) begin
                        if (candidate_state_read_rsp_tag !=
                                {13'd0, state_slice_index} ||
                            candidate_state_read_rsp_row_mask !=
                                candidate_state_read_row_mask) begin
                            error <= 1'b1;
                            error_id <= ERROR_SRAM_RESPONSE;
                            state <= ERROR_DRAIN;
                        end else begin
                            case (state_slice_index)
                                3'd0:
                                    for (integer stripe = 0; stripe < 4;
                                         stripe = stripe + 1)
                                        working_row_state[stripe][0 +: 128] <=
                                            candidate_state_read_rsp_data[
                                                stripe*128 +: 128];
                                3'd1:
                                    for (integer stripe = 0; stripe < 4;
                                         stripe = stripe + 1)
                                        working_row_state[stripe][128 +: 128] <=
                                            candidate_state_read_rsp_data[
                                                stripe*128 +: 128];
                                3'd2:
                                    for (integer stripe = 0; stripe < 4;
                                         stripe = stripe + 1)
                                        working_row_state[stripe][256 +: 128] <=
                                            candidate_state_read_rsp_data[
                                                stripe*128 +: 128];
                                3'd3:
                                    for (integer stripe = 0; stripe < 4;
                                         stripe = stripe + 1)
                                        working_row_state[4+stripe][0 +: 128] <=
                                            candidate_state_read_rsp_data[
                                                stripe*128 +: 128];
                                3'd4:
                                    for (integer stripe = 0; stripe < 4;
                                         stripe = stripe + 1)
                                        working_row_state[4+stripe][128 +: 128] <=
                                            candidate_state_read_rsp_data[
                                                stripe*128 +: 128];
                                default:
                                    for (integer stripe = 0; stripe < 4;
                                         stripe = stripe + 1)
                                        working_row_state[4+stripe][256 +: 128] <=
                                            candidate_state_read_rsp_data[
                                                stripe*128 +: 128];
                            endcase
                            if (state_slice_index == 5) begin
                                lane_index <= '0;
                                merge_lane_index <= '0;
                                if (merge_sum_pass)
                                    state <= merge_collect_max ?
                                        MERGE_BLOCK_MAX_REQUEST :
                                        MERGE_WINNER_SCAN;
                                else if (active_quotient == 0)
                                    state <= INITIALIZE_LANE;
                                else
                                    state <= UPDATE_MAX_REQUEST;
                            end else begin
                                state_slice_index <= state_slice_index + 1'b1;
                                state <= READ_STATE_REQUEST;
                            end
                        end
                    end
                    INITIALIZE_LANE: begin
                        if (lane_index == 7) begin
                            state_slice_index <= '0;
                            state <= WRITE_STATE;
                        end else
                            lane_index <= lane_index + 1'b1;
                    end
                    UPDATE_MAX_REQUEST: begin
                        if (!saved_logit_column_mask[lane_index]) begin
                            if (lane_index == 7) begin
                                state_slice_index <= '0;
                                state <= WRITE_STATE;
                            end else
                                lane_index <= lane_index + 1'b1;
                        end else if (max_request_fire)
                            state <= UPDATE_MAX_RESPONSE;
                    end
                    UPDATE_MAX_RESPONSE: if (max_response_fire) begin
                        if (max_rsp.tag !=
                                (16'h1000 | {13'd0, lane_index})) begin
                            error <= 1'b1;
                            error_id <= ERROR_SHARED_RESPONSE;
                            state <= ERROR_DRAIN;
                        end else begin
                            for (integer row = 0; row < 8; row = row + 1)
                                lane_new_max[row] <= max_rsp.values[row*16 +: 16];
                            state <= UPDATE_OLD_DELTA_REQUEST;
                        end
                    end
                    UPDATE_OLD_DELTA_REQUEST: if (bf16_request_fire)
                        state <= UPDATE_OLD_DELTA_RESPONSE;
                    UPDATE_OLD_DELTA_RESPONSE: if (bf16_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            lane_old_delta[row] <=
                                bf16_rsp.values[row*8*16 +: 16];
                        state <= UPDATE_NEW_DELTA_REQUEST;
                    end
                    UPDATE_NEW_DELTA_REQUEST: if (bf16_request_fire)
                        state <= UPDATE_NEW_DELTA_RESPONSE;
                    UPDATE_NEW_DELTA_RESPONSE: if (bf16_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            lane_new_delta[row] <=
                                bf16_rsp.values[row*8*16 +: 16];
                        state <= UPDATE_OLD_EXP_REQUEST;
                    end
                    UPDATE_OLD_EXP_REQUEST: if (exp_request_fire)
                        state <= UPDATE_OLD_EXP_RESPONSE;
                    UPDATE_OLD_EXP_RESPONSE: if (exp_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            lane_old_exp[row] <=
                                exp_rsp.values[row*8*16 +: 16];
                        state <= UPDATE_NEW_EXP_REQUEST;
                    end
                    UPDATE_NEW_EXP_REQUEST: if (exp_request_fire)
                        state <= UPDATE_NEW_EXP_RESPONSE;
                    UPDATE_NEW_EXP_RESPONSE: if (exp_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            lane_new_exp[row] <=
                                exp_rsp.values[row*8*16 +: 16];
                        state <= UPDATE_SUM_REQUEST;
                    end
                    UPDATE_SUM_REQUEST: if (bf16_request_fire)
                        state <= UPDATE_SUM_RESPONSE;
                    UPDATE_SUM_RESPONSE: if (bf16_response_fire) begin
                        if (lane_index == 7) begin
                            state_slice_index <= '0;
                            state <= WRITE_STATE;
                        end else begin
                            lane_index <= lane_index + 1'b1;
                            state <= UPDATE_MAX_REQUEST;
                        end
                    end
                    WRITE_STATE: if (state_write_fire ||
                        !(|candidate_state_write_row_mask)) begin
                        if (state_slice_index == 5) begin
                            if ({1'b0, expected_row_group} + 5'd1 ==
                                row_group_count) begin
                                expected_row_group <= '0;
                                if (expected_panel + 1'b1 == panel_count) begin
                                    merge_row_group <= '0;
                                    merge_lane_block <= '0;
                                    merge_sum_pass <= 1'b1;
                                    merge_collect_max <= 1'b1;
                                    saved_logit_row_mask <= saved_row_count >= 8 ?
                                        8'hff : (8'h01 << saved_row_count) - 1'b1;
                                    state_slice_index <= '0;
                                    for (integer row = 0; row < 8; row = row + 1)
                                        for (integer block = 0; block < 8;
                                             block = block + 1) begin
                                            block_value[row][block] <= '0;
                                        end
                                    for (integer row = 0; row < 8; row = row + 1)
                                        selected_metadata[row] <= '0;
                                    state <= READ_STATE_REQUEST;
                                end else begin
                                    expected_panel <= expected_panel + 1'b1;
                                    state <= RECEIVE_LOGIT;
                                end
                            end else begin
                                expected_row_group <= expected_row_group + 1'b1;
                                state <= RECEIVE_LOGIT;
                            end
                        end else
                            state_slice_index <= state_slice_index + 1'b1;
                    end
                    MERGE_BLOCK_MAX_REQUEST: if (max_request_fire)
                        state <= MERGE_BLOCK_MAX_RESPONSE;
                    MERGE_BLOCK_MAX_RESPONSE: if (max_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1) begin
                            block_value[row][merge_lane_block] <=
                                max_rsp.values[row*16 +: 16];
                            if (saved_logit_row_mask[row] &&
                                selected_metadata_valid(
                                    working_row_state[row]))
                                selected_metadata[row] <=
                                    select_selected_metadata(
                                        working_row_state[row]);
                        end
                        if (merge_lane_block == last_valid_lane_block) begin
                            state <= MERGE_GLOBAL_MAX_REQUEST;
                        end else begin
                            merge_lane_block <= merge_lane_block + 1'b1;
                            state_slice_index <= '0;
                            state <= READ_STATE_REQUEST;
                        end
                    end
                    MERGE_GLOBAL_MAX_REQUEST: if (max_request_fire)
                        state <= MERGE_GLOBAL_MAX_RESPONSE;
                    MERGE_GLOBAL_MAX_RESPONSE: if (max_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1) begin
                            global_maximum[row] <= max_rsp.values[row*16 +: 16];
                            winner_token[row] <= 17'h1ffff;
                        end
                        merge_lane_block <= '0;
                        merge_lane_index <= '0;
                        state_slice_index <= '0;
                        merge_collect_max <= 1'b0;
                        state <= READ_STATE_REQUEST;
                    end
                    MERGE_WINNER_SCAN: begin
                        for (integer row = 0; row < 8; row = row + 1)
                            if (saved_logit_row_mask[row] &&
                                ({11'd0, merge_lane_block,
                                    merge_lane_index} < saved_vocabulary_size) &&
                                bf16_equal(
                                    select_lane_maximum(working_row_state[row],
                                        merge_lane_index),
                                    global_maximum[row]) &&
                                ({select_lane_quotient(working_row_state[row],
                                      merge_lane_index),
                                  merge_lane_block, merge_lane_index} <
                                    winner_token[row]))
                                winner_token[row] <= {
                                    select_lane_quotient(working_row_state[row],
                                        merge_lane_index),
                                    merge_lane_block, merge_lane_index};
                        if (merge_lane_index == 7)
                            state <= MERGE_DELTA_REQUEST;
                        else
                            merge_lane_index <= merge_lane_index + 1'b1;
                    end
                    MERGE_DELTA_REQUEST: if (bf16_request_fire)
                        state <= MERGE_DELTA_RESPONSE;
                    MERGE_DELTA_RESPONSE: if (bf16_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            for (integer lane = 0; lane < 8; lane = lane + 1)
                                merge_value[row][lane] <=
                                    bf16_rsp.values[(row*8+lane)*16 +: 16];
                        state <= MERGE_EXP_REQUEST;
                    end
                    MERGE_EXP_REQUEST: if (exp_request_fire)
                        state <= MERGE_EXP_RESPONSE;
                    MERGE_EXP_RESPONSE: if (exp_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            for (integer lane = 0; lane < 8; lane = lane + 1)
                                merge_value[row][lane] <=
                                    exp_rsp.values[(row*8+lane)*16 +: 16];
                        state <= MERGE_MULTIPLY_REQUEST;
                    end
                    MERGE_MULTIPLY_REQUEST: if (bf16_request_fire)
                        state <= MERGE_MULTIPLY_RESPONSE;
                    MERGE_MULTIPLY_RESPONSE: if (bf16_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            for (integer lane = 0; lane < 8; lane = lane + 1)
                                merge_value[row][lane] <=
                                    bf16_rsp.values[(row*8+lane)*16 +: 16];
                        merge_product_mask <= bf16_rsp.lane_mask;
                        state <= MERGE_BLOCK_SUM_REQUEST;
                    end
                    MERGE_BLOCK_SUM_REQUEST: if (reduction_request_fire)
                        state <= MERGE_BLOCK_SUM_RESPONSE;
                    MERGE_BLOCK_SUM_RESPONSE: if (reduction_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            block_value[row][merge_lane_block] <=
                                reduction_rsp.values[row*16 +: 16];
                        if (merge_lane_block == last_valid_lane_block)
                            state <= MERGE_FINAL_SUM_REQUEST;
                        else begin
                            merge_lane_block <= merge_lane_block + 1'b1;
                            merge_lane_index <= '0;
                            state_slice_index <= '0;
                            state <= READ_STATE_REQUEST;
                        end
                    end
                    MERGE_FINAL_SUM_REQUEST: if (reduction_request_fire)
                        state <= MERGE_FINAL_SUM_RESPONSE;
                    MERGE_FINAL_SUM_RESPONSE: if (reduction_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            block_value[row][0] <=
                                reduction_rsp.values[row*16 +: 16];
                        state <= MERGE_RECIPROCAL_REQUEST;
                    end
                    MERGE_RECIPROCAL_REQUEST: if (reciprocal_request_fire)
                        state <= MERGE_RECIPROCAL_RESPONSE;
                    MERGE_RECIPROCAL_RESPONSE: if (reciprocal_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            raw_confidence[row] <=
                                reciprocal_rsp.values[row*16 +: 16];
                        state <= SELECTED_DELTA_REQUEST;
                    end
                    SELECTED_DELTA_REQUEST: if (bf16_request_fire)
                        state <= SELECTED_DELTA_RESPONSE;
                    SELECTED_DELTA_RESPONSE: if (bf16_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            merge_value[row][0] <=
                                bf16_rsp.values[row*8*16 +: 16];
                        state <= SELECTED_EXP_REQUEST;
                    end
                    SELECTED_EXP_REQUEST: if (exp_request_fire)
                        state <= SELECTED_EXP_RESPONSE;
                    SELECTED_EXP_RESPONSE: if (exp_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            merge_value[row][0] <=
                                exp_rsp.values[row*8*16 +: 16];
                        state <= SELECTED_MULTIPLY_REQUEST;
                    end
                    SELECTED_MULTIPLY_REQUEST: if (bf16_request_fire)
                        state <= SELECTED_MULTIPLY_RESPONSE;
                    SELECTED_MULTIPLY_RESPONSE: if (bf16_response_fire) begin
                        for (integer row = 0; row < 8; row = row + 1)
                            selected_probability[row] <=
                                bf16_rsp.values[row*8*16 +: 16];
                        result_row <= '0;
                        suppressed_index <= '0;
                        suppressed_match <= 1'b0;
                        state <= saved_suppressed_count == 0 ? SEND_RESULT :
                            SUPPRESSED_REQUEST;
                    end
                    SUPPRESSED_REQUEST: if (suppressed_request_fire)
                        state <= SUPPRESSED_RESPONSE;
                    SUPPRESSED_RESPONSE: if (suppressed_response_fire) begin
                        if (suppressed_rsp_token_id == winner_token[result_row])
                            suppressed_match <= 1'b1;
                        if (suppressed_index + 1'b1 == saved_suppressed_count)
                            state <= SEND_RESULT;
                        else begin
                            suppressed_index <= suppressed_index + 1'b1;
                            state <= SUPPRESSED_REQUEST;
                        end
                    end
                    SEND_RESULT: if (result_valid && result_ready) begin
                        suppressed_index <= '0;
                        suppressed_match <= 1'b0;
                        if (8'(result_row) + 8'd1 <
                            (saved_row_count - {merge_row_group, 3'b000} >= 8 ?
                                8 : saved_row_count -
                                    {merge_row_group, 3'b000})) begin
                            result_row <= result_row + 1'b1;
                            state <= saved_suppressed_count == 0 ? SEND_RESULT :
                                SUPPRESSED_REQUEST;
                        end else if ({1'b0, merge_row_group} + 5'd1 ==
                            row_group_count) begin
                            state <= COMPLETE;
                        end else begin
                            merge_row_group <= merge_row_group + 1'b1;
                            merge_lane_block <= '0;
                            merge_lane_index <= '0;
                            state_slice_index <= '0;
                            merge_collect_max <= 1'b1;
                            saved_logit_row_mask <=
                                saved_row_count -
                                    ({merge_row_group + 1'b1, 3'b000}) >= 8 ?
                                8'hff : (8'h01 <<
                                    (saved_row_count -
                                        {merge_row_group + 1'b1, 3'b000})) - 1'b1;
                            for (integer row = 0; row < 8; row = row + 1)
                                for (integer block = 0; block < 8;
                                     block = block + 1) begin
                                    block_value[row][block] <= '0;
                                end
                            for (integer row = 0; row < 8; row = row + 1)
                                selected_metadata[row] <= '0;
                            state <= READ_STATE_REQUEST;
                        end
                    end
                    COMPLETE: if (done_valid && done_ready)
                        state <= IDLE;
                    ERROR_DRAIN: begin
                        bf16_abort_pending <= bf16_outstanding;
                        if (!state_read_outstanding && !bf16_outstanding &&
                            !max_outstanding && !reduction_outstanding &&
                            !exp_outstanding && !reciprocal_outstanding &&
                            !suppressed_outstanding &&
                            !candidate_state_read_rsp_valid && !bf16_rsp_valid &&
                            !max_rsp_valid && !reduction_rsp_valid &&
                            !exp_rsp_valid && !reciprocal_rsp_valid &&
                            !suppressed_rsp_valid) begin
                            bf16_abort_pending <= 1'b0;
                            state <= COMPLETE;
                        end
                    end
                    ABORT_DRAIN: begin
                        if (bf16_abort_ack) bf16_abort_pending <= 1'b0;
                        if (!state_read_outstanding && !bf16_outstanding &&
                            !max_outstanding && !reduction_outstanding &&
                            !exp_outstanding && !reciprocal_outstanding &&
                            !suppressed_outstanding &&
                            !candidate_state_read_rsp_valid && !bf16_rsp_valid &&
                            !max_rsp_valid && !reduction_rsp_valid &&
                            !exp_rsp_valid && !reciprocal_rsp_valid &&
                            !suppressed_rsp_valid && !bf16_abort_pending) begin
                            abort_ack <= 1'b1;
                            state <= ABORT_WAIT_LOW;
                        end
                    end
                    ABORT_WAIT_LOW: if (!abort_request)
                        state <= IDLE;
                    default: begin
                        error <= 1'b1;
                        error_id <= ERROR_SHARED_RESPONSE;
                        state <= ERROR_DRAIN;
                    end
                endcase
            end
            if (!abort_request && lane_update_valid) begin
                case (lane_index)
                    3'd0:
                        for (integer row = 0; row < 8; row = row + 1)
                            working_row_state[row][0 +: 48] <=
                                lane_update_value[row];
                    3'd1:
                        for (integer row = 0; row < 8; row = row + 1)
                            working_row_state[row][48 +: 48] <=
                                lane_update_value[row];
                    3'd2:
                        for (integer row = 0; row < 8; row = row + 1)
                            working_row_state[row][96 +: 48] <=
                                lane_update_value[row];
                    3'd3:
                        for (integer row = 0; row < 8; row = row + 1)
                            working_row_state[row][144 +: 48] <=
                                lane_update_value[row];
                    3'd4:
                        for (integer row = 0; row < 8; row = row + 1)
                            working_row_state[row][192 +: 48] <=
                                lane_update_value[row];
                    3'd5:
                        for (integer row = 0; row < 8; row = row + 1)
                            working_row_state[row][240 +: 48] <=
                                lane_update_value[row];
                    3'd6:
                        for (integer row = 0; row < 8; row = row + 1)
                            working_row_state[row][288 +: 48] <=
                                lane_update_value[row];
                    default:
                        for (integer row = 0; row < 8; row = row + 1)
                            working_row_state[row][336 +: 48] <=
                                lane_update_value[row];
                endcase
            end
        end
    end

    initial begin
        if ($bits(start_row_count) != 8 || $bits(start_vocabulary_size) != 17)
            $error("candidate reducer parameter widths are invalid");
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst && state == SELECTED_DELTA_REQUEST)
            for (integer row = 0; row < 8; row = row + 1)
                assert (!saved_logit_row_mask[row] ||
                        selected_metadata[row][16])
                    else $error(
                        "candidate selected logit was not restored for row %0d",
                        row);
    end

    assert property (@(posedge clk) disable iff (rst)
        logit_valid && !logit_ready |=>
            logit_valid && $stable({logit_row_base, logit_vocab_base,
                logit_values, logit_lane_mask, logit_selected_token_ids}))
        else $error("candidate logit payload changed while stalled");
    assert property (@(posedge clk) disable iff (rst)
        result_valid && !result_ready |=>
            result_valid && $stable({result_row_index, result_top1_token_id,
                result_top_logit_bf16, result_raw_confidence_bf16,
                result_selected_token_logit_bf16,
                result_selected_probability_bf16,
                result_suppressed_winner,
                result_action_confidence_bf16}))
        else $error("candidate result changed while stalled");
`endif
endmodule

`default_nettype wire
