`default_nettype none

module draft_verify_state_controller #(
    parameter integer MAX_POSITIONS = 32
) (
    input  logic clk,
    input  logic rst,
    input  logic abort_request,
    output logic abort_ack,

    input  logic start_valid,
    output logic start_ready,
    input  logic [5:0] start_row_count,
    input  logic [31:0] start_masked_mask,
    input  logic [31:0] start_tentative_mask,
    input  logic [31:0] start_locked_mask,
    input  logic [31:0] start_mask_token_id,
    input  logic [16:0] start_vocabulary_size,
    input  logic [15:0] start_high_confidence_threshold_bf16,
    input  logic [15:0] start_tail_high_confidence_threshold_bf16,
    input  logic [15:0] start_low_confidence_threshold_bf16,
    input  logic [15:0] start_verify_threshold_bf16,
    input  logic [15:0] start_stability_bonus_bf16,
    input  logic [15:0] start_budget_scale_bf16,
    input  logic [15:0] start_scheduled_quota,
    input  logic [5:0] start_max_handoff_tokens,
    input  logic [15:0] start_remaining_forwards,
    input  logic [15:0] start_step_index,
    input  logic [15:0] start_tail_after_step,
    input  logic [15:0] start_maturity_age,
    input  logic start_tail_threshold_enable,
    input  logic start_tail_all,
    input  logic start_tail_bypass_all,
    input  logic start_tail_bypass_stable_only,
    input  logic [1:0] start_closeout_kind,
    input  logic start_canonical_future,
    input  logic start_source_a_handoff,
    input  logic start_transfer_only,
    input  logic [31:0] start_observed_mask,

    input  logic row_valid,
    output logic row_ready,
    input  logic [4:0] row_index,
    input  logic [31:0] row_token_id,
    input  logic [31:0] row_last_top1,
    input  logic signed [15:0] row_precision_age,
    input  logic [10:0] row_token_position,
    input  logic [1:0] row_origin,
    // Executed precision for observed rows; stored precision otherwise.
    input  logic [3:0] activation_bits,
    input  logic row_cache_valid,
    input  logic row_refresh_required,
    input  logic row_prediction_flag,
    input  logic [31:0] row_candidate_top1,
    input  logic [15:0] row_selected_probability_bf16,
    input  logic [15:0] row_action_confidence_bf16,
    input  logic row_action_confidence_valid,
    input  logic row_source_a,
    input  logic row_source_a_pending,
    input  logic row_suppressed_winner,

    output logic bf16_abort_request,
    input  logic bf16_abort_ack,
    output logic bf16_req_valid,
    input  logic bf16_req_ready,
    output hardware_types_pkg::bf16_request_t bf16_req,
    input  logic bf16_rsp_valid,
    output logic bf16_rsp_ready,
    input  hardware_types_pkg::bf16_response_t bf16_rsp,

    output logic next_token_valid,
    input  logic next_token_ready,
    output logic [4:0] next_token_index,
    output logic [31:0] next_token_token_id,
    output logic [31:0] next_token_last_top1,
    output logic signed [15:0] next_token_precision_age,
    output logic [10:0] next_token_token_position,
    output logic [1:0] next_token_state,
    output logic [1:0] next_token_origin,
    output logic [3:0] next_activation_bits,
    output logic next_token_cache_valid,
    output logic next_token_refresh_required,
    output logic next_token_prediction_flag,
    output logic next_token_source_a_pending,
    output logic [1:0] next_token_change_flags,
    output logic [15:0] next_token_change_confidence_bf16,
    output logic [15:0] next_token_action_confidence_bf16,
    output logic next_token_action_confidence_valid,

    output logic done_valid,
    input  logic done_ready,
    output logic error,
    output logic [7:0] error_id,
    output logic [31:0] confirmed_mask,
    output logic [31:0] remasked_mask,
    output logic [31:0] selected_mask,
    output logic [31:0] direct_locked_mask,
    output logic [31:0] stable_tentative_mask,
    output logic [31:0] fallback_tentative_mask,
    output logic [31:0] mandatory_refresh_mask,
    output logic [31:0] cache_commit_mask,
    output logic [31:0] cache_invalidate_mask,
    output logic [31:0] cache_keep_mask,
    output logic [31:0] token_changed_mask,
    output logic [31:0] tail_closed_mask
);
    import hardware_types_pkg::*;

    localparam logic [1:0] TOKEN_MASKED = 2'd0;
    localparam logic [1:0] TOKEN_TENTATIVE = 2'd1;
    localparam logic [1:0] TOKEN_LOCKED = 2'd2;
    localparam logic [1:0] ORIGIN_NONE = 2'd0;
    localparam logic [1:0] ORIGIN_HIGH = 2'd1;
    localparam logic [1:0] ORIGIN_STABLE = 2'd2;
    localparam logic [1:0] ORIGIN_FALLBACK = 2'd3;
    localparam logic [15:0] A4_DIRECT_THRESHOLD_BF16 = 16'h3f66;

    localparam logic [7:0] ERROR_START = 8'h01;
    localparam logic [7:0] ERROR_ROW = 8'h02;
    localparam logic [7:0] ERROR_DUPLICATE_POSITION = 8'h03;
    localparam logic [7:0] ERROR_BF16_RESPONSE = 8'h04;
    localparam logic [7:0] ERROR_BF16_QUOTA = 8'h05;

    typedef enum logic [4:0] {
        IDLE,
        LOAD_ROWS,
        DUPLICATE_SCAN,
        TAIL_APPLY,
        CLOSEOUT_APPLY,
        CONFIRM_SCAN,
        CLASSIFY_SCAN,
        SCORE_ADD_REQUEST,
        SCORE_ADD_RESPONSE,
        QUOTA_DIVIDE,
        QUOTA_MULTIPLY_REQUEST,
        QUOTA_MULTIPLY_RESPONSE,
        SELECT_SCAN_START,
        SELECT_SCAN,
        SELECT_SCAN_FINISH,
        APPLY_SCAN,
        SEND_ROWS,
        COMPLETE,
        ABORT_DRAIN,
        ABORT_WAIT_LOW
    } state_t;
    state_t state;

    logic [5:0] saved_row_count;
    logic [31:0] saved_masked_mask;
    logic [31:0] saved_tentative_mask;
    logic [31:0] saved_locked_mask;
    logic [31:0] saved_mask_token_id;
    logic [16:0] saved_vocabulary_size;
    logic [15:0] saved_high_confidence_threshold_bf16;
    logic [15:0] saved_tail_high_confidence_threshold_bf16;
    logic [15:0] saved_low_confidence_threshold_bf16;
    logic [15:0] saved_verify_threshold_bf16;
    logic [15:0] saved_stability_bonus;
    logic [15:0] saved_budget_scale;
    logic [5:0] saved_scheduled_quota;
    logic handoff_limit_enabled;
    logic [5:0] handoff_remaining, effective_scheduled_quota;
    assign effective_scheduled_quota = handoff_limit_enabled && handoff_remaining < saved_scheduled_quota ?
        handoff_remaining : saved_scheduled_quota;
    logic [15:0] saved_remaining_forwards;
    logic [15:0] saved_step_index;
    logic [15:0] saved_tail_after_step;
    logic [15:0] saved_maturity_age;
    logic saved_tail_threshold_enable;
    logic saved_tail_all;
    logic saved_tail_bypass_all, saved_tail_bypass_stable_only;
    logic tail_blocked_after_update;
    logic [1:0] saved_closeout_kind;
    logic saved_canonical_future;
    logic saved_source_a_handoff;
    logic saved_transfer_only;
    logic [31:0] saved_observed_mask;

    typedef struct packed {
        logic [31:0] token_id;
        logic [31:0] last_top1;
        logic signed [15:0] precision_age;
        logic [10:0] token_position;
        logic [1:0] token_state;
        logic [1:0] origin;
        logic [3:0] activation_bits;
        logic cache_valid;
        logic refresh_required;
        logic [31:0] candidate_top1;
        // Classification overwrites the input probability with selection score.
        logic [15:0] selected_probability;
        logic [15:0] action_confidence;
        logic action_confidence_valid;
        logic source_a;
        logic source_a_pending;
        // Classification overwrites the input flag with stable observation.
        logic suppressed;
    } position_record_t;

    position_record_t position_memory [0:MAX_POSITIONS-1];
    position_record_t selected_record;
    position_record_t output_record;
    logic [10:0] duplicate_outer_logical;
    logic [10:0] duplicate_inner_logical;

    logic [31:0] high_mask;
    logic [31:0] stable_class_mask;
    logic [31:0] eligible_mask;

    logic [4:0] load_count;
    logic [4:0] scan_index;
    logic [4:0] duplicate_outer;
    logic [4:0] duplicate_inner;
    logic [5:0] active_masked_count;
    logic [5:0] division_remaining;
    logic [5:0] minimum_required;
    logic [5:0] base_quota;
    logic [5:0] maximum_quota;
    logic [5:0] selected_count;
    logic select_fallback;
    logic best_valid;
    logic [4:0] best_index;
    logic [15:0] best_score;
    logic [10:0] best_token_position;
    logic [4:0] output_index;
    logic bf16_outstanding;
    logic abort_needs_bf16;
    logic terminal_error;
    logic [7:0] terminal_error_id;

    logic [31:0] start_valid_position_mask;
    logic start_configuration_error;
    logic row_fire;
    logic bf16_request_fire;
    logic bf16_response_fire;
    logic next_token_fire;
    logic [15:0] active_high_confidence_threshold_bf16;

    function automatic logic [31:0] position_mask(input logic [5:0] count);
        logic [31:0] value;
        begin
            if (count == 6'd32)
                position_mask = 32'hffff_ffff;
            else begin
                value = (32'd1 << count) - 32'd1;
                position_mask = value;
            end
        end
    endfunction

    function automatic logic bf16_finite(input logic [15:0] value);
        bf16_finite = value[14:7] != 8'hff;
    endfunction

    function automatic logic bf16_zero(input logic [15:0] value);
        bf16_zero = value[14:0] == 15'd0;
    endfunction

    function automatic logic bf16_equal(input logic [15:0] lhs,
                                          input logic [15:0] rhs);
        bf16_equal = (bf16_zero(lhs) && bf16_zero(rhs)) || lhs == rhs;
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

    function automatic logic bf16_ge(input logic [15:0] lhs,
                                       input logic [15:0] rhs);
        bf16_ge = bf16_equal(lhs, rhs) || bf16_greater(lhs, rhs);
    endfunction

    function automatic logic bf16_nonnegative_finite(input logic [15:0] value);
        bf16_nonnegative_finite = !value[15] && bf16_finite(value);
    endfunction

    function automatic logic [15:0] integer_bf16(input logic [5:0] value);
        case (value)
            6'd1: integer_bf16 = 16'h3f80;
            6'd2: integer_bf16 = 16'h4000;
            6'd3: integer_bf16 = 16'h4040;
            6'd4: integer_bf16 = 16'h4080;
            6'd5: integer_bf16 = 16'h40a0;
            6'd6: integer_bf16 = 16'h40c0;
            6'd7: integer_bf16 = 16'h40e0;
            6'd8: integer_bf16 = 16'h4100;
            6'd9: integer_bf16 = 16'h4110;
            6'd10: integer_bf16 = 16'h4120;
            6'd11: integer_bf16 = 16'h4130;
            6'd12: integer_bf16 = 16'h4140;
            6'd13: integer_bf16 = 16'h4150;
            6'd14: integer_bf16 = 16'h4160;
            6'd15: integer_bf16 = 16'h4170;
            6'd16: integer_bf16 = 16'h4180;
            6'd17: integer_bf16 = 16'h4188;
            6'd18: integer_bf16 = 16'h4190;
            6'd19: integer_bf16 = 16'h4198;
            6'd20: integer_bf16 = 16'h41a0;
            6'd21: integer_bf16 = 16'h41a8;
            6'd22: integer_bf16 = 16'h41b0;
            6'd23: integer_bf16 = 16'h41b8;
            6'd24: integer_bf16 = 16'h41c0;
            6'd25: integer_bf16 = 16'h41c8;
            6'd26: integer_bf16 = 16'h41d0;
            6'd27: integer_bf16 = 16'h41d8;
            6'd28: integer_bf16 = 16'h41e0;
            6'd29: integer_bf16 = 16'h41e8;
            6'd30: integer_bf16 = 16'h41f0;
            6'd31: integer_bf16 = 16'h41f8;
            default: integer_bf16 = 16'h4200;
        endcase
    endfunction

    function automatic logic [5:0] ceil_positive_bf16_32(
        input logic [15:0] value
    );
        logic [7:0] significand;
        logic signed [9:0] exponent_value;
        logic [7:0] integer_value;
        logic fractional;
        integer shift_count;
        begin
            if (bf16_zero(value)) begin
                ceil_positive_bf16_32 = 6'd0;
            end else if (!bf16_nonnegative_finite(value)) begin
                ceil_positive_bf16_32 = 6'd63;
            end else begin
                exponent_value = $signed({2'b00, value[14:7]}) - 10'sd127;
                significand = value[14:7] == 0 ? {1'b0, value[6:0]} :
                    {1'b1, value[6:0]};
                if (exponent_value < 0) begin
                    ceil_positive_bf16_32 = 6'd1;
                end else if (exponent_value >= 5) begin
                    ceil_positive_bf16_32 = 6'd32;
                end else begin
                    shift_count = 7 - integer'(exponent_value);
                    integer_value = significand >> shift_count;
                    fractional = |(significand & (8'hff >> (8-shift_count)));
                    ceil_positive_bf16_32 = 6'(integer_value) + fractional;
                end
            end
        end
    endfunction

    function automatic logic [3:0] next_token_precision(
        input logic [1:0] row_state_value,
        input logic signed [15:0] age_value,
        input logic [15:0] maturity_age
    );
        if (row_state_value == TOKEN_MASKED)
            next_token_precision = 4'd4;
        else if (row_state_value == TOKEN_TENTATIVE)
            next_token_precision = 4'd8;
        else if (age_value >= 0 && $unsigned(age_value) >= maturity_age)
            next_token_precision = 4'd4;
        else
            next_token_precision = 4'd8;
    endfunction

    assign start_valid_position_mask = position_mask(start_row_count);
    assign start_configuration_error =
        (start_source_a_handoff && !start_canonical_future) ||
        start_max_handoff_tokens > 32 || (start_max_handoff_tokens != 0 && !start_canonical_future) ||
        start_row_count == 0 || start_row_count > 6'(MAX_POSITIONS) ||
        ((start_masked_mask | start_tentative_mask | start_locked_mask) !=
            start_valid_position_mask) ||
        |(start_masked_mask & start_tentative_mask) ||
        |(start_masked_mask & start_locked_mask) ||
        |(start_tentative_mask & start_locked_mask) ||
        start_vocabulary_size == 0 ||
        start_mask_token_id >= {15'd0, start_vocabulary_size} ||
        !bf16_nonnegative_finite(start_high_confidence_threshold_bf16) ||
        !bf16_nonnegative_finite(start_tail_high_confidence_threshold_bf16) ||
        !bf16_nonnegative_finite(start_low_confidence_threshold_bf16) ||
        !bf16_nonnegative_finite(start_verify_threshold_bf16) ||
        !bf16_nonnegative_finite(start_stability_bonus_bf16) ||
        !bf16_nonnegative_finite(start_budget_scale_bf16) ||
        bf16_zero(start_budget_scale_bf16) ||
        !bf16_ge(start_budget_scale_bf16, 16'h3f80) ||
        bf16_greater(start_low_confidence_threshold_bf16, start_high_confidence_threshold_bf16) ||
        start_scheduled_quota > 16'(MAX_POSITIONS) ||
        start_remaining_forwards == 0 || start_maturity_age == 0 ||
        (start_tail_all && |start_masked_mask) ||
        (start_tail_bypass_stable_only && (start_tail_bypass_all || start_tail_all ||
            start_closeout_kind != 0 || start_transfer_only || start_canonical_future)) ||
        start_closeout_kind == 2'd3 ||
        (start_closeout_kind != 0 && (start_tail_all || start_tail_bypass_all || start_canonical_future)) ||
        (start_closeout_kind == 2 && |start_tentative_mask) ||
        (start_transfer_only && (|start_tentative_mask || start_canonical_future ||
            start_closeout_kind != 0 || start_tail_all || start_tail_bypass_all || start_tail_threshold_enable)) ||
        (start_canonical_future && (start_tail_all || start_tail_bypass_all || start_tail_threshold_enable ||
            (start_observed_mask & ~start_valid_position_mask) != 0 || start_budget_scale_bf16 != 16'h3f80));

    assign start_ready = state == IDLE && !abort_request;
    assign row_ready = state == LOAD_ROWS && !terminal_error && !abort_request;
    assign row_fire = row_valid && row_ready;
    assign active_high_confidence_threshold_bf16 = saved_tail_threshold_enable &&
        saved_step_index >= saved_tail_after_step ?
            saved_tail_high_confidence_threshold_bf16 : saved_high_confidence_threshold_bf16;

    always_comb begin : position_reads
        selected_record = '0;
        output_record = '0;
        duplicate_outer_logical = '0;
        duplicate_inner_logical = '0;
        for (integer entry = 0; entry < MAX_POSITIONS; entry++) begin
            if (scan_index == 5'(entry))
                selected_record = position_memory[entry];
            if (output_index == 5'(entry))
                output_record = position_memory[entry];
            if (duplicate_outer == 5'(entry))
                duplicate_outer_logical =
                    position_memory[entry].token_position;
            if (duplicate_inner == 5'(entry))
                duplicate_inner_logical =
                    position_memory[entry].token_position;
        end
    end

    always_comb begin
        bf16_req_valid = 1'b0;
        bf16_req = '0;
        if (state == SCORE_ADD_REQUEST && !abort_request) begin
            bf16_req_valid = 1'b1;
            bf16_req.operation = BF16_VECTOR_ADD;
            bf16_req.values[15:0] = selected_record.action_confidence;
            bf16_req.paired_values[15:0] = saved_stability_bonus;
            bf16_req.lane_mask[0] = 1'b1;
            bf16_req.tag = 16'ha100 | {11'd0, scan_index};
        end else if (state == QUOTA_MULTIPLY_REQUEST && !abort_request) begin
            bf16_req_valid = 1'b1;
            bf16_req.operation = BF16_VECTOR_MULTIPLY;
            bf16_req.values[15:0] = integer_bf16(base_quota);
            bf16_req.factor0_values[15:0] = saved_budget_scale;
            bf16_req.lane_mask[0] = 1'b1;
            bf16_req.tag = 16'ha200;
        end
    end
    assign bf16_request_fire = bf16_req_valid && bf16_req_ready;
    assign bf16_rsp_ready = state == SCORE_ADD_RESPONSE ||
        state == QUOTA_MULTIPLY_RESPONSE || state == ABORT_DRAIN;
    assign bf16_response_fire = bf16_rsp_valid && bf16_rsp_ready;
    assign bf16_abort_request = state == ABORT_DRAIN && abort_needs_bf16;

    assign next_token_valid = state == SEND_ROWS;
    assign next_token_index = output_index;
    assign next_token_token_id = output_record.token_id;
    assign next_token_last_top1 = output_record.last_top1;
    assign next_token_precision_age = output_record.precision_age;
    assign next_token_token_position = output_record.token_position;
    assign next_token_state = output_record.token_state;
    assign next_token_origin = output_record.origin;
    assign next_activation_bits = output_record.activation_bits;
    assign next_token_cache_valid = output_record.cache_valid;
    assign next_token_refresh_required = output_record.refresh_required;
    assign next_token_source_a_pending = output_record.source_a_pending;
    assign next_token_prediction_flag =
        output_record.token_state != TOKEN_LOCKED;
    assign next_token_change_flags = {remasked_mask[output_index], token_changed_mask[output_index]};
    assign next_token_change_confidence_bf16 = remasked_mask[output_index] ? output_record.selected_probability : 16'h3f80;
    assign next_token_action_confidence_bf16 = output_record.action_confidence;
    assign next_token_action_confidence_valid = output_record.action_confidence_valid;
    assign next_token_fire = next_token_valid && next_token_ready;

    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;

    always_ff @(posedge clk) begin
        logic input_error;
        logic keep_supported;
        logic stable_value;
        logic high_value;
        logic stable_low_value;
        logic a4_high_value;
        logic a4_direct_value;
        logic allowed_value;
        logic better_value;
        logic direct_value;
        logic tail_blocked_value;
        logic [5:0] base_value;
        logic [5:0] quota_ceiling;
        logic [31:0] next_token_value;
        logic [31:0] next_last_top1_value;
        logic signed [15:0] next_age_value;
        logic [1:0] next_state_value;
        logic [1:0] next_origin_value;
        logic [3:0] next_bits_value;
        logic next_cache_valid_value;
        logic next_refresh_required_value;
        logic cache_commit_value;
        logic cache_invalidate_value;
        logic cache_keep_value;
        logic position_write_valid;
        logic [4:0] position_write_index;
        position_record_t position_write_data;

        position_write_valid = 1'b0;
        position_write_index = scan_index;
        position_write_data = selected_record;

        if (rst) begin
            state <= IDLE;
            abort_ack <= 1'b0;
            saved_row_count <= '0;
            saved_masked_mask <= '0;
            saved_tentative_mask <= '0;
            saved_locked_mask <= '0;
            saved_mask_token_id <= '0;
            saved_vocabulary_size <= '0;
            saved_high_confidence_threshold_bf16 <= '0;
            saved_tail_high_confidence_threshold_bf16 <= '0;
            saved_low_confidence_threshold_bf16 <= '0;
            saved_verify_threshold_bf16 <= '0;
            saved_stability_bonus <= '0;
            saved_budget_scale <= '0;
            saved_scheduled_quota <= '0; handoff_remaining <= '0; handoff_limit_enabled <= 1'b0;
            saved_canonical_future <= 1'b0;
            saved_source_a_handoff <= 1'b0;
            saved_transfer_only <= 1'b0;
            saved_closeout_kind <= '0;
            saved_observed_mask <= '0;
            saved_remaining_forwards <= '0;
            saved_step_index <= '0;
            saved_tail_after_step <= '0;
            saved_maturity_age <= '0;
            saved_tail_threshold_enable <= 1'b0;
            saved_tail_all <= 1'b0;
            saved_tail_bypass_all <= 1'b0;
            saved_tail_bypass_stable_only <= 1'b0;
            tail_blocked_after_update <= 1'b0;
            confirmed_mask <= '0;
            remasked_mask <= '0;
            selected_mask <= '0;
            direct_locked_mask <= '0;
            stable_tentative_mask <= '0;
            fallback_tentative_mask <= '0;
            mandatory_refresh_mask <= '0;
            cache_commit_mask <= '0;
            cache_invalidate_mask <= '0;
            cache_keep_mask <= '0;
            token_changed_mask <= '0;
            tail_closed_mask <= '0;
            high_mask <= '0;
            stable_class_mask <= '0;
            eligible_mask <= '0;
            load_count <= '0;
            scan_index <= '0;
            duplicate_outer <= '0;
            duplicate_inner <= '0;
            active_masked_count <= '0;
            division_remaining <= '0;
            minimum_required <= '0;
            base_quota <= '0;
            maximum_quota <= '0;
            selected_count <= '0;
            select_fallback <= 1'b0;
            best_valid <= 1'b0;
            best_index <= '0;
            best_score <= '0;
            best_token_position <= '0;
            output_index <= '0;
            bf16_outstanding <= 1'b0;
            abort_needs_bf16 <= 1'b0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
        end else begin
            abort_ack <= 1'b0;

            if (abort_request && state != IDLE && state != ABORT_DRAIN &&
                state != ABORT_WAIT_LOW) begin
                abort_needs_bf16 <= bf16_outstanding;
                state <= bf16_outstanding ? ABORT_DRAIN : ABORT_WAIT_LOW;
                if (!bf16_outstanding)
                    abort_ack <= 1'b1;
            end else begin
                case (state)
                    IDLE: if (start_valid && start_ready) begin
                        saved_row_count <= start_row_count;
                        saved_masked_mask <= start_masked_mask;
                        saved_canonical_future <= start_canonical_future;
                        saved_source_a_handoff <= start_source_a_handoff;
                        saved_transfer_only <= start_transfer_only;
                        saved_closeout_kind <= start_closeout_kind;
                        saved_observed_mask <= start_canonical_future ? start_observed_mask :
                            start_closeout_kind == 1 ? start_tentative_mask | start_locked_mask :
                            start_closeout_kind == 2 ? start_masked_mask | start_locked_mask : start_valid_position_mask;
                        saved_tentative_mask <= start_tentative_mask;
                        saved_locked_mask <= start_locked_mask;
                        saved_mask_token_id <= start_mask_token_id;
                        saved_vocabulary_size <= start_vocabulary_size;
                        saved_high_confidence_threshold_bf16 <= start_high_confidence_threshold_bf16;
                        saved_tail_high_confidence_threshold_bf16 <= start_tail_high_confidence_threshold_bf16;
                        saved_low_confidence_threshold_bf16 <= start_low_confidence_threshold_bf16;
                        saved_verify_threshold_bf16 <= start_verify_threshold_bf16;
                        saved_stability_bonus <= start_stability_bonus_bf16;
                        saved_budget_scale <= start_budget_scale_bf16;
                        saved_scheduled_quota <= start_scheduled_quota[5:0];
                        handoff_limit_enabled <= start_max_handoff_tokens != 0;
                        handoff_remaining <= start_max_handoff_tokens;
                        saved_remaining_forwards <= start_remaining_forwards;
                        saved_step_index <= start_step_index;
                        saved_tail_after_step <= start_tail_after_step;
                        saved_maturity_age <= start_maturity_age;
                        saved_tail_threshold_enable <=
                            start_tail_threshold_enable;
                        saved_tail_all <= start_tail_all;
                        saved_tail_bypass_all <= start_tail_bypass_all;
                        saved_tail_bypass_stable_only <= start_tail_bypass_stable_only;
                        tail_blocked_after_update <= 1'b0;
                        confirmed_mask <= '0;
                        remasked_mask <= '0;
                        selected_mask <= '0;
                        direct_locked_mask <= '0;
                        stable_tentative_mask <= '0;
                        fallback_tentative_mask <= '0;
                        mandatory_refresh_mask <= '0;
                        cache_commit_mask <= '0;
                        cache_invalidate_mask <= '0;
                        cache_keep_mask <= '0;
                        token_changed_mask <= '0;
                        tail_closed_mask <= '0;
                        high_mask <= '0;
                        stable_class_mask <= '0;
                        eligible_mask <= '0;
                        load_count <= '0;
                        active_masked_count <= '0;
                        terminal_error <= start_configuration_error;
                        terminal_error_id <= start_configuration_error ?
                            ERROR_START : 8'd0;
                        state <= start_configuration_error ? COMPLETE : LOAD_ROWS;
                    end

                    LOAD_ROWS: if (row_fire) begin
                        // Count incoming tentative tokens before any confirmations;
                        // A-pending consumes capacity even when not observed.
                        if (saved_tentative_mask[row_index] && handoff_remaining != 0)
                            handoff_remaining <= handoff_remaining - 6'd1;
                        input_error = row_index != load_count[4:0] ||
                            (row_source_a && !saved_canonical_future) ||
                            (row_source_a_pending && (!saved_tentative_mask[row_index] ||
                                (saved_canonical_future && (!saved_source_a_handoff || saved_observed_mask[row_index])))) ||
                            row_token_id >= {15'd0, saved_vocabulary_size} ||
                            (row_last_top1 != 32'hffff_ffff &&
                             row_last_top1 >= {15'd0, saved_vocabulary_size}) ||
                            (activation_bits != 4 && activation_bits != 8) ||
                            row_prediction_flag !=
                                (!saved_locked_mask[row_index] && saved_observed_mask[row_index]) ||
                            (saved_masked_mask[row_index] &&
                             row_token_id != saved_mask_token_id) ||
                            (saved_observed_mask[row_index] && (saved_masked_mask[row_index] ||
                              saved_tentative_mask[row_index]) &&
                             (row_candidate_top1 >=
                                {15'd0, saved_vocabulary_size} ||
                              !bf16_nonnegative_finite(
                                row_selected_probability_bf16) ||
                              !bf16_nonnegative_finite(
                                row_action_confidence_bf16)));
                        position_write_valid = 1'b1;
                        position_write_index = load_count;
                        position_write_data = '0;
                        position_write_data.token_id = row_token_id;
                        position_write_data.last_top1 = row_last_top1;
                        position_write_data.precision_age = row_precision_age;
                        position_write_data.token_position =
                            row_token_position;
                        position_write_data.token_state =
                            saved_masked_mask[row_index] ? TOKEN_MASKED :
                            saved_tentative_mask[row_index] ? TOKEN_TENTATIVE :
                                TOKEN_LOCKED;
                        position_write_data.origin = row_origin;
                        position_write_data.activation_bits = activation_bits;
                        position_write_data.cache_valid = row_cache_valid;
                        position_write_data.refresh_required =
                            row_refresh_required;
                        position_write_data.candidate_top1 = row_candidate_top1;
                        position_write_data.selected_probability =
                            row_selected_probability_bf16;
                        position_write_data.action_confidence =
                            row_action_confidence_bf16;
                        position_write_data.action_confidence_valid = row_action_confidence_valid;
                        position_write_data.source_a = row_source_a;
                        // Becoming current clears pending without confirming the draft.
                        position_write_data.source_a_pending = row_source_a_pending && saved_canonical_future;
                        position_write_data.suppressed = row_suppressed_winner;
                        if (saved_masked_mask[row_index] && (!saved_canonical_future ||
                            (saved_observed_mask[row_index] && bf16_ge(row_action_confidence_bf16, saved_low_confidence_threshold_bf16))))
                            active_masked_count <= active_masked_count + 6'd1;
                        if (input_error) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_ROW;
                            state <= COMPLETE;
                        end else if ({1'b0, load_count} + 6'd1 >=
                                     saved_row_count) begin
                            if (saved_row_count == 1) begin
                                scan_index <= '0;
                                state <= saved_tail_all ? TAIL_APPLY : saved_closeout_kind == 2 ? CLOSEOUT_APPLY : CONFIRM_SCAN;
                            end else begin
                                duplicate_outer <= 5'd1;
                                duplicate_inner <= 5'd0;
                                state <= DUPLICATE_SCAN;
                            end
                        end else begin
                            load_count <= load_count + 5'd1;
                        end
                    end

                    DUPLICATE_SCAN: begin
                        if (duplicate_outer_logical ==
                            duplicate_inner_logical) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_DUPLICATE_POSITION;
                            state <= COMPLETE;
                        end else if ({1'b0, duplicate_inner} + 6'd1 <
                                     {1'b0, duplicate_outer}) begin
                            duplicate_inner <= duplicate_inner + 5'd1;
                        end else if ({1'b0, duplicate_outer} + 6'd1 <
                                     saved_row_count) begin
                            duplicate_outer <= duplicate_outer + 5'd1;
                            duplicate_inner <= '0;
                        end else begin
                            scan_index <= '0;
                            state <= saved_tail_all ? TAIL_APPLY : saved_closeout_kind == 2 ? CLOSEOUT_APPLY : CONFIRM_SCAN;
                        end
                    end

                    TAIL_APPLY: begin
                        next_state_value = selected_record.token_state;
                        next_age_value = selected_record.precision_age;
                        next_bits_value = selected_record.activation_bits;
                        next_cache_valid_value = selected_record.cache_valid;
                        next_refresh_required_value =
                            selected_record.refresh_required;
                        if (selected_record.token_state == TOKEN_TENTATIVE) begin
                            next_state_value = TOKEN_LOCKED;
                            next_age_value = 16'sd0;
                            next_bits_value = 4'd8;
                            tail_closed_mask[scan_index] <= 1'b1;
                        end
                        if (next_cache_valid_value &&
                                     !next_refresh_required_value) begin
                            cache_keep_mask[scan_index] <= 1'b1;
                        end
                        position_write_valid = 1'b1;
                        position_write_data.token_state = next_state_value;
                        position_write_data.precision_age = next_age_value;
                        position_write_data.activation_bits = next_bits_value;
                        position_write_data.cache_valid = next_cache_valid_value;
                        position_write_data.refresh_required =
                            next_refresh_required_value;
                        if (next_refresh_required_value)
                            mandatory_refresh_mask[scan_index] <= 1'b1;
                        if ({1'b0, scan_index} + 6'd1 >= saved_row_count) begin
                            output_index <= '0;
                            state <= SEND_ROWS;
                        end else begin
                            scan_index <= scan_index + 5'd1;
                        end
                    end

                    CONFIRM_SCAN: begin
                        // Inherited drafts wait for an A8 execution before confirmation.
                        if (saved_tentative_mask[scan_index] && saved_observed_mask[scan_index] &&
                            (saved_canonical_future || selected_record.activation_bits == 4'd8)) begin
                            keep_supported = selected_record.candidate_top1 ==
                                    selected_record.token_id &&
                                !selected_record.suppressed &&
                                bf16_ge(selected_record.selected_probability,
                                    saved_verify_threshold_bf16);
                            position_write_valid = 1'b1;
                            if (saved_closeout_kind == 0 || !keep_supported)
                                position_write_data.last_top1 = 32'hffff_ffff;
                            if (keep_supported) begin
                                position_write_data.token_state = TOKEN_LOCKED;
                                position_write_data.precision_age = 16'sd0;
                                confirmed_mask[scan_index] <= 1'b1;
                                cache_commit_mask[scan_index] <= 1'b1;
                            end else begin
                                position_write_data.token_state = TOKEN_MASKED;
                                position_write_data.token_id =
                                    saved_mask_token_id;
                                position_write_data.precision_age = -16'sd1;
                                position_write_data.origin = ORIGIN_NONE;
                                remasked_mask[scan_index] <= 1'b1;
                                cache_invalidate_mask[scan_index] <= 1'b1;
                                if (selected_record.token_id !=
                                    saved_mask_token_id) begin
                                    token_changed_mask[scan_index] <= 1'b1;
                                    mandatory_refresh_mask[scan_index] <= 1'b1;
                                end
                            end
                        end
                        if ({1'b0, scan_index} + 6'd1 >= saved_row_count) begin
                            scan_index <= '0;
                            state <= saved_closeout_kind == 1 ? CLOSEOUT_APPLY : CLASSIFY_SCAN;
                        end else begin
                            scan_index <= scan_index + 5'd1;
                        end
                    end

                    CLOSEOUT_APPLY: begin
                        next_token_value = selected_record.token_id;
                        next_state_value = selected_record.token_state;
                        next_age_value = selected_record.precision_age;
                        next_cache_valid_value = selected_record.cache_valid;
                        next_refresh_required_value = selected_record.refresh_required;
                        cache_commit_value = confirmed_mask[scan_index];
                        cache_invalidate_value = remasked_mask[scan_index];
                        if (saved_locked_mask[scan_index]) begin
                            if (next_age_value < 16'sh7fff) next_age_value = next_age_value + 16'sd1;
                            if (next_refresh_required_value) begin
                                next_cache_valid_value = 1'b1;
                                next_refresh_required_value = 1'b0;
                            end
                        end
                        if (saved_closeout_kind == 2 && saved_masked_mask[scan_index]) begin
                            next_token_value = selected_record.candidate_top1;
                            next_state_value = TOKEN_LOCKED;
                            next_age_value = 16'sd0;
                            selected_mask[scan_index] <= 1'b1;
                            tail_closed_mask[scan_index] <= 1'b1;
                            if (next_token_value != selected_record.token_id) begin
                                token_changed_mask[scan_index] <= 1'b1;
                                cache_invalidate_value = 1'b1;
                            end
                        end
                        if (cache_commit_value) begin
                            next_cache_valid_value = 1'b1; next_refresh_required_value = 1'b0;
                        end
                        if (cache_invalidate_value) begin
                            next_cache_valid_value = 1'b0; next_refresh_required_value = 1'b1;
                        end
                        mandatory_refresh_mask[scan_index] <= next_refresh_required_value;
                        cache_commit_mask[scan_index] <= cache_commit_value;
                        cache_invalidate_mask[scan_index] <= cache_invalidate_value;
                        cache_keep_mask[scan_index] <= next_cache_valid_value && !next_refresh_required_value && !cache_commit_value;
                        position_write_valid = 1'b1;
                        position_write_data.token_id = next_token_value;
                        position_write_data.token_state = next_state_value;
                        position_write_data.precision_age = next_age_value;
                        position_write_data.activation_bits = next_token_precision(next_state_value, next_age_value, saved_maturity_age);
                        position_write_data.cache_valid = next_cache_valid_value;
                        position_write_data.refresh_required = next_refresh_required_value;
                        if ({1'b0, scan_index} + 6'd1 >= saved_row_count) begin
                            output_index <= '0; state <= SEND_ROWS;
                        end else scan_index <= scan_index + 5'd1;
                    end

                    CLASSIFY_SCAN: begin
                        if (saved_transfer_only) begin
                            position_write_valid = 1'b1;
                            position_write_data.selected_probability = selected_record.action_confidence;
                            eligible_mask[scan_index] <= saved_masked_mask[scan_index];
                            if ({1'b0, scan_index} + 6'd1 >= saved_row_count) begin
                                maximum_quota <= effective_scheduled_quota > active_masked_count ? active_masked_count : effective_scheduled_quota;
                                base_quota <= '0;
                                selected_count <= '0;
                                select_fallback <= 1'b0;
                                state <= SELECT_SCAN_START;
                            end else scan_index <= scan_index + 5'd1;
                        end else if (saved_masked_mask[scan_index] && saved_observed_mask[scan_index]) begin
                            stable_value = selected_record.last_top1 ==
                                selected_record.candidate_top1;
                            high_value = bf16_ge(
                                selected_record.action_confidence,
                                saved_canonical_future ? saved_low_confidence_threshold_bf16 : active_high_confidence_threshold_bf16);
                            stable_low_value = stable_value && bf16_ge(
                                selected_record.action_confidence,
                                saved_low_confidence_threshold_bf16) &&
                                !bf16_ge(selected_record.action_confidence,
                                    saved_high_confidence_threshold_bf16);
                            a4_high_value = high_value &&
                                selected_record.activation_bits == 4;
                            a4_direct_value =
                                selected_record.activation_bits == 4 &&
                                bf16_ge(selected_record.action_confidence,
                                    A4_DIRECT_THRESHOLD_BF16);
                            position_write_valid = 1'b1;
                            position_write_data.suppressed = stable_value;
                            high_mask[scan_index] <=
                                (saved_canonical_future && high_value) ||
                                (high_value && selected_record.activation_bits == 8) ||
                                a4_direct_value;
                            stable_class_mask[scan_index] <=
                                saved_canonical_future ? stable_low_value :
                                (stable_low_value || (a4_high_value && !a4_direct_value));
                            eligible_mask[scan_index] <=
                                saved_canonical_future ? high_value :
                                (high_value && selected_record.activation_bits == 8) ||
                                a4_direct_value || stable_low_value || a4_high_value;
                            if (stable_value) begin
                                state <= SCORE_ADD_REQUEST;
                            end else begin
                                position_write_data.selected_probability =
                                    selected_record.action_confidence;
                                if ({1'b0, scan_index} + 6'd1 >=
                                    saved_row_count) begin
                                    division_remaining <= active_masked_count;
                                    minimum_required <= '0;
                                    state <= QUOTA_DIVIDE;
                                end else begin
                                    scan_index <= scan_index + 5'd1;
                                end
                            end
                        end else if ({1'b0, scan_index} + 6'd1 >=
                                     saved_row_count) begin
                            division_remaining <= active_masked_count;
                            minimum_required <= '0;
                            state <= QUOTA_DIVIDE;
                        end else begin
                            scan_index <= scan_index + 5'd1;
                        end
                    end

                    SCORE_ADD_REQUEST: if (bf16_request_fire) begin
                        bf16_outstanding <= 1'b1;
                        state <= SCORE_ADD_RESPONSE;
                    end

                    SCORE_ADD_RESPONSE: if (bf16_response_fire) begin
                        bf16_outstanding <= 1'b0;
                        if (bf16_rsp.tag !=
                                (16'ha100 | {11'd0, scan_index}) ||
                            !bf16_rsp.lane_mask[0] ||
                            !bf16_nonnegative_finite(bf16_rsp.values[15:0])) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_BF16_RESPONSE;
                            state <= COMPLETE;
                        end else begin
                            position_write_valid = 1'b1;
                            position_write_data.selected_probability =
                                bf16_rsp.values[15:0];
                            if ({1'b0, scan_index} + 6'd1 >=
                                saved_row_count) begin
                                division_remaining <= active_masked_count;
                                minimum_required <= '0;
                                state <= QUOTA_DIVIDE;
                            end else begin
                                scan_index <= scan_index + 5'd1;
                                state <= CLASSIFY_SCAN;
                            end
                        end
                    end

                    QUOTA_DIVIDE: begin
                        if (division_remaining == 0) begin
                            base_value = effective_scheduled_quota;
                            if (base_value < 1)
                                base_value = 6'd1;
                            if (base_value < minimum_required)
                                base_value = minimum_required;
                            base_quota <= base_value;
                            if (saved_canonical_future && effective_scheduled_quota == 0) begin
                                maximum_quota <= '0;
                                scan_index <= '0;
                                state <= APPLY_SCAN;
                            end else state <= QUOTA_MULTIPLY_REQUEST;
                        end else begin
                            minimum_required <= minimum_required + 6'd1;
                            if ({10'd0, division_remaining} >
                                saved_remaining_forwards)
                                division_remaining <= 6'(
                                    {10'd0, division_remaining} -
                                    saved_remaining_forwards);
                            else
                                division_remaining <= '0;
                        end
                    end

                    QUOTA_MULTIPLY_REQUEST: if (bf16_request_fire) begin
                        bf16_outstanding <= 1'b1;
                        state <= QUOTA_MULTIPLY_RESPONSE;
                    end

                    QUOTA_MULTIPLY_RESPONSE: if (bf16_response_fire) begin
                        bf16_outstanding <= 1'b0;
                        quota_ceiling = ceil_positive_bf16_32(
                            bf16_rsp.values[15:0]);
                        if (bf16_rsp.tag != 16'ha200 ||
                            !bf16_rsp.lane_mask[0] || quota_ceiling > 32) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_BF16_QUOTA;
                            state <= COMPLETE;
                        end else begin
                            maximum_quota <= quota_ceiling > active_masked_count ?
                                active_masked_count : quota_ceiling;
                            selected_mask <= '0;
                            selected_count <= '0;
                            select_fallback <= 1'b0;
                            state <= SELECT_SCAN_START;
                        end
                    end

                    SELECT_SCAN_START: begin
                        if ((!select_fallback &&
                             selected_count >= maximum_quota) ||
                            (select_fallback &&
                             (selected_count >= base_quota ||
                              selected_count >= active_masked_count))) begin
                            if (!select_fallback && !saved_canonical_future && !saved_transfer_only) begin
                                select_fallback <= 1'b1;
                            end else begin
                                scan_index <= '0;
                                state <= APPLY_SCAN;
                            end
                        end else begin
                            scan_index <= '0;
                            best_valid <= 1'b0;
                            best_index <= '0;
                            best_score <= '0;
                            best_token_position <= '0;
                            state <= SELECT_SCAN;
                        end
                    end

                    SELECT_SCAN: begin
                        allowed_value = !selected_mask[scan_index] &&
                            (select_fallback ? saved_masked_mask[scan_index] :
                                               eligible_mask[scan_index]);
                        better_value = allowed_value && (!best_valid ||
                            bf16_greater(selected_record.selected_probability,
                                best_score) ||
                            (bf16_equal(selected_record.selected_probability,
                                best_score) &&
                             selected_record.token_position <
                                best_token_position));
                        if (better_value) begin
                            best_valid <= 1'b1;
                            best_index <= scan_index;
                            best_score <=
                                selected_record.selected_probability;
                            best_token_position <=
                                selected_record.token_position;
                        end
                        if ({1'b0, scan_index} + 6'd1 >= saved_row_count)
                            state <= SELECT_SCAN_FINISH;
                        else
                            scan_index <= scan_index + 5'd1;
                    end

                    SELECT_SCAN_FINISH: begin
                        if (best_valid) begin
                            selected_mask[best_index] <= 1'b1;
                            selected_count <= selected_count + 6'd1;
                            state <= SELECT_SCAN_START;
                        end else if (!select_fallback && !saved_canonical_future && !saved_transfer_only) begin
                            select_fallback <= 1'b1;
                            state <= SELECT_SCAN_START;
                        end else begin
                            scan_index <= '0;
                            state <= APPLY_SCAN;
                        end
                    end

                    APPLY_SCAN: begin
                        next_token_value = selected_record.token_id;
                        next_last_top1_value = selected_record.last_top1;
                        next_age_value = selected_record.precision_age;
                        next_state_value = selected_record.token_state;
                        next_origin_value = selected_record.origin;
                        next_cache_valid_value = selected_record.cache_valid;
                        next_refresh_required_value =
                            selected_record.refresh_required;
                        cache_commit_value = cache_commit_mask[scan_index];
                        cache_invalidate_value =
                            cache_invalidate_mask[scan_index];
                        cache_keep_value = cache_keep_mask[scan_index];

                        if (saved_locked_mask[scan_index] && saved_observed_mask[scan_index] &&
                            next_refresh_required_value) begin
                            next_cache_valid_value = 1'b1;
                            next_refresh_required_value = 1'b0;
                        end

                        if (saved_transfer_only && saved_masked_mask[scan_index]) begin
                            cache_keep_value = 1'b1;
                            next_cache_valid_value = 1'b1;
                            next_refresh_required_value = 1'b0;
                            if (selected_mask[scan_index]) begin
                                next_token_value = selected_record.candidate_top1;
                                next_state_value = TOKEN_LOCKED;
                                next_origin_value = ORIGIN_HIGH;
                                next_age_value = 16'sd0;
                                next_last_top1_value = 32'hffff_ffff;
                                direct_locked_mask[scan_index] <= 1'b1;
                            end else begin
                                next_last_top1_value = selected_record.candidate_top1;
                            end
                        end else if (!saved_transfer_only && saved_masked_mask[scan_index] && saved_observed_mask[scan_index]) begin
                            if (selected_mask[scan_index]) begin
                                direct_value =
                                    (saved_canonical_future && high_mask[scan_index]) ||
                                    (selected_record.activation_bits == 4 &&
                                     bf16_ge(
                                         selected_record.action_confidence,
                                         A4_DIRECT_THRESHOLD_BF16)) ||
                                    (high_mask[scan_index] &&
                                     selected_record.activation_bits == 8 &&
                                     (bf16_ge(
                                         selected_record.action_confidence,
                                         saved_high_confidence_threshold_bf16) ||
                                      selected_record.suppressed));
                                if (saved_canonical_future && saved_source_a_handoff && selected_record.source_a)
                                    direct_value = 1'b0;
                                position_write_data.source_a_pending = saved_canonical_future &&
                                    saved_source_a_handoff && selected_record.source_a;
                                next_token_value =
                                    selected_record.candidate_top1;
                                next_last_top1_value = 32'hffff_ffff;
                                cache_invalidate_value = 1'b1;
                                if (next_token_value != selected_record.token_id)
                                    token_changed_mask[scan_index] <= 1'b1;
                                if (direct_value) begin
                                    next_state_value = TOKEN_LOCKED;
                                    next_age_value = 16'sd0;
                                    next_origin_value = ORIGIN_HIGH;
                                    direct_locked_mask[scan_index] <= 1'b1;
                                end else begin
                                    next_state_value = TOKEN_TENTATIVE;
                                    next_age_value = -16'sd1;
                                    if (stable_class_mask[scan_index]) begin
                                        next_origin_value = ORIGIN_STABLE;
                                        stable_tentative_mask[scan_index] <= 1'b1;
                                    end else begin
                                        next_origin_value = ORIGIN_FALLBACK;
                                        fallback_tentative_mask[scan_index] <= 1'b1;
                                    end
                                end
                            end else begin
                                next_last_top1_value =
                                    selected_record.candidate_top1;
                                cache_keep_value = 1'b1;
                                next_cache_valid_value = 1'b1;
                                next_refresh_required_value = 1'b0;
                            end
                        end else if (saved_locked_mask[scan_index]) begin
                            if ((saved_transfer_only || saved_observed_mask[scan_index]) && next_state_value == TOKEN_LOCKED &&
                                next_age_value < 16'sh7fff)
                                next_age_value = next_age_value + 16'sd1;
                        end

                        if (next_state_value != TOKEN_MASKED)
                            next_last_top1_value = 32'hffff_ffff;
                        next_bits_value = saved_transfer_only ? 4'd8 : next_token_precision(next_state_value,
                            next_age_value, saved_maturity_age);
                        if (saved_locked_mask[scan_index] &&
                            next_state_value == TOKEN_LOCKED &&
                            next_cache_valid_value &&
                            !next_refresh_required_value)
                            cache_keep_value = 1'b1;
                        if (cache_commit_value) begin
                            next_cache_valid_value = 1'b1;
                            next_refresh_required_value = 1'b0;
                        end
                        if (cache_invalidate_value) begin
                            next_cache_valid_value = 1'b0;
                            next_refresh_required_value = 1'b1;
                        end
                        if (next_token_value != selected_record.token_id) begin
                            token_changed_mask[scan_index] <= 1'b1;
                            mandatory_refresh_mask[scan_index] <= 1'b1;
                        end
                        if (next_token_value != selected_record.token_id) begin
                            mandatory_refresh_mask[scan_index] <= 1'b1;
                            cache_invalidate_value = 1'b1;
                            cache_commit_value = 1'b0;
                            cache_keep_value = 1'b0;
                            next_cache_valid_value = 1'b0;
                            next_refresh_required_value = 1'b1;
                        end
                        if (next_refresh_required_value)
                            mandatory_refresh_mask[scan_index] <= 1'b1;
                        position_write_valid = 1'b1;
                        position_write_data.token_id = next_token_value;
                        position_write_data.last_top1 = next_last_top1_value;
                        position_write_data.precision_age = next_age_value;
                        position_write_data.token_state = next_state_value;
                        position_write_data.origin = next_origin_value;
                        position_write_data.activation_bits = next_bits_value;
                        position_write_data.cache_valid = next_cache_valid_value;
                        position_write_data.refresh_required =
                            next_refresh_required_value;
                        cache_commit_mask[scan_index] <= cache_commit_value;
                        cache_invalidate_mask[scan_index] <=
                            cache_invalidate_value;
                        cache_keep_mask[scan_index] <= cache_keep_value;
                        // Inherited A4 drafts still need their A8 confirmation;
                        // tail bypass must not close that pending handoff.
                        tail_blocked_value = next_state_value == TOKEN_MASKED ||
                            (!saved_canonical_future && saved_tentative_mask[scan_index] &&
                             selected_record.activation_bits == 4'd4 && next_state_value == TOKEN_TENTATIVE) ||
                            (saved_tail_bypass_stable_only && next_state_value == TOKEN_TENTATIVE &&
                             next_origin_value != ORIGIN_STABLE);
                        if (tail_blocked_value)
                            tail_blocked_after_update <= 1'b1;
                        if ({1'b0, scan_index} + 6'd1 >= saved_row_count) begin
                            output_index <= '0;
                            scan_index <= '0;
                            state <= (saved_tail_bypass_all || saved_tail_bypass_stable_only) &&
                                !tail_blocked_after_update && !tail_blocked_value ?
                                TAIL_APPLY : SEND_ROWS;
                        end else begin
                            scan_index <= scan_index + 5'd1;
                        end
                    end

                    SEND_ROWS: if (next_token_fire) begin
                        if ({1'b0, output_index} + 6'd1 >= saved_row_count)
                            state <= COMPLETE;
                        else
                            output_index <= output_index + 5'd1;
                    end

                    COMPLETE: if (done_valid && done_ready)
                        state <= IDLE;

                    ABORT_DRAIN: begin
                        if (bf16_response_fire)
                            bf16_outstanding <= 1'b0;
                        if (bf16_abort_ack) begin
                            bf16_outstanding <= 1'b0;
                            abort_needs_bf16 <= 1'b0;
                            abort_ack <= 1'b1;
                            state <= ABORT_WAIT_LOW;
                        end
                    end

                    ABORT_WAIT_LOW: if (!abort_request)
                        state <= IDLE;

                    default: state <= IDLE;
                endcase
            end
            for (integer entry = 0; entry < MAX_POSITIONS; entry++) begin
                if (position_write_valid &&
                    position_write_index == 5'(entry))
                    position_memory[entry] <= position_write_data;
            end
        end
    end

    initial begin
        if (MAX_POSITIONS != 32)
            $error("draft_verify_state_controller MAX_POSITIONS must be 32");
        if ($bits(position_record_t) != 169)
            $error("draft_verify_state_controller position record width must be 169");
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (!(bf16_outstanding && bf16_req_valid))
                else $error("PSME issued a BF16 request while one was outstanding");
            assert ((confirmed_mask & remasked_mask) == 0)
                else $error("PSME confirmed and remasked masks overlap");
            assert ((cache_commit_mask & cache_invalidate_mask) == 0)
                else $error("PSME cache commit and invalidate masks overlap");
            if (next_token_valid)
                assert ({1'b0, next_token_index} < saved_row_count)
                    else $error("PSME next-token index exceeds token count");
        end
    end
`endif
endmodule

`default_nettype wire
