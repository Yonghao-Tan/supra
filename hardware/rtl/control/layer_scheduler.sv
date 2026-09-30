`default_nettype none
module layer_scheduler (
    input  logic        clk,
    input  logic        rst,

    input  logic        start_valid,
    output logic        start_ready,
    input  logic [5:0]  start_layer_base,
    input  logic [5:0]  start_layer_count,
    input  logic [10:0] start_step_mask,
    input  logic [10:0] start_cache_release_mask,
    input  logic [10:0] start_qkv_residual_spill_mask,
    input  logic [10:0] start_rms_qkv_prefetch_q_mask,
    input  logic [10:0] start_q_head_preprocess_mask,

    output logic        command_valid,
    input  logic        command_ready,
    output logic [5:0]  command_layer,
    output logic [3:0]  layer_operator_index,
    output logic        command_is_mixed_matmul,
    output logic        command_cache_release,
    output logic        command_qkv_residual_spill,
    output logic        command_rms_qkv_prefetch_q,
    output logic        command_q_head_preprocess,

    input  logic        command_done_valid,
    output logic        command_done_ready,
    input  logic        command_done_error,
    input  logic [15:0] command_done_error_id,

    output logic        done_valid,
    input  logic        done_ready,
    output logic        done_error,
    output logic [15:0] done_error_id
);
    import layer_schedule_pkg::*;

    typedef enum logic [1:0] {IDLE, ISSUE, WAIT_ACTION, COMPLETE} state_t;
    state_t state;
    logic [5:0] last_layer;
    logic [10:0] step_mask;
    logic [10:0] cache_release_mask;
    logic [10:0] qkv_residual_spill_mask;
    logic [10:0] rms_qkv_prefetch_q_mask;
    logic [10:0] q_head_preprocess_mask;

    function automatic logic [3:0] first_enabled_step(
        input logic [10:0] mask
    );
        logic found;
        begin
            first_enabled_step = 4'd0;
            found = 1'b0;
            for (integer step = 0; step < LAYER_OPERATOR_COUNT; step = step + 1) begin
                if (!found && mask[step]) begin
                    first_enabled_step = 4'(step);
                    found = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [3:0] next_enabled_step(
        input logic [10:0] mask,
        input logic [3:0] current
    );
        logic found;
        begin
            next_enabled_step = current;
            found = 1'b0;
            for (integer step = 0; step < LAYER_OPERATOR_COUNT; step = step + 1) begin
                if (!found && step > integer'(current) && mask[step]) begin
                    next_enabled_step = 4'(step);
                    found = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic has_later_step(
        input logic [10:0] mask,
        input logic [3:0] current
    );
        begin
            has_later_step = 1'b0;
            for (integer step = 0; step < LAYER_OPERATOR_COUNT; step = step + 1)
                if (step > integer'(current) && mask[step])
                    has_later_step = 1'b1;
        end
    endfunction

    assign start_ready = state == IDLE;
    assign command_valid = state == ISSUE;
    assign command_done_ready = state == WAIT_ACTION;
    assign done_valid = state == COMPLETE;
    assign command_is_mixed_matmul =
        layer_operator_index == LAYER_OPERATOR_ATTENTION_OUTPUT_MATMUL_RESIDUAL ||
        layer_operator_index == LAYER_OPERATOR_FFN_GATE_MATMUL ||
        layer_operator_index == LAYER_OPERATOR_FFN_UP_MATMUL ||
        layer_operator_index == LAYER_OPERATOR_FFN_DOWN_MATMUL_RESIDUAL;
    assign command_cache_release = layer_operator_index < 4'(LAYER_OPERATOR_COUNT) ?
        cache_release_mask[layer_operator_index] : 1'b0;
    assign command_qkv_residual_spill = layer_operator_index < 4'(LAYER_OPERATOR_COUNT) ?
        qkv_residual_spill_mask[layer_operator_index] : 1'b0;
    assign command_rms_qkv_prefetch_q = layer_operator_index < 4'(LAYER_OPERATOR_COUNT) ?
        rms_qkv_prefetch_q_mask[layer_operator_index] : 1'b0;
    assign command_q_head_preprocess = layer_operator_index < 4'(LAYER_OPERATOR_COUNT) ?
        q_head_preprocess_mask[layer_operator_index] : 1'b0;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            last_layer <= 6'd0;
            step_mask <= '0;
            cache_release_mask <= '0;
            qkv_residual_spill_mask <= '0;
            rms_qkv_prefetch_q_mask <= '0;
            q_head_preprocess_mask <= '0;
            command_layer <= 6'd0;
            layer_operator_index <= 4'd0;
            done_error <= 1'b0;
            done_error_id <= 16'd0;
        end else begin
            case (state)
                IDLE: if (start_valid && start_ready) begin
                    done_error <= 1'b0;
                    done_error_id <= 16'd0;
                    if (start_layer_count < 1 || start_layer_count > 32 ||
                        {1'b0, start_layer_base} +
                            {1'b0, start_layer_count} > 7'd32 ||
                        start_step_mask == 11'd0 ||
                        (start_cache_release_mask & ~start_step_mask) != 11'd0 ||
                        (start_qkv_residual_spill_mask &
                            ~start_step_mask) != 11'd0 ||
                        (start_rms_qkv_prefetch_q_mask &
                            ~start_step_mask) != 11'd0 ||
                        (start_q_head_preprocess_mask &
                            ~start_step_mask) != 11'd0) begin
                        done_error <= 1'b1;
                        done_error_id <= 16'h0101;
                        state <= COMPLETE;
                    end else begin
                        last_layer <= start_layer_base +
                            start_layer_count - 1'b1;
                        step_mask <= start_step_mask;
                        cache_release_mask <= start_cache_release_mask;
                        qkv_residual_spill_mask <=
                            start_qkv_residual_spill_mask;
                        rms_qkv_prefetch_q_mask <=
                            start_rms_qkv_prefetch_q_mask;
                        q_head_preprocess_mask <=
                            start_q_head_preprocess_mask;
                        command_layer <= start_layer_base;
                        layer_operator_index <= first_enabled_step(start_step_mask);
                        state <= ISSUE;
                    end
                end

                ISSUE: if (command_valid && command_ready)
                    state <= WAIT_ACTION;

                WAIT_ACTION: if (command_done_valid && command_done_ready) begin
                    if (command_done_error) begin
                        done_error <= 1'b1;
                        done_error_id <= command_done_error_id;
                        state <= COMPLETE;
                    end else if (!has_later_step(step_mask, layer_operator_index)) begin
                        if (command_layer == last_layer) begin
                            state <= COMPLETE;
                        end else begin
                            command_layer <= command_layer + 1'b1;
                            layer_operator_index <= first_enabled_step(step_mask);
                            state <= ISSUE;
                        end
                    end else begin
                        layer_operator_index <= next_enabled_step(step_mask, layer_operator_index);
                        state <= ISSUE;
                    end
                end

                COMPLETE: if (done_valid && done_ready)
                    state <= IDLE;

                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    logic [14:0] stalled_command;
    logic [16:0] stalled_completion;
    always_ff @(posedge clk) begin
        if (!rst) begin
            if ($past(!rst && command_valid && !command_ready))
                assert (command_valid &&
                    {command_layer, layer_operator_index,
                     command_is_mixed_matmul,
                     command_cache_release,
                     command_qkv_residual_spill,
                     command_rms_qkv_prefetch_q,
                     command_q_head_preprocess} ==
                    stalled_command)
                    else $error("fixed block command changed while stalled");
            if (command_valid && !command_ready)
                stalled_command <= {command_layer, layer_operator_index,
                    command_is_mixed_matmul,
                    command_cache_release,
                    command_qkv_residual_spill,
                    command_rms_qkv_prefetch_q,
                    command_q_head_preprocess};
            if ($past(!rst && done_valid && !done_ready))
                assert (done_valid && {done_error, done_error_id} == stalled_completion)
                    else $error("fixed schedule completion changed while stalled");
            if (done_valid && !done_ready)
                stalled_completion <= {done_error, done_error_id};
        end
    end
`endif
endmodule

`default_nettype wire
