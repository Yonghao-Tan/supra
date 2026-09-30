`default_nettype none

module bf16_vector_pipe #(
    parameter integer LANES = 64,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic                  abort_request,
    output logic                  abort_ack,
    input  logic                  req_valid,
    output logic                  req_ready,
    input  logic [2:0]            req_operation,
    input  logic [LANES*16-1:0]   req_values,
    input  logic [LANES*16-1:0]   req_paired_values,
    input  logic [LANES*16-1:0]   req_factor0_values,
    input  logic [LANES*16-1:0]   req_factor1_values,
    input  logic [LANES-1:0]      req_lane_mask,
    input  logic [TAG_WIDTH-1:0]  req_tag,
    output logic                  rsp_valid,
    input  logic                  rsp_ready,
    output logic [LANES*16-1:0]   rsp_values,
    output logic [LANES-1:0]      rsp_lane_mask,
    output logic [TAG_WIDTH-1:0]  rsp_tag,
    output logic                  idle,
    output logic                  trace_sample_valid,
    output logic [1:0]            trace_sample_stage,
    output logic [LANES*16-1:0]   trace_sample_values,
    output logic [LANES-1:0]      trace_sample_lane_mask,
    output logic [TAG_WIDTH-1:0]  trace_sample_tag
);
    localparam logic [2:0] OP_ADD = 3'd0;
    localparam logic [2:0] OP_MULTIPLY = 3'd1;
    localparam logic [2:0] OP_ROPE = 3'd2;
    localparam logic [2:0] OP_SILU = 3'd3;
    localparam logic [2:0] OP_MUL_ADD = 3'd4;
    localparam logic [2:0] OP_RMSNORM = 3'd5;
    localparam integer META_WIDTH = 3 + TAG_WIDTH;
    localparam integer RMS_FACTOR_WIDTH = LANES*16 + TAG_WIDTH;
    localparam integer BF16_MULTIPLY_INFLIGHT = 3;

    logic [2:0] active_operation;
    logic [15:0] inflight_count;
    logic abort_seen;
    logic phase_accepts_request;
    logic queued_valid, queued_ready;
    logic [2:0] queued_operation;
    logic [LANES*16-1:0] queued_values;
    logic [LANES*16-1:0] queued_paired_values;
    logic [LANES*16-1:0] queued_factor0_values;
    logic [LANES*16-1:0] queued_factor1_values;
    logic [LANES-1:0] queued_lane_mask;
    logic [TAG_WIDTH-1:0] queued_tag;
    logic dispatch_fire;
    logic response_fire;

    logic add_req_valid;
    logic add_req_ready;
    logic [LANES*16-1:0] add_req_lhs;
    logic [LANES*16-1:0] add_req_rhs;
    logic [LANES-1:0] add_req_mask;
    logic [META_WIDTH-1:0] add_req_meta;
    logic add_rsp_valid;
    logic add_rsp_ready;
    logic [LANES*16-1:0] add_rsp_values;
    logic [LANES-1:0] add_rsp_mask;
    logic [META_WIDTH-1:0] add_rsp_meta;

    logic mul0_req_valid;
    logic mul0_req_ready;
    logic [META_WIDTH-1:0] mul0_req_meta;
    logic mul0_rsp_valid;
    logic mul0_rsp_ready;
    logic [LANES*16-1:0] mul0_rsp_values;
    logic [LANES-1:0] mul0_rsp_mask;
    logic [META_WIDTH-1:0] mul0_rsp_meta;
    logic mul1_req_valid;
    logic mul1_req_ready;
    logic [META_WIDTH-1:0] mul1_req_meta;
    logic mul1_rsp_valid;
    logic mul1_rsp_ready;
    logic [LANES*16-1:0] mul1_rsp_values;
    logic [LANES-1:0] mul1_rsp_mask;
    logic [META_WIDTH-1:0] mul1_rsp_meta;

    logic [2:0] mul0_rsp_operation;
    logic [TAG_WIDTH-1:0] mul0_rsp_tag;
    logic [2:0] mul1_rsp_operation;
    logic [TAG_WIDTH-1:0] mul1_rsp_tag;
    logic [2:0] add_rsp_operation;
    logic [TAG_WIDTH-1:0] add_rsp_tag_value;
    logic compound_pair_valid;
    logic compound_pair_ready;
    logic rms_factor_input_valid;
    logic rms_factor_input_ready;
    logic [RMS_FACTOR_WIDTH-1:0] rms_factor_input_data;
    logic rms_factor_output_valid;
    logic rms_factor_output_ready;
    logic [RMS_FACTOR_WIDTH-1:0] rms_factor_output_data;
    logic [$clog2(BF16_MULTIPLY_INFLIGHT):0] rms_factor_occupancy;
    logic [LANES*16-1:0] rms_factor_values;
    logic [TAG_WIDTH-1:0] rms_factor_tag;

    assign queued_valid = req_valid;
    assign queued_operation = req_operation;
    assign queued_values = req_values;
    assign queued_paired_values = req_paired_values;
    assign queued_factor0_values = req_factor0_values;
    assign queued_factor1_values = req_factor1_values;
    assign queued_lane_mask = req_lane_mask;
    assign queued_tag = req_tag;
    assign req_ready = queued_ready && !abort_request;

    assign mul0_req_meta = {queued_operation, queued_tag};
    assign mul1_req_meta = {queued_operation, queued_tag};
    assign {mul0_rsp_operation, mul0_rsp_tag} = mul0_rsp_meta;
    assign {mul1_rsp_operation, mul1_rsp_tag} = mul1_rsp_meta;
    assign {add_rsp_operation, add_rsp_tag_value} = add_rsp_meta;
    assign {rms_factor_tag, rms_factor_values} = rms_factor_output_data;

    assign phase_accepts_request = inflight_count == 0 ||
        queued_operation == active_operation ||
        (inflight_count == 1 && response_fire);
    always_comb begin
        queued_ready = 1'b0;
        if (!abort_request && queued_operation <= OP_RMSNORM &&
            phase_accepts_request) begin
            case (queued_operation)
                OP_ADD: queued_ready = add_req_ready;
                OP_ROPE, OP_SILU, OP_MUL_ADD:
                    queued_ready = mul0_req_ready && mul1_req_ready;
                OP_RMSNORM: queued_ready = mul0_req_ready && rms_factor_input_ready;
                default: queued_ready = mul0_req_ready;
            endcase
        end
    end
    assign dispatch_fire = queued_valid && queued_ready;

    assign mul0_req_valid = queued_valid && phase_accepts_request &&
        !abort_request && queued_operation != OP_ADD &&
        queued_operation <= OP_RMSNORM &&
        ((queued_operation != OP_ROPE && queued_operation != OP_SILU &&
          queued_operation != OP_MUL_ADD) || mul1_req_ready) &&
        (queued_operation != OP_RMSNORM || rms_factor_input_ready);
    assign rms_factor_input_valid = queued_valid && phase_accepts_request &&
        !abort_request && queued_operation == OP_RMSNORM && mul0_req_ready;
    assign rms_factor_input_data = {queued_tag, queued_factor1_values};
    assign mul1_req_valid = (queued_valid && phase_accepts_request &&
        !abort_request &&
        (queued_operation == OP_ROPE || queued_operation == OP_SILU ||
         queued_operation == OP_MUL_ADD) && mul0_req_ready) ||
        (mul0_rsp_valid && mul0_rsp_operation == OP_RMSNORM &&
         rms_factor_output_valid && !abort_request);

    assign compound_pair_valid = mul0_rsp_valid &&
        (mul0_rsp_operation == OP_ROPE || mul0_rsp_operation == OP_SILU ||
         mul0_rsp_operation == OP_MUL_ADD) && mul1_rsp_valid;
    assign compound_pair_ready = add_req_ready;

    always_comb begin
        add_req_valid = 1'b0;
        add_req_lhs = '0;
        add_req_rhs = '0;
        add_req_mask = '0;
        add_req_meta = '0;
        if (queued_valid && phase_accepts_request && !abort_request &&
            queued_operation == OP_ADD) begin
            add_req_valid = 1'b1;
            add_req_lhs = queued_values;
            add_req_rhs = queued_paired_values;
            add_req_mask = queued_lane_mask;
            add_req_meta = {queued_operation, queued_tag};
        end else if (compound_pair_valid) begin
            add_req_valid = 1'b1;
            add_req_lhs = mul0_rsp_values;
            add_req_rhs = mul1_rsp_values;
            add_req_mask = mul0_rsp_mask;
            add_req_meta = mul0_rsp_meta;
        end
    end

    always_comb begin
        mul0_rsp_ready = 1'b0;
        mul1_rsp_ready = mul1_rsp_operation == OP_RMSNORM ? rsp_ready : 1'b0;
        if (mul0_rsp_operation == OP_MULTIPLY) begin
            mul0_rsp_ready = rsp_ready;
        end else if (mul0_rsp_operation == OP_RMSNORM) begin
            mul0_rsp_ready = mul1_req_ready && rms_factor_output_valid;
        end else if (mul0_rsp_operation == OP_ROPE ||
                     mul0_rsp_operation == OP_SILU ||
                     mul0_rsp_operation == OP_MUL_ADD) begin
            mul0_rsp_ready = compound_pair_ready && mul1_rsp_valid;
            mul1_rsp_ready = compound_pair_ready && mul0_rsp_valid;
        end
    end
    assign rms_factor_output_ready = mul0_rsp_valid &&
        mul0_rsp_operation == OP_RMSNORM && mul1_req_ready;

    assign add_rsp_ready = active_operation == OP_MULTIPLY ||
        active_operation == OP_RMSNORM ? 1'b0 : rsp_ready;
    always_comb begin
        rsp_valid = 1'b0;
        rsp_values = '0;
        rsp_lane_mask = '0;
        rsp_tag = '0;
        if (!abort_request && active_operation == OP_MULTIPLY) begin
            rsp_valid = mul0_rsp_valid;
            rsp_values = mul0_rsp_values;
            rsp_lane_mask = mul0_rsp_mask;
            rsp_tag = mul0_rsp_tag;
        end else if (!abort_request && active_operation == OP_RMSNORM) begin
            rsp_valid = mul1_rsp_valid;
            rsp_values = mul1_rsp_values;
            rsp_lane_mask = mul1_rsp_mask;
            rsp_tag = mul1_rsp_tag;
        end else if (!abort_request) begin
            rsp_valid = add_rsp_valid;
            rsp_values = add_rsp_values;
            rsp_lane_mask = add_rsp_mask;
            rsp_tag = add_rsp_tag_value;
        end
    end
    assign response_fire = rsp_valid && rsp_ready;
    assign idle = !abort_request && inflight_count == 0;

`ifdef SYNTHESIS
    always_comb begin
        trace_sample_valid = 1'b0;
        trace_sample_stage = 2'd0;
        trace_sample_values = '0;
        trace_sample_lane_mask = '0;
        trace_sample_tag = '0;
    end
`else
    always_comb begin
        trace_sample_valid = 1'b0;
        trace_sample_stage = 2'd0;
        trace_sample_values = '0;
        trace_sample_lane_mask = '0;
        trace_sample_tag = '0;
        if (mul0_rsp_valid && mul0_rsp_ready) begin
            trace_sample_valid = 1'b1;
            trace_sample_values = mul0_rsp_values;
            trace_sample_lane_mask = mul0_rsp_mask;
            trace_sample_tag = mul0_rsp_tag;
        end else if (mul1_rsp_valid && mul1_rsp_ready) begin
            trace_sample_valid = 1'b1;
            trace_sample_stage = 2'd1;
            trace_sample_values = mul1_rsp_values;
            trace_sample_lane_mask = mul1_rsp_mask;
            trace_sample_tag = mul1_rsp_tag;
        end else if (add_rsp_valid && add_rsp_ready) begin
            trace_sample_valid = 1'b1;
            trace_sample_stage = 2'd2;
            trace_sample_values = add_rsp_values;
            trace_sample_lane_mask = add_rsp_mask;
            trace_sample_tag = add_rsp_tag_value;
        end
    end
`endif

    bf16_add_pipe #(.LANES(LANES), .TAG_WIDTH(META_WIDTH)) shared_add (
        .clk(clk), .rst(rst || abort_request), .req_valid(add_req_valid),
        .req_ready(add_req_ready), .req_lhs(add_req_lhs), .req_rhs(add_req_rhs),
        .req_lane_mask(add_req_mask), .req_tag(add_req_meta), .rsp_valid(add_rsp_valid),
        .rsp_ready(add_rsp_ready), .rsp_values(add_rsp_values),
        .rsp_lane_mask(add_rsp_mask), .rsp_tag(add_rsp_meta));

    bf16_mul_pipe #(.LANES(LANES), .TAG_WIDTH(META_WIDTH)) multiply_left (
        .clk(clk), .rst(rst || abort_request), .req_valid(mul0_req_valid),
        .req_ready(mul0_req_ready), .req_lhs(queued_values),
        .req_rhs(queued_factor0_values), .req_lane_mask(queued_lane_mask),
        .req_tag(mul0_req_meta), .rsp_valid(mul0_rsp_valid),
        .rsp_ready(mul0_rsp_ready), .rsp_values(mul0_rsp_values),
        .rsp_lane_mask(mul0_rsp_mask), .rsp_tag(mul0_rsp_meta));

    bf16_mul_pipe #(.LANES(LANES), .TAG_WIDTH(META_WIDTH)) multiply_right (
        .clk(clk), .rst(rst || abort_request), .req_valid(mul1_req_valid),
        .req_ready(mul1_req_ready),
        .req_lhs(mul0_rsp_valid && mul0_rsp_operation == OP_RMSNORM ?
            mul0_rsp_values :
            (queued_operation == OP_ROPE ? queued_paired_values :
                queued_factor1_values)),
        .req_rhs(mul0_rsp_valid && mul0_rsp_operation == OP_RMSNORM ?
            rms_factor_values :
            (queued_operation == OP_ROPE ? queued_factor1_values :
                {(LANES){16'h3f80}})),
        .req_lane_mask(mul0_rsp_valid && mul0_rsp_operation == OP_RMSNORM ?
            mul0_rsp_mask : queued_lane_mask),
        .req_tag(mul0_rsp_valid && mul0_rsp_operation == OP_RMSNORM ?
            mul0_rsp_meta : mul1_req_meta), .rsp_valid(mul1_rsp_valid),
        .rsp_ready(mul1_rsp_ready), .rsp_values(mul1_rsp_values),
        .rsp_lane_mask(mul1_rsp_mask), .rsp_tag(mul1_rsp_meta));

    // RMSNorm needs two serial BF16 materialization points. One factor entry
    // is retained for each in-flight request in the first multiply pipeline.
    ready_valid_fifo #(
        .DATA_WIDTH(RMS_FACTOR_WIDTH),
        .DEPTH(BF16_MULTIPLY_INFLIGHT)
    ) rms_factor_alignment (
        .clk(clk), .rst(rst || abort_request), .input_valid(rms_factor_input_valid),
        .input_ready(rms_factor_input_ready), .input_data(rms_factor_input_data),
        .output_valid(rms_factor_output_valid), .output_ready(rms_factor_output_ready),
        .output_data(rms_factor_output_data), .occupancy(rms_factor_occupancy));

    always_ff @(posedge clk) begin
        if (rst) begin
            active_operation <= OP_ADD;
            inflight_count <= '0;
            abort_ack <= 1'b0;
            abort_seen <= 1'b0;
        end else if (abort_request) begin
            inflight_count <= '0;
            abort_ack <= !abort_seen;
            abort_seen <= 1'b1;
        end else begin
            abort_ack <= 1'b0;
            abort_seen <= 1'b0;
            if (dispatch_fire &&
                (inflight_count == 0 ||
                 (inflight_count == 1 && response_fire)))
                active_operation <= queued_operation;
            case ({dispatch_fire, response_fire})
                2'b10: inflight_count <= inflight_count + 1'b1;
                2'b01: inflight_count <= inflight_count - 1'b1;
                default: begin end
            endcase
        end
    end

    initial begin
        if (LANES < 1 || TAG_WIDTH < 1)
            $error("bf16_vector_pipe requires positive LANES and TAG_WIDTH");
    end

`ifndef SYNTHESIS
    logic stalled;
    logic [LANES*16-1:0] held_values;
    logic [LANES-1:0] held_lane_mask;
    logic [TAG_WIDTH-1:0] held_tag;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled <= 1'b0;
            held_values <= '0;
            held_lane_mask <= '0;
            held_tag <= '0;
        end else begin
            if (dispatch_fire && inflight_count != 0 &&
                !(inflight_count == 1 && response_fire))
                assert (queued_operation == active_operation)
                    else $error("bf16_vector_pipe changed operation before drain");
            if (mul0_rsp_valid &&
                (mul0_rsp_operation == OP_ROPE ||
                 mul0_rsp_operation == OP_SILU ||
                 mul0_rsp_operation == OP_MUL_ADD) && mul1_rsp_valid)
                assert (mul0_rsp_tag == mul1_rsp_tag &&
                        mul0_rsp_mask == mul1_rsp_mask &&
                        mul1_rsp_operation == mul0_rsp_operation)
                    else $error("bf16_vector_pipe compound product metadata mismatch");
            if (mul0_rsp_valid && mul0_rsp_operation == OP_RMSNORM &&
                rms_factor_output_valid)
                assert (mul0_rsp_tag == rms_factor_tag)
                    else $error("bf16_vector_pipe RMSNorm gamma tag mismatch");
            if (stalled && !abort_request)
                assert (rsp_valid && rsp_values == held_values &&
                        rsp_lane_mask == held_lane_mask && rsp_tag == held_tag)
                    else $error("bf16_vector_pipe changed a stalled response");
            stalled <= rsp_valid && !rsp_ready && !abort_request;
            if (rsp_valid && !rsp_ready) begin
                held_values <= rsp_values;
                held_lane_mask <= rsp_lane_mask;
                held_tag <= rsp_tag;
            end
        end
    end
`endif
endmodule

`default_nettype wire
