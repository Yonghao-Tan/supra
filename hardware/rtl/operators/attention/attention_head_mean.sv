`default_nettype none

// One group contains eight query/key pairs, streamed in head order 0..31.
// Four 8xFP32 partial sums (128 B) preserve the production mean reduction.
// The add response forwards into a same-partial request four cycles later.
module attention_head_mean (
    input logic clk, rst, abort_request,
    input logic start_valid,
    output logic start_ready,
    input logic [7:0] start_lane_mask,
    input logic [15:0] start_tag,
    input logic values_valid,
    output logic values_ready,
    input logic [127:0] values_bf16,
    output logic result_valid,
    input logic result_ready,
    output logic [255:0] result_mean_fp32,
    output logic [127:0] result_mean_bf16,
    output logic [63:0] result_score_u8,
    output logic [7:0] result_lane_mask,
    output logic [15:0] result_tag
);
    typedef enum logic [3:0] {
        IDLE, HEADS, HEAD_DRAIN, SUM_REQUEST, SUM_WAIT, DIVIDE_MEAN,
        SCORE_PRODUCT, SCORE_ROUND, OUTPUT_ROUND, COMPLETE
    } state_t;
    state_t state;
    logic [255:0] partial_sum [0:3];
    logic [255:0] head_sum;
    logic [5:0] head_request, head_response;
    logic [1:0] sum_index;
    logic [7:0] lane_mask;
    logic add_req_valid, add_req_ready, add_rsp_valid;
    logic [255:0] add_lhs, add_rhs, add_values;
    logic [2:0] add_tag, add_result_tag;

    assign start_ready = state == IDLE && !abort_request;
    assign values_ready = state == HEADS && add_req_ready && !abort_request;
    assign result_valid = state == COMPLETE && !abort_request;
    assign result_lane_mask = lane_mask;
    assign add_req_valid = !abort_request &&
        ((state == HEADS && values_valid) || state == SUM_REQUEST);
    assign add_tag = state == HEADS ? {1'b0, head_request[1:0]} : {1'b1, sum_index};
    always_comb begin
        add_lhs = state == HEADS ? partial_sum[head_request[1:0]] : head_sum;
        if (state == HEADS && head_request < 6'd4) add_lhs = '0;
        else if (state == HEADS && add_rsp_valid && add_result_tag == add_tag)
            add_lhs = add_values;
        add_rhs = partial_sum[sum_index];
        if (state == HEADS)
            for (integer lane = 0; lane < 8; lane++)
                add_rhs[lane*32 +: 32] = {values_bf16[lane*16 +: 16], 16'd0};
    end

    fp32_positive_add_pipe #(.LANES(8), .TAG_WIDTH(3)) add (
        .clk, .rst(rst || abort_request), .req_valid(add_req_valid), .req_ready(add_req_ready),
        .req_lhs(add_lhs), .req_rhs(add_rhs), .req_lane_mask(lane_mask), .req_tag(add_tag),
        .rsp_valid(add_rsp_valid), .rsp_ready(1'b1), .rsp_values(add_values),
        .rsp_lane_mask(), .rsp_tag(add_result_tag)
    );

    always_ff @(posedge clk) begin
        if (rst || abort_request) begin
            state <= IDLE; head_request <= '0; head_response <= '0;
            sum_index <= '0; lane_mask <= '0; result_tag <= '0; head_sum <= '0;
        end else begin
            if (add_rsp_valid && !add_result_tag[2]) begin
                partial_sum[add_result_tag[1:0]] <= add_values;
                head_response <= head_response + 6'd1;
            end
            case (state)
                IDLE: if (start_valid && start_ready) begin
                    lane_mask <= start_lane_mask; result_tag <= start_tag;
                    head_request <= '0; head_response <= '0; state <= HEADS;
                end
                HEADS: if (values_valid && values_ready) begin
                    head_request <= head_request + 6'd1;
                    if (head_request == 6'd31) state <= HEAD_DRAIN;
                end
                HEAD_DRAIN: if (head_response == 6'd32) begin
                    head_sum <= partial_sum[0]; sum_index <= 2'd1; state <= SUM_REQUEST;
                end
                SUM_REQUEST: if (add_req_ready) state <= SUM_WAIT;
                SUM_WAIT: if (add_rsp_valid) begin
                    head_sum <= add_values;
                    if (sum_index == 2'd3) state <= DIVIDE_MEAN;
                    else begin sum_index <= sum_index + 2'd1; state <= SUM_REQUEST; end
                end
                DIVIDE_MEAN: state <= SCORE_PRODUCT;
                SCORE_PRODUCT: state <= SCORE_ROUND;
                SCORE_ROUND: state <= OUTPUT_ROUND;
                OUTPUT_ROUND: state <= COMPLETE;
                COMPLETE: if (result_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end

    for (genvar lane = 0; lane < 8; lane++) begin : convert_lane
        logic [31:0] sum_value, mean_value, scaled_value;
        logic [23:0] sum_significand;
        logic [4:0] divide_shift;
        logic [23:0] divide_truncated, divide_mask;
        logic divide_round_up;
        logic [24:0] divided_significand;
        logic [31:0] divided_value;
        logic [31:0] product;
        logic [7:0] mean_exponent;
        logic [24:0] product_rounded;
        logic [8:0] product_exponent;
        logic [31:0] product_value;
        logic [15:0] bf16_mean;
        logic [23:0] score_significand, score_truncated, score_mask;
        logic [7:0] score_shift;
        logic score_round_up;
        logic [24:0] score_integer;

        assign sum_value = head_sum[lane*32 +: 32];
        assign sum_significand = {sum_value[30:23] != 0, sum_value[22:0]};
        assign divide_shift = sum_value[30:23] == 0 ? 5'd5 : 5'd6 - sum_value[27:23];
        assign divide_truncated = sum_significand >> divide_shift;
        assign divide_mask = 24'hffffff >> (5'd24 - divide_shift);
        assign divide_round_up = divide_shift != 0 && sum_significand[divide_shift-1] &&
            (divide_truncated[0] || |(sum_significand & (divide_mask >> 1)));
        assign divided_significand = {1'b0, divide_truncated} + {24'd0, divide_round_up};
        assign divided_value = sum_value[30:23] > 8'd5 ?
            {1'b0, 8'(sum_value[30:23] - 8'd5), sum_value[22:0]} :
            {8'd0, divided_significand[23:0]};
        assign bf16_mean = mean_value[31:16] +
            {15'd0, mean_value[15] && (|mean_value[14:0] || mean_value[16])};

        // For mean < 2^-16, score is strictly below 0.5 and rounds to zero.
        // This avoids a subnormal multiply without changing any UINT8 score.
        always_comb begin
            product_rounded = product[31] ?
                {1'b0, product[31:8]} + {24'd0, product[7] && (|product[6:0] || product[8])} :
                {1'b0, product[30:7]} + {24'd0, product[6] && (|product[5:0] || product[7])};
            product_exponent = {1'b0, mean_exponent} + (product[31] ? 9'd8 : 9'd7) +
                {8'd0, product_rounded[24]};
            product_value = product_rounded[24] ?
                {1'b0, product_exponent[7:0], product_rounded[23:1]} :
                {1'b0, product_exponent[7:0], product_rounded[22:0]};
            if (mean_exponent < 8'd111) product_value = '0;
        end
        assign score_significand = {scaled_value[30:23] != 0, scaled_value[22:0]};
        assign score_shift = 8'd150 - scaled_value[30:23];
        assign score_truncated = score_significand >> score_shift;
        assign score_mask = score_shift >= 8'd24 ? 24'hffffff : 24'hffffff >> (8'd24 - score_shift);
        assign score_round_up = score_shift > 0 && score_shift <= 24 &&
            score_significand[score_shift-1] && (score_truncated[0] || |(score_significand & (score_mask >> 1)));
        assign score_integer = {1'b0, score_truncated} + {24'd0, score_round_up};

        always_ff @(posedge clk) begin
            if (rst) begin
                mean_value <= '0; product <= '0; mean_exponent <= '0; scaled_value <= '0;
                result_mean_fp32[lane*32 +: 32] <= '0;
                result_mean_bf16[lane*16 +: 16] <= '0;
                result_score_u8[lane*8 +: 8] <= '0;
            end else if (!abort_request) begin
                if (state == DIVIDE_MEAN) mean_value <= divided_value;
                if (state == SCORE_PRODUCT) begin
                    product <= {1'b1, mean_value[22:0]} * 8'd255;
                    mean_exponent <= mean_value[30:23];
                end
                if (state == SCORE_ROUND) scaled_value <= product_value;
                if (state == OUTPUT_ROUND) begin
                    result_mean_fp32[lane*32 +: 32] <= lane_mask[lane] ? mean_value : 32'd0;
                    result_mean_bf16[lane*16 +: 16] <= lane_mask[lane] ? bf16_mean : 16'd0;
                    result_score_u8[lane*8 +: 8] <= !lane_mask[lane] ? 8'd0 :
                        score_integer > 25'd255 ? 8'd255 : score_integer[7:0];
                end
            end
        end
`ifndef SYNTHESIS
        always_ff @(posedge clk)
            if (!rst && values_valid && values_ready && lane_mask[lane])
                assert (values_bf16[lane*16 +: 16] <= 16'h3f80)
                    else $error("attention_head_mean requires BF16 probabilities in [0,1]");
`endif
    end
endmodule

`default_nettype wire
