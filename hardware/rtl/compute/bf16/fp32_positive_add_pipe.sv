`default_nettype none

// FP32 RNE addition for nonnegative finite probabilities and partial sums.
// Four stages separate exponent ordering, alignment, addition, and rounding.
// The refresh path supplies the accumulation order and SRAM addresses.
module fp32_positive_add_pipe #(
    parameter integer LANES = 8,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       req_valid,
    output logic                       req_ready,
    input  logic [LANES*32-1:0]        req_lhs,
    input  logic [LANES*32-1:0]        req_rhs,
    input  logic [LANES-1:0]           req_lane_mask,
    input  logic [TAG_WIDTH-1:0]       req_tag,
    output logic                       rsp_valid,
    input  logic                       rsp_ready,
    output logic [LANES*32-1:0]        rsp_values,
    output logic [LANES-1:0]           rsp_lane_mask,
    output logic [TAG_WIDTH-1:0]       rsp_tag
);
    logic advance_pipe;
    logic [3:0] valid_pipe;
    logic [LANES-1:0] mask_pipe [0:3];
    logic [TAG_WIDTH-1:0] tag_pipe [0:3];

    assign advance_pipe = !valid_pipe[3] || rsp_ready;
    assign req_ready = advance_pipe && !rst;
    assign rsp_valid = valid_pipe[3] && !rst;
    assign rsp_lane_mask = mask_pipe[3];
    assign rsp_tag = tag_pipe[3];

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_pipe <= '0;
        end else if (advance_pipe) begin
            valid_pipe <= {valid_pipe[2:0], req_valid};
            if (req_valid) begin
                mask_pipe[0] <= req_lane_mask;
                tag_pipe[0] <= req_tag;
            end
            for (integer stage = 1; stage < 4; stage++) begin
                if (valid_pipe[stage-1]) begin
                    mask_pipe[stage] <= mask_pipe[stage-1];
                    tag_pipe[stage] <= tag_pipe[stage-1];
                end
            end
        end
    end

    for (genvar lane = 0; lane < LANES; lane++) begin : add_lane
        logic [31:0] lhs, rhs;
        logic [7:0] lhs_exp, rhs_exp;
        logic lhs_larger;
        logic [23:0] large_sig, small_sig;
        logic [7:0] ordered_exp, exponent_difference;
        logic [26:0] aligned_large, aligned_small;
        logic [7:0] aligned_exp;
        logic [26:0] extended_small, shifted_small;
        logic [26:0] lost_mask;
        logic [27:0] full_sum;
        logic [26:0] normalized_sig;
        logic [8:0] normalized_exp;
        logic [24:0] rounded_sig;
        logic round_up;
        logic [8:0] rounded_exp;
        logic [31:0] rounded_value;

        assign lhs = req_lhs[lane*32 +: 32];
        assign rhs = req_rhs[lane*32 +: 32];
        assign lhs_exp = lhs[30:23] == 0 ? 8'd1 : lhs[30:23];
        assign rhs_exp = rhs[30:23] == 0 ? 8'd1 : rhs[30:23];
        assign lhs_larger = lhs_exp >= rhs_exp;
        assign extended_small = {small_sig, 3'b000};

        always_comb begin
            lost_mask = '0;
            if (exponent_difference >= 8'd27) begin
                shifted_small = {26'd0, |extended_small};
            end else begin
                lost_mask = 27'h7ffffff >> (8'd27 - exponent_difference);
                shifted_small = extended_small >> exponent_difference;
                shifted_small[0] = shifted_small[0] | (|(extended_small & lost_mask));
            end
        end

        assign full_sum = {1'b0, aligned_large} + {1'b0, aligned_small};
        assign round_up = normalized_sig[2] &&
            (normalized_sig[1] || normalized_sig[0] || normalized_sig[3]);
        assign rounded_sig = {1'b0, normalized_sig[26:3]} + {24'd0, round_up};
        assign rounded_exp = normalized_exp + {8'd0, rounded_sig[24]};

        always_comb begin
            if (rounded_exp >= 9'd255)
                rounded_value = 32'h7f800000;
            else if (rounded_sig[24])
                rounded_value = {1'b0, rounded_exp[7:0], rounded_sig[23:1]};
            else if (!rounded_sig[23])
                rounded_value = {9'd0, rounded_sig[22:0]};
            else
                rounded_value = {1'b0, rounded_exp[7:0], rounded_sig[22:0]};
        end

        always_ff @(posedge clk) begin
            if (!rst && advance_pipe) begin
                if (req_valid) begin
                    large_sig <= lhs_larger ?
                        {lhs[30:23] != 0, lhs[22:0]} : {rhs[30:23] != 0, rhs[22:0]};
                    small_sig <= lhs_larger ?
                        {rhs[30:23] != 0, rhs[22:0]} : {lhs[30:23] != 0, lhs[22:0]};
                    ordered_exp <= lhs_larger ? lhs_exp : rhs_exp;
                    exponent_difference <= lhs_larger ? lhs_exp - rhs_exp : rhs_exp - lhs_exp;
                end
                if (valid_pipe[0]) begin
                    aligned_large <= {large_sig, 3'b000};
                    aligned_small <= shifted_small;
                    aligned_exp <= ordered_exp;
                end
                if (valid_pipe[1]) begin
                    normalized_sig <= full_sum[27] ?
                        {full_sum[27:2], |full_sum[1:0]} : full_sum[26:0];
                    normalized_exp <= {1'b0, aligned_exp} + {8'd0, full_sum[27]};
                end
                if (valid_pipe[2])
                    rsp_values[lane*32 +: 32] <= mask_pipe[2][lane] ? rounded_value : 32'd0;
            end
        end

`ifndef SYNTHESIS
        always_ff @(posedge clk) begin
            if (!rst && req_valid && req_ready && req_lane_mask[lane]) begin
                assert (!lhs[31] && lhs[30:23] != 8'hff &&
                        !rhs[31] && rhs[30:23] != 8'hff)
                    else $error("fp32_positive_add_pipe lane %0d requires nonnegative finite inputs", lane);
            end
        end
`endif
    end

    initial begin
        if (LANES < 1 || TAG_WIDTH < 1)
            $error("fp32_positive_add_pipe requires positive LANES and TAG_WIDTH");
    end
endmodule

`default_nettype wire
