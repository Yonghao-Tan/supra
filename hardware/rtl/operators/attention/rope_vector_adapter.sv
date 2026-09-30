`default_nettype none
module rope_vector_adapter #(
    parameter integer PAIR_LANES = 8,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         req_valid,
    output logic                         req_ready,
    input  logic [PAIR_LANES*16-1:0]     req_first_values,
    input  logic [PAIR_LANES*16-1:0]     req_second_values,
    input  logic [PAIR_LANES*16-1:0]     req_cos_values,
    input  logic [PAIR_LANES*16-1:0]     req_sin_values,
    input  logic [PAIR_LANES-1:0]        req_pair_mask,
    input  logic [TAG_WIDTH-1:0]         req_tag,
    output logic                         rsp_valid,
    input  logic                         rsp_ready,
    output logic [PAIR_LANES*16-1:0]     rsp_first_values,
    output logic [PAIR_LANES*16-1:0]     rsp_second_values,
    output logic [PAIR_LANES-1:0]        rsp_pair_mask,
    output logic [TAG_WIDTH-1:0]         rsp_tag,
    output logic                         arithmetic_req_valid,
    input  logic                         arithmetic_req_ready,
    output logic [2:0]                   arithmetic_req_operation,
    output logic [2*PAIR_LANES*16-1:0]   arithmetic_req_values,
    output logic [2*PAIR_LANES*16-1:0]   arithmetic_req_paired_values,
    output logic [2*PAIR_LANES*16-1:0]   arithmetic_req_factor0_values,
    output logic [2*PAIR_LANES*16-1:0]   arithmetic_req_factor1_values,
    output logic [2*PAIR_LANES-1:0]      arithmetic_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         arithmetic_req_tag,
    input  logic                         arithmetic_rsp_valid,
    output logic                         arithmetic_rsp_ready,
    input  logic [2*PAIR_LANES*16-1:0]   arithmetic_rsp_values,
    input  logic [2*PAIR_LANES-1:0]      arithmetic_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]         arithmetic_rsp_tag
);
    localparam logic [2:0] OP_ROPE = 3'd2;

    assign req_ready = arithmetic_req_ready;
    assign rsp_valid = arithmetic_rsp_valid;
    assign arithmetic_req_valid = req_valid;
    assign arithmetic_rsp_ready = rsp_ready;
    assign arithmetic_req_operation = OP_ROPE;
    assign arithmetic_req_tag = req_tag;
    assign rsp_first_values = arithmetic_rsp_values[0 +: PAIR_LANES*16];
    assign rsp_second_values = arithmetic_rsp_values[PAIR_LANES*16 +: PAIR_LANES*16];
    assign rsp_pair_mask = arithmetic_rsp_lane_mask[0 +: PAIR_LANES];
    assign rsp_tag = arithmetic_rsp_tag;

    genvar pair_lane;
    generate
        for (pair_lane = 0; pair_lane < PAIR_LANES; pair_lane = pair_lane + 1) begin : pair_mapping
            assign arithmetic_req_values[pair_lane*16 +: 16] =
                req_first_values[pair_lane*16 +: 16];
            assign arithmetic_req_paired_values[pair_lane*16 +: 16] =
                req_second_values[pair_lane*16 +: 16];
            assign arithmetic_req_factor0_values[pair_lane*16 +: 16] =
                req_cos_values[pair_lane*16 +: 16];
            assign arithmetic_req_factor1_values[pair_lane*16 +: 16] = {
                ~req_sin_values[pair_lane*16 + 15],
                req_sin_values[pair_lane*16 +: 15]
            };

            assign arithmetic_req_values[(PAIR_LANES+pair_lane)*16 +: 16] =
                req_second_values[pair_lane*16 +: 16];
            assign arithmetic_req_paired_values[(PAIR_LANES+pair_lane)*16 +: 16] =
                req_first_values[pair_lane*16 +: 16];
            assign arithmetic_req_factor0_values[(PAIR_LANES+pair_lane)*16 +: 16] =
                req_cos_values[pair_lane*16 +: 16];
            assign arithmetic_req_factor1_values[(PAIR_LANES+pair_lane)*16 +: 16] =
                req_sin_values[pair_lane*16 +: 16];
            assign arithmetic_req_lane_mask[pair_lane] = req_pair_mask[pair_lane];
            assign arithmetic_req_lane_mask[PAIR_LANES+pair_lane] = req_pair_mask[pair_lane];
        end
    endgenerate

    initial begin
        if (PAIR_LANES < 1 || TAG_WIDTH < 1)
            $error("rope_vector_adapter requires positive PAIR_LANES and TAG_WIDTH");
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst && arithmetic_rsp_valid) begin
            assert (arithmetic_rsp_lane_mask[PAIR_LANES-1:0] ==
                    arithmetic_rsp_lane_mask[2*PAIR_LANES-1:PAIR_LANES])
                else $error("rope_vector_adapter output pair mask mismatch");
        end
    end
`endif
endmodule

`default_nettype wire
