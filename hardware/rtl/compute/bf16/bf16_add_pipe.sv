`default_nettype none

module bf16_add_pipe #(
    parameter integer LANES = 8,
    parameter integer TAG_WIDTH = 32
) (
    input  logic                      clk,
    input  logic                      rst,
    input  logic                      req_valid,
    output logic                      req_ready,
    input  logic [LANES*16-1:0]       req_lhs,
    input  logic [LANES*16-1:0]       req_rhs,
    input  logic [LANES-1:0]          req_lane_mask,
    input  logic [TAG_WIDTH-1:0]      req_tag,
    output logic                      rsp_valid,
    input  logic                      rsp_ready,
    output logic [LANES*16-1:0]       rsp_values,
    output logic [LANES-1:0]          rsp_lane_mask,
    output logic [TAG_WIDTH-1:0]      rsp_tag
);
    localparam integer ALIGN_WIDTH = 25;

    logic advance_pipe;
    logic [LANES*ALIGN_WIDTH-1:0] aligned_comb;
    logic [LANES*ALIGN_WIDTH-1:0] aligned_values;
    logic [LANES*ALIGN_WIDTH-1:0] normalize_prepare_comb;
    logic [LANES*ALIGN_WIDTH-1:0] normalized_values;
    logic [LANES*16-1:0] round_finish_comb;
    logic align_valid;
    logic [LANES-1:0] aligned_lane_mask;
    logic [TAG_WIDTH-1:0] aligned_tag;
    logic normalize_valid;
    logic [LANES-1:0] normalized_lane_mask;
    logic [TAG_WIDTH-1:0] normalized_tag;

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : add_lanes
            bf16_add_align align_lane (
                .lhs(req_lhs[lane*16 +: 16]),
                .rhs(req_rhs[lane*16 +: 16]),
                .aligned(aligned_comb[lane*ALIGN_WIDTH +: ALIGN_WIDTH])
            );
            bf16_add_normalize_prepare normalize_lane (
                .aligned(aligned_values[lane*ALIGN_WIDTH +: ALIGN_WIDTH]),
                .normalized(normalize_prepare_comb[lane*ALIGN_WIDTH +: ALIGN_WIDTH])
            );
            bf16_add_round_finish round_lane (
                .normalized(normalized_values[lane*ALIGN_WIDTH +: ALIGN_WIDTH]),
                .result(round_finish_comb[lane*16 +: 16])
            );
        end
    endgenerate

    assign advance_pipe = !normalize_valid || rsp_ready;
    assign req_ready = advance_pipe;
    assign rsp_valid = normalize_valid;
    assign rsp_values = round_finish_comb;
    assign rsp_lane_mask = normalized_lane_mask;
    assign rsp_tag = normalized_tag;

    always_ff @(posedge clk) begin
        if (rst) begin
            align_valid <= 1'b0;
            normalize_valid <= 1'b0;
        end else if (advance_pipe) begin
            normalize_valid <= align_valid;
            if (align_valid) begin
                normalized_values <= normalize_prepare_comb;
                normalized_lane_mask <= aligned_lane_mask;
                normalized_tag <= aligned_tag;
            end

            align_valid <= req_valid;
            if (req_valid) begin
                aligned_values <= aligned_comb;
                aligned_lane_mask <= req_lane_mask;
                aligned_tag <= req_tag;
            end
        end
    end

    initial begin
        if (LANES < 1 || TAG_WIDTH < 1)
            $error("bf16_add_pipe parameter relation is invalid");
    end
endmodule

`default_nettype wire
`default_nettype none

module bf16_add_align (
    input  logic [15:0] lhs,
    input  logic [15:0] rhs,
    output logic [24:0] aligned
);
    function automatic [12:0] shift_right_sticky13(
        input logic [12:0] value,
        input logic [9:0] shift
    );
        case (shift)
            10'd0: shift_right_sticky13 = value;
            10'd1: shift_right_sticky13 =
                {1'd0, value[12:2], |value[1:0]};
            10'd2: shift_right_sticky13 =
                {2'd0, value[12:3], |value[2:0]};
            10'd3: shift_right_sticky13 =
                {3'd0, value[12:4], |value[3:0]};
            10'd4: shift_right_sticky13 =
                {4'd0, value[12:5], |value[4:0]};
            10'd5: shift_right_sticky13 =
                {5'd0, value[12:6], |value[5:0]};
            10'd6: shift_right_sticky13 =
                {6'd0, value[12:7], |value[6:0]};
            10'd7: shift_right_sticky13 =
                {7'd0, value[12:8], |value[7:0]};
            10'd8: shift_right_sticky13 =
                {8'd0, value[12:9], |value[8:0]};
            10'd9: shift_right_sticky13 =
                {9'd0, value[12:10], |value[9:0]};
            10'd10: shift_right_sticky13 =
                {10'd0, value[12:11], |value[10:0]};
            10'd11: shift_right_sticky13 =
                {11'd0, value[12], |value[11:0]};
            default: shift_right_sticky13 = value != 0 ? 13'd1 : 13'd0;
        endcase
    endfunction

    function automatic [24:0] align_and_add;
        input [15:0] a;
        input [15:0] b;
        reg special_result_valid;
        reg [15:0] special_result;
        reg sign_large;
        reg sign_small;
        reg [7:0] exponent_a;
        reg [7:0] exponent_b;
        reg [7:0] mantissa_a;
        reg [7:0] mantissa_b;
        reg [7:0] mantissa_large;
        reg [7:0] mantissa_small;
        reg [12:0] large_extended;
        reg [12:0] small_extended;
        reg [12:0] sum_value;
        reg signed [9:0] exponent_large;
        reg signed [9:0] exponent_small;
        reg [9:0] difference;
        begin
            special_result_valid = 1'b1;
            special_result = 16'h0000;
            sign_large = 1'b0;
            exponent_large = 10'sd0;
            sum_value = 13'd0;
            exponent_a = a[14:7];
            exponent_b = b[14:7];

            if (exponent_a == 8'hff && a[6:0] != 0) begin
                special_result = a | 16'h0040;
            end else if (exponent_b == 8'hff && b[6:0] != 0) begin
                special_result = b | 16'h0040;
            end else if (exponent_a == 8'hff && exponent_b == 8'hff && a[15] != b[15]) begin
                special_result = 16'hffc0;
            end else if (exponent_a == 8'hff) begin
                special_result = a;
            end else if (exponent_b == 8'hff) begin
                special_result = b;
            end else if (a[14:0] == 0 && b[14:0] == 0) begin
                special_result = {a[15] & b[15], 15'h0000};
            end else if (a[14:0] == 0) begin
                special_result = b;
            end else if (b[14:0] == 0) begin
                special_result = a;
            end else begin
                special_result_valid = 1'b0;
                mantissa_a = exponent_a == 0 ? {1'b0, a[6:0]} : {1'b1, a[6:0]};
                mantissa_b = exponent_b == 0 ? {1'b0, b[6:0]} : {1'b1, b[6:0]};
                exponent_large = exponent_a == 0 ? -10'sd126 : $signed({2'b00, exponent_a}) - 10'sd127;
                exponent_small = exponent_b == 0 ? -10'sd126 : $signed({2'b00, exponent_b}) - 10'sd127;
                if (exponent_large > exponent_small ||
                    (exponent_large == exponent_small && mantissa_a >= mantissa_b)) begin
                    mantissa_large = mantissa_a;
                    mantissa_small = mantissa_b;
                    sign_large = a[15];
                    sign_small = b[15];
                end else begin
                    mantissa_large = mantissa_b;
                    mantissa_small = mantissa_a;
                    sign_large = b[15];
                    sign_small = a[15];
                    exponent_large = exponent_small;
                    exponent_small = exponent_a == 0 ? -10'sd126 :
                        $signed({2'b00, exponent_a}) - 10'sd127;
                end
                difference = exponent_large - exponent_small;
                large_extended = {2'b00, mantissa_large, 3'b000};
                small_extended = {2'b00, mantissa_small, 3'b000};
                small_extended = shift_right_sticky13(
                    small_extended, difference);
                sum_value = sign_large == sign_small ?
                    large_extended + small_extended : large_extended - small_extended;
            end

            align_and_add = special_result_valid ?
                {1'b1, 8'd0, special_result} :
                {1'b0, sign_large, exponent_large, sum_value};
        end
    endfunction

    assign aligned = align_and_add(lhs, rhs);
endmodule

`default_nettype wire
`default_nettype none

module bf16_add_normalize_prepare (
    input  logic [24:0] aligned,
    output logic [24:0] normalized
);
    function automatic [12:0] shift_left13(
        input logic [12:0] value,
        input logic [3:0] shift
    );
        case (shift)
            4'd0: shift_left13 = value;
            4'd1: shift_left13 = {value[11:0], 1'd0};
            4'd2: shift_left13 = {value[10:0], 2'd0};
            4'd3: shift_left13 = {value[9:0], 3'd0};
            4'd4: shift_left13 = {value[8:0], 4'd0};
            4'd5: shift_left13 = {value[7:0], 5'd0};
            4'd6: shift_left13 = {value[6:0], 6'd0};
            4'd7: shift_left13 = {value[5:0], 7'd0};
            4'd8: shift_left13 = {value[4:0], 8'd0};
            4'd9: shift_left13 = {value[3:0], 9'd0};
            default: shift_left13 = {value[2:0], 10'd0};
        endcase
    endfunction

    function automatic [3:0] normalization_shift11(input logic [10:0] value);
        logic [15:0] padded;
        logic [7:0] octet;
        logic [3:0] nibble;
        logic [3:0] highest_bit;
        begin
            padded = {5'd0, value};
            highest_bit = 4'd0;
            if (|padded[15:8]) begin
                highest_bit[3] = 1'b1;
                octet = padded[15:8];
            end else begin
                octet = padded[7:0];
            end
            if (|octet[7:4]) begin
                highest_bit[2] = 1'b1;
                nibble = octet[7:4];
            end else begin
                nibble = octet[3:0];
            end
            if (|nibble[3:2]) begin
                highest_bit[1] = 1'b1;
                highest_bit[0] = nibble[3];
            end else begin
                highest_bit[0] = nibble[1];
            end
            normalization_shift11 = 4'd10 - highest_bit;
        end
    endfunction

    logic special_result_valid;
    logic [15:0] special_result;
    logic sign_large;
    logic signed [9:0] exponent_value;
    logic [12:0] sum_value;
    logic sticky;
    logic [3:0] requested_shift;
    logic signed [10:0] exponent_shift_limit;
    logic [3:0] normalization_shift;

    always_comb begin
        special_result_valid = aligned[24];
        special_result = aligned[15:0];
        {sign_large, exponent_value, sum_value} = aligned[23:0];
        sticky = 1'b0;
        requested_shift = 4'd0;
        exponent_shift_limit = 11'sd0;
        normalization_shift = 4'd0;

        if (!special_result_valid && sum_value != 0) begin
            if (sum_value[11]) begin
                sticky = sum_value[0];
                sum_value = sum_value >> 1;
                if (sticky)
                    sum_value[0] = 1'b1;
                exponent_value = exponent_value + 10'sd1;
            end else begin
                requested_shift = normalization_shift11(sum_value[10:0]);
                exponent_shift_limit = $signed(exponent_value) + 11'sd126;
                normalization_shift = {7'd0, requested_shift} < exponent_shift_limit ?
                    requested_shift : exponent_shift_limit[3:0];
                sum_value = shift_left13(sum_value, normalization_shift);
                exponent_value = exponent_value -
                    $signed({6'd0, normalization_shift});
            end
        end

        normalized = special_result_valid ?
            {1'b1, 8'd0, special_result} :
            {1'b0, sign_large, exponent_value, sum_value};
    end
endmodule

`default_nettype wire
`default_nettype none

module bf16_add_round_finish (
    input  logic [24:0] normalized,
    output logic [15:0] result
);
    logic special_result_valid;
    logic [15:0] special_result;
    logic sign_large;
    logic signed [9:0] exponent_value;
    logic [12:0] sum_value;
    logic [8:0] rounded;
    logic signed [10:0] biased_exponent;

    always_comb begin
        special_result_valid = normalized[24];
        special_result = normalized[15:0];
        {sign_large, exponent_value, sum_value} = normalized[23:0];
        result = special_result;
        rounded = 9'd0;
        biased_exponent = 11'sd0;

        if (!special_result_valid) begin
            if (sum_value == 0) begin
                result = 16'h0000;
            end else begin
                rounded = {1'b0, sum_value[10:3]};
                if (sum_value[2:0] > 3'b100 ||
                    (sum_value[2:0] == 3'b100 && sum_value[3]))
                    rounded = rounded + 9'd1;
                if (rounded[8]) begin
                    rounded = rounded >> 1;
                    exponent_value = exponent_value + 10'sd1;
                end
                biased_exponent = exponent_value + 11'sd127;
                if (biased_exponent >= 11'sd255)
                    result = {sign_large, 8'hff, 7'h00};
                else if (exponent_value <= -10'sd126 && !sum_value[10])
                    result = {sign_large, 8'h00, rounded[6:0]};
                else
                    result = {sign_large, biased_exponent[7:0], rounded[6:0]};
            end
        end
    end
endmodule

`default_nettype wire
