`default_nettype none

module bf16_mul_pipe #(
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
    localparam integer PREPARE_WIDTH = 28;
    localparam integer PRODUCT_METADATA_WIDTH = 18;
    localparam integer NORMALIZED_WIDTH = 19;

    function automatic [15:0] quiet_nan(input logic [15:0] value);
        quiet_nan = value | 16'h0040;
    endfunction

    function automatic [2:0] normalization_shift8(input logic [7:0] value);
        logic [3:0] nibble;
        logic [2:0] highest_bit;
        begin
            highest_bit = 3'd0;
            if (|value[7:4]) begin
                highest_bit[2] = 1'b1;
                nibble = value[7:4];
            end else begin
                nibble = value[3:0];
            end
            if (|nibble[3:2]) begin
                highest_bit[1] = 1'b1;
                highest_bit[0] = nibble[3];
            end else begin
                highest_bit[0] = nibble[1];
            end
            normalization_shift8 = 3'd7 - highest_bit;
        end
    endfunction

    function automatic [PREPARE_WIDTH-1:0] prepare_multiply(
        input logic [15:0] lhs,
        input logic [15:0] rhs
    );
        logic result_sign;
        logic special_valid;
        logic [15:0] special_result;
        logic [7:0] exponent_lhs;
        logic [7:0] exponent_rhs;
        logic [7:0] significand_lhs;
        logic [7:0] significand_rhs;
        integer exponent_value;
        logic [2:0] lhs_normalization_shift;
        logic [2:0] rhs_normalization_shift;
        begin
            result_sign = lhs[15] ^ rhs[15];
            exponent_lhs = lhs[14:7];
            exponent_rhs = rhs[14:7];
            significand_lhs = exponent_lhs == 0 ? {1'b0, lhs[6:0]} : {1'b1, lhs[6:0]};
            significand_rhs = exponent_rhs == 0 ? {1'b0, rhs[6:0]} : {1'b1, rhs[6:0]};
            exponent_value = (exponent_lhs == 0 ? -126 : exponent_lhs - 127) +
                             (exponent_rhs == 0 ? -126 : exponent_rhs - 127);
            special_valid = 1'b1;
            special_result = 16'h0000;
            lhs_normalization_shift = 3'd0;
            rhs_normalization_shift = 3'd0;

            if (exponent_lhs == 8'hff && lhs[6:0] != 0) begin
                special_result = quiet_nan(lhs);
            end else if (exponent_rhs == 8'hff && rhs[6:0] != 0) begin
                special_result = quiet_nan(rhs);
            end else if ((exponent_lhs == 8'hff && rhs[14:0] == 0) ||
                         (exponent_rhs == 8'hff && lhs[14:0] == 0)) begin
                special_result = 16'hffc0;
            end else if (exponent_lhs == 8'hff || exponent_rhs == 8'hff) begin
                special_result = {result_sign, 8'hff, 7'h00};
            end else if (lhs[14:0] == 0 || rhs[14:0] == 0) begin
                special_result = {result_sign, 15'h0000};
            end else begin
                special_valid = 1'b0;
                lhs_normalization_shift = normalization_shift8(significand_lhs);
                rhs_normalization_shift = normalization_shift8(significand_rhs);
                significand_lhs = significand_lhs << lhs_normalization_shift;
                significand_rhs = significand_rhs << rhs_normalization_shift;
                exponent_value = exponent_value -
                    integer'(lhs_normalization_shift) -
                    integer'(rhs_normalization_shift);
            end
            if (special_valid)
                prepare_multiply = {1'b1, special_result, 11'd0};
            else
                prepare_multiply = {1'b0, result_sign,
                    exponent_value[9:0], significand_lhs, significand_rhs};
        end
    endfunction

    logic [LANES*PREPARE_WIDTH-1:0] prepare_values;
    logic prepared_valid;
    logic [LANES*PREPARE_WIDTH-1:0] prepared_values;
    logic [LANES-1:0] prepared_lane_mask;
    logic [TAG_WIDTH-1:0] prepared_tag;
    logic product_valid;
    logic [LANES*PRODUCT_METADATA_WIDTH-1:0] product_metadata;
    logic [LANES*16-1:0] product_values;
    logic [LANES-1:0] product_lane_mask;
    logic [TAG_WIDTH-1:0] product_tag;
    logic [LANES*NORMALIZED_WIDTH-1:0] normalized_comb;
    logic normalized_valid;
    logic [LANES*NORMALIZED_WIDTH-1:0] normalized_values;
    logic [LANES-1:0] normalized_lane_mask;
    logic [TAG_WIDTH-1:0] normalized_tag;
    logic [LANES*16-1:0] finished_values;
    logic normalized_ready;
    logic product_ready;
    logic prepared_ready;

    assign normalized_ready = !normalized_valid || rsp_ready;
    assign product_ready = !product_valid || normalized_ready;
    assign prepared_ready = !prepared_valid || product_ready;
    assign req_ready = prepared_ready;
    assign rsp_valid = normalized_valid;
    assign rsp_values = finished_values;
    assign rsp_lane_mask = normalized_lane_mask;
    assign rsp_tag = normalized_tag;

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : multiply_lanes
            assign prepare_values[lane*PREPARE_WIDTH +: PREPARE_WIDTH] =
                prepare_multiply(
                    req_lhs[lane*16 +: 16],
                    req_rhs[lane*16 +: 16]);
            bf16_mul_normalize_prepare normalize_lane (
                .product_metadata(
                    product_metadata[lane*PRODUCT_METADATA_WIDTH +: PRODUCT_METADATA_WIDTH]),
                .significand_product(product_values[lane*16 +: 16]),
                .normalized(normalized_comb[lane*NORMALIZED_WIDTH +: NORMALIZED_WIDTH])
            );
            bf16_mul_round_finish finish_lane (
                .normalized(normalized_values[lane*NORMALIZED_WIDTH +: NORMALIZED_WIDTH]),
                .result(finished_values[lane*16 +: 16])
            );
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (rst) begin
            prepared_valid <= 1'b0;
            product_valid <= 1'b0;
            normalized_valid <= 1'b0;
        end else begin
            if (normalized_ready) begin
                normalized_valid <= product_valid;
                if (product_valid) begin
                    normalized_values <= normalized_comb;
                    normalized_lane_mask <= product_lane_mask;
                    normalized_tag <= product_tag;
                end
            end
            if (product_ready) begin
                product_valid <= prepared_valid;
                if (prepared_valid) begin
                    for (integer product_lane = 0; product_lane < LANES; product_lane = product_lane + 1) begin
                        if (prepared_values[product_lane*PREPARE_WIDTH + 27])
                            product_metadata[product_lane*PRODUCT_METADATA_WIDTH +:
                                PRODUCT_METADATA_WIDTH] <= {
                                    1'b1,
                                    prepared_values[product_lane*PREPARE_WIDTH + 11 +: 16],
                                    1'b0};
                        else
                            product_metadata[product_lane*PRODUCT_METADATA_WIDTH +:
                                PRODUCT_METADATA_WIDTH] <= {
                                    1'b0, 6'd0,
                                    prepared_values[product_lane*PREPARE_WIDTH + 16 +: 11]};
                        product_values[product_lane*16 +: 16] <=
                            prepared_values[product_lane*PREPARE_WIDTH +: 8] *
                            prepared_values[product_lane*PREPARE_WIDTH + 8 +: 8];
                    end
                    product_lane_mask <= prepared_lane_mask;
                    product_tag <= prepared_tag;
                end
            end
            if (prepared_ready) begin
                prepared_valid <= req_valid;
                if (req_valid) begin
                    prepared_values <= prepare_values;
                    prepared_lane_mask <= req_lane_mask;
                    prepared_tag <= req_tag;
                end
            end
        end
    end

    initial begin
        if (LANES < 1 || TAG_WIDTH < 1)
            $error("bf16_mul_pipe parameter relation is invalid");
    end
endmodule

`default_nettype wire
`default_nettype none

module bf16_mul_normalize_prepare (
    input  logic [17:0] product_metadata,
    input  logic [15:0] significand_product,
    output logic [18:0] normalized
);
    logic special_valid;
    logic [15:0] special_result;
    logic result_sign;
    logic signed [9:0] exponent_value;
    logic [7:0] exponent_field;
    logic [7:0] quotient;
    logic round_up;
    integer biased_exponent;
    integer shift_count;
    integer total_shift;

    always_comb begin
        special_valid = product_metadata[17];
        special_result = product_metadata[16:1];
        result_sign = product_metadata[10];
        exponent_value = $signed(product_metadata[9:0]);
        exponent_field = 8'd0;
        quotient = 8'd0;
        biased_exponent = 0;
        shift_count = 0;
        total_shift = 0;
        round_up = 1'b0;

        if (!special_valid) begin
            if (significand_product[15]) begin
                quotient = significand_product[15:8];
                round_up = significand_product[7] &&
                    (|significand_product[6:0] || significand_product[8]);
                exponent_value = exponent_value + 10'sd1;
            end else begin
                quotient = significand_product[14:7];
                round_up = significand_product[6] &&
                    (|significand_product[5:0] || significand_product[7]);
            end

            biased_exponent = exponent_value + 127;
            if (biased_exponent <= 0) begin
                shift_count = 1 - biased_exponent;
                total_shift = (significand_product[15] ? 8 : 7) + shift_count;
                case (total_shift)
                    8: begin
                        quotient = significand_product[15:8];
                        round_up = significand_product[7] &&
                            (|significand_product[6:0] || significand_product[8]);
                    end
                    9: begin
                        quotient = {1'd0, significand_product[15:9]};
                        round_up = significand_product[8] &&
                            (|significand_product[7:0] || significand_product[9]);
                    end
                    10: begin
                        quotient = {2'd0, significand_product[15:10]};
                        round_up = significand_product[9] &&
                            (|significand_product[8:0] || significand_product[10]);
                    end
                    11: begin
                        quotient = {3'd0, significand_product[15:11]};
                        round_up = significand_product[10] &&
                            (|significand_product[9:0] || significand_product[11]);
                    end
                    12: begin
                        quotient = {4'd0, significand_product[15:12]};
                        round_up = significand_product[11] &&
                            (|significand_product[10:0] || significand_product[12]);
                    end
                    13: begin
                        quotient = {5'd0, significand_product[15:13]};
                        round_up = significand_product[12] &&
                            (|significand_product[11:0] || significand_product[13]);
                    end
                    14: begin
                        quotient = {6'd0, significand_product[15:14]};
                        round_up = significand_product[13] &&
                            (|significand_product[12:0] || significand_product[14]);
                    end
                    15: begin
                        quotient = {7'd0, significand_product[15]};
                        round_up = significand_product[14] &&
                            (|significand_product[13:0] || significand_product[15]);
                    end
                    16: begin
                        quotient = 8'd0;
                        round_up = significand_product[15] &&
                            |significand_product[14:0];
                    end
                    default: begin
                        quotient = 8'd0;
                        round_up = 1'b0;
                    end
                endcase
            end else if (biased_exponent >= 255) begin
                exponent_field = 8'hff;
            end else begin
                exponent_field = biased_exponent[7:0];
            end
        end

        if (special_valid) begin
            normalized = {1'b1, special_result, 2'd0};
        end else begin
            normalized = {
                1'b0,
                result_sign,
                exponent_field,
                quotient,
                round_up
            };
        end
    end
endmodule

`default_nettype wire
`default_nettype none

module bf16_mul_round_finish (
    input  logic [18:0] normalized,
    output logic [15:0] result
);
    logic special_valid;
    logic [15:0] special_result;
    logic result_sign;
    logic [7:0] exponent_field;
    logic [7:0] quotient;
    logic round_up;
    logic [8:0] rounded;
    logic [7:0] mantissa;
    logic [8:0] rounded_exponent;

    always_comb begin
        special_valid = normalized[18];
        special_result = normalized[17:2];
        {result_sign, exponent_field, quotient, round_up} = normalized[17:0];
        result = special_result;
        rounded = 9'd0;
        mantissa = quotient;
        rounded_exponent = {1'b0, exponent_field};

        if (!special_valid) begin
            if (exponent_field == 0) begin
                rounded = {1'b0, quotient};
                if (round_up)
                    rounded = rounded + 9'd1;
                result = {result_sign, 7'h00, rounded[7:0]};
            end else if (exponent_field == 8'hff) begin
                result = {result_sign, 8'hff, 7'h00};
            end else begin
                rounded = {1'b0, quotient};
                if (round_up)
                    rounded = rounded + 9'd1;
                if (rounded[8]) begin
                    mantissa = rounded[8:1];
                    rounded_exponent = {1'b0, exponent_field} + 9'd1;
                end else begin
                    mantissa = rounded[7:0];
                end
                if (rounded_exponent >= 9'd255)
                    result = {result_sign, 8'hff, 7'h00};
                else
                    result = {result_sign, rounded_exponent[7:0], mantissa[6:0]};
            end
        end
    end
endmodule

`default_nettype wire
