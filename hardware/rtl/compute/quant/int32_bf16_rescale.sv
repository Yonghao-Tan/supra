`default_nettype none

module int32_bf16_rescale #(
    parameter integer LANES = 16,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                      clk,
    input  logic                      rst,
    input  logic                      abort_request,
    output logic                      abort_ack,
    input  logic                      req_valid,
    output logic                      req_ready,
    input  logic [LANES*32-1:0]       req_accumulators,
    input  logic [8*16-1:0]           req_activation_scales,
    input  logic [8*16-1:0]           req_weight_scales,
    input  logic [15:0]               req_qk_scale_bf16,
    input  logic [1:0]                req_rescale_mode,
    input  logic [LANES-1:0]          req_lane_mask,
    input  logic [TAG_WIDTH-1:0]      req_tag,
    output logic                      rsp_valid,
    input  logic                      rsp_ready,
    output logic [LANES*16-1:0]      rsp_values,
    output logic [LANES-1:0]         rsp_lane_mask,
    output logic [TAG_WIDTH-1:0]     rsp_tag,
    output logic                      idle
);
    localparam integer PHYSICAL_LANES = LANES < 1 ? 1 : (LANES < 16 ? LANES : 16);
    localparam integer CHUNKS = (LANES + PHYSICAL_LANES - 1) / PHYSICAL_LANES;
    localparam logic [2:0] CHUNK_COUNT = 3'(CHUNKS);
    localparam bit HAS_PARTIAL_CHUNK = LANES % PHYSICAL_LANES != 0;
    localparam logic [1:0] RESCALE_LINEAR = 2'd0;
    localparam logic [1:0] RESCALE_QK = 2'd2;
    typedef enum logic [1:0] {IDLE, RUN_PASS, HOLD_RESPONSE} state_t;
    state_t state;
    logic [1:0] pass_index;
    logic [2:0] step_index;
    logic [LANES*32-1:0] accumulator_or_fp32;
    logic [127:0] saved_scale0, saved_scale1;
    logic [15:0] saved_qk_scale_bf16;
    logic [1:0] saved_rescale_mode;
    logic [LANES-1:0] saved_lane_mask;
    logic [TAG_WIDTH-1:0] saved_tag;
    logic [PHYSICAL_LANES*32-1:0] load_int32;
    logic [PHYSICAL_LANES*32-1:0] load_converted;
    logic [PHYSICAL_LANES*32-1:0] load_values;
    logic [PHYSICAL_LANES*16-1:0] load_scales;
    logic [PHYSICAL_LANES*32-1:0] multiplier_lhs_reg;
    logic [PHYSICAL_LANES*16-1:0] multiplier_scale_reg;
    logic [PHYSICAL_LANES*45-1:0] multiply_stage_reg;
    logic [PHYSICAL_LANES-1:0] special_valid;
    logic [PHYSICAL_LANES*32-1:0] special_result;
    logic [PHYSICAL_LANES-1:0] result_sign;
    logic [PHYSICAL_LANES*11-1:0] exponent_sum;
    logic [PHYSICAL_LANES*24-1:0] significand_lhs;
    logic [PHYSICAL_LANES*8-1:0] significand_rhs;
    logic [PHYSICAL_LANES*32-1:0] product;
    logic [PHYSICAL_LANES*32-1:0] multiplier_result;
    logic [PHYSICAL_LANES*16-1:0] multiplier_result_bf16;
    logic abort_seen;

    assign req_ready = state == IDLE && !abort_request;
    assign idle = req_ready;
    assign rsp_valid = state == HOLD_RESPONSE && !abort_request;

    for (genvar lane = 0; lane < PHYSICAL_LANES; lane++) begin : arithmetic_lanes
        int32_to_fp32_rne accumulator_convert (
            .value(load_int32[lane*32 +: 32]),
            .result(load_converted[lane*32 +: 32]));
        fp32_bf16_multiply_prepare multiply_prepare (
            .lhs(multiplier_lhs_reg[lane*32 +: 32]),
            .rhs(multiplier_scale_reg[lane*16 +: 16]),
            .special_valid(special_valid[lane]),
            .special_result(special_result[lane*32 +: 32]),
            .result_sign(result_sign[lane]),
            .exponent_sum(exponent_sum[lane*11 +: 11]),
            .significand_lhs(significand_lhs[lane*24 +: 24]),
            .significand_rhs(significand_rhs[lane*8 +: 8]));
        assign product[lane*32 +: 32] =
            significand_lhs[lane*24 +: 24] * significand_rhs[lane*8 +: 8];
        fp32_bf16_multiply_finish multiply_finish (
            .special_valid(multiply_stage_reg[lane*45 + 44]),
            .special_result(multiply_stage_reg[lane*45 + 12 +: 32]),
            .result_sign(multiply_stage_reg[lane*45 + 43]),
            .exponent_sum(multiply_stage_reg[lane*45 + 32 +: 11]),
            .significand_product(multiply_stage_reg[lane*45 +: 32]),
            .result(multiplier_result[lane*32 +: 32]));
        fp32_to_bf16_rne result_convert (
            .value(multiplier_result[lane*32 +: 32]),
            .result(multiplier_result_bf16[lane*16 +: 16]));
    end

    always_comb begin
        load_int32 = '0;
        load_values = '0;
        load_scales = '0;
        for (integer lane = 0; lane < PHYSICAL_LANES; lane++) begin
            if (step_index < CHUNK_COUNT &&
                (!HAS_PARTIAL_CHUNK ||
                 integer'(step_index)*PHYSICAL_LANES+lane < LANES)) begin
                load_int32[lane*32 +: 32] =
                    accumulator_or_fp32[(integer'(step_index)*PHYSICAL_LANES+lane)*32 +: 32];
                load_values[lane*32 +: 32] = pass_index == 0 ?
                    load_converted[lane*32 +: 32] :
                    accumulator_or_fp32[(integer'(step_index)*PHYSICAL_LANES+lane)*32 +: 32];
                case (pass_index)
                    0: load_scales[lane*16 +: 16] =
                        saved_scale0[((integer'(step_index)*PHYSICAL_LANES+lane)/8)*16 +: 16];
                    1: load_scales[lane*16 +: 16] =
                        saved_scale1[((integer'(step_index)*PHYSICAL_LANES+lane)%8)*16 +: 16];
                    default: load_scales[lane*16 +: 16] = saved_qk_scale_bf16;
                endcase
            end
        end
    end

    initial begin
        if (LANES < 1 || LANES > 64 || TAG_WIDTH < 1)
            $error("Rescale requires 1..64 external lanes and a positive tag width");
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            pass_index <= '0;
            step_index <= '0;
            saved_rescale_mode <= RESCALE_LINEAR;
            abort_ack <= 1'b0;
            abort_seen <= 1'b0;
        end else if (abort_request) begin
            state <= IDLE;
            abort_ack <= !abort_seen;
            abort_seen <= 1'b1;
        end else begin
            abort_ack <= 1'b0;
            abort_seen <= 1'b0;
            case (state)
                IDLE: if (req_valid) begin
                    accumulator_or_fp32 <= req_accumulators;
                    saved_scale0 <= req_activation_scales;
                    saved_scale1 <= req_weight_scales;
                    saved_qk_scale_bf16 <= req_qk_scale_bf16;
                    saved_rescale_mode <= req_rescale_mode;
                    saved_lane_mask <= req_lane_mask;
                    saved_tag <= req_tag;
                    pass_index <= '0;
                    step_index <= '0;
                    state <= RUN_PASS;
                end
                RUN_PASS: begin
                    if (step_index < CHUNK_COUNT) begin
                        multiplier_lhs_reg <= load_values;
                        multiplier_scale_reg <= load_scales;
                    end
                    if (step_index >= 3'd1 && step_index <= CHUNK_COUNT) begin
                        for (integer lane = 0; lane < PHYSICAL_LANES; lane++)
                            if (!HAS_PARTIAL_CHUNK ||
                                (integer'(step_index)-1)*PHYSICAL_LANES+lane < LANES)
                                multiply_stage_reg[lane*45 +: 45] <=
                                    special_valid[lane] ?
                                    {1'b1, special_result[lane*32 +: 32], 12'd0} :
                                    {1'b0, result_sign[lane], exponent_sum[lane*11 +: 11],
                                     product[lane*32 +: 32]};
                    end
                    if (step_index >= 2) begin
                        for (integer lane = 0; lane < PHYSICAL_LANES; lane++) begin
                            if (!HAS_PARTIAL_CHUNK ||
                                (integer'(step_index)-2)*PHYSICAL_LANES+lane < LANES) begin
                                if (pass_index == 0 ||
                                    (pass_index == 1 && saved_rescale_mode == RESCALE_QK))
                                    accumulator_or_fp32[
                                        ((integer'(step_index)-2)*PHYSICAL_LANES+lane)*32 +: 32] <=
                                        pass_index == 0 && saved_rescale_mode == RESCALE_LINEAR ?
                                        {multiplier_result_bf16[lane*16 +: 16], 16'd0} :
                                        multiplier_result[lane*32 +: 32];
                                else
                                    rsp_values[
                                        ((integer'(step_index)-2)*PHYSICAL_LANES+lane)*16 +: 16] <=
                                        multiplier_result_bf16[lane*16 +: 16];
                            end
                        end
                    end
                    if (step_index == CHUNK_COUNT+3'd1) begin
                        step_index <= '0;
                        if (pass_index == 0 ||
                            (pass_index == 1 && saved_rescale_mode == RESCALE_QK))
                            pass_index <= pass_index + 1'b1;
                        else begin
                            rsp_lane_mask <= saved_lane_mask;
                            rsp_tag <= saved_tag;
                            state <= HOLD_RESPONSE;
                        end
                    end else begin
                        step_index <= step_index + 1'b1;
                    end
                end
                HOLD_RESPONSE: if (rsp_ready)
                    state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    logic stalled_response;
    logic [LANES*16-1:0] stalled_values;
    logic [LANES-1:0] stalled_lane_mask;
    logic [TAG_WIDTH-1:0] stalled_tag;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_response <= 1'b0;
            stalled_values <= '0;
            stalled_lane_mask <= '0;
            stalled_tag <= '0;
        end else begin
            if (stalled_response && !abort_request)
                assert (rsp_valid && rsp_values == stalled_values &&
                        rsp_lane_mask == stalled_lane_mask && rsp_tag == stalled_tag)
                    else $error("Rescale changed a stalled response");
            if (req_valid && req_ready)
                assert (req_rescale_mode <= RESCALE_QK)
                    else $error("Rescale received unsupported mode");
            stalled_response <= rsp_valid && !rsp_ready && !abort_request;
            stalled_values <= rsp_values;
            stalled_lane_mask <= rsp_lane_mask;
            stalled_tag <= rsp_tag;
        end
    end
`endif
endmodule

`default_nettype wire
`default_nettype none

module int32_to_fp32_rne (
    input  wire signed [31:0] value,
    output wire [31:0]        result
);
    function automatic [4:0] highest_set_bit32(input logic [31:0] input_value);
        logic [15:0] half;
        logic [7:0] octet;
        logic [3:0] nibble;
        logic [4:0] index;
        begin
            index = 5'd0;
            if (|input_value[31:16]) begin
                index[4] = 1'b1;
                half = input_value[31:16];
            end else begin
                half = input_value[15:0];
            end
            if (|half[15:8]) begin
                index[3] = 1'b1;
                octet = half[15:8];
            end else begin
                octet = half[7:0];
            end
            if (|octet[7:4]) begin
                index[2] = 1'b1;
                nibble = octet[7:4];
            end else begin
                nibble = octet[3:0];
            end
            if (|nibble[3:2]) begin
                index[1] = 1'b1;
                index[0] = nibble[3];
            end else begin
                index[0] = nibble[1];
            end
            highest_set_bit32 = index;
        end
    endfunction

    function automatic [31:0] convert;
        input signed [31:0] input_value;
        reg sign_value;
        reg [31:0] magnitude;
        reg [32:0] rounded_significand;
        reg [31:0] quotient;
        reg [31:0] remainder;
        reg [31:0] half_value;
        integer leading_bit;
        integer shift_count;
        integer exponent_value;
        begin
            sign_value = input_value[31];
            magnitude = sign_value ? (~input_value + 32'd1) : input_value;
            rounded_significand = 33'd0;
            quotient = 32'd0;
            remainder = 32'd0;
            half_value = 32'd0;
            leading_bit = 0;
            shift_count = 0;
            exponent_value = 0;
            if (magnitude == 0) begin
                convert = {sign_value, 31'd0};
            end else begin
                leading_bit = integer'(highest_set_bit32(magnitude));
                exponent_value = leading_bit + 127;
                if (leading_bit <= 23) begin
                    quotient = magnitude << (23 - leading_bit);
                end else begin
                    shift_count = leading_bit - 23;
                    quotient = magnitude >> shift_count;
                    remainder = magnitude & ((32'd1 << shift_count) - 32'd1);
                    half_value = 32'd1 << (shift_count - 1);
                    rounded_significand = {1'b0, quotient};
                    if (remainder > half_value ||
                        (remainder == half_value && quotient[0]))
                        rounded_significand = rounded_significand + 33'd1;
                    if (rounded_significand[24]) begin
                        quotient = {8'd0, rounded_significand[24:1]};
                        exponent_value = exponent_value + 1;
                    end else begin
                        quotient = {8'd0, rounded_significand[23:0]};
                    end
                end
                convert = {sign_value, exponent_value[7:0], quotient[22:0]};
            end
        end
    endfunction

    assign result = convert(value);
endmodule

`default_nettype wire
`default_nettype none

module fp32_bf16_multiply_prepare (
    input  logic signed [31:0] lhs,
    input  logic        [15:0] rhs,
    output logic               special_valid,
    output logic        [31:0] special_result,
    output logic               result_sign,
    output logic signed [10:0] exponent_sum,
    output logic        [23:0] significand_lhs,
    output logic         [7:0] significand_rhs
);
    function automatic [4:0] normalization_shift24(input logic [23:0] value);
        logic [31:0] padded;
        logic [15:0] half;
        logic [7:0] octet;
        logic [3:0] nibble;
        logic [4:0] highest_bit;
        begin
            padded = {8'd0, value};
            highest_bit = 5'd0;
            if (|padded[31:16]) begin
                highest_bit[4] = 1'b1;
                half = padded[31:16];
            end else begin
                half = padded[15:0];
            end
            if (|half[15:8]) begin
                highest_bit[3] = 1'b1;
                octet = half[15:8];
            end else begin
                octet = half[7:0];
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
            normalization_shift24 = 5'd23 - highest_bit;
        end
    endfunction

    function automatic [2:0] normalization_shift8(input logic [7:0] value);
        begin
            casez (value)
                8'b1???????: normalization_shift8 = 3'd0;
                8'b01??????: normalization_shift8 = 3'd1;
                8'b001?????: normalization_shift8 = 3'd2;
                8'b0001????: normalization_shift8 = 3'd3;
                8'b00001???: normalization_shift8 = 3'd4;
                8'b000001??: normalization_shift8 = 3'd5;
                8'b0000001?: normalization_shift8 = 3'd6;
                default:     normalization_shift8 = 3'd7;
            endcase
        end
    endfunction

    logic [7:0] exponent_lhs;
    logic [7:0] exponent_rhs;
    logic signed [10:0] unbiased_lhs;
    logic signed [10:0] unbiased_rhs;
    logic [4:0] lhs_normalization_shift;
    logic [2:0] rhs_normalization_shift;

    always_comb begin
        result_sign = lhs[31] ^ rhs[15];
        exponent_lhs = lhs[30:23];
        exponent_rhs = rhs[14:7];
        significand_lhs = exponent_lhs == 0 ? {1'b0, lhs[22:0]} : {1'b1, lhs[22:0]};
        significand_rhs = exponent_rhs == 0 ? {1'b0, rhs[6:0]} : {1'b1, rhs[6:0]};
        unbiased_lhs = exponent_lhs == 0 ? -11'sd126 :
            $signed({3'b000, exponent_lhs}) - 11'sd127;
        unbiased_rhs = exponent_rhs == 0 ? -11'sd126 :
            $signed({3'b000, exponent_rhs}) - 11'sd127;
        special_valid = 1'b1;
        special_result = 32'd0;
        exponent_sum = 11'sd0;
        lhs_normalization_shift = 5'd0;
        rhs_normalization_shift = 3'd0;

        if (exponent_lhs == 8'hff && lhs[22:0] != 0) begin
            special_result = lhs | 32'h00400000;
        end else if (exponent_rhs == 8'hff && rhs[6:0] != 0) begin
            special_result = {rhs, 16'd0} | 32'h00400000;
        end else if ((exponent_lhs == 8'hff && rhs[14:0] == 0) ||
                     (exponent_rhs == 8'hff && lhs[30:0] == 0)) begin
            special_result = 32'hffc00000;
        end else if (exponent_lhs == 8'hff || exponent_rhs == 8'hff) begin
            special_result = {result_sign, 8'hff, 23'd0};
        end else if (lhs[30:0] == 0 || rhs[14:0] == 0) begin
            special_result = {result_sign, 31'd0};
        end else begin
            special_valid = 1'b0;
            if (exponent_lhs == 0) begin
                lhs_normalization_shift = normalization_shift24(significand_lhs);
                significand_lhs = significand_lhs << lhs_normalization_shift;
                unbiased_lhs = unbiased_lhs -
                    $signed({6'b000000, lhs_normalization_shift});
            end
            if (exponent_rhs == 0) begin
                rhs_normalization_shift = normalization_shift8(significand_rhs);
                significand_rhs = significand_rhs << rhs_normalization_shift;
                unbiased_rhs = unbiased_rhs -
                    $signed({8'b00000000, rhs_normalization_shift});
            end
            exponent_sum = unbiased_lhs + unbiased_rhs;
        end
    end
endmodule

`default_nettype wire
`default_nettype none

module fp32_bf16_multiply_finish (
    input  logic               special_valid,
    input  logic        [31:0] special_result,
    input  logic               result_sign,
    input  logic signed [10:0] exponent_sum,
    input  logic        [31:0] significand_product,
    output logic        [31:0] result
);
    logic [31:0] quotient;
    logic [31:0] remainder;
    logic [31:0] halfway;
    logic [32:0] rounded;
    integer result_exponent;
    integer biased_exponent;
    integer shift_count;

    always_comb begin
        result = special_result;
        quotient = 32'd0;
        remainder = 32'd0;
        halfway = 32'd0;
        rounded = 33'd0;
        result_exponent = integer'(exponent_sum) +
            (significand_product[31] ? 1 : 0);
        biased_exponent = 0;
        shift_count = 0;

        if (!special_valid) begin
            if (result_exponent > 127) begin
                result = {result_sign, 8'hff, 23'd0};
            end else if (result_exponent >= -126) begin
                shift_count = significand_product[31] ? 8 : 7;
                quotient = significand_product >> shift_count;
                remainder = significand_product & ((32'd1 << shift_count) - 32'd1);
                halfway = 32'd1 << (shift_count - 1);
                rounded = {1'b0, quotient};
                if (remainder > halfway || (remainder == halfway && quotient[0]))
                    rounded = rounded + 33'd1;
                if (rounded[24]) begin
                    quotient = rounded[32:1];
                    result_exponent = result_exponent + 1;
                end else begin
                    quotient = rounded[31:0];
                end
                biased_exponent = result_exponent + 127;
                if (biased_exponent >= 255)
                    result = {result_sign, 8'hff, 23'd0};
                else
                    result = {result_sign, biased_exponent[7:0], quotient[22:0]};
            end else begin
                shift_count = -integer'(exponent_sum) - 119;
                if (shift_count > 32) begin
                    result = {result_sign, 31'd0};
                end else begin
                    quotient = significand_product >> shift_count;
                    remainder = significand_product & ((32'd1 << shift_count) - 32'd1);
                    halfway = 32'd1 << (shift_count - 1);
                    rounded = {1'b0, quotient};
                    if (remainder > halfway || (remainder == halfway && quotient[0]))
                        rounded = rounded + 33'd1;
                    if (rounded[23])
                        result = {result_sign, 8'h01, 23'd0};
                    else
                        result = {result_sign, 8'h00, rounded[22:0]};
                end
            end
        end
    end
endmodule

`default_nettype wire
`default_nettype none

module fp32_to_bf16_rne (
    input  wire [31:0] value,
    output wire [15:0] result
);
    wire [15:0] upper = value[31:16];
    wire [15:0] lower = value[15:0];
    assign result = value[30:23] == 8'hff && value[22:0] != 0 ?
        upper | 16'h0040 :
        (lower > 16'h8000 || (lower == 16'h8000 && upper[0]) ? upper + 16'd1 : upper);
endmodule

`default_nettype wire
