`default_nettype none

module bf16_divide_u16_iterative (
    input  logic        clk,
    input  logic        rst,
    input  logic        req_valid,
    output logic        req_ready,
    input  logic [15:0] req_value,
    input  logic [15:0] req_divisor,
    output logic        rsp_valid,
    input  logic        rsp_ready,
    output logic [15:0] rsp_value
);
    function automatic [4:0] ceil_log2_u16(input logic [15:0] value);
        logic [15:0] decremented;
        logic [7:0] octet;
        logic [3:0] nibble;
        logic [3:0] highest_bit;
        begin
            if (value <= 16'd1) begin
                ceil_log2_u16 = 5'd0;
            end else begin
                decremented = value - 16'd1;
                highest_bit = 4'd0;
                if (|decremented[15:8]) begin
                    highest_bit[3] = 1'b1;
                    octet = decremented[15:8];
                end else begin
                    octet = decremented[7:0];
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
                ceil_log2_u16 = {1'b0, highest_bit} + 5'd1;
            end
        end
    endfunction

    logic busy;
    logic prepare_pending;
    logic [15:0] saved_value;
    logic [15:0] saved_divisor;
    logic [4:0] bit_index;
    logic [23:0] numerator;
    logic [23:0] quotient;
    logic [24:0] remainder;
    logic [15:0] divisor;
    logic signed [10:0] output_exponent;

    logic [24:0] shifted_remainder;
    logic divide_take;
    logic [24:0] next_remainder;
    logic [23:0] next_quotient;
    logic round_up;
    logic [24:0] rounded_quotient;
    logic normalize_result;
    logic [6:0] result_mantissa;
    logic signed [11:0] biased_exponent;
    assign shifted_remainder = {remainder[23:0], numerator[bit_index]};
    assign divide_take = shifted_remainder >= {9'd0, divisor};
    assign next_remainder = divide_take ? shifted_remainder - {9'd0, divisor} : shifted_remainder;
    assign next_quotient = divide_take ? quotient | (24'd1 << bit_index) : quotient;
    assign round_up = {next_remainder, 1'b0} > {10'd0, divisor} ||
        ({next_remainder, 1'b0} == {10'd0, divisor} && next_quotient[0]);
    assign rounded_quotient = {1'b0, next_quotient} + {24'd0, round_up};
    assign normalize_result = rounded_quotient >= 25'd256;
    assign result_mantissa = normalize_result ? rounded_quotient[7:1] : rounded_quotient[6:0];
    assign biased_exponent = output_exponent + 12'sd127 + $signed({11'd0, normalize_result});
    assign req_ready = !busy && !prepare_pending && (!rsp_valid || rsp_ready);

    integer ceil_log2_divisor;
    integer selected_shift;
    logic [31:0] prepared_numerator;
    logic signed [10:0] input_exponent;
    logic [7:0] input_mantissa;
    always_comb begin
        ceil_log2_divisor = 0;
        ceil_log2_divisor = integer'(ceil_log2_u16(saved_divisor));
        input_mantissa = saved_value[14:7] == 0 ?
            {1'b0, saved_value[6:0]} : {1'b1, saved_value[6:0]};
        input_exponent = saved_value[14:7] == 0 ? -11'sd126 :
            $signed({3'd0, saved_value[14:7]}) - 11'sd127;
        selected_shift = ceil_log2_divisor;
        prepared_numerator = {24'd0, input_mantissa} << selected_shift;
        if (prepared_numerator >= ({16'd0, saved_divisor} << 8)) begin
            selected_shift = selected_shift - 1;
            prepared_numerator = {24'd0, input_mantissa} << selected_shift;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            prepare_pending <= 1'b0;
            saved_value <= 16'd0;
            saved_divisor <= 16'd1;
            bit_index <= 5'd0;
            numerator <= 24'd0;
            quotient <= 24'd0;
            remainder <= 25'd0;
            divisor <= 16'd1;
            output_exponent <= 11'sd0;
            rsp_valid <= 1'b0;
            rsp_value <= 16'd0;
        end else begin
            if (rsp_valid && rsp_ready)
                rsp_valid <= 1'b0;
            if (req_valid && req_ready) begin
                if (req_divisor == 0 || req_value[15] || req_value[14:7] == 8'hff) begin
                    rsp_value <= 16'h7fc0;
                    rsp_valid <= 1'b1;
                end else if (req_value[14:0] == 0) begin
                    rsp_value <= 16'h0000;
                    rsp_valid <= 1'b1;
                end else begin
                    saved_value <= req_value;
                    saved_divisor <= req_divisor;
                    prepare_pending <= 1'b1;
                end
            end else if (prepare_pending) begin
                numerator <= prepared_numerator[23:0];
                quotient <= 24'd0;
                remainder <= 25'd0;
                divisor <= saved_divisor;
                output_exponent <= input_exponent - selected_shift;
                bit_index <= 5'd23;
                busy <= 1'b1;
                prepare_pending <= 1'b0;
            end else if (busy) begin
                quotient <= next_quotient;
                remainder <= next_remainder;
                if (bit_index == 0) begin
                    if (biased_exponent <= 0)
                        rsp_value <= 16'h0000;
                    else if (biased_exponent >= 255)
                        rsp_value <= 16'h7f80;
                    else
                        rsp_value <= {1'b0, biased_exponent[7:0], result_mantissa};
                    rsp_valid <= 1'b1;
                    busy <= 1'b0;
                end else begin
                    bit_index <= bit_index - 5'd1;
                end
            end
        end
    end
endmodule

`default_nettype wire
