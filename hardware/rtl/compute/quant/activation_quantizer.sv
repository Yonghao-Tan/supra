`default_nettype none

module activation_quantizer #(
    parameter integer LANES = 64,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   abort_request,
    output logic                   abort_ack,
    input  logic                   scale_req_valid,
    output logic                   scale_req_ready,
    input  logic [7:0]             scale_req_a4_row_mask,
    input  logic                   scale_req_use_static_scale,
    input  logic [8*16-1:0]        scale_req_static_scales_bf16,
    input  logic [8*16-1:0]        scale_req_row_max_abs,
    input  logic [7:0]             scale_req_row_mask,
    input  logic [15:0]            scale_req_clip_ratio_bf16,
    output logic                   scale_rsp_valid,
    input  logic                   scale_rsp_ready,
    output logic [8*16-1:0]        scale_rsp_values_bf16,
    input  logic                   quantized_req_valid,
    output logic                   quantized_req_ready,
    input  logic [LANES*16-1:0]    quantized_req_values_bf16,
    input  logic [LANES-1:0]       quantized_req_lane_mask,
    input  logic [TAG_WIDTH-1:0]   quantized_req_tag,
    output logic                   quantized_rsp_valid,
    input  logic                   quantized_rsp_ready,
    output logic [LANES*8-1:0]     quantized_rsp_values,
    output logic [LANES-1:0]       quantized_rsp_lane_mask,
    output logic [TAG_WIDTH-1:0]   quantized_rsp_tag,
    output logic                   idle
);
    localparam integer ROWS = 8;
    localparam integer ELEMENTS_PER_ROW = LANES / ROWS;
    localparam integer QUOTIENT_STAGES = 4;
    localparam logic [1:0] QUANT_INACTIVE = 2'b00;
    localparam logic [1:0] QUANT_NORMAL = 2'b01;
    localparam logic [1:0] QUANT_DOUBLE_DENOMINATOR = 2'b10;
    localparam logic [1:0] QUANT_SATURATE = 2'b11;

    logic setup_busy;
    logic config_valid;
    logic [7:0] config_a4_row_mask;
    logic [7:0] config_row_mask;
    logic [8*16-1:0] config_scales;
    logic [7:0] scale_pending;
    logic [7:0] scale_capture;
    logic [7:0] scale_pending_after_capture;
    logic [7:0] scale_engine_req_ready;
    logic [7:0] scale_engine_rsp_valid;
    logic [8*16-1:0] scale_engine_rsp_value;
    logic scale_request_fire;
    logic scale_response_fire;
    logic config_clip_enable;
    logic [127:0] config_row_max;
    logic [127:0] config_clip_limits;
    logic [15:0] clip_ratio;
    logic clip_setup_busy;
    logic [3:0] clip_issue_row;
    logic clip_mul_ready, clip_mul_valid;
    logic [15:0] clip_mul_value;
    logic [2:0] clip_mul_row;
    logic clip_limits_ready;
    logic scale_engine_start;
    logic request_clip_enable;
    logic clip_input_valid;
    logic [LANES*16-1:0] clip_input_values;
    logic [LANES-1:0] clip_input_lane_mask;
    logic [TAG_WIDTH-1:0] clip_input_tag;
    logic source_valid;
    logic [LANES*16-1:0] source_values;
    logic [LANES-1:0] source_lane_mask;
    logic [TAG_WIDTH-1:0] source_tag;

    logic prepare_valid;
    logic [LANES-1:0] prepare_lane_mask;
    logic [TAG_WIDTH-1:0] prepare_tag;
    logic [9:0] prepare_remainder [0:LANES-1];
    logic [7:0] prepare_quotient_bits [0:LANES-1];
    logic prepare_sign [0:LANES-1];
    logic [1:0] prepare_action [0:LANES-1];
    logic [9:0] prepared_remainder [0:LANES-1];
    logic [7:0] prepared_quotient_bits [0:LANES-1];
    logic prepared_sign [0:LANES-1];
    logic [1:0] prepared_action [0:LANES-1];
    logic [7:0] config_scale_mantissa [0:ROWS-1];

    logic [QUOTIENT_STAGES-1:0] quotient_valid;
    logic [LANES-1:0] quotient_lane_mask [0:QUOTIENT_STAGES-1];
    logic [TAG_WIDTH-1:0] quotient_tag [0:QUOTIENT_STAGES-1];
    logic [9:0] quotient_remainder [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [7:0] quotient_bits [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic quotient_sign [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [1:0] quotient_action [0:QUOTIENT_STAGES-1][0:LANES-1];

    logic [9:0] step_source_denominator [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [9:0] step_source_remainder [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [7:0] step_source_quotient_bits [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic step_source_sign [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [1:0] step_source_action [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [10:0] step_shifted_remainder [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [10:0] step_denominator_twice [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [10:0] step_denominator_thrice [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [1:0] step_digit [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [10:0] step_subtracted_remainder [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [9:0] step_next_remainder [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [7:0] step_next_quotient_bits [0:QUOTIENT_STAGES-1][0:LANES-1];
    logic [9:0] final_denominator [0:LANES-1];

    logic pipeline_advance;
    logic pipeline_busy;
    logic abort_seen;

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

    function automatic [7:0] normalize_mantissa(input logic [7:0] value);
        normalize_mantissa = value << normalization_shift8(value);
    endfunction

    function automatic signed [10:0] normalized_exponent(input logic [15:0] value);
        logic signed [10:0] exponent_value;
        begin
            exponent_value = value[14:7] == 0 ? -11'sd126 :
                $signed({3'b000, value[14:7]}) - 11'sd127;
            normalized_exponent = exponent_value - $signed({8'd0,
                normalization_shift8(value[14:7] == 0 ?
                    {1'b0, value[6:0]} : {1'b1, value[6:0]})});
        end
    endfunction

    always_comb begin
        integer row;
        logic [15:0] scale;
        for (row = 0; row < ROWS; row = row + 1) begin
            scale = config_scales[row*16 +: 16];
            config_scale_mantissa[row] = normalize_mantissa(
                scale[14:7] == 0 ? {1'b0, scale[6:0]} :
                {1'b1, scale[6:0]});
        end
    end

    generate
        for (genvar row = 0; row < ROWS; row = row + 1) begin : scale_rows
            dynamic_activation_scale_iterative scale_engine (
                .clk(clk), .rst(rst || abort_request),
                .req_valid(scale_engine_start &&
                    (clip_limits_ready ? config_row_mask[row] : scale_req_row_mask[row])),
                .req_ready(scale_engine_req_ready[row]),
                .req_a4(clip_limits_ready ? config_a4_row_mask[row] : scale_req_a4_row_mask[row]),
                .req_max_abs(clip_limits_ready ?
                    (config_a4_row_mask[row] ? config_clip_limits[row*16 +: 16] :
                        config_row_max[row*16 +: 16]) : scale_req_row_max_abs[row*16 +: 16]),
                .rsp_valid(scale_engine_rsp_valid[row]),
                .rsp_ready(!scale_pending[row] || scale_capture[row]),
                .rsp_scale_bf16(scale_engine_rsp_value[row*16 +: 16]));
            assign scale_capture[row] = setup_busy && scale_pending[row] &&
                scale_engine_rsp_valid[row];
        end
    endgenerate

    // One pipelined multiplier prepares eight row limits while data is idle.
    bf16_mul_pipe #(.LANES(1), .TAG_WIDTH(3)) clip_limit_multiply (
        .clk(clk), .rst(rst || abort_request),
        .req_valid(clip_setup_busy && clip_issue_row < 8), .req_ready(clip_mul_ready),
        .req_lhs(config_row_max[clip_issue_row[2:0]*16 +: 16]),
        .req_rhs(clip_ratio), .req_lane_mask(1'b1), .req_tag(clip_issue_row[2:0]),
        .rsp_valid(clip_mul_valid), .rsp_ready(1'b1), .rsp_values(clip_mul_value),
        .rsp_lane_mask(), .rsp_tag(clip_mul_row));
    assign request_clip_enable = scale_req_clip_ratio_bf16 != 0 &&
        !scale_req_use_static_scale;
    assign scale_engine_start = (scale_request_fire && !request_clip_enable &&
        !scale_req_use_static_scale) || clip_limits_ready;
    assign source_valid = config_clip_enable ? clip_input_valid : quantized_req_valid;
    assign source_values = config_clip_enable ? clip_input_values : quantized_req_values_bf16;
    assign source_lane_mask = config_clip_enable ? clip_input_lane_mask : quantized_req_lane_mask;
    assign source_tag = config_clip_enable ? clip_input_tag : quantized_req_tag;
    assign pipeline_busy = clip_input_valid || prepare_valid || (|quotient_valid) || quantized_rsp_valid;
    assign scale_req_ready = !abort_request && !setup_busy && !scale_rsp_valid &&
        !pipeline_busy && (scale_req_use_static_scale ||
        ((scale_engine_req_ready | ~scale_req_row_mask) == 8'hff));
    assign scale_request_fire = scale_req_valid && scale_req_ready;
    assign scale_response_fire = scale_rsp_valid && scale_rsp_ready;
    assign scale_pending_after_capture = scale_pending & ~scale_capture;
    assign pipeline_advance = !quantized_rsp_valid || quantized_rsp_ready;
    assign quantized_req_ready = !abort_request && config_valid &&
        (config_clip_enable ? !clip_input_valid || pipeline_advance : pipeline_advance);
    assign idle = !abort_request && !setup_busy && !scale_rsp_valid &&
        !pipeline_busy;

    always_comb begin
        integer lane;
        integer row_index;
        logic [15:0] value;
        logic [15:0] scale;
        logic [7:0] value_mantissa;
        logic signed [11:0] exponent_difference;
        logic [15:0] numerator;
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            row_index = lane / ELEMENTS_PER_ROW;
            value = source_values[lane*16 +: 16];
            scale = config_scales[row_index*16 +: 16];
            value_mantissa = normalize_mantissa(value[14:7] == 0 ?
                {1'b0, value[6:0]} : {1'b1, value[6:0]});
            exponent_difference = normalized_exponent(value) -
                normalized_exponent(scale);
            numerator = exponent_difference >= 0 && exponent_difference <= 7 ?
                ({8'd0, value_mantissa} << exponent_difference[2:0]) :
                {8'd0, value_mantissa};
            prepare_remainder[lane] = {2'b00, numerator[15:8]};
            prepare_quotient_bits[lane] = numerator[7:0];
            prepare_sign[lane] = value[15];
            prepare_action[lane] = QUANT_INACTIVE;
            if (source_lane_mask[lane] && config_row_mask[row_index] &&
                value[14:0] != 0 && !scale[15] && scale[14:0] != 0 &&
                scale[14:7] != 8'hff && exponent_difference >= -1) begin
                if (exponent_difference > 7)
                    prepare_action[lane] = QUANT_SATURATE;
                else if (exponent_difference == -1)
                    prepare_action[lane] = QUANT_DOUBLE_DENOMINATOR;
                else
                    prepare_action[lane] = QUANT_NORMAL;
            end
        end
    end

    generate
        for (genvar lane = 0; lane < LANES; lane = lane + 1) begin : prepare_lanes
            always_ff @(posedge clk) begin
                if (pipeline_advance && source_valid && config_valid && !abort_request) begin
                    prepared_remainder[lane] <= prepare_remainder[lane];
                    prepared_quotient_bits[lane] <= prepare_quotient_bits[lane];
                    prepared_sign[lane] <= prepare_sign[lane];
                    prepared_action[lane] <= prepare_action[lane];
                end
            end
        end
        for (genvar stage = 0; stage < QUOTIENT_STAGES; stage = stage + 1) begin : quotient_steps
            if (stage == 0) begin : stage_zero_metadata
                always_ff @(posedge clk) begin
                    if (rst || abort_request) begin
                        quotient_valid[stage] <= 1'b0;
                        quotient_lane_mask[stage] <= '0;
                        quotient_tag[stage] <= '0;
                    end else if (pipeline_advance) begin
                        quotient_valid[stage] <= prepare_valid;
                        quotient_lane_mask[stage] <= prepare_lane_mask;
                        quotient_tag[stage] <= prepare_tag;
                    end
                end
            end else begin : later_stage_metadata
                always_ff @(posedge clk) begin
                    if (rst || abort_request) begin
                        quotient_valid[stage] <= 1'b0;
                        quotient_lane_mask[stage] <= '0;
                        quotient_tag[stage] <= '0;
                    end else if (pipeline_advance) begin
                        quotient_valid[stage] <= quotient_valid[stage-1];
                        quotient_lane_mask[stage] <= quotient_lane_mask[stage-1];
                        quotient_tag[stage] <= quotient_tag[stage-1];
                    end
                end
            end
            for (genvar lane = 0; lane < LANES; lane = lane + 1) begin : quotient_lanes
                localparam integer ROW_INDEX = lane / ELEMENTS_PER_ROW;
                if (stage == 0) begin
                    assign step_source_remainder[stage][lane] = prepared_remainder[lane];
                    assign step_source_quotient_bits[stage][lane] =
                        prepared_quotient_bits[lane];
                    assign step_source_sign[stage][lane] = prepared_sign[lane];
                    assign step_source_action[stage][lane] = prepared_action[lane];
                end else begin
                    assign step_source_remainder[stage][lane] =
                        quotient_remainder[stage-1][lane];
                    assign step_source_quotient_bits[stage][lane] =
                        quotient_bits[stage-1][lane];
                    assign step_source_sign[stage][lane] = quotient_sign[stage-1][lane];
                    assign step_source_action[stage][lane] =
                        quotient_action[stage-1][lane];
                end
                assign step_source_denominator[stage][lane] =
                    step_source_action[stage][lane] == QUANT_DOUBLE_DENOMINATOR ?
                    {1'b0, config_scale_mantissa[ROW_INDEX], 1'b0} :
                    {2'b00, config_scale_mantissa[ROW_INDEX]};
                assign step_shifted_remainder[stage][lane] =
                    {step_source_remainder[stage][lane][8:0],
                     step_source_quotient_bits[stage][lane][7:6]};
                assign step_denominator_twice[stage][lane] =
                    {step_source_denominator[stage][lane], 1'b0};
                assign step_denominator_thrice[stage][lane] =
                    {1'b0, step_source_denominator[stage][lane]} +
                    step_denominator_twice[stage][lane];
                assign step_digit[stage][lane] =
                    step_shifted_remainder[stage][lane] >=
                        step_denominator_thrice[stage][lane] ? 2'd3 :
                    step_shifted_remainder[stage][lane] >=
                        step_denominator_twice[stage][lane] ? 2'd2 :
                    step_shifted_remainder[stage][lane] >=
                        {1'b0, step_source_denominator[stage][lane]} ? 2'd1 :
                    2'd0;
                assign step_subtracted_remainder[stage][lane] =
                    step_shifted_remainder[stage][lane] -
                    (step_digit[stage][lane] == 2'd3 ?
                        step_denominator_thrice[stage][lane] :
                     step_digit[stage][lane] == 2'd2 ?
                        step_denominator_twice[stage][lane] :
                     step_digit[stage][lane] == 2'd1 ?
                        {1'b0, step_source_denominator[stage][lane]} : 11'd0);
                assign step_next_remainder[stage][lane] =
                    step_subtracted_remainder[stage][lane][9:0];
                assign step_next_quotient_bits[stage][lane] =
                    {step_source_quotient_bits[stage][lane][5:0],
                     step_digit[stage][lane]};
                always_ff @(posedge clk) begin
                    if (pipeline_advance) begin
                        quotient_remainder[stage][lane] <= step_next_remainder[stage][lane];
                        quotient_bits[stage][lane] <=
                            step_next_quotient_bits[stage][lane];
                        quotient_sign[stage][lane] <= step_source_sign[stage][lane];
                        quotient_action[stage][lane] <=
                            step_source_action[stage][lane];
                    end
                end
            end
        end
    endgenerate

    for (genvar lane = 0; lane < LANES; lane = lane + 1) begin : final_denominators
        localparam integer ROW_INDEX = lane / ELEMENTS_PER_ROW;
        assign final_denominator[lane] =
            quotient_action[QUOTIENT_STAGES-1][lane] ==
                QUANT_DOUBLE_DENOMINATOR ?
            {1'b0, config_scale_mantissa[ROW_INDEX], 1'b0} :
            {2'b00, config_scale_mantissa[ROW_INDEX]};
    end

    always_ff @(posedge clk) begin
        integer row;
        integer lane;
        logic [10:0] doubled_remainder;
        logic round_up;
        logic [8:0] rounded_magnitude;
        logic [7:0] limited_magnitude;
        if (rst) begin
            setup_busy <= 1'b0;
            config_valid <= 1'b0;
            clip_setup_busy <= 1'b0;
            clip_issue_row <= '0;
            clip_limits_ready <= 1'b0;
            clip_input_valid <= 1'b0;
            scale_pending <= 8'd0;
            scale_rsp_valid <= 1'b0;
            prepare_valid <= 1'b0;
            quantized_rsp_valid <= 1'b0;
            abort_ack <= 1'b0;
            abort_seen <= 1'b0;
        end else if (abort_request) begin
            setup_busy <= 1'b0;
            config_valid <= 1'b0;
            clip_setup_busy <= 1'b0;
            clip_limits_ready <= 1'b0;
            clip_input_valid <= 1'b0;
            scale_pending <= 8'd0;
            scale_rsp_valid <= 1'b0;
            prepare_valid <= 1'b0;
            quantized_rsp_valid <= 1'b0;
            abort_ack <= !abort_seen;
            abort_seen <= 1'b1;
        end else begin
            clip_limits_ready <= 1'b0;
            if (clip_setup_busy && clip_issue_row < 8 && clip_mul_ready)
                clip_issue_row <= clip_issue_row + 1'b1;
            if (clip_mul_valid && clip_setup_busy) begin
                config_clip_limits[clip_mul_row*16 +: 16] <= clip_mul_value;
                if (clip_mul_row == 7) begin
                    clip_setup_busy <= 1'b0;
                    clip_limits_ready <= 1'b1;
                end
            end
            if (config_clip_enable && quantized_req_ready) begin
                clip_input_valid <= quantized_req_valid;
                if (quantized_req_valid) begin
                    clip_input_lane_mask <= quantized_req_lane_mask;
                    clip_input_tag <= quantized_req_tag;
                    for (integer k = 0; k < LANES; k = k + 1)
                        clip_input_values[k*16 +: 16] <=
                            config_a4_row_mask[k/ELEMENTS_PER_ROW] &&
                            quantized_req_values_bf16[k*16 +: 15] >
                                config_clip_limits[(k/ELEMENTS_PER_ROW)*16 +: 15] ?
                            {quantized_req_values_bf16[k*16+15],
                                config_clip_limits[(k/ELEMENTS_PER_ROW)*16 +: 15]} :
                            quantized_req_values_bf16[k*16 +: 16];
                end
            end
            abort_ack <= 1'b0;
            abort_seen <= 1'b0;
            if (scale_response_fire) begin
                scale_rsp_valid <= 1'b0;
                config_valid <= 1'b1;
                config_scales <= scale_rsp_values_bf16;
            end
            if (scale_request_fire) begin
                config_clip_enable <= request_clip_enable;
                config_row_max <= scale_req_row_max_abs;
                clip_ratio <= scale_req_clip_ratio_bf16;
                clip_issue_row <= 0;
                clip_setup_busy <= request_clip_enable;
                setup_busy <= 1'b1;
                config_valid <= 1'b0;
                config_a4_row_mask <= scale_req_a4_row_mask;
                config_row_mask <= scale_req_row_mask;
                scale_pending <= scale_req_row_mask &
                    {ROWS{!scale_req_use_static_scale}};
                for (row = 0; row < ROWS; row = row + 1)
                    scale_rsp_values_bf16[row*16 +: 16] <=
                        !scale_req_row_mask[row] ? 16'h3f80 :
                        scale_req_use_static_scale ? scale_req_static_scales_bf16[row*16 +: 16] :
                        16'h0000;
            end else if (setup_busy) begin
                for (row = 0; row < ROWS; row = row + 1)
                    if (scale_capture[row])
                        scale_rsp_values_bf16[row*16 +: 16] <=
                            scale_engine_rsp_value[row*16 +: 16];
                scale_pending <= scale_pending_after_capture;
                if (scale_pending_after_capture == 0 && !clip_setup_busy && !clip_limits_ready) begin
                    setup_busy <= 1'b0;
                    scale_rsp_valid <= 1'b1;
                end
            end

            if (pipeline_advance) begin
                quantized_rsp_valid <= quotient_valid[QUOTIENT_STAGES-1];
                if (quotient_valid[QUOTIENT_STAGES-1]) begin
                    quantized_rsp_lane_mask <= quotient_lane_mask[QUOTIENT_STAGES-1];
                    quantized_rsp_tag <= quotient_tag[QUOTIENT_STAGES-1];
                    for (lane = 0; lane < LANES; lane = lane + 1) begin
                        doubled_remainder =
                            {quotient_remainder[QUOTIENT_STAGES-1][lane], 1'b0};
                        round_up = doubled_remainder >
                            {1'b0, final_denominator[lane]} ||
                            (doubled_remainder ==
                             {1'b0, final_denominator[lane]} &&
                             quotient_bits[QUOTIENT_STAGES-1][lane][0]);
                        rounded_magnitude =
                            {1'b0, quotient_bits[QUOTIENT_STAGES-1][lane]} +
                            {8'd0, round_up};
                        limited_magnitude = config_a4_row_mask[
                            lane/ELEMENTS_PER_ROW] ?
                            8'd7 : 8'd127;
                        if (quotient_action[QUOTIENT_STAGES-1][lane] ==
                            QUANT_INACTIVE)
                            quantized_rsp_values[lane*8 +: 8] <= 8'd0;
                        else if (quotient_action[QUOTIENT_STAGES-1][lane] ==
                                 QUANT_SATURATE ||
                                 rounded_magnitude > {1'b0, limited_magnitude})
                            quantized_rsp_values[lane*8 +: 8] <=
                                quotient_sign[QUOTIENT_STAGES-1][lane] ?
                                -$signed(limited_magnitude) : limited_magnitude;
                        else
                            quantized_rsp_values[lane*8 +: 8] <=
                                quotient_sign[QUOTIENT_STAGES-1][lane] ?
                                -$signed(rounded_magnitude[7:0]) : rounded_magnitude[7:0];
                    end
                end

                prepare_valid <= source_valid && config_valid;
                if (source_valid && config_valid) begin
                    prepare_lane_mask <= source_lane_mask;
                    prepare_tag <= source_tag;
                end
            end
        end
    end

    initial begin
        if (LANES != 64 || TAG_WIDTH < 1)
            $error("activation_quantizer requires LANES=64 and positive TAG_WIDTH");
    end

`ifndef SYNTHESIS
    logic previous_quantized_stalled;
    logic [LANES*8-1:0] previous_values;
    logic [LANES-1:0] previous_lane_mask;
    logic [TAG_WIDTH-1:0] previous_tag;
    always_ff @(posedge clk) begin
        if (rst) begin
            previous_quantized_stalled <= 1'b0;
            previous_values <= '0;
            previous_lane_mask <= '0;
            previous_tag <= '0;
        end else begin
            if (previous_quantized_stalled && !abort_request)
                assert (quantized_rsp_valid && quantized_rsp_values == previous_values &&
                        quantized_rsp_lane_mask == previous_lane_mask &&
                        quantized_rsp_tag == previous_tag)
                    else $error("activation_quantizer response changed while stalled");
            previous_quantized_stalled <= quantized_rsp_valid && !quantized_rsp_ready && !abort_request;
            previous_values <= quantized_rsp_values;
            previous_lane_mask <= quantized_rsp_lane_mask;
            previous_tag <= quantized_rsp_tag;
        end
    end
`endif
endmodule

`default_nettype wire
`default_nettype none

module dynamic_activation_scale_iterative (
    input  logic        clk,
    input  logic        rst,
    input  logic        req_valid,
    output logic        req_ready,
    input  logic        req_a4,
    input  logic [15:0] req_max_abs,
    output logic        rsp_valid,
    input  logic        rsp_ready,
    output logic [15:0] rsp_scale_bf16
);
    logic busy;
    logic req_a4_saved;
    logic [3:0] bit_index;
    logic [15:0] numerator;
    logic [15:0] quotient;
    logic [15:0] remainder;
    logic signed [9:0] output_exponent;
    logic subnormal_result_saved;

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

    function automatic [7:0] normalize_mantissa(input logic [7:0] value);
        normalize_mantissa = value << normalization_shift8(value);
    endfunction

    logic [7:0] input_mantissa;
    logic [7:0] normalized_input_mantissa;
    logic signed [9:0] normalized_input_exponent;
    logic [8:0] request_mantissa;
    logic [2:0] request_subnormal_shift;
    logic [15:0] request_subnormal_numerator;
    logic [15:0] request_subnormal_limit;
    logic request_subnormal_result;
    logic [7:0] divisor;
    logic [16:0] shifted_remainder;
    logic divide_take;
    logic [16:0] next_remainder;
    logic [15:0] next_quotient;
    logic round_up;
    logic [16:0] rounded_quotient;
    logic normalize_result;
    logic [6:0] result_mantissa;
    logic signed [10:0] biased_exponent;

    always_comb begin
        input_mantissa = req_max_abs[14:7] == 0 ? {1'b0, req_max_abs[6:0]} : {1'b1, req_max_abs[6:0]};
        normalized_input_mantissa = normalize_mantissa(input_mantissa);
        normalized_input_exponent = (req_max_abs[14:7] == 0 ? -10'sd126 :
            $signed({2'b00, req_max_abs[14:7]}) - 10'sd127) -
            $signed({7'd0, normalization_shift8(input_mantissa)});
        request_mantissa = req_max_abs[14:7] == 0 ?
            {2'b00, req_max_abs[6:0]} : {1'b0, 1'b1, req_max_abs[6:0]};
        request_subnormal_shift = req_max_abs[14:7] == 0 ? 3'd0 :
            req_max_abs[14:7] <= 8'd7 ? req_max_abs[9:7] - 3'd1 : 3'd0;
        request_subnormal_numerator =
            {7'd0, request_mantissa} << request_subnormal_shift;
        request_subnormal_limit = req_a4 ? 16'd896 : 16'd16256;
        request_subnormal_result = req_max_abs[14:7] <= 8'd7 &&
            request_subnormal_numerator < request_subnormal_limit;
        divisor = req_a4_saved ? 8'd7 : 8'd127;
        shifted_remainder = {remainder, numerator[bit_index]};
        divide_take = shifted_remainder >= {9'd0, divisor};
        next_remainder = divide_take ? shifted_remainder - {9'd0, divisor} : shifted_remainder;
        next_quotient = divide_take ? quotient | (16'd1 << bit_index) : quotient;
        round_up = next_remainder * 2 > {9'd0, divisor} ||
            (next_remainder * 2 == {9'd0, divisor} && next_quotient[0]);
        rounded_quotient = {1'b0, next_quotient} + {16'd0, round_up};
        normalize_result = rounded_quotient >= 17'd256;
        result_mantissa = normalize_result ? rounded_quotient[7:1] : rounded_quotient[6:0];
        biased_exponent = output_exponent + 11'sd127 + $signed({10'd0, normalize_result});
    end

    assign req_ready = !busy && !rsp_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            req_a4_saved <= 1'b0;
            bit_index <= 4'd0;
            numerator <= 16'd0;
            quotient <= 16'd0;
            remainder <= 16'd0;
            output_exponent <= 10'sd0;
            subnormal_result_saved <= 1'b0;
            rsp_valid <= 1'b0;
            rsp_scale_bf16 <= 16'h3f80;
        end else begin
            if (rsp_valid && rsp_ready)
                rsp_valid <= 1'b0;
            if (req_valid && req_ready) begin
                req_a4_saved <= req_a4;
                if (req_max_abs[15] || req_max_abs[14:0] == 0) begin
                    rsp_scale_bf16 <= 16'h3f80;
                    rsp_valid <= 1'b1;
                end else begin
                    subnormal_result_saved <= request_subnormal_result;
                    if (request_subnormal_result) begin
                        numerator <= request_subnormal_numerator;
                        output_exponent <= -10'sd127;
                    end else if (req_a4) begin
                        if (normalized_input_mantissa >= 8'd224) begin
                            numerator <= {6'd0, normalized_input_mantissa, 2'd0};
                            output_exponent <= normalized_input_exponent - 10'sd2;
                        end else begin
                            numerator <= {5'd0, normalized_input_mantissa, 3'd0};
                            output_exponent <= normalized_input_exponent - 10'sd3;
                        end
                    end else if (normalized_input_mantissa >= 8'd254) begin
                        numerator <= {2'd0, normalized_input_mantissa, 6'd0};
                        output_exponent <= normalized_input_exponent - 10'sd6;
                    end else begin
                        numerator <= {1'd0, normalized_input_mantissa, 7'd0};
                        output_exponent <= normalized_input_exponent - 10'sd7;
                    end
                    quotient <= 16'd0;
                    remainder <= 16'd0;
                    bit_index <= 4'd15;
                    busy <= 1'b1;
                end
            end else if (busy) begin
                quotient <= next_quotient;
                remainder <= next_remainder[15:0];
                if (bit_index == 0) begin
                    if (subnormal_result_saved)
                        rsp_scale_bf16 <= rounded_quotient == 0 ?
                            16'h3f80 : rounded_quotient[15:0];
                    else if (biased_exponent <= 0)
                        rsp_scale_bf16 <= rounded_quotient == 0 ?
                            16'h3f80 : rounded_quotient[15:0];
                    else if (biased_exponent >= 255)
                        rsp_scale_bf16 <= 16'h7f80;
                    else
                        rsp_scale_bf16 <= {1'b0, biased_exponent[7:0], result_mantissa};
                    rsp_valid <= 1'b1;
                    busy <= 1'b0;
                end else begin
                    bit_index <= bit_index - 4'd1;
                end
            end
        end
    end
endmodule

`default_nettype wire
