`default_nettype none

module pe_product_pipeline #(
    parameter integer TAG_WIDTH = 16
) (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       req_valid,
    output logic                       req_ready,
    input  logic [1:0]                 req_mode,
    input  logic [8*32*4-1:0]          req_activation_payload,
    input  logic [32*8*4-1:0]          req_weight_payload,
    input  logic [7:0]                 req_row_mask,
    input  logic [31:0]                req_k_mask,
    input  logic [7:0]                 req_col_mask,
    input  logic                       req_first_k_step,
    input  logic                       req_last_k_step,
    input  logic                       req_mixed_phase,
    input  logic                       req_mixed_phase_first,
    input  logic [7:0]                 req_mixed_a8_rows,
    input  logic [8*16-1:0]            req_activation_scales,
    input  logic [8*16-1:0]            req_weight_scales,
    input  logic [TAG_WIDTH-1:0]       req_tag,
    output logic                       rsp_valid,
    input  logic                       rsp_ready,
    output logic [64*18-1:0]           rsp_step_sums,
    output logic [63:0]                rsp_sum_mask,
    output logic [1:0]                 rsp_mode,
    output logic                       rsp_first_k_step,
    output logic                       rsp_last_k_step,
    output logic                       rsp_mixed_phase,
    output logic                       rsp_mixed_phase_first,
    output logic [7:0]                 rsp_mixed_a8_rows,
    output logic [8*16-1:0]            rsp_activation_scales,
    output logic [8*16-1:0]            rsp_weight_scales,
    output logic [TAG_WIDTH-1:0]       rsp_tag,
    output logic                       idle
);
    localparam logic [1:0] MODE_W4A4 = 2'd0;
    localparam logic [1:0] MODE_W4A8 = 2'd1;
    localparam logic [1:0] MODE_W8A8 = 2'd2;
    localparam logic [1:0] MODE_MIXED_W4 = 2'd3;
    localparam integer META_WIDTH =
        2 + 8 + 8 + 1 + 1 + 1 + 1 + 8 + 8*16 + 8*16 + TAG_WIDTH;

    logic s1_valid;
    logic s2_valid;
    logic pipeline_advance;
    logic [META_WIDTH-1:0] s1_meta;
    logic [META_WIDTH-1:0] s2_meta;

    // Stage 1 stores four exact, mode-normalized partial sums per output.
    logic [64*4*16-1:0] s1_partial_sums;
    logic [64*8*15-1:0] stage1_group_sums;
    logic [64*4*16-1:0] stage1_values;
    logic [64*8-1:0] stage1_overflow;
    logic [64*32*16-1:0] slice_values;
    logic [8*32*4-1:0] formatted_activation;
    logic [8*32*4-1:0] masked_activation;
    logic [8*32-1:0] formatted_activation_unsigned;
    logic [8*32-1:0] formatted_k_active;
    logic [8*32-1:0] formatted_shift4;
    logic [8*32-1:0] formatted_shift8;
    logic [32*8*4-1:0] formatted_weight;
    logic [32*8*4-1:0] masked_weight;
    logic [32*8*4-1:0] masked_mixed_a4_weight;
    logic [32*8-1:0] formatted_weight_unsigned;
    logic [7:0] mixed_a4_weight_rows;

    // Stage 2 performs the complete 4-to-1 balanced reduction.
    logic [64*2*17-1:0] stage2_level2;
    logic [64*18-1:0] stage2_values;
    logic [64*18-1:0] s2_step_sums;
    logic [7:0] s2_row_mask;
    logic [7:0] s2_col_mask;

    // The two product stages advance together. The only stall source is the
    // accumulator/accum_result boundary in mixed_precision_pe.
    assign pipeline_advance = !s2_valid || rsp_ready;
    assign req_ready = pipeline_advance;
    assign rsp_valid = s2_valid;
    assign rsp_step_sums = s2_step_sums;
    assign idle = !(s1_valid || s2_valid);

    assign {rsp_mode, s2_row_mask, s2_col_mask,
            rsp_first_k_step, rsp_last_k_step,
            rsp_mixed_phase, rsp_mixed_phase_first, rsp_mixed_a8_rows,
            rsp_activation_scales, rsp_weight_scales, rsp_tag} = s2_meta;
    assign rsp_sum_mask = {
        {8{s2_row_mask[7]}}, {8{s2_row_mask[6]}},
        {8{s2_row_mask[5]}}, {8{s2_row_mask[4]}},
        {8{s2_row_mask[3]}}, {8{s2_row_mask[2]}},
        {8{s2_row_mask[1]}}, {8{s2_row_mask[0]}}
    } & {8{s2_col_mask}};

    generate
        for (genvar row = 0; row < 8; row = row + 1) begin : activation_rows
            assign mixed_a4_weight_rows[row] =
                req_mode == MODE_MIXED_W4 && !req_mixed_a8_rows[row];
            for (genvar slice = 0; slice < 32; slice = slice + 1) begin : activation_slices
                localparam integer W4A8_K = slice / 2;
                localparam integer W4A8_PART = slice % 2;
                localparam integer FIXED_K = slice / 4;
                localparam integer FIXED_PART = slice % 4;
                always_comb begin
                    formatted_activation[(row*32+slice)*4 +: 4] = 4'd0;
                    formatted_activation_unsigned[row*32+slice] = 1'b0;
                    formatted_k_active[row*32+slice] = 1'b0;
                    formatted_shift4[row*32+slice] = 1'b0;
                    formatted_shift8[row*32+slice] = 1'b0;
                    case (req_mode)
                        MODE_W4A4: begin
                            formatted_activation[(row*32+slice)*4 +: 4] =
                                req_activation_payload[(row*32+slice)*4 +: 4];
                            formatted_k_active[row*32+slice] =
                                req_k_mask[slice];
                        end
                        MODE_W4A8: begin
                            formatted_activation[(row*32+slice)*4 +: 4] =
                                W4A8_PART == 0 ?
                                    req_activation_payload[
                                        (row*16+W4A8_K)*8+4 +: 4] :
                                    req_activation_payload[
                                        (row*16+W4A8_K)*8 +: 4];
                            formatted_activation_unsigned[row*32+slice] =
                                W4A8_PART != 0;
                            formatted_k_active[row*32+slice] =
                                req_k_mask[W4A8_K];
                            formatted_shift4[row*32+slice] =
                                W4A8_PART == 0;
                        end
                        MODE_W8A8: begin
                            formatted_activation[(row*32+slice)*4 +: 4] =
                                FIXED_PART < 2 ?
                                    req_activation_payload[
                                        (row*8+FIXED_K)*8+4 +: 4] :
                                    req_activation_payload[
                                        (row*8+FIXED_K)*8 +: 4];
                            formatted_activation_unsigned[row*32+slice] =
                                FIXED_PART >= 2;
                            formatted_k_active[row*32+slice] =
                                req_k_mask[FIXED_K];
                            formatted_shift8[row*32+slice] = FIXED_PART == 0;
                            formatted_shift4[row*32+slice] =
                                FIXED_PART == 1 || FIXED_PART == 2;
                        end
                        MODE_MIXED_W4: begin
                            if (req_mixed_a8_rows[row]) begin
                                formatted_activation[
                                    (row*32+slice)*4 +: 4] =
                                    W4A8_PART == 0 ?
                                        req_activation_payload[
                                            (row*16+W4A8_K)*8+4 +: 4] :
                                        req_activation_payload[
                                            (row*16+W4A8_K)*8 +: 4];
                                formatted_activation_unsigned[row*32+slice] =
                                    W4A8_PART != 0;
                                formatted_k_active[row*32+slice] =
                                    req_mixed_phase ?
                                        req_k_mask[16+W4A8_K] :
                                        req_k_mask[W4A8_K];
                                formatted_shift4[row*32+slice] =
                                    W4A8_PART == 0;
                            end else begin
                                formatted_activation[
                                    (row*32+slice)*4 +: 4] =
                                    req_activation_payload[
                                        (row*32+slice)*4 +: 4];
                                formatted_k_active[row*32+slice] =
                                    req_k_mask[slice];
                            end
                        end
                        default: begin end
                    endcase
                end
                assign masked_activation[(row*32+slice)*4 +: 4] =
                    req_row_mask[row] && formatted_k_active[row*32+slice] ?
                    formatted_activation[(row*32+slice)*4 +: 4] : 4'd0;
            end
        end

        for (genvar column = 0; column < 8; column = column + 1) begin : weight_columns
            for (genvar slice = 0; slice < 32; slice = slice + 1) begin : weight_slices
                localparam integer W4A8_K = slice / 2;
                localparam integer FIXED_K = slice / 4;
                localparam integer FIXED_PART = slice % 4;
                always_comb begin
                    formatted_weight[(slice*8+column)*4 +: 4] = 4'd0;
                    formatted_weight_unsigned[slice*8+column] = 1'b0;
                    case (req_mode)
                        MODE_W4A4:
                            formatted_weight[(slice*8+column)*4 +: 4] =
                                req_weight_payload[(slice*8+column)*4 +: 4];
                        MODE_W4A8:
                            formatted_weight[(slice*8+column)*4 +: 4] =
                                req_weight_payload[(W4A8_K*8+column)*4 +: 4];
                        MODE_W8A8: begin
                            formatted_weight[(slice*8+column)*4 +: 4] =
                                (FIXED_PART == 0 || FIXED_PART == 2) ?
                                    req_weight_payload[
                                        (FIXED_K*8+column)*8+4 +: 4] :
                                    req_weight_payload[
                                        (FIXED_K*8+column)*8 +: 4];
                            formatted_weight_unsigned[slice*8+column] =
                                FIXED_PART == 1 || FIXED_PART == 3;
                        end
                        MODE_MIXED_W4:
                            formatted_weight[(slice*8+column)*4 +: 4] =
                                req_mixed_phase ?
                                    req_weight_payload[
                                        ((16+W4A8_K)*8+column)*4 +: 4] :
                                    req_weight_payload[
                                        (W4A8_K*8+column)*4 +: 4];
                        default: begin end
                    endcase
                end
                assign masked_weight[(slice*8+column)*4 +: 4] =
                    req_col_mask[column] ?
                    formatted_weight[(slice*8+column)*4 +: 4] : 4'd0;
                assign masked_mixed_a4_weight[(slice*8+column)*4 +: 4] =
                    req_col_mask[column] ?
                    req_weight_payload[(slice*8+column)*4 +: 4] : 4'd0;
            end
        end

        for (genvar row = 0; row < 8; row = row + 1) begin : product_rows
            for (genvar column = 0; column < 8; column = column + 1) begin : product_columns
                for (genvar slice = 0; slice < 32; slice = slice + 1) begin : product_slices
                    logic [3:0] activation_quantized;
                    logic activation_unsigned;
                    logic [3:0] weight_quantized;
                    logic weight_unsigned;
                    logic signed [3:0] activation_operand;
                    logic signed [3:0] weight_operand;
                    logic signed [7:0] base_product;
                    logic signed [9:0] activation_unsigned_correction;
                    logic signed [9:0] weight_unsigned_correction;
                    logic signed [9:0] unsigned_cross_correction;
                    logic signed [9:0] nibble_product;
                    logic signed [15:0] product_extended;
                    logic signed [15:0] contribution;

                    always_comb begin
                        activation_quantized = masked_activation[
                            (row*32+slice)*4 +: 4];
                        activation_unsigned =
                            formatted_activation_unsigned[row*32+slice];
                        weight_quantized = mixed_a4_weight_rows[row] ?
                            masked_mixed_a4_weight[
                                (slice*8+column)*4 +: 4] :
                            masked_weight[(slice*8+column)*4 +: 4];
                        weight_unsigned = mixed_a4_weight_rows[row] ? 1'b0 :
                            formatted_weight_unsigned[slice*8+column];

                        activation_operand = $signed(activation_quantized);
                        weight_operand = $signed(weight_quantized);
                        base_product = activation_operand * weight_operand;
                        activation_unsigned_correction =
                            activation_unsigned && activation_quantized[3] ?
                                $signed({{6{weight_operand[3]}},
                                         weight_operand}) <<< 4 : 10'sd0;
                        weight_unsigned_correction =
                            weight_unsigned && weight_quantized[3] ?
                                $signed({{6{activation_operand[3]}},
                                         activation_operand}) <<< 4 : 10'sd0;
                        unsigned_cross_correction =
                            activation_unsigned && activation_quantized[3] &&
                            weight_unsigned && weight_quantized[3] ?
                                10'sd256 : 10'sd0;
                        nibble_product =
                            $signed({{2{base_product[7]}}, base_product}) +
                            activation_unsigned_correction +
                            weight_unsigned_correction +
                            unsigned_cross_correction;
                        product_extended =
                            $signed({{6{nibble_product[9]}}, nibble_product});
                        if (formatted_shift8[row*32+slice])
                            contribution = product_extended <<< 8;
                        else if (formatted_shift4[row*32+slice])
                            contribution = product_extended <<< 4;
                        else
                            contribution = product_extended;
                    end

                    assign slice_values[
                        ((row*8+column)*32+slice)*16 +: 16] = contribution;
                end
            end
        end

        for (genvar lane = 0; lane < 64; lane = lane + 1) begin : stage1_lanes
            for (genvar group = 0; group < 8; group = group + 1) begin : stage1_groups
                wire signed [15:0] contribution0 = $signed(
                    slice_values[(lane*32+group*4)*16 +: 16]);
                wire signed [15:0] contribution1 = $signed(
                    slice_values[(lane*32+group*4+1)*16 +: 16]);
                wire signed [15:0] contribution2 = $signed(
                    slice_values[(lane*32+group*4+2)*16 +: 16]);
                wire signed [15:0] contribution3 = $signed(
                    slice_values[(lane*32+group*4+3)*16 +: 16]);
                wire signed [15:0] pair0 = contribution0 + contribution1;
                wire signed [15:0] pair1 = contribution2 + contribution3;
                wire signed [15:0] group_sum = pair0 + pair1;

                assign stage1_group_sums[(lane*8+group)*15 +: 15] =
                    group_sum[14:0];
                assign stage1_overflow[lane*8+group] =
                    group_sum[15] != group_sum[14];
            end
        end

        for (genvar lane = 0; lane < 64; lane = lane + 1) begin : stage2_lanes
            for (genvar pair = 0; pair < 4; pair = pair + 1) begin : level1_pairs
                assign stage1_values[(lane*4+pair)*16 +: 16] =
                    $signed({stage1_group_sums[(lane*8+pair*2)*15+14],
                             stage1_group_sums[(lane*8+pair*2)*15 +: 15]}) +
                    $signed({stage1_group_sums[(lane*8+pair*2+1)*15+14],
                             stage1_group_sums[(lane*8+pair*2+1)*15 +: 15]});
            end
            for (genvar pair = 0; pair < 2; pair = pair + 1) begin : level2_pairs
                assign stage2_level2[(lane*2+pair)*17 +: 17] =
                    $signed({s1_partial_sums[(lane*4+pair*2)*16+15],
                             s1_partial_sums[(lane*4+pair*2)*16 +: 16]}) +
                    $signed({s1_partial_sums[(lane*4+pair*2+1)*16+15],
                             s1_partial_sums[(lane*4+pair*2+1)*16 +: 16]});
            end
            assign stage2_values[lane*18 +: 18] =
                $signed({stage2_level2[(lane*2)*17+16],
                         stage2_level2[(lane*2)*17 +: 17]}) +
                $signed({stage2_level2[(lane*2+1)*17+16],
                         stage2_level2[(lane*2+1)*17 +: 17]});
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0;
            s2_valid <= 1'b0;
        end else if (pipeline_advance) begin
            s1_valid <= req_valid;
            if (req_valid) begin
                s1_partial_sums <= stage1_values;
                s1_meta <= {req_mode, req_row_mask, req_col_mask,
                            req_first_k_step, req_last_k_step,
                            req_mixed_phase, req_mixed_phase_first,
                            req_mixed_a8_rows,
                            req_activation_scales, req_weight_scales, req_tag};
            end
            s2_valid <= s1_valid;
            if (s1_valid) begin
                s2_step_sums <= stage2_values;
                s2_meta <= s1_meta;
            end
        end
    end

    initial begin
        if (TAG_WIDTH < 1)
            $error("pe_product_pipeline requires TAG_WIDTH >= 1");
    end

`ifndef SYNTHESIS
    logic stalled_response;
    logic [64*18+META_WIDTH-1:0] stalled_payload;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_response <= 1'b0;
            stalled_payload <= '0;
        end else begin
            if (req_valid && req_ready) begin
                assert (req_mode == MODE_MIXED_W4 ||
                        (!req_mixed_phase && !req_mixed_phase_first &&
                         req_mixed_a8_rows == 8'd0))
                    else $error("pe_product_pipeline received mixed row fields for mode %0d",
                                req_mode);
                assert (!(|stage1_overflow))
                    else $error("pe_product_pipeline Stage 1 exceeded signed 15 bit");
            end
            if (stalled_response)
                assert (rsp_valid && {rsp_step_sums, s2_meta} == stalled_payload)
                    else $error("pe_product_pipeline changed stalled response");
            stalled_response <= rsp_valid && !rsp_ready;
            stalled_payload <= {rsp_step_sums, s2_meta};
        end
    end
`endif
endmodule

`default_nettype wire
