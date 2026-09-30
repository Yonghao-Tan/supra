`default_nettype none

// Runs one batch of at most eight Softmax rows. Each main pass transfers one
// 8-row x 8-key tile per accepted request. The active client supplies score/probability
// SRAM access and the shared BF16, max, reduction, and quantizer resources.
module softmax_row_pipeline #(
    parameter integer MAX_LENGTH = 2048,
    parameter integer DIM_WIDTH = 12,
    parameter integer TAG_WIDTH = 16
) (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    output logic          abort_ack,
    input  logic          start_valid,
    output logic          start_ready,
    input  logic [3:0]    row_count,
    input  logic [DIM_WIDTH-1:0] row_length,
    input  logic          capture_probability_enable,
    output logic          capture_probability_valid,
    input  logic          capture_probability_ready,

    output logic          source_req_valid,
    input  logic          source_req_ready,
    output logic [1:0]    source_req_pass,
    output logic [DIM_WIDTH-1:0] source_req_key,
    output logic [TAG_WIDTH-1:0] source_req_tag,
    input  logic          source_rsp_valid,
    output logic          source_rsp_ready,
    input  logic [1023:0] source_rsp_values,
    input  logic [63:0]   source_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0] source_rsp_tag,

    output logic          bf16_write_valid,
    input  logic          bf16_write_ready,
    output logic          bf16_write_probability,
    output logic [DIM_WIDTH-1:0] bf16_write_key,
    output logic [1023:0] bf16_write_values,
    output logic [63:0]   bf16_write_lane_mask,
    output logic [TAG_WIDTH-1:0] bf16_write_tag,

    output logic          quantized_write_valid,
    input  logic          quantized_write_ready,
    output logic [DIM_WIDTH-1:0] quantized_write_key,
    output logic [511:0]  quantized_write_values,
    output logic [63:0]   quantized_write_lane_mask,
    output logic [TAG_WIDTH-1:0] quantized_write_tag,
    output logic          scale_write_valid,
    input  logic          scale_write_ready,
    output logic [127:0]  scale_write_values,
    output logic [7:0]    scale_write_row_mask,

    output logic          scratch_read_valid,
    input  logic          scratch_read_ready,
    output logic          scratch_read_bank,
    output logic [9:0]    scratch_read_left_address,
    output logic [9:0]    scratch_read_right_address,
    output logic [TAG_WIDTH-1:0] scratch_read_tag,
    input  logic          scratch_rsp_valid,
    output logic          scratch_rsp_ready,
    input  logic [127:0]  scratch_rsp_left_data,
    input  logic [127:0]  scratch_rsp_right_data,
    input  logic [TAG_WIDTH-1:0] scratch_rsp_tag,
    output logic          scratch_write_valid,
    input  logic          scratch_write_ready,
    output logic          scratch_write_bank,
    output logic [9:0]    scratch_write_address,
    output logic [127:0]  scratch_write_data,
    output logic [15:0]   scratch_write_byte_enable,

    output logic          vector_req_valid,
    input  logic          vector_req_ready,
    output logic [2:0]    vector_req_operation,
    output logic [1023:0] vector_req_values,
    output logic [1023:0] vector_req_paired_values,
    output logic [1023:0] vector_req_factor0_values,
    output logic [1023:0] vector_req_factor1_values,
    output logic [63:0]   vector_req_lane_mask,
    output logic [TAG_WIDTH-1:0] vector_req_tag,
    input  logic          vector_rsp_valid,
    output logic          vector_rsp_ready,
    input  logic [1023:0] vector_rsp_values,
    input  logic [63:0]   vector_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0] vector_rsp_tag,

    output logic          max_req_valid,
    input  logic          max_req_ready,
    output logic          max_req_magnitude,
    output logic [1023:0] max_req_values,
    output logic [63:0]   max_req_lane_mask,
    output logic [TAG_WIDTH-1:0] max_req_tag,
    input  logic          max_rsp_valid,
    output logic          max_rsp_ready,
    input  logic [127:0]  max_rsp_values,
    input  logic [7:0]    max_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0] max_rsp_tag,

    output logic          reduction_req_valid,
    input  logic          reduction_req_ready,
    output logic [1023:0] reduction_req_values,
    output logic [63:0]   reduction_req_lane_mask,
    output logic [TAG_WIDTH-1:0] reduction_req_tag,
    input  logic          reduction_rsp_valid,
    output logic          reduction_rsp_ready,
    input  logic [127:0]  reduction_rsp_values,
    input  logic [7:0]    reduction_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0] reduction_rsp_tag,

    output logic          exp_req_valid,
    input  logic          exp_req_ready,
    output logic [1023:0] exp_req_delta_values,
    output logic [63:0]   exp_req_lane_mask,
    output logic [TAG_WIDTH-1:0] exp_req_tag,
    input  logic          exp_rsp_valid,
    output logic          exp_rsp_ready,
    input  logic [1023:0] exp_rsp_values,
    input  logic [63:0]   exp_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0] exp_rsp_tag,

    output logic          reciprocal_req_valid,
    input  logic          reciprocal_req_ready,
    output logic [127:0]  reciprocal_req_sum_values,
    output logic [7:0]    reciprocal_req_row_mask,
    output logic [TAG_WIDTH-1:0] reciprocal_req_tag,
    input  logic          reciprocal_rsp_valid,
    output logic          reciprocal_rsp_ready,
    input  logic [127:0]  reciprocal_rsp_values,
    input  logic [7:0]    reciprocal_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0] reciprocal_rsp_tag,

    output logic          quant_scale_req_valid,
    input  logic          quant_scale_req_ready,
    output logic [127:0]  quant_scale_req_row_max_abs,
    output logic [7:0]    quant_scale_req_row_mask,
    input  logic          quant_scale_rsp_valid,
    output logic          quant_scale_rsp_ready,
    input  logic [127:0]  quant_scale_rsp_values,
    output logic          quant_values_req_valid,
    input  logic          quant_values_req_ready,
    output logic [1023:0] quant_values_req_values,
    output logic [63:0]   quant_values_req_lane_mask,
    output logic [TAG_WIDTH-1:0] quant_values_req_tag,
    input  logic          quant_values_rsp_valid,
    output logic          quant_values_rsp_ready,
    input  logic [511:0]  quant_values_rsp_values,
    input  logic [63:0]   quant_values_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0] quant_values_rsp_tag,

    output logic          done_pulse,
    output logic          error,
    output logic [63:0]   accepted_source_tile_count,
    output logic [63:0]   accepted_bf16_write_count,
    output logic [63:0]   accepted_quantized_write_count,
    output logic [63:0]   accepted_scratch_read_count,
    output logic [63:0]   accepted_scratch_write_count,
    output logic [63:0]   accepted_vector_request_count,
    output logic [63:0]   accepted_max_request_count,
    output logic [63:0]   accepted_reduction_request_count,
    output logic [63:0]   accepted_quantized_value_count
);
    localparam logic [2:0] OP_ADD = 3'd0;
    localparam logic [2:0] OP_MULTIPLY = 3'd1;
    localparam logic [1:0] READ_SCORE_MAX = 2'd0;
    localparam logic [1:0] READ_SCORE_EXP = 2'd1;
    localparam logic [1:0] READ_EXP_NORMALIZE = 2'd2;
    localparam logic [1:0] READ_PROBABILITY_QUANTIZED = 2'd3;

    typedef enum logic [3:0] {
        IDLE, MAX_PASS, EXP_PASS, PAD_ROOTS, REDUCE_LEVEL,
        RECIPROCAL_REQUEST, RECIPROCAL_RESPONSE, NORMALIZE_PASS,
        SCALE_REQUEST, SCALE_RESPONSE,
        SCALE_WRITE, QUANTIZED_PASS, COMPLETE, ABORT_DRAIN, ABORT_WAIT_LOW,
        CAPTURE_PROBABILITY
    } state_t;
    state_t state;

    logic [3:0] saved_row_count;
    logic [DIM_WIDTH-1:0] saved_row_length;
    logic [7:0] row_mask;
    logic [9:0] tile_count;
    logic [9:0] padded_root_count;
    logic [9:0] source_request_count;
    logic [9:0] source_response_count;
    logic [9:0] bf16_write_count;
    logic [9:0] max_response_count;
    logic [9:0] scratch_write_count;
    logic [9:0] level_pair_count;
    logic [9:0] level_read_count;
    logic [9:0] quantized_write_count;
    logic source_scratch_bank;
    logic destination_scratch_bank;
    logic [127:0] row_maximum;
    logic [127:0] probability_maximum;
    logic [127:0] row_sum;
    logic [127:0] reciprocal;
    logic [127:0] quant_scales;
    logic internal_error;
    logic saved_capture_probability_enable;

    logic [10:0] source_outstanding;
    logic [10:0] vector_outstanding;
    logic [10:0] max_outstanding;
    logic [10:0] reduction_outstanding;
    logic [2:0] exp_outstanding;
    logic reciprocal_outstanding;
    logic [10:0] quant_outstanding;
    logic [10:0] scratch_outstanding;
    logic quant_scale_outstanding;

    logic source_request_fire, source_response_fire, vector_request_fire;
    logic vector_response_fire, max_request_fire, max_response_fire;
    logic reduction_request_fire, reduction_response_fire;
    logic scratch_read_fire, scratch_write_fire, bf16_write_fire;
    logic exp_request_fire, exp_response_fire;
    logic reciprocal_request_fire, reciprocal_response_fire;
    logic quant_values_request_fire, quant_values_response_fire, quantized_write_fire;
    wire start_fire = start_valid && start_ready;

    function automatic [9:0] next_power_of_two(input [9:0] value);
        logic [9:0] result;
        begin
            result = 10'd1;
            for (integer bit_index = 0; bit_index < 10; bit_index = bit_index + 1)
                if (result < value)
                    result = result << 1;
            next_power_of_two = result;
        end
    endfunction

    function automatic logic bf16_greater(input logic [15:0] lhs,
                                            input logic [15:0] rhs);
        begin
            if (lhs[14:0] == 0 && rhs[14:0] == 0)
                bf16_greater = 1'b0;
            else if (lhs[15] != rhs[15])
                bf16_greater = rhs[15];
            else if (!lhs[15])
                bf16_greater = lhs[14:0] > rhs[14:0];
            else
                bf16_greater = lhs[14:0] < rhs[14:0];
        end
    endfunction

    assign start_ready = state == IDLE && !abort_request;
    assign capture_probability_valid = state == CAPTURE_PROBABILITY && !abort_request;
    assign row_mask = saved_row_count == 8 ? 8'hff :
        (8'h01 << saved_row_count) - 1'b1;
    assign source_req_valid = (state == MAX_PASS || state == EXP_PASS ||
        state == NORMALIZE_PASS || state == QUANTIZED_PASS) &&
        source_request_count < tile_count && !abort_request;
    assign source_req_pass = state == MAX_PASS ? READ_SCORE_MAX :
        state == EXP_PASS ? READ_SCORE_EXP :
        state == NORMALIZE_PASS ? READ_EXP_NORMALIZE : READ_PROBABILITY_QUANTIZED;
    assign source_req_key = DIM_WIDTH'({source_request_count, 3'b000});
    assign source_req_tag = TAG_WIDTH'(source_request_count);
    assign source_request_fire = source_req_valid && source_req_ready;

    always_comb begin
        source_rsp_ready = 1'b0;
        vector_req_valid = 1'b0;
        vector_req_operation = OP_ADD;
        vector_req_values = source_rsp_values;
        vector_req_paired_values = '0;
        vector_req_factor0_values = '0;
        vector_req_factor1_values = '0;
        vector_req_lane_mask = source_rsp_lane_mask;
        vector_req_tag = source_rsp_tag;
        max_req_valid = 1'b0;
        max_req_magnitude = state == NORMALIZE_PASS;
        max_req_values = source_rsp_values;
        max_req_lane_mask = source_rsp_lane_mask;
        max_req_tag = source_rsp_tag;
        quant_values_req_valid = 1'b0;
        quant_values_req_values = source_rsp_values;
        quant_values_req_lane_mask = source_rsp_lane_mask;
        quant_values_req_tag = source_rsp_tag;
        if (state == MAX_PASS) begin
            max_req_valid = source_rsp_valid;
            source_rsp_ready = max_req_ready;
        end else if (state == EXP_PASS) begin
            vector_req_valid = source_rsp_valid;
            source_rsp_ready = vector_req_ready;
            for (integer row = 0; row < 8; row = row + 1)
                for (integer element = 0; element < 8; element = element + 1)
                    vector_req_paired_values[(row*8+element)*16 +: 16] =
                        {~row_maximum[row*16+15], row_maximum[row*16 +: 15]};
        end else if (state == NORMALIZE_PASS) begin
            vector_req_valid = source_rsp_valid;
            vector_req_operation = OP_MULTIPLY;
            source_rsp_ready = vector_req_ready;
            for (integer row = 0; row < 8; row = row + 1)
                for (integer element = 0; element < 8; element = element + 1)
                    vector_req_factor0_values[(row*8+element)*16 +: 16] =
                        reciprocal[row*16 +: 16];
            max_req_valid = vector_rsp_valid && bf16_write_ready;
            max_req_magnitude = 1'b1;
            max_req_values = vector_rsp_values;
            max_req_lane_mask = vector_rsp_lane_mask;
            max_req_tag = vector_rsp_tag;
        end else if (state == QUANTIZED_PASS) begin
            quant_values_req_valid = source_rsp_valid;
            source_rsp_ready = quant_values_req_ready;
        end else if (state == ABORT_DRAIN) begin
            source_rsp_ready = 1'b1;
        end
    end
    assign source_response_fire = source_rsp_valid && source_rsp_ready;
    assign vector_request_fire = vector_req_valid && vector_req_ready;
    assign quant_values_request_fire = quant_values_req_valid && quant_values_req_ready;

    assign exp_req_valid = state == EXP_PASS && vector_rsp_valid &&
        !abort_request;
    assign exp_req_delta_values = vector_rsp_values;
    assign exp_req_lane_mask = vector_rsp_lane_mask;
    assign exp_req_tag = vector_rsp_tag;
    assign exp_request_fire = exp_req_valid && exp_req_ready;
    assign exp_rsp_ready = state == EXP_PASS ?
        (bf16_write_ready && reduction_req_ready) : state == ABORT_DRAIN;
    assign exp_response_fire = exp_rsp_valid && exp_rsp_ready;
    assign vector_rsp_ready = state == EXP_PASS ? exp_req_ready :
        state == NORMALIZE_PASS ? (bf16_write_ready && max_req_ready) :
        state == ABORT_DRAIN;
    assign vector_response_fire = vector_rsp_valid && vector_rsp_ready;

    always_comb begin
        bf16_write_valid = 1'b0;
        bf16_write_probability = 1'b0;
        bf16_write_key = '0;
        bf16_write_values = '0;
        bf16_write_lane_mask = '0;
        bf16_write_tag = '0;
        reduction_req_valid = 1'b0;
        reduction_req_values = '0;
        reduction_req_lane_mask = '0;
        reduction_req_tag = '0;
        max_rsp_ready = state == MAX_PASS || state == NORMALIZE_PASS ||
            state == ABORT_DRAIN;
        if (state == EXP_PASS) begin
            bf16_write_valid = exp_rsp_valid && reduction_req_ready;
            bf16_write_key = DIM_WIDTH'(exp_rsp_tag << 3);
            bf16_write_values = exp_rsp_values;
            bf16_write_lane_mask = exp_rsp_lane_mask;
            bf16_write_tag = exp_rsp_tag;
            reduction_req_valid = exp_rsp_valid && bf16_write_ready;
            reduction_req_values = exp_rsp_values;
            reduction_req_lane_mask = exp_rsp_lane_mask;
            reduction_req_tag = exp_rsp_tag;
        end else if (state == NORMALIZE_PASS) begin
            bf16_write_valid = vector_rsp_valid && max_req_ready;
            bf16_write_probability = 1'b1;
            bf16_write_key = DIM_WIDTH'(vector_rsp_tag << 3);
            bf16_write_values = vector_rsp_values;
            bf16_write_lane_mask = vector_rsp_lane_mask;
            bf16_write_tag = vector_rsp_tag;
        end else if (state == REDUCE_LEVEL) begin
            reduction_req_valid = scratch_rsp_valid;
            reduction_req_tag = scratch_rsp_tag;
            for (integer row = 0; row < 8; row = row + 1) begin
                reduction_req_values[(row*8)*16 +: 16] =
                    scratch_rsp_left_data[row*16 +: 16];
                reduction_req_values[(row*8+1)*16 +: 16] =
                    scratch_rsp_right_data[row*16 +: 16];
                reduction_req_lane_mask[row*8] = row_mask[row];
                reduction_req_lane_mask[row*8+1] = row_mask[row];
            end
        end
    end
    assign bf16_write_fire = bf16_write_valid && bf16_write_ready;
    assign max_request_fire = max_req_valid && max_req_ready;
    assign max_response_fire = max_rsp_valid && max_rsp_ready;
    assign reduction_request_fire = reduction_req_valid && reduction_req_ready;

    assign scratch_read_valid = state == REDUCE_LEVEL &&
        level_read_count < level_pair_count && !abort_request;
    assign scratch_read_bank = source_scratch_bank;
    assign scratch_read_left_address = level_read_count << 1;
    assign scratch_read_right_address = (level_read_count << 1) + 1'b1;
    assign scratch_read_tag = TAG_WIDTH'(level_read_count);
    assign scratch_read_fire = scratch_read_valid && scratch_read_ready;
    assign scratch_rsp_ready = state == REDUCE_LEVEL ? reduction_req_ready :
        state == ABORT_DRAIN;

    always_comb begin
        scratch_write_valid = 1'b0;
        scratch_write_bank = state == EXP_PASS ? 1'b0 : destination_scratch_bank;
        scratch_write_address = reduction_rsp_tag[9:0];
        scratch_write_data = reduction_rsp_values;
        scratch_write_byte_enable = {{2{reduction_rsp_row_mask[7]}},
            {2{reduction_rsp_row_mask[6]}}, {2{reduction_rsp_row_mask[5]}},
            {2{reduction_rsp_row_mask[4]}}, {2{reduction_rsp_row_mask[3]}},
            {2{reduction_rsp_row_mask[2]}}, {2{reduction_rsp_row_mask[1]}},
            {2{reduction_rsp_row_mask[0]}}};
        reduction_rsp_ready = state == ABORT_DRAIN;
        if (state == EXP_PASS || state == REDUCE_LEVEL) begin
            scratch_write_valid = reduction_rsp_valid;
            reduction_rsp_ready = scratch_write_ready;
        end else if (state == PAD_ROOTS) begin
            scratch_write_valid = scratch_write_count < padded_root_count;
            scratch_write_bank = 1'b0;
            scratch_write_address = scratch_write_count;
            scratch_write_data = '0;
            scratch_write_byte_enable = 16'hffff;
        end
    end
    assign reduction_response_fire = reduction_rsp_valid && reduction_rsp_ready;
    assign scratch_write_fire = scratch_write_valid && scratch_write_ready;

    assign reciprocal_req_valid = state == RECIPROCAL_REQUEST &&
        !abort_request;
    assign reciprocal_req_sum_values = row_sum;
    assign reciprocal_req_row_mask = row_mask;
    assign reciprocal_req_tag = '0;
    assign reciprocal_request_fire = reciprocal_req_valid &&
        reciprocal_req_ready;
    assign reciprocal_rsp_ready = state == RECIPROCAL_RESPONSE ||
        state == ABORT_DRAIN;
    assign reciprocal_response_fire = reciprocal_rsp_valid &&
        reciprocal_rsp_ready;
    assign quant_scale_req_valid = state == SCALE_REQUEST && !abort_request;
    assign quant_scale_req_row_max_abs = probability_maximum;
    assign quant_scale_req_row_mask = row_mask;
    assign quant_scale_rsp_ready = state == SCALE_RESPONSE || state == ABORT_DRAIN;
    assign scale_write_valid = state == SCALE_WRITE && !abort_request;
    assign scale_write_values = quant_scales;
    assign scale_write_row_mask = row_mask;
    assign quant_values_rsp_ready = state == QUANTIZED_PASS ? quantized_write_ready :
        state == ABORT_DRAIN;
    assign quantized_write_valid = state == QUANTIZED_PASS && quant_values_rsp_valid;
    assign quantized_write_key = DIM_WIDTH'(quant_values_rsp_tag << 3);
    assign quantized_write_values = quant_values_rsp_values;
    assign quantized_write_lane_mask = quant_values_rsp_lane_mask;
    assign quantized_write_tag = quant_values_rsp_tag;
    assign quant_values_response_fire = quant_values_rsp_valid && quant_values_rsp_ready;
    assign quantized_write_fire = quantized_write_valid && quantized_write_ready;

`ifdef SYNTHESIS
    always_comb begin
        accepted_source_tile_count = '0;
        accepted_bf16_write_count = '0;
        accepted_quantized_write_count = '0;
        accepted_scratch_read_count = '0;
        accepted_scratch_write_count = '0;
        accepted_vector_request_count = '0;
        accepted_max_request_count = '0;
        accepted_reduction_request_count = '0;
        accepted_quantized_value_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            saved_capture_probability_enable <= 1'b0;
            state <= IDLE;
            saved_row_count <= '0;
            saved_row_length <= '0;
            tile_count <= '0;
            padded_root_count <= '0;
            source_request_count <= '0;
            source_response_count <= '0;
            bf16_write_count <= '0;
            max_response_count <= '0;
            scratch_write_count <= '0;
            level_pair_count <= '0;
            level_read_count <= '0;
            quantized_write_count <= '0;
            source_scratch_bank <= 1'b0;
            destination_scratch_bank <= 1'b1;
            row_maximum <= '0;
            probability_maximum <= '0;
            row_sum <= '0;
            reciprocal <= '0;
            quant_scales <= '0;
            internal_error <= 1'b0;
            source_outstanding <= '0;
            vector_outstanding <= '0;
            max_outstanding <= '0;
            reduction_outstanding <= '0;
            exp_outstanding <= '0;
            reciprocal_outstanding <= 1'b0;
            quant_outstanding <= '0;
            scratch_outstanding <= '0;
            quant_scale_outstanding <= 1'b0;
            abort_ack <= 1'b0;
            done_pulse <= 1'b0;
            error <= 1'b0;
`ifndef SYNTHESIS
            accepted_source_tile_count <= '0;
            accepted_bf16_write_count <= '0;
            accepted_quantized_write_count <= '0;
            accepted_scratch_read_count <= '0;
            accepted_scratch_write_count <= '0;
            accepted_vector_request_count <= '0;
            accepted_max_request_count <= '0;
            accepted_reduction_request_count <= '0;
            accepted_quantized_value_count <= '0;
`endif
        end else begin
            done_pulse <= 1'b0;
            abort_ack <= 1'b0;

            case ({source_request_fire, source_response_fire})
                2'b10: source_outstanding <= source_outstanding + 1'b1;
                2'b01: source_outstanding <= source_outstanding - 1'b1;
                default: begin end
            endcase
            case ({vector_request_fire, vector_response_fire})
                2'b10: vector_outstanding <= vector_outstanding + 1'b1;
                2'b01: vector_outstanding <= vector_outstanding - 1'b1;
                default: begin end
            endcase
            case ({max_request_fire, max_response_fire})
                2'b10: max_outstanding <= max_outstanding + 1'b1;
                2'b01: max_outstanding <= max_outstanding - 1'b1;
                default: begin end
            endcase
            case ({reduction_request_fire, reduction_response_fire})
                2'b10: reduction_outstanding <= reduction_outstanding + 1'b1;
                2'b01: reduction_outstanding <= reduction_outstanding - 1'b1;
                default: begin end
            endcase
            case ({exp_request_fire, exp_response_fire})
                2'b10: exp_outstanding <= exp_outstanding + 1'b1;
                2'b01: exp_outstanding <= exp_outstanding - 1'b1;
                default: begin end
            endcase
            case ({reciprocal_request_fire, reciprocal_response_fire})
                2'b10: reciprocal_outstanding <= 1'b1;
                2'b01: reciprocal_outstanding <= 1'b0;
                default: begin end
            endcase
            case ({quant_values_request_fire, quant_values_response_fire})
                2'b10: quant_outstanding <= quant_outstanding + 1'b1;
                2'b01: quant_outstanding <= quant_outstanding - 1'b1;
                default: begin end
            endcase
            case ({scratch_read_fire, scratch_rsp_valid && scratch_rsp_ready})
                2'b10: scratch_outstanding <= scratch_outstanding + 1'b1;
                2'b01: scratch_outstanding <= scratch_outstanding - 1'b1;
                default: begin end
            endcase
            case ({quant_scale_req_valid && quant_scale_req_ready,
                   quant_scale_rsp_valid && quant_scale_rsp_ready})
                2'b10: quant_scale_outstanding <= 1'b1;
                2'b01: quant_scale_outstanding <= 1'b0;
                default: begin end
            endcase

            if (source_request_fire) begin
                source_request_count <= source_request_count + 1'b1;
`ifndef SYNTHESIS
                accepted_source_tile_count <= accepted_source_tile_count + 1'b1;
`endif
            end
            if (source_response_fire) begin
                source_response_count <= source_response_count + 1'b1;
                if (source_rsp_tag != TAG_WIDTH'(source_response_count))
                    internal_error <= 1'b1;
            end
            if (bf16_write_fire) begin
                bf16_write_count <= bf16_write_count + 1'b1;
`ifndef SYNTHESIS
                accepted_bf16_write_count <= accepted_bf16_write_count + 1'b1;
`endif
            end
`ifndef SYNTHESIS
            if (max_request_fire)
                accepted_max_request_count <= accepted_max_request_count + 1'b1;
`endif
            if (max_response_fire) begin
                max_response_count <= max_response_count + 1'b1;
                if (max_rsp_tag != TAG_WIDTH'(max_response_count) ||
                    max_rsp_row_mask != row_mask)
                    internal_error <= 1'b1;
                for (integer row = 0; row < 8; row = row + 1)
                    if (row_mask[row]) begin
                        if (state == MAX_PASS && (max_response_count == 0 ||
                            bf16_greater(max_rsp_values[row*16 +: 16],
                                row_maximum[row*16 +: 16])))
                            row_maximum[row*16 +: 16] <=
                                max_rsp_values[row*16 +: 16];
                        if (state == NORMALIZE_PASS &&
                            max_rsp_values[row*16 +: 15] >
                                probability_maximum[row*16 +: 15])
                            probability_maximum[row*16 +: 16] <=
                                max_rsp_values[row*16 +: 16];
                    end
            end
            if (scratch_read_fire) begin
                level_read_count <= level_read_count + 1'b1;
`ifndef SYNTHESIS
                accepted_scratch_read_count <= accepted_scratch_read_count + 1'b1;
`endif
            end
            if (scratch_write_fire) begin
                scratch_write_count <= scratch_write_count + 1'b1;
`ifndef SYNTHESIS
                accepted_scratch_write_count <= accepted_scratch_write_count + 1'b1;
`endif
            end
`ifndef SYNTHESIS
            if (vector_request_fire)
                accepted_vector_request_count <= accepted_vector_request_count + 1'b1;
            if (reduction_request_fire)
                accepted_reduction_request_count <=
                    accepted_reduction_request_count + 1'b1;
            if (quant_values_request_fire)
                accepted_quantized_value_count <= accepted_quantized_value_count + 1'b1;
`endif
            if (quantized_write_fire) begin
                quantized_write_count <= quantized_write_count + 1'b1;
`ifndef SYNTHESIS
                accepted_quantized_write_count <= accepted_quantized_write_count + 1'b1;
`endif
                if (quant_values_rsp_tag != TAG_WIDTH'(quantized_write_count))
                    internal_error <= 1'b1;
            end

            if (abort_request && state != IDLE && state != ABORT_DRAIN &&
                state != ABORT_WAIT_LOW) begin
                state <= ABORT_DRAIN;
            end else begin
                case (state)
                    IDLE: if (start_fire) begin
                        saved_capture_probability_enable <= capture_probability_enable;
                        error <= 1'b0;
                        internal_error <= 1'b0;
                        if (row_count == 0 || row_count > 8 || row_length == 0 ||
                            row_length > DIM_WIDTH'(MAX_LENGTH)) begin
                            error <= 1'b1;
                            done_pulse <= 1'b1;
                        end else begin
                            saved_row_count <= row_count;
                            saved_row_length <= row_length;
                            tile_count <= 10'((row_length + DIM_WIDTH'(7)) >> 3);
                            padded_root_count <= next_power_of_two(
                                10'((row_length + DIM_WIDTH'(7)) >> 3));
                            source_request_count <= '0;
                            source_response_count <= '0;
                            max_response_count <= '0;
                            row_maximum <= '0;
                            state <= MAX_PASS;
                        end
                    end
                    MAX_PASS: if (max_response_fire &&
                        max_response_count + 1'b1 == tile_count) begin
                        source_request_count <= '0;
                        source_response_count <= '0;
                        bf16_write_count <= '0;
                        scratch_write_count <= '0;
                        state <= EXP_PASS;
                    end
                    EXP_PASS: if (scratch_write_fire &&
                        scratch_write_count + 1'b1 == tile_count) begin
                        if (padded_root_count == 1) begin
                            row_sum <= reduction_rsp_values;
                            state <= RECIPROCAL_REQUEST;
                        end else if (tile_count < padded_root_count)
                            state <= PAD_ROOTS;
                        else begin
                            level_pair_count <= padded_root_count >> 1;
                            level_read_count <= '0;
                            scratch_write_count <= '0;
                            source_scratch_bank <= 1'b0;
                            destination_scratch_bank <= 1'b1;
                            state <= REDUCE_LEVEL;
                        end
                    end
                    PAD_ROOTS: if (scratch_write_fire &&
                        scratch_write_count + 1'b1 == padded_root_count) begin
                        level_pair_count <= padded_root_count >> 1;
                        level_read_count <= '0;
                        scratch_write_count <= '0;
                        source_scratch_bank <= 1'b0;
                        destination_scratch_bank <= 1'b1;
                        state <= REDUCE_LEVEL;
                    end
                    REDUCE_LEVEL: if (scratch_write_fire &&
                        scratch_write_count + 1'b1 == level_pair_count) begin
                        if (level_pair_count == 1) begin
                            row_sum <= reduction_rsp_values;
                            state <= RECIPROCAL_REQUEST;
                        end else begin
                            level_pair_count <= level_pair_count >> 1;
                            level_read_count <= '0;
                            scratch_write_count <= '0;
                            source_scratch_bank <= destination_scratch_bank;
                            destination_scratch_bank <= source_scratch_bank;
                        end
                    end
                    RECIPROCAL_REQUEST: if (reciprocal_request_fire)
                        state <= RECIPROCAL_RESPONSE;
                    RECIPROCAL_RESPONSE: if (reciprocal_response_fire) begin
                        reciprocal <= reciprocal_rsp_values;
                        if (reciprocal_rsp_row_mask != row_mask ||
                            reciprocal_rsp_tag != '0)
                            internal_error <= 1'b1;
                        source_request_count <= '0;
                        source_response_count <= '0;
                        bf16_write_count <= '0;
                        max_response_count <= '0;
                        probability_maximum <= '0;
                        state <= NORMALIZE_PASS;
                    end
                    NORMALIZE_PASS: if (max_response_fire &&
                        max_response_count + 1'b1 == tile_count)
                        state <= SCALE_REQUEST;
                    SCALE_REQUEST: if (quant_scale_req_valid &&
                        quant_scale_req_ready)
                        state <= SCALE_RESPONSE;
                    SCALE_RESPONSE: if (quant_scale_rsp_valid &&
                        quant_scale_rsp_ready) begin
                        quant_scales <= quant_scale_rsp_values;
                        state <= SCALE_WRITE;
                    end
                    SCALE_WRITE: if (scale_write_valid && scale_write_ready) begin
                        source_request_count <= '0;
                        source_response_count <= '0;
                        quantized_write_count <= '0;
                        state <= saved_capture_probability_enable ? CAPTURE_PROBABILITY : QUANTIZED_PASS;
                    end
                    CAPTURE_PROBABILITY: if (capture_probability_ready) state <= QUANTIZED_PASS;
                    QUANTIZED_PASS: if (quantized_write_fire &&
                        quantized_write_count + 1'b1 == tile_count)
                        state <= COMPLETE;
                    COMPLETE: begin
                        done_pulse <= 1'b1;
                        error <= internal_error;
                        state <= IDLE;
                    end
                    ABORT_DRAIN: if (source_outstanding == 0 &&
                        vector_outstanding == 0 && max_outstanding == 0 &&
                        reduction_outstanding == 0 && exp_outstanding == 0 &&
                        !reciprocal_outstanding && quant_outstanding == 0 &&
                        scratch_outstanding == 0 && !quant_scale_outstanding &&
                        !source_rsp_valid && !vector_rsp_valid && !max_rsp_valid &&
                        !reduction_rsp_valid && !exp_rsp_valid &&
                        !reciprocal_rsp_valid && !quant_values_rsp_valid &&
                        !scratch_rsp_valid && !quant_scale_rsp_valid) begin
                        abort_ack <= 1'b1;
                        state <= ABORT_WAIT_LOW;
                    end
                    ABORT_WAIT_LOW: if (!abort_request)
                        state <= IDLE;
                    default: begin
                        error <= 1'b1;
                        done_pulse <= 1'b1;
                        state <= IDLE;
                    end
                endcase
            end
        end
    end

    initial begin
        if (MAX_LENGTH != 2048 || DIM_WIDTH < 12 || TAG_WIDTH < 10)
            $error("softmax_row_pipeline parameter configuration is invalid");
    end

`ifndef SYNTHESIS
    assert property (@(posedge clk) disable iff (rst)
        (state == IDLE && start_valid && !start_ready) |=>
            (state == IDLE &&
             $stable({saved_row_count, saved_row_length, tile_count,
                      padded_root_count})))
        else $error("softmax_row_pipeline changed accepted command state without a start handshake");

    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (source_request_count <= tile_count)
                else $error("softmax source request count overflow");
            assert (scratch_write_count <= padded_root_count)
                else $error("softmax scratch write count overflow");
            if (source_response_fire)
                assert ((source_rsp_lane_mask[7:0] ==
                    ((source_rsp_tag + 1'b1 == TAG_WIDTH'(tile_count) &&
                      saved_row_length[2:0] != 0 ?
                        ((8'h01 << saved_row_length[2:0]) - 1'b1) : 8'hff) &
                     {8{row_mask[0]}})))
                    else $error("softmax source tail mask mismatch");
        end
    end
`endif
endmodule

// The LUT address path is explicitly split into prepare, scale, and finish
// stages by softmax_row_pipeline.
module softmax_lut_address_prepare (
    input logic [15:0] delta, output logic special_case,
    output logic [7:0] special_address,
    output logic [16:0] unscaled_numerator,
    output logic [4:0] total_shift
);
    logic [7:0] mantissa;
    integer exponent_value;
    integer value_denominator_shift;
    always_comb begin
        special_case = 1'b1;
        special_address = 8'd255;
        unscaled_numerator = 17'd0;
        total_shift = 5'd0;
        mantissa = 8'd0;
        exponent_value = 0;
        value_denominator_shift = 0;
        if (delta[15] == 0 || delta[14:0] == 0)
            special_address = 8'd255;
        else if (delta[14:7] >= 8'h83)
            special_address = 8'd0;
        else begin
            mantissa = delta[14:7] == 0 ? {1'b0, delta[6:0]} :
                {1'b1, delta[6:0]};
            exponent_value = delta[14:7] == 0 ? -126 :
                {24'd0, delta[14:7]} - 127;
            if (exponent_value < -5)
                special_address = 8'd255;
            else begin
                special_case = 1'b0;
                value_denominator_shift = 7 - exponent_value;
                unscaled_numerator = (17'd1 << (value_denominator_shift + 4)) -
                    {9'd0, mantissa};
                total_shift = 5'(value_denominator_shift + 4);
            end
        end
    end
endmodule

module softmax_lut_address_scale (
    input logic [16:0] unscaled_numerator,
    output logic [24:0] numerator
);
    always_comb
        numerator = ({8'd0, unscaled_numerator} << 8) -
            {8'd0, unscaled_numerator};
endmodule

module softmax_lut_address_finish (
    input logic special_case, input logic [7:0] special_address,
    input logic [24:0] numerator, input logic [4:0] total_shift,
    output logic [7:0] address
);
    logic [24:0] quotient;
    logic [24:0] remainder;
    logic [24:0] denominator;
    always_comb begin
        address = special_address;
        quotient = 25'd0;
        remainder = 25'd0;
        denominator = 25'd1;
        if (!special_case) begin
            quotient = numerator >> total_shift;
            remainder = numerator & ((25'd1 << total_shift) - 1'b1);
            denominator = 25'd1 << total_shift;
            if ((remainder << 1) > denominator ||
                ((remainder << 1) == denominator && quotient[0]))
                quotient = quotient + 1'b1;
            address = quotient > 255 ? 8'd255 : quotient[7:0];
        end
    end
endmodule

module softmax_reciprocal_bf16 (
    input logic [15:0] sum_value, input logic [15:0] lut_value,
    output logic [7:0] lut_address, output logic [15:0] reciprocal
);
    logic [15:0] numerator;
    logic [15:0] quotient;
    logic [15:0] remainder;
    integer sum_exponent;
    integer reciprocal_exponent;
    always_comb begin
        lut_address = 8'd0;
        reciprocal = 16'h7fc0;
        numerator = 16'd0;
        quotient = 16'd0;
        remainder = 16'd0;
        sum_exponent = 0;
        reciprocal_exponent = 0;
        if (!sum_value[15] && sum_value[14:7] != 8'h00 &&
            sum_value[14:7] != 8'hff) begin
            numerator = {9'd0, sum_value[6:0]} * 16'd255;
            quotient = numerator >> 7;
            remainder = numerator & 16'h007f;
            if (remainder > 16'd64 || (remainder == 16'd64 && quotient[0]))
                quotient = quotient + 1'b1;
            lut_address = quotient > 16'd255 ? 8'd255 : quotient[7:0];
            sum_exponent = {24'd0, sum_value[14:7]} - 127;
            reciprocal_exponent = {24'd0, lut_value[14:7]} - sum_exponent;
            if (lut_value[14:7] == 8'hff)
                reciprocal = lut_value;
            else if (reciprocal_exponent >= 255)
                reciprocal = {lut_value[15], 8'hff, 7'h00};
            else if (reciprocal_exponent <= 0)
                reciprocal = {lut_value[15], 15'h0000};
            else
                reciprocal = {lut_value[15], reciprocal_exponent[7:0],
                    lut_value[6:0]};
        end
    end
endmodule

`default_nettype wire
