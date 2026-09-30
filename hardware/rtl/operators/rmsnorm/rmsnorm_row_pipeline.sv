`default_nettype none

// Processes one batch of up to eight rows. Each source transaction carries
// eight channels from every active row, arranged as row-major 8x8 BF16 data.
module rmsnorm_row_pipeline #(
    parameter integer MAX_ELEMENTS = 4096,
    parameter integer DIM_WIDTH = 16,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [3:0]                   row_count,
    input  logic [DIM_WIDTH-1:0]         element_count,
    input  logic [15:0]                  epsilon_bf16,
    input  logic                         gamma_bypass,

    output logic                         source_request_valid,
    input  logic                         source_request_ready,
    output logic                         source_request_output_pass,
    output logic [DIM_WIDTH-1:0]         source_request_element,
    output logic [TAG_WIDTH-1:0]         source_request_tag,
    input  logic                         source_response_valid,
    output logic                         source_response_ready,
    input  logic [1023:0]                source_values,
    input  logic [127:0]                 source_gamma,
    input  logic [63:0]                  source_lane_mask,
    input  logic [TAG_WIDTH-1:0]         source_response_tag,

    output logic                         scratch_read_valid,
    input  logic                         scratch_read_ready,
    output logic                         scratch_read_bank,
    output logic [9:0]                   scratch_read_left_address,
    output logic [9:0]                   scratch_read_right_address,
    output logic [TAG_WIDTH-1:0]         scratch_read_tag,
    input  logic                         scratch_read_response_valid,
    output logic                         scratch_read_response_ready,
    input  logic [127:0]                 scratch_read_left_data,
    input  logic [127:0]                 scratch_read_right_data,
    input  logic [TAG_WIDTH-1:0]         scratch_read_response_tag,
    output logic                         scratch_write_valid,
    input  logic                         scratch_write_ready,
    output logic                         scratch_write_bank,
    output logic [9:0]                   scratch_write_address,
    output logic [127:0]                 scratch_write_data,
    output logic [15:0]                  scratch_write_byte_enable,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [DIM_WIDTH-1:0]         output_element,
    output logic [1023:0]                output_values,
    output logic [63:0]                  output_lane_mask,
    output logic [TAG_WIDTH-1:0]         output_tag,

    output logic                         vector_req_valid,
    input  logic                         vector_req_ready,
    output logic [2:0]                   vector_req_operation,
    output logic [1023:0]                vector_req_values,
    output logic [1023:0]                vector_req_paired_values,
    output logic [1023:0]                vector_req_factor0_values,
    output logic [1023:0]                vector_req_factor1_values,
    output logic [63:0]                  vector_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         vector_req_tag,
    input  logic                         vector_rsp_valid,
    output logic                         vector_rsp_ready,
    input  logic [1023:0]                vector_rsp_values,
    input  logic [63:0]                  vector_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]         vector_rsp_tag,

    output logic                         reduction_req_valid,
    input  logic                         reduction_req_ready,
    output logic [1023:0]                reduction_req_values,
    output logic [63:0]                  reduction_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         reduction_req_tag,
    input  logic                         reduction_rsp_valid,
    output logic                         reduction_rsp_ready,
    input  logic [127:0]                 reduction_rsp_values,
    input  logic [7:0]                   reduction_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0]         reduction_rsp_tag,

    output logic                         done_pulse,
    output logic                         error,
    output logic [63:0]                  accepted_source_tile_count,
    output logic [63:0]                  accepted_output_tile_count,
    output logic [63:0]                  accepted_scratch_read_count,
    output logic [63:0]                  accepted_scratch_write_count,
    output logic [63:0]                  accepted_vector_request_count,
    output logic [63:0]                  accepted_reduction_request_count
);
    localparam logic [2:0] OP_ADD = 3'd0;
    localparam logic [2:0] OP_MULTIPLY = 3'd1;
    localparam logic [2:0] OP_RMSNORM = 3'd5;

    typedef enum logic [4:0] {
        IDLE, LEAF_PASS, PAD_ROOTS, REDUCE_LEVEL,
        DIVIDE_REQUEST, DIVIDE_WAIT,
        EPSILON_REQUEST, EPSILON_WAIT,
        HALF_REQUEST, HALF_WAIT,
        SQUARE_INVERSE_REQUEST, SQUARE_INVERSE_WAIT,
        NEWTON_PRODUCT_REQUEST, NEWTON_PRODUCT_WAIT,
        CORRECTION_REQUEST, CORRECTION_WAIT,
        INVERSE_REQUEST, INVERSE_WAIT,
        OUTPUT_PASS
    } state_t;
    state_t state;

    logic [3:0] saved_row_count;
    logic [DIM_WIDTH-1:0] saved_element_count;
    logic [15:0] saved_epsilon;
    logic saved_gamma_bypass;
    logic [9:0] tile_count;
    logic [9:0] padded_root_count;
    logic [9:0] source_request_count;
    logic [9:0] scratch_write_count;
    logic [9:0] level_pair_count;
    logic [9:0] level_read_count;
    logic source_scratch_bank;
    logic destination_scratch_bank;

    logic [127:0] row_sum;
    logic [127:0] row_mean;
    logic [127:0] mean_epsilon;
    logic [127:0] initial_inverse;
    logic [127:0] half_value;
    logic [127:0] inverse_square;
    logic [127:0] newton_product;
    logic [127:0] correction;
    logic [127:0] inverse_rms;
    logic [7:0] row_mask;

    logic [7:0] divider_req_ready;
    logic [7:0] divider_rsp_valid;
    logic [127:0] divider_rsp_values;
    logic divider_request_fire;
    logic divider_response_fire;
    logic source_request_fire;
    logic scratch_read_fire;
    logic scratch_write_fire;
    logic vector_request_fire;
    logic vector_response_fire;
`ifndef SYNTHESIS
    logic reduction_request_fire;
`endif
    logic reduction_response_fire;
    logic output_fire;

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

    function automatic [15:0] magic_initial_bf16(input [15:0] input_value);
        logic [31:0] input_bits;
        logic [31:0] estimate_bits;
        logic [15:0] upper;
        logic [15:0] lower;
        begin
            input_bits = {input_value, 16'h0000};
            estimate_bits = 32'h5f3759df - (input_bits >> 1);
            upper = estimate_bits[31:16];
            lower = estimate_bits[15:0];
            if (lower > 16'h8000 || (lower == 16'h8000 && upper[0]))
                upper = upper + 16'd1;
            magic_initial_bf16 = upper;
        end
    endfunction

    assign start_ready = state == IDLE;
    assign row_mask = saved_row_count == 8 ? 8'hff :
        (8'h01 << saved_row_count) - 1'b1;

    assign source_request_valid = (state == LEAF_PASS || state == OUTPUT_PASS) &&
        source_request_count < tile_count;
    assign source_request_output_pass = state == OUTPUT_PASS;
    assign source_request_element = DIM_WIDTH'({source_request_count, 3'b000});
    assign source_request_tag = TAG_WIDTH'(source_request_count);
    assign source_request_fire = source_request_valid && source_request_ready;

    assign scratch_read_valid = state == REDUCE_LEVEL &&
        level_read_count < level_pair_count;
    assign scratch_read_bank = source_scratch_bank;
    assign scratch_read_left_address = level_read_count << 1;
    assign scratch_read_right_address = (level_read_count << 1) + 1'b1;
    assign scratch_read_tag = TAG_WIDTH'(level_read_count);
    assign scratch_read_fire = scratch_read_valid && scratch_read_ready;

    always_comb begin
        vector_req_valid = 1'b0;
        vector_req_operation = OP_MULTIPLY;
        vector_req_values = '0;
        vector_req_paired_values = '0;
        vector_req_factor0_values = '0;
        vector_req_factor1_values = '0;
        vector_req_lane_mask = '0;
        vector_req_tag = '0;
        source_response_ready = 1'b0;

        if (state == LEAF_PASS) begin
            vector_req_valid = source_response_valid;
            vector_req_values = source_values;
            vector_req_factor0_values = source_values;
            vector_req_lane_mask = source_lane_mask;
            vector_req_tag = source_response_tag;
            source_response_ready = vector_req_ready;
        end else if (state == OUTPUT_PASS) begin
            vector_req_valid = source_response_valid;
            vector_req_operation = saved_gamma_bypass ? OP_MULTIPLY :
                OP_RMSNORM;
            vector_req_values = source_values;
            vector_req_lane_mask = source_lane_mask;
            vector_req_tag = source_response_tag;
            source_response_ready = vector_req_ready;
            for (integer row = 0; row < 8; row = row + 1)
                for (integer element = 0; element < 8; element = element + 1) begin
                    vector_req_factor0_values[(row*8+element)*16 +: 16] =
                        inverse_rms[row*16 +: 16];
                    vector_req_factor1_values[(row*8+element)*16 +: 16] =
                        source_gamma[element*16 +: 16];
                end
        end else begin
            case (state)
                EPSILON_REQUEST: begin
                    vector_req_valid = 1'b1;
                    vector_req_operation = OP_ADD;
                    for (integer row = 0; row < 8; row = row + 1) begin
                        vector_req_values[(row*8)*16 +: 16] = row_mean[row*16 +: 16];
                        vector_req_paired_values[(row*8)*16 +: 16] = saved_epsilon;
                        vector_req_lane_mask[row*8] = row_mask[row];
                    end
                end
                HALF_REQUEST: begin
                    vector_req_valid = 1'b1;
                    for (integer row = 0; row < 8; row = row + 1) begin
                        vector_req_values[(row*8)*16 +: 16] = mean_epsilon[row*16 +: 16];
                        vector_req_factor0_values[(row*8)*16 +: 16] = 16'h3f00;
                        vector_req_lane_mask[row*8] = row_mask[row];
                    end
                end
                SQUARE_INVERSE_REQUEST: begin
                    vector_req_valid = 1'b1;
                    for (integer row = 0; row < 8; row = row + 1) begin
                        vector_req_values[(row*8)*16 +: 16] = initial_inverse[row*16 +: 16];
                        vector_req_factor0_values[(row*8)*16 +: 16] = initial_inverse[row*16 +: 16];
                        vector_req_lane_mask[row*8] = row_mask[row];
                    end
                end
                NEWTON_PRODUCT_REQUEST: begin
                    vector_req_valid = 1'b1;
                    for (integer row = 0; row < 8; row = row + 1) begin
                        vector_req_values[(row*8)*16 +: 16] = half_value[row*16 +: 16];
                        vector_req_factor0_values[(row*8)*16 +: 16] = inverse_square[row*16 +: 16];
                        vector_req_lane_mask[row*8] = row_mask[row];
                    end
                end
                CORRECTION_REQUEST: begin
                    vector_req_valid = 1'b1;
                    vector_req_operation = OP_ADD;
                    for (integer row = 0; row < 8; row = row + 1) begin
                        vector_req_values[(row*8)*16 +: 16] = 16'h3fc0;
                        vector_req_paired_values[(row*8)*16 +: 16] =
                            {~newton_product[row*16+15], newton_product[row*16 +: 15]};
                        vector_req_lane_mask[row*8] = row_mask[row];
                    end
                end
                INVERSE_REQUEST: begin
                    vector_req_valid = 1'b1;
                    for (integer row = 0; row < 8; row = row + 1) begin
                        vector_req_values[(row*8)*16 +: 16] = initial_inverse[row*16 +: 16];
                        vector_req_factor0_values[(row*8)*16 +: 16] = correction[row*16 +: 16];
                        vector_req_lane_mask[row*8] = row_mask[row];
                    end
                end
                default: begin end
            endcase
            vector_req_tag = TAG_WIDTH'(state);
        end
    end

    assign vector_request_fire = vector_req_valid && vector_req_ready;
    assign vector_rsp_ready = state == LEAF_PASS ? reduction_req_ready :
        state == OUTPUT_PASS ? output_ready :
        state == EPSILON_WAIT || state == HALF_WAIT ||
        state == SQUARE_INVERSE_WAIT || state == NEWTON_PRODUCT_WAIT ||
        state == CORRECTION_WAIT || state == INVERSE_WAIT;
    assign vector_response_fire = vector_rsp_valid && vector_rsp_ready;

    always_comb begin
        reduction_req_valid = 1'b0;
        reduction_req_values = '0;
        reduction_req_lane_mask = '0;
        reduction_req_tag = '0;
        scratch_read_response_ready = 1'b0;
        if (state == LEAF_PASS) begin
            reduction_req_valid = vector_rsp_valid;
            reduction_req_values = vector_rsp_values;
            reduction_req_lane_mask = vector_rsp_lane_mask;
            reduction_req_tag = vector_rsp_tag;
        end else if (state == REDUCE_LEVEL) begin
            reduction_req_valid = scratch_read_response_valid;
            reduction_req_tag = scratch_read_response_tag;
            scratch_read_response_ready = reduction_req_ready;
            for (integer row = 0; row < 8; row = row + 1) begin
                reduction_req_values[(row*8)*16 +: 16] =
                    scratch_read_left_data[row*16 +: 16];
                reduction_req_values[(row*8+1)*16 +: 16] =
                    scratch_read_right_data[row*16 +: 16];
                reduction_req_lane_mask[row*8] = row_mask[row];
                reduction_req_lane_mask[row*8+1] = row_mask[row];
            end
        end
    end
`ifndef SYNTHESIS
    assign reduction_request_fire = reduction_req_valid && reduction_req_ready;
`endif

    always_comb begin
        scratch_write_valid = 1'b0;
        scratch_write_bank = 1'b0;
        scratch_write_address = '0;
        scratch_write_data = '0;
        scratch_write_byte_enable = '0;
        reduction_rsp_ready = 1'b0;
        if (state == LEAF_PASS || state == REDUCE_LEVEL) begin
            scratch_write_valid = reduction_rsp_valid;
            scratch_write_bank = state == LEAF_PASS ? 1'b0 : destination_scratch_bank;
            scratch_write_address = reduction_rsp_tag[9:0];
            scratch_write_data = reduction_rsp_values;
            scratch_write_byte_enable = {{2{reduction_rsp_row_mask[7]}},
                {2{reduction_rsp_row_mask[6]}}, {2{reduction_rsp_row_mask[5]}},
                {2{reduction_rsp_row_mask[4]}}, {2{reduction_rsp_row_mask[3]}},
                {2{reduction_rsp_row_mask[2]}}, {2{reduction_rsp_row_mask[1]}},
                {2{reduction_rsp_row_mask[0]}}};
            reduction_rsp_ready = scratch_write_ready;
        end else if (state == PAD_ROOTS) begin
            scratch_write_valid = scratch_write_count < padded_root_count;
            scratch_write_address = scratch_write_count;
            scratch_write_byte_enable = 16'hffff;
        end
    end
    assign scratch_write_fire = scratch_write_valid && scratch_write_ready;
    assign reduction_response_fire = reduction_rsp_valid && reduction_rsp_ready;

    assign output_valid = state == OUTPUT_PASS && vector_rsp_valid;
    assign output_values = vector_rsp_values;
    assign output_lane_mask = vector_rsp_lane_mask;
    assign output_tag = vector_rsp_tag;
    assign output_element = DIM_WIDTH'(vector_rsp_tag << 3);
    assign output_fire = output_valid && output_ready;

    for (genvar row = 0; row < 8; row = row + 1) begin : g_mean_divider
        bf16_divide_u16_iterative divider (
            .clk(clk), .rst(rst), .req_valid(state == DIVIDE_REQUEST),
            .req_ready(divider_req_ready[row]), .req_value(row_sum[row*16 +: 16]),
            .req_divisor(saved_element_count), .rsp_valid(divider_rsp_valid[row]),
            .rsp_ready(state == DIVIDE_WAIT && &divider_rsp_valid),
            .rsp_value(divider_rsp_values[row*16 +: 16]));
    end
    assign divider_request_fire = state == DIVIDE_REQUEST && &divider_req_ready;
    assign divider_response_fire = state == DIVIDE_WAIT && &divider_rsp_valid;

`ifdef SYNTHESIS
    always_comb begin
        accepted_source_tile_count = '0;
        accepted_output_tile_count = '0;
        accepted_scratch_read_count = '0;
        accepted_scratch_write_count = '0;
        accepted_vector_request_count = '0;
        accepted_reduction_request_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            saved_row_count <= '0;
            saved_element_count <= '0;
            saved_epsilon <= '0;
            saved_gamma_bypass <= 1'b0;
            tile_count <= '0;
            padded_root_count <= '0;
            source_request_count <= '0;
            scratch_write_count <= '0;
            level_pair_count <= '0;
            level_read_count <= '0;
            source_scratch_bank <= 1'b0;
            destination_scratch_bank <= 1'b1;
            row_sum <= '0;
            row_mean <= '0;
            mean_epsilon <= '0;
            initial_inverse <= '0;
            half_value <= '0;
            inverse_square <= '0;
            newton_product <= '0;
            correction <= '0;
            inverse_rms <= '0;
            done_pulse <= 1'b0;
            error <= 1'b0;
`ifndef SYNTHESIS
            accepted_source_tile_count <= '0;
            accepted_output_tile_count <= '0;
            accepted_scratch_read_count <= '0;
            accepted_scratch_write_count <= '0;
            accepted_vector_request_count <= '0;
            accepted_reduction_request_count <= '0;
`endif
        end else begin
            done_pulse <= 1'b0;
            if (source_request_fire) begin
                source_request_count <= source_request_count + 1'b1;
`ifndef SYNTHESIS
                accepted_source_tile_count <= accepted_source_tile_count + 1'b1;
`endif
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
                accepted_reduction_request_count <= accepted_reduction_request_count + 1'b1;
            if (output_fire)
                accepted_output_tile_count <= accepted_output_tile_count + 1'b1;
`endif

            case (state)
                IDLE: if (start_valid) begin
                    error <= 1'b0;
                    if (row_count == 0 || row_count > 8 || element_count == 0 ||
                        element_count > MAX_ELEMENTS) begin
                        error <= 1'b1;
                        done_pulse <= 1'b1;
                    end else begin
                        saved_row_count <= row_count;
                        saved_element_count <= element_count;
                        saved_epsilon <= epsilon_bf16;
                        saved_gamma_bypass <= gamma_bypass;
                        tile_count <= 10'((element_count + 16'd7) >> 3);
                        padded_root_count <= next_power_of_two(
                            10'((element_count + 16'd7) >> 3));
                        source_request_count <= '0;
                        scratch_write_count <= '0;
                        state <= LEAF_PASS;
                    end
                end
                LEAF_PASS: begin
                    if (source_response_valid && source_response_ready &&
                        source_response_tag >= tile_count)
                        error <= 1'b1;
                    if (scratch_write_fire && scratch_write_count + 1'b1 == tile_count) begin
                        scratch_write_count <= tile_count;
                        if (tile_count < padded_root_count)
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
                REDUCE_LEVEL: begin
                    if (reduction_response_fire && level_pair_count == 1)
                        row_sum <= reduction_rsp_values;
                    if (scratch_write_fire && scratch_write_count + 1'b1 == level_pair_count) begin
                        if (level_pair_count == 1) begin
                            state <= DIVIDE_REQUEST;
                        end else begin
                            level_pair_count <= level_pair_count >> 1;
                            level_read_count <= '0;
                            scratch_write_count <= '0;
                            source_scratch_bank <= destination_scratch_bank;
                            destination_scratch_bank <= source_scratch_bank;
                        end
                    end
                end
                DIVIDE_REQUEST: if (divider_request_fire)
                    state <= DIVIDE_WAIT;
                DIVIDE_WAIT: if (divider_response_fire) begin
                    row_mean <= divider_rsp_values;
                    state <= EPSILON_REQUEST;
                end
                EPSILON_REQUEST: if (vector_request_fire)
                    state <= EPSILON_WAIT;
                EPSILON_WAIT: if (vector_response_fire) begin
                    for (integer row = 0; row < 8; row = row + 1) begin
                        mean_epsilon[row*16 +: 16] <= vector_rsp_values[(row*8)*16 +: 16];
                        initial_inverse[row*16 +: 16] <=
                            magic_initial_bf16(vector_rsp_values[(row*8)*16 +: 16]);
                    end
                    state <= HALF_REQUEST;
                end
                HALF_REQUEST: if (vector_request_fire)
                    state <= HALF_WAIT;
                HALF_WAIT: if (vector_response_fire) begin
                    for (integer row = 0; row < 8; row = row + 1)
                        half_value[row*16 +: 16] <= vector_rsp_values[(row*8)*16 +: 16];
                    state <= SQUARE_INVERSE_REQUEST;
                end
                SQUARE_INVERSE_REQUEST: if (vector_request_fire)
                    state <= SQUARE_INVERSE_WAIT;
                SQUARE_INVERSE_WAIT: if (vector_response_fire) begin
                    for (integer row = 0; row < 8; row = row + 1)
                        inverse_square[row*16 +: 16] <= vector_rsp_values[(row*8)*16 +: 16];
                    state <= NEWTON_PRODUCT_REQUEST;
                end
                NEWTON_PRODUCT_REQUEST: if (vector_request_fire)
                    state <= NEWTON_PRODUCT_WAIT;
                NEWTON_PRODUCT_WAIT: if (vector_response_fire) begin
                    for (integer row = 0; row < 8; row = row + 1)
                        newton_product[row*16 +: 16] <= vector_rsp_values[(row*8)*16 +: 16];
                    state <= CORRECTION_REQUEST;
                end
                CORRECTION_REQUEST: if (vector_request_fire)
                    state <= CORRECTION_WAIT;
                CORRECTION_WAIT: if (vector_response_fire) begin
                    for (integer row = 0; row < 8; row = row + 1)
                        correction[row*16 +: 16] <= vector_rsp_values[(row*8)*16 +: 16];
                    state <= INVERSE_REQUEST;
                end
                INVERSE_REQUEST: if (vector_request_fire)
                    state <= INVERSE_WAIT;
                INVERSE_WAIT: if (vector_response_fire) begin
                    for (integer row = 0; row < 8; row = row + 1)
                        inverse_rms[row*16 +: 16] <= vector_rsp_values[(row*8)*16 +: 16];
                    source_request_count <= '0;
                    state <= OUTPUT_PASS;
                end
                OUTPUT_PASS: if (output_fire &&
                    vector_rsp_tag + 1'b1 == tile_count) begin
                    done_pulse <= 1'b1;
                    state <= IDLE;
                end
                default: begin
                    error <= 1'b1;
                    done_pulse <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

    initial begin
        if (MAX_ELEMENTS < 8 || MAX_ELEMENTS > 4096 || DIM_WIDTH < 13 || TAG_WIDTH < 10)
            $error("rmsnorm_row_pipeline parameter configuration is invalid");
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (reduction_request_fire && state == LEAF_PASS)
                for (integer row = 0; row < 8; row++)
                    assert (reduction_req_lane_mask[row*8 +: 8] ==
                            {8{row_mask[row]}})
                        else $error(
                            "rmsnorm_row_pipeline leaf reduction mask mismatch row=%0d actual=%h expected=%h tag=%h",
                            row, reduction_req_lane_mask[row*8 +: 8],
                            {8{row_mask[row]}}, reduction_req_tag);
            if (source_response_valid && source_response_ready)
                for (integer row = 0; row < 8; row++)
                    assert (source_lane_mask[row*8 +: 8] == (
                        (source_response_tag + 1'b1 == tile_count &&
                         saved_element_count[2:0] != 0 ?
                            ((8'h01 << saved_element_count[2:0]) - 1'b1) :
                            8'hff) & {8{row_mask[row]}}))
                        else $error(
                            "rmsnorm_row_pipeline source mask mismatch row=%0d actual=%h expected=%h tag=%h",
                            row, source_lane_mask[row*8 +: 8],
                            (source_response_tag + 1'b1 == tile_count &&
                             saved_element_count[2:0] != 0 ?
                                ((8'h01 << saved_element_count[2:0]) - 1'b1) :
                                8'hff) & {8{row_mask[row]}},
                            source_response_tag);
            if (reduction_rsp_valid && reduction_rsp_ready)
                assert (reduction_rsp_row_mask == row_mask)
                    else $error(
                        "rmsnorm_row_pipeline reduction row mask mismatch actual=%h expected=%h state=%0d tag=%h",
                        reduction_rsp_row_mask, row_mask, state,
                        reduction_rsp_tag);
            assert (source_request_count <= tile_count)
                else $error("rmsnorm_row_pipeline source request count overflow");
            assert (scratch_write_count <= padded_root_count)
                else $error("rmsnorm_row_pipeline scratch write count overflow");
        end
    end
`endif
endmodule

`default_nettype wire
