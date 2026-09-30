`default_nettype none

// Accepts one N8 output stripe at a time. Gate and Up use paired-stripe DDR
// slots. Residual modes first load one old-output stripe, perform one shared
// BF16 add per row, then write the combined stripe to local SRAM or DDR.
module matmul_output_controller #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer MAX_ROWS = 48
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    abort_request,
    output logic                    abort_ack,
    input  logic                    cfg_valid,
    output logic                    cfg_ready,
    input  logic [2:0]              cfg_mode,
    input  logic [5:0]              cfg_physical_row_base,
    input  logic [5:0]              cfg_row_count,
    input  logic                    cfg_fused_product,
    input  logic [2:0]              cfg_batch_count,
    input  logic [5:0]              cfg_second_batch_rows,
    input  logic [5:0]              cfg_third_batch_rows,
    input  logic [5:0]              cfg_fourth_batch_rows,
    input  logic [5:0]              cfg_fifth_batch_rows,
    input  logic [5:0]              cfg_sixth_batch_rows,
    input  logic [15:0]             cfg_output_channels,
    input  logic [ADDR_WIDTH-1:0]   cfg_output_base,
    input  logic [ADDR_WIDTH-1:0]   cfg_output_limit,
    input  logic [31:0]             cfg_local_row_stride,
    input  logic                    cfg_residual_from_ddr,
    input  logic [ADDR_WIDTH-1:0]   cfg_residual_base,
    input  logic [ADDR_WIDTH-1:0]   cfg_residual_limit,
    input  logic [5:0]              cfg_residual_storage_rows,

    input  logic                    in_valid,
    input  logic [2:0]              in_batch_index,
    output logic                    in_ready,
    input  logic [5:0]              in_physical_row,
    input  logic [15:0]             in_output_channel,
    input  logic [ADDR_WIDTH-1:0]   in_row_byte_base,
    input  logic [127:0]            in_data,
    input  logic [15:0]             in_byte_enable,

    output logic                    residual_read_request_valid,
    output logic                    residual_ddr_stage,
    input  logic                    residual_read_request_ready,
    output logic [ADDR_WIDTH-1:0]   residual_read_request_address,
    output logic [31:0]             residual_read_request_bytes,
    output logic [7:0]              residual_read_request_tag,
    output logic                    residual_read_local,
    input  logic                    residual_read_request_done,
    input  logic                    residual_read_request_error,
    input  logic                    residual_read_data_valid,
    output logic                    residual_read_data_ready,
    input  logic [127:0]            residual_read_data,
    input  logic [15:0]             residual_read_byte_enable,
    input  logic                    residual_read_data_last,
    input  logic [7:0]              residual_read_data_tag,

    output logic                    arithmetic_req_valid,
    input  logic                    arithmetic_req_ready,
    output logic [2:0]              arithmetic_req_operation,
    output logic [255:0]            arithmetic_req_values,
    output logic [255:0]            arithmetic_req_paired_values,
    output logic [255:0]            arithmetic_req_factor0_values,
    output logic [255:0]            arithmetic_req_factor1_values,
    output logic [15:0]             arithmetic_req_lane_mask,
    output logic [15:0]             arithmetic_req_tag,
    input  logic                    arithmetic_rsp_valid,
    output logic                    arithmetic_rsp_ready,
    input  logic [255:0]            arithmetic_rsp_values,
    input  logic [15:0]             arithmetic_rsp_lane_mask,
    input  logic [15:0]             arithmetic_rsp_tag,
    input  logic                    arithmetic_abort_ack,

    output logic                    local_write_valid,
    input  logic                    local_write_ready,
    output logic [ADDR_WIDTH-1:0]   local_write_byte_address,
    output logic [127:0]            local_write_data,
    output logic [15:0]             local_write_byte_enable,

    output logic                    output_write_valid,
    input  logic                    output_write_ready,
    output logic [ADDR_WIDTH-1:0]   output_write_byte_address,
    output logic [31:0]             output_write_transaction_bytes,
    output logic [127:0]            output_write_data,
    output logic [15:0]             output_write_byte_enable,
    output logic                    output_write_first,
    output logic                    output_write_last,
    input  logic                    output_write_done,
    input  logic                    output_write_error,

    output logic                    staging_write_valid,
    input  logic                    staging_write_ready,
    output logic [9:0]              staging_write_address,
    output logic [127:0]            staging_write_data,
    output logic [15:0]             staging_write_byte_enable,
    output logic                    staging_read_request_valid,
    input  logic                    staging_read_request_ready,
    output logic [9:0]              staging_read_address,
    input  logic                    staging_read_response_valid,
    input  logic [127:0]            staging_read_data,

    output logic                    row_commit,
    output logic                    done_pulse,
    output logic                    busy,
    output logic                    error,
    output logic [3:0]              error_id
);
    localparam logic [2:0] MODE_LOCAL = 3'd0;
    localparam logic [2:0] MODE_PAIRED_GATE = 3'd1;
    localparam logic [2:0] MODE_PAIRED_UP = 3'd2;
    localparam logic [2:0] MODE_RESIDUAL_LOCAL = 3'd3;
    localparam logic [2:0] MODE_RESIDUAL_DDR = 3'd4;
    localparam logic [2:0] MODE_FUSED_PRODUCT = 3'd5;
    localparam logic [2:0] MODE_DDR = 3'd6;
    localparam logic [2:0] BF16_ADD = 3'd0;
    localparam logic [2:0] BF16_MULTIPLY = 3'd1;
    localparam logic [2:0] BF16_SILU = 3'd3;
    localparam logic [3:0] ERROR_CONFIGURATION = 4'h1;
    localparam logic [3:0] ERROR_READ = 4'h2;
    localparam logic [3:0] ERROR_ARITHMETIC = 4'h3;
    localparam logic [3:0] ERROR_WRITE = 4'h4;
    localparam logic [6:0] RESIDUAL_BASE = 7'd0;
    localparam logic [6:0] RESIDUAL_ALT_BASE = 7'd48;
    localparam logic [9:0] RESULT_BASE = 10'd32;
    localparam logic [9:0] FUSED_GATE_BASE = 10'd96;

    typedef enum logic [4:0] {
        IDLE,
        ISSUE_RESIDUAL_READ,
        RECEIVE_RESIDUAL,
        WAIT_RESIDUAL_DONE,
        ACCEPT_FRAGMENT,
        READ_RESIDUAL_WORD,
        WAIT_RESIDUAL_WORD,
        ISSUE_ADD,
        WAIT_ADD,
        READ_OUTPUT_WORD,
        WAIT_OUTPUT_WORD,
        SEND_OUTPUT_WORD,
        WAIT_OUTPUT_DONE,
        WAIT_PREFETCHED_RESIDUAL,
        ABORT_DRAIN,
        ABORT_WAIT_LOW,
        FUSED_WAIT_GATE,
        FUSED_SILU_REQUEST,
        FUSED_SILU_RESPONSE,
        FUSED_PRODUCT_REQUEST,
        FUSED_PRODUCT_RESPONSE
    } state_t;

    state_t state;
    logic [2:0] mode;
    logic fused_product;
    logic [5:0] row_base;
    logic [5:0] row_count;
    logic [5:0] first_batch_rows;
    logic [5:0] second_batch_rows;
    logic [5:0] third_batch_rows;
    logic [5:0] fourth_batch_rows;
    logic [5:0] fifth_batch_rows;
    logic [5:0] sixth_batch_rows;
    logic [2:0] output_batch_count;
    logic [2:0] output_batch_index;
    logic [ADDR_WIDTH:0] second_batch_base;
    logic [ADDR_WIDTH:0] third_batch_base;
    logic [ADDR_WIDTH:0] fourth_batch_base;
    logic [ADDR_WIDTH:0] fifth_batch_base;
    logic [ADDR_WIDTH:0] sixth_batch_base;
    logic [ADDR_WIDTH-1:0] first_batch_next_address;
    logic [ADDR_WIDTH-1:0] second_batch_next_address;
    logic [ADDR_WIDTH-1:0] third_batch_next_address;
    logic [ADDR_WIDTH-1:0] fourth_batch_next_address;
    logic [ADDR_WIDTH-1:0] fifth_batch_next_address;
    logic [ADDR_WIDTH-1:0] sixth_batch_next_address;
    logic [2:0] switch_batch_index;
    logic [5:0] switch_batch_rows;
    logic [ADDR_WIDTH-1:0] switch_batch_address;
    logic [15:0] output_channels;
    logic [15:0] stripe_count;
    logic [15:0] stripe_index;
    logic [31:0] stripe_bytes;
    logic [31:0] stripe_stride_bytes;
    logic [ADDR_WIDTH-1:0] output_base;
    logic [ADDR_WIDTH-1:0] output_limit;
    logic [31:0] local_row_stride;
    logic [5:0] input_row;
    logic [5:0] fragment_input_row;
    logic [15:0] fragment_input_stripe;
    logic fragment_input_active;
    logic [5:0] residual_receive_row;
    logic [5:0] output_row;
    logic [127:0] saved_fragment;
    logic [127:0] saved_fragment_second;
    logic [127:0] saved_residual;
    logic [127:0] saved_residual_second;
    logic [127:0] saved_output;
    logic [127:0] saved_output_second;
    logic [127:0] saved_gate;
    logic [127:0] saved_up;
    logic [127:0] saved_silu;
    logic [5:0] pair_first_row;
    logic collecting_second_row;
    logic pair_has_second_row;
    logic output_second_row;
    logic residual_read_active;
    logic residual_read_is_prefetch;
    logic residual_terminal_seen;
    logic residual_terminal_error;
    logic residual_buffer_select;
    logic residual_prefetch_pending;
    logic residual_prefetched_valid;
    logic residual_prefetched_error;
    logic [12:0] residual_prefetch_stripe;
    logic output_write_active;
    logic output_terminal_seen;
    logic output_terminal_error;
    logic arithmetic_outstanding;
    logic staging_read_outstanding;
    logic abort_seen;
    logic residual_mode;
    logic ddr_output_mode;
    logic [3:0] configuration_stage;
    logic [ADDR_WIDTH:0] configured_end;
    logic [38:0] local_last_end;
    logic [24:0] required_bytes;
    logic [38:0] local_first_address;
    logic [6:0] configuration_row;
    logic [38:0] configuration_row_address;
    logic configuration_shape_valid;
    logic configuration_valid;
    logic configuration_accepted;
    logic configured_end_within_limit;
    logic local_end_within_limit;
    logic input_metadata_valid;
    logic fragment_input_metadata_valid;
    logic [2:0] fragment_input_batch;
    logic [5:0] fragment_input_rows;
    logic [ADDR_WIDTH-1:0] residual_output_offset;
    logic [15:0] stripe_byte_enable;
    logic [15:0] fragment_input_byte_enable;
    logic fragment_fifo_input_valid;
    logic fragment_fifo_input_ready;
    logic fragment_fifo_output_valid;
    logic fragment_fifo_output_ready;
    logic [127:0] fragment_fifo_output_data;
    logic [3:0] fragment_fifo_occupancy;
    logic fragment_fifo_input_fire;
    logic fragment_fifo_output_fire;
    logic final_stripe;
    logic final_input_row;
    logic final_output_row;
    logic residual_data_metadata_valid;
    logic [ADDR_WIDTH-1:0] stripe_address;
    logic [ADDR_WIDTH-1:0] stripe_address_step;
    logic [ADDR_WIDTH-1:0] write_byte_address;
    logic [ADDR_WIDTH-1:0] residual_current_address;
    logic [ADDR_WIDTH-1:0] residual_prefetch_address;
    logic [ADDR_WIDTH-1:0] residual_address_step;
    logic residual_from_ddr;
    logic [ADDR_WIDTH-1:0] residual_base;
    logic [ADDR_WIDTH-1:0] residual_limit;
    logic [5:0] residual_storage_rows;
    logic [ADDR_WIDTH:0] residual_configured_end;
    logic [24:0] residual_required_bytes;
    logic residual_end_within_limit;
    logic residual_prefetch_request;
    logic [7:0] residual_request_stripe;
    logic [9:0] residual_write_base;
    logic [9:0] residual_read_base;
    logic residual_prefetch_data_error;
    logic [8:0] batch_row_offset;
    logic [9:0] configured_total_rows;
    logic [9:0] fused_gate_address;
    logic [15:0] fused_lane_mask;
    logic [255:0] fused_silu_inputs;
    logic [255:0] fused_silu_slopes;
    logic [255:0] fused_silu_intercepts;
    logic [31:0] fused_silu_coefficients [0:7];

    always_comb begin
        residual_mode = mode == MODE_RESIDUAL_LOCAL ||
            mode == MODE_RESIDUAL_DDR;
        ddr_output_mode = mode == MODE_PAIRED_GATE ||
            mode == MODE_PAIRED_UP || mode == MODE_RESIDUAL_DDR ||
            mode == MODE_FUSED_PRODUCT || mode == MODE_DDR;
        final_stripe = stripe_index + 1'b1 == stripe_count;
        final_input_row = input_row + 1'b1 == row_count;
        final_output_row = output_row + 1'b1 == row_count;

        configured_total_rows =
            ({4'd0, first_batch_rows} +
             (output_batch_count >= 3'd2 ? {4'd0, second_batch_rows} : 10'd0)) +
            ((output_batch_count >= 3'd3 ? {4'd0, third_batch_rows} : 10'd0) +
             (output_batch_count >= 3'd4 ? {4'd0, fourth_batch_rows} : 10'd0)) +
            ((output_batch_count >= 3'd5 ? {4'd0, fifth_batch_rows} : 10'd0) +
             (output_batch_count >= 3'd6 ? {4'd0, sixth_batch_rows} : 10'd0));

        configuration_shape_valid =
            mode <= MODE_DDR &&
            row_count != 0 &&
            ({1'b0, row_base} + {1'b0, row_count}) <= 7'(MAX_ROWS) &&
            output_channels != 0 &&
            output_limit > output_base;
        configured_end_within_limit = !configured_end[ADDR_WIDTH] &&
            (configured_end[63:32] < output_limit[63:32] ||
             (configured_end[63:32] == output_limit[63:32] &&
              configured_end[31:0] <= output_limit[31:0]));
        residual_end_within_limit = !residual_configured_end[ADDR_WIDTH] &&
            (residual_configured_end[63:32] < residual_limit[63:32] ||
             (residual_configured_end[63:32] == residual_limit[63:32] &&
              residual_configured_end[31:0] <= residual_limit[31:0]));
        local_end_within_limit = output_base[63:32] == 0 &&
            output_limit[63:32] == 0 && local_last_end[38:32] == 0 &&
            local_last_end[31:0] <= output_limit[31:0];
        configuration_valid = configuration_shape_valid &&
            output_batch_count >= 3'd1 && output_batch_count <= 3'd6 &&
            (fused_product == (mode == MODE_FUSED_PRODUCT)) &&
            (!fused_product || configured_total_rows <= 10'd288) &&
            (!fused_product || (output_channels[2:0] == 0 && row_base == 0)) &&
            (output_batch_count == 3'd1 ||
             ((mode == MODE_PAIRED_GATE || mode == MODE_PAIRED_UP ||
               mode == MODE_FUSED_PRODUCT || mode == MODE_DDR ||
               (mode == MODE_LOCAL && output_batch_count == 3'd2) ||
               mode == MODE_RESIDUAL_DDR) &&
              second_batch_rows != 0 && second_batch_rows <= 6'(MAX_ROWS) &&
              (output_batch_count < 3'd3 ||
               (third_batch_rows != 0 && third_batch_rows <= 6'(MAX_ROWS))) &&
              (output_batch_count < 3'd4 ||
               (fourth_batch_rows != 0 && fourth_batch_rows <= 6'(MAX_ROWS))) &&
              (output_batch_count < 3'd5 ||
               (fifth_batch_rows != 0 && fifth_batch_rows <= 6'(MAX_ROWS))) &&
              (output_batch_count < 3'd6 ||
               (sixth_batch_rows != 0 && sixth_batch_rows <= 6'(MAX_ROWS))) &&
              row_base == 0)) &&
            ((mode == MODE_LOCAL || mode == MODE_RESIDUAL_LOCAL) ?
                (local_row_stride >= (32'(output_channels) << 1) &&
                 local_end_within_limit) :
                (configured_end_within_limit &&
                 output_base[7:0] == 0)) &&
            (!residual_from_ddr ||
             (residual_base[7:0] == 0 && residual_limit > residual_base &&
              residual_end_within_limit)) &&
            (residual_storage_rows == 0 ||
             (residual_from_ddr && output_batch_count == 3'd1 &&
              residual_storage_rows >= row_count &&
              residual_storage_rows <= 6'(MAX_ROWS)));

        stripe_byte_enable = 16'hffff;
        if (final_stripe && output_channels[2:0] != 0)
            stripe_byte_enable = 16'hffff >>
                (16 - {output_channels[2:0], 1'b0});

        fused_lane_mask = 16'h00ff;
        if (final_stripe && output_channels[2:0] != 0)
            fused_lane_mask = 16'h00ff >> (8 - output_channels[2:0]);

        fragment_input_byte_enable = 16'hffff;
        if (fragment_input_stripe + 1'b1 == stripe_count &&
            output_channels[2:0] != 0)
            fragment_input_byte_enable = 16'hffff >>
                (16 - {output_channels[2:0], 1'b0});

        input_metadata_valid =
            in_batch_index == output_batch_index &&
            in_physical_row == row_base + input_row &&
            in_output_channel == (fused_product ?
                {1'b0, stripe_index[12:1], 3'b000} :
                {stripe_index[12:0], 3'b000}) &&
            in_byte_enable == stripe_byte_enable;
        fragment_input_metadata_valid =
            in_batch_index == fragment_input_batch &&
            in_physical_row == row_base + fragment_input_row &&
            in_output_channel == {fragment_input_stripe[12:0], 3'b000} &&
            in_byte_enable == fragment_input_byte_enable;
        case (fragment_input_batch)
            3'd1: fragment_input_rows = second_batch_rows;
            3'd2: fragment_input_rows = third_batch_rows;
            3'd3: fragment_input_rows = fourth_batch_rows;
            3'd4: fragment_input_rows = fifth_batch_rows;
            3'd5: fragment_input_rows = sixth_batch_rows;
            default: fragment_input_rows = first_batch_rows;
        endcase
        residual_prefetch_request = residual_prefetch_pending &&
            !residual_read_active && state != ISSUE_RESIDUAL_READ;
        residual_request_stripe = residual_prefetch_request ?
            residual_prefetch_stripe[7:0] : stripe_index[7:0];
        residual_write_base = residual_read_is_prefetch ?
            (residual_buffer_select ? RESIDUAL_BASE : RESIDUAL_ALT_BASE) :
            (residual_buffer_select ? RESIDUAL_ALT_BASE : RESIDUAL_BASE);
        residual_read_base = residual_buffer_select ?
            RESIDUAL_ALT_BASE : RESIDUAL_BASE;
        residual_data_metadata_valid =
            residual_read_byte_enable == 16'hffff &&
            residual_read_data_tag == (residual_read_is_prefetch ?
                residual_prefetch_stripe[7:0] : stripe_index[7:0]) &&
            residual_read_data_last ==
                (residual_receive_row + 1'b1 == row_count);
        residual_prefetch_data_error = residual_read_is_prefetch &&
            residual_read_active && residual_read_data_valid &&
            residual_read_data_ready && !residual_data_metadata_valid;

        fused_gate_address = FUSED_GATE_BASE +
            {1'b0, batch_row_offset} + {4'd0, input_row};

        switch_batch_index =
            output_batch_index + 3'd1 < output_batch_count ?
                output_batch_index + 3'd1 : 3'd0;
        case (switch_batch_index)
            3'd1: begin
                switch_batch_rows = second_batch_rows;
                switch_batch_address = second_batch_next_address;
            end
            3'd2: begin
                switch_batch_rows = third_batch_rows;
                switch_batch_address = third_batch_next_address;
            end
            3'd3: begin
                switch_batch_rows = fourth_batch_rows;
                switch_batch_address = fourth_batch_next_address;
            end
            3'd4: begin
                switch_batch_rows = fifth_batch_rows;
                switch_batch_address = fifth_batch_next_address;
            end
            3'd5: begin
                switch_batch_rows = sixth_batch_rows;
                switch_batch_address = sixth_batch_next_address;
            end
            default: begin
                switch_batch_rows = first_batch_rows;
                switch_batch_address = first_batch_next_address;
            end
        endcase

    end

    always_comb begin
        fused_silu_inputs = {128'd0, saved_gate};
        fused_silu_slopes = '0;
        fused_silu_intercepts = '0;
        for (integer lane = 0; lane < 8; lane = lane + 1) begin
            fused_silu_coefficients[lane] =
                bf16_silu_pwl_pkg::coefficients(saved_gate[lane*16 +: 16]);
            fused_silu_inputs[lane*16 +: 16] =
                bf16_silu_pwl_pkg::input_value(saved_gate[lane*16 +: 16]);
            fused_silu_slopes[lane*16 +: 16] =
                fused_silu_coefficients[lane][15:0];
            fused_silu_intercepts[lane*16 +: 16] =
                fused_silu_coefficients[lane][31:16];
        end
    end

    assign cfg_ready = state == IDLE && configuration_stage == 0 &&
        !abort_request;
    assign busy = (state != IDLE && state != ABORT_WAIT_LOW) ||
        configuration_stage != 0;

    assign residual_read_request_valid =
        (state == ISSUE_RESIDUAL_READ || residual_prefetch_request) &&
        !abort_request;
    assign residual_ddr_stage =
        (mode == MODE_RESIDUAL_DDR || residual_from_ddr) &&
        (state == ISSUE_RESIDUAL_READ || state == RECEIVE_RESIDUAL ||
         state == WAIT_RESIDUAL_DONE || residual_prefetch_pending ||
         (residual_read_active && residual_read_is_prefetch));
    assign residual_read_request_address = residual_prefetch_request ?
        residual_prefetch_address : residual_current_address;
    assign residual_read_request_bytes = stripe_bytes;
    assign residual_read_request_tag = residual_request_stripe;
    assign residual_read_local = !residual_from_ddr &&
        mode == MODE_RESIDUAL_LOCAL;
    assign residual_read_data_ready =
        (state == RECEIVE_RESIDUAL || residual_read_is_prefetch ||
         state == ABORT_DRAIN) &&
        residual_read_active && staging_write_ready;

    assign in_ready = residual_mode ?
        (fragment_input_active && fragment_fifo_input_ready &&
         fragment_input_metadata_valid && !abort_request) :
        fused_product && stripe_index[0] ?
        (state == ACCEPT_FRAGMENT && input_metadata_valid &&
         staging_read_request_ready && !abort_request) :
        (state == ACCEPT_FRAGMENT && input_metadata_valid &&
         staging_write_ready && !abort_request);

    assign fragment_fifo_input_valid = residual_mode &&
        fragment_input_active && in_valid && fragment_input_metadata_valid &&
        !abort_request;
    assign fragment_fifo_input_fire = fragment_fifo_input_valid &&
        fragment_fifo_input_ready;
    assign fragment_fifo_output_ready = residual_mode &&
        ((state == ACCEPT_FRAGMENT && !abort_request) || state == ABORT_DRAIN);
    assign fragment_fifo_output_fire = fragment_fifo_output_valid &&
        fragment_fifo_output_ready;

    ready_valid_fifo #(.DATA_WIDTH(128), .DEPTH(8)) fragment_fifo (
        .clk(clk),
        .rst(rst || state == IDLE),
        .input_valid(fragment_fifo_input_valid),
        .input_ready(fragment_fifo_input_ready),
        .input_data(in_data),
        .output_valid(fragment_fifo_output_valid),
        .output_ready(fragment_fifo_output_ready),
        .output_data(fragment_fifo_output_data),
        .occupancy(fragment_fifo_occupancy)
    );

    assign staging_write_valid =
        ((state == RECEIVE_RESIDUAL || residual_read_is_prefetch) &&
         residual_read_data_valid &&
         residual_read_active && !abort_request) ||
        (state == ACCEPT_FRAGMENT && in_valid && !residual_mode &&
         !(fused_product && stripe_index[0]) &&
         input_metadata_valid && !abort_request);
    assign staging_write_address =
        (state == RECEIVE_RESIDUAL || residual_read_is_prefetch) ?
        residual_write_base + residual_receive_row :
        fused_product ? fused_gate_address : RESULT_BASE + input_row;
    assign staging_write_data =
        (state == RECEIVE_RESIDUAL || residual_read_is_prefetch) ?
        residual_read_data :
        in_data;
    assign staging_write_byte_enable = fused_product &&
        state == ACCEPT_FRAGMENT ? in_byte_enable : 16'hffff;

    assign staging_read_request_valid =
        ((state == READ_RESIDUAL_WORD || state == READ_OUTPUT_WORD) ||
         (state == ACCEPT_FRAGMENT && fused_product && stripe_index[0] &&
          in_valid && input_metadata_valid)) && !abort_request;
    assign staging_read_address =
        state == READ_RESIDUAL_WORD ? residual_read_base + input_row :
        state == READ_OUTPUT_WORD ? RESULT_BASE + output_row :
        fused_gate_address;

    assign arithmetic_req_valid =
        (state == ISSUE_ADD || state == FUSED_SILU_REQUEST ||
         state == FUSED_PRODUCT_REQUEST) && !abort_request;
    assign arithmetic_req_operation = state == FUSED_SILU_REQUEST ?
        BF16_SILU : state == FUSED_PRODUCT_REQUEST ? BF16_MULTIPLY : BF16_ADD;
    assign arithmetic_req_values = state == FUSED_SILU_REQUEST ?
        fused_silu_inputs : state == FUSED_PRODUCT_REQUEST ?
        {128'd0, saved_silu} : {saved_fragment_second, saved_fragment};
    assign arithmetic_req_paired_values =
        {saved_residual_second, saved_residual};
    assign arithmetic_req_factor0_values = state == FUSED_SILU_REQUEST ?
        fused_silu_slopes : state == FUSED_PRODUCT_REQUEST ?
        {128'd0, saved_up} : '0;
    assign arithmetic_req_factor1_values = state == FUSED_SILU_REQUEST ?
        fused_silu_intercepts : '0;
    assign arithmetic_req_lane_mask =
        fused_product ? fused_lane_mask :
        pair_has_second_row ? 16'hffff : 16'h00ff;
    assign arithmetic_req_tag = {stripe_index[9:0], pair_first_row};
    assign arithmetic_rsp_ready = state == WAIT_ADD ||
        state == FUSED_SILU_RESPONSE || state == FUSED_PRODUCT_RESPONSE ||
        state == ABORT_DRAIN;

    assign local_write_valid =
        state == SEND_OUTPUT_WORD && !ddr_output_mode && !abort_request;
    assign local_write_byte_address = write_byte_address;
    assign local_write_data =
        output_second_row ? saved_output_second : saved_output;
    assign local_write_byte_enable = stripe_byte_enable;

    assign output_write_valid =
        state == SEND_OUTPUT_WORD && ddr_output_mode && !abort_request &&
        !output_terminal_error && !output_write_error;
    assign output_write_byte_address = write_byte_address;
    assign output_write_transaction_bytes = stripe_bytes;
    assign output_write_data =
        output_second_row ? saved_output_second : saved_output;
    assign output_write_byte_enable = stripe_byte_enable;
    assign output_write_first = output_row == 0;
    assign output_write_last = final_output_row;

    assign row_commit =
        ((local_write_valid && local_write_ready) ||
         (output_write_valid && output_write_ready)) &&
        final_stripe;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            mode <= MODE_LOCAL;
            fused_product <= 1'b0;
            row_base <= '0;
            row_count <= '0;
            first_batch_rows <= '0;
            second_batch_rows <= '0;
            third_batch_rows <= '0;
            fourth_batch_rows <= '0;
            fifth_batch_rows <= '0;
            sixth_batch_rows <= '0;
            output_batch_count <= 3'd1;
            output_batch_index <= 3'd0;
            batch_row_offset <= '0;
            second_batch_base <= '0;
            third_batch_base <= '0;
            fourth_batch_base <= '0;
            fifth_batch_base <= '0;
            sixth_batch_base <= '0;
            first_batch_next_address <= '0;
            second_batch_next_address <= '0;
            third_batch_next_address <= '0;
            fourth_batch_next_address <= '0;
            fifth_batch_next_address <= '0;
            sixth_batch_next_address <= '0;
            output_channels <= '0;
            stripe_count <= '0;
            stripe_index <= '0;
            stripe_bytes <= '0;
            stripe_stride_bytes <= '0;
            output_base <= '0;
            output_limit <= '0;
            local_row_stride <= '0;
            input_row <= '0;
            fragment_input_row <= '0;
            fragment_input_batch <= 3'd0;
            residual_output_offset <= '0;
            fragment_input_stripe <= '0;
            fragment_input_active <= 1'b0;
            residual_receive_row <= '0;
            output_row <= '0;
            saved_fragment <= '0;
            saved_fragment_second <= '0;
            saved_residual <= '0;
            saved_residual_second <= '0;
            saved_output <= '0;
            saved_output_second <= '0;
            saved_gate <= '0;
            saved_up <= '0;
            saved_silu <= '0;
            pair_first_row <= '0;
            collecting_second_row <= 1'b0;
            pair_has_second_row <= 1'b0;
            output_second_row <= 1'b0;
            residual_read_active <= 1'b0;
            residual_read_is_prefetch <= 1'b0;
            residual_terminal_seen <= 1'b0;
            residual_terminal_error <= 1'b0;
            residual_buffer_select <= 1'b0;
            residual_prefetch_pending <= 1'b0;
            residual_prefetched_valid <= 1'b0;
            residual_prefetched_error <= 1'b0;
            residual_prefetch_stripe <= '0;
            output_write_active <= 1'b0;
            output_terminal_seen <= 1'b0;
            output_terminal_error <= 1'b0;
            arithmetic_outstanding <= 1'b0;
            staging_read_outstanding <= 1'b0;
            configuration_stage <= '0;
            configured_end <= '0;
            local_last_end <= '0;
            required_bytes <= '0;
            local_first_address <= '0;
            configuration_row <= '0;
            configuration_row_address <= '0;
            configuration_accepted <= 1'b0;
            stripe_address <= '0;
            stripe_address_step <= '0;
            write_byte_address <= '0;
            residual_current_address <= '0;
            residual_prefetch_address <= '0;
            residual_address_step <= '0;
            residual_from_ddr <= 1'b0;
            residual_base <= '0;
            residual_limit <= '0;
            residual_storage_rows <= '0;
            residual_configured_end <= '0;
            residual_required_bytes <= '0;
            abort_ack <= 1'b0;
            abort_seen <= 1'b0;
            done_pulse <= 1'b0;
            error <= 1'b0;
            error_id <= '0;
        end else begin
            abort_ack <= 1'b0;
            done_pulse <= 1'b0;

            if (staging_read_request_valid && staging_read_request_ready)
                staging_read_outstanding <= 1'b1;
            if (staging_read_response_valid)
                staging_read_outstanding <= 1'b0;
            if (arithmetic_rsp_valid && arithmetic_rsp_ready)
                arithmetic_outstanding <= 1'b0;

            if (residual_read_request_done || residual_read_request_error) begin
                residual_read_active <= 1'b0;
                if (residual_read_is_prefetch) begin
                    residual_prefetched_valid <= !residual_read_request_error;
                    residual_prefetched_error <= residual_read_request_error;
                end else begin
                    residual_terminal_seen <= 1'b1;
                    if (residual_read_request_error)
                        residual_terminal_error <= 1'b1;
                end
            end
            if (output_write_done || output_write_error) begin
                output_write_active <= 1'b0;
                output_terminal_seen <= 1'b1;
                if (output_write_error)
                    output_terminal_error <= 1'b1;
            end

            if (residual_read_request_valid && residual_read_request_ready) begin
                residual_read_active <= 1'b1;
                residual_read_is_prefetch <= residual_prefetch_request;
                residual_receive_row <= '0;
                if (residual_prefetch_request)
                    residual_prefetch_pending <= 1'b0;
                else begin
                    residual_terminal_seen <= 1'b0;
                    residual_terminal_error <= 1'b0;
                end
            end

            if (residual_read_is_prefetch && residual_read_active &&
                residual_read_data_valid && residual_read_data_ready &&
                residual_data_metadata_valid && !residual_read_data_last)
                residual_receive_row <= residual_receive_row + 1'b1;

            if (fragment_fifo_input_fire) begin
                if (fragment_input_row + 1'b1 == fragment_input_rows) begin
                    fragment_input_row <= '0;
                    if (fragment_input_batch + 3'd1 < output_batch_count)
                        fragment_input_batch <= fragment_input_batch + 3'd1;
                    else begin
                        fragment_input_batch <= 3'd0;
                        if (fragment_input_stripe + 1'b1 == stripe_count)
                            fragment_input_active <= 1'b0;
                        else
                            fragment_input_stripe <= fragment_input_stripe + 1'b1;
                    end
                end else begin
                    fragment_input_row <= fragment_input_row + 1'b1;
                end
            end

            if (abort_request &&
                ((state != IDLE && state != ABORT_DRAIN &&
                  state != ABORT_WAIT_LOW) || configuration_stage != 0)) begin
                configuration_stage <= '0;
                fragment_input_active <= 1'b0;
                state <= ABORT_DRAIN;
            end else if (residual_prefetch_data_error && state != IDLE &&
                         state != ABORT_DRAIN && state != ABORT_WAIT_LOW) begin
                error <= 1'b1;
                error_id <= ERROR_READ;
                fragment_input_active <= 1'b0;
                state <= ABORT_DRAIN;
            end else if (residual_prefetched_error && state != IDLE &&
                         state != ABORT_DRAIN && state != ABORT_WAIT_LOW) begin
                error <= 1'b1;
                error_id <= ERROR_READ;
                fragment_input_active <= 1'b0;
                state <= ABORT_DRAIN;
            end else if (mode == MODE_RESIDUAL_DDR &&
                         output_batch_count > 3'd1 &&
                         (output_write_error || residual_read_request_error) &&
                         state != IDLE && state != ABORT_DRAIN &&
                         state != ABORT_WAIT_LOW) begin
                error <= 1'b1;
                error_id <= residual_read_request_error ? ERROR_READ : ERROR_WRITE;
                fragment_input_active <= 1'b0;
                state <= ABORT_DRAIN;
            end else if (residual_mode && fragment_input_active && in_valid &&
                         !fragment_input_metadata_valid && state != IDLE &&
                         state != ABORT_DRAIN && state != ABORT_WAIT_LOW) begin
                error <= 1'b1;
                error_id <= ERROR_CONFIGURATION;
                fragment_input_active <= 1'b0;
                if (mode == MODE_RESIDUAL_DDR && output_batch_count > 3'd1)
                    state <= ABORT_DRAIN;
                else begin
                    done_pulse <= 1'b1;
                    state <= IDLE;
                end
            end else case (state)
                IDLE: begin
                    if (cfg_valid && cfg_ready) begin
                        mode <= cfg_mode;
                        row_base <= cfg_physical_row_base;
                        row_count <= cfg_row_count;
                        first_batch_rows <= cfg_row_count;
                        second_batch_rows <= cfg_second_batch_rows;
                        third_batch_rows <= cfg_third_batch_rows;
                        fourth_batch_rows <= cfg_fourth_batch_rows;
                        fifth_batch_rows <= cfg_fifth_batch_rows;
                        sixth_batch_rows <= cfg_sixth_batch_rows;
                        output_batch_count <= cfg_batch_count == 0 ?
                            3'd1 : cfg_batch_count;
                        output_batch_index <= 3'd0;
                        batch_row_offset <= '0;
                        output_channels <= cfg_output_channels;
                        stripe_count <= cfg_fused_product ?
                            (16'((({1'b0, cfg_output_channels} + 17'd7) >> 3)) << 1) :
                            16'((({1'b0, cfg_output_channels} + 17'd7) >> 3));
                        fused_product <= cfg_fused_product;
                        stripe_index <= '0;
                        stripe_bytes <= 32'(cfg_row_count) << 4;
                        stripe_stride_bytes <=
                            (32'(cfg_row_count) + 32'(cfg_row_count[0])) << 4;
                        output_base <= cfg_output_base;
                        output_limit <= cfg_output_limit;
                        local_row_stride <= cfg_local_row_stride;
                        fragment_input_row <= '0;
                        fragment_input_batch <= 3'd0;
                        fragment_input_stripe <= '0;
                        fragment_input_active <= 1'b0;
                        residual_from_ddr <= cfg_residual_from_ddr;
                        residual_base <= cfg_residual_base;
                        residual_limit <= cfg_residual_limit;
                        residual_storage_rows <= cfg_residual_storage_rows;
                        input_row <= '0;
                        residual_receive_row <= '0;
                        output_row <= '0;
                        residual_read_active <= 1'b0;
                        residual_read_is_prefetch <= 1'b0;
                        residual_terminal_seen <= 1'b0;
                        residual_terminal_error <= 1'b0;
                        residual_buffer_select <= 1'b0;
                        residual_prefetch_pending <= 1'b0;
                        residual_prefetched_valid <= 1'b0;
                        residual_prefetched_error <= 1'b0;
                        residual_prefetch_stripe <= 13'd1;
                        output_write_active <= 1'b0;
                        output_terminal_seen <= 1'b0;
                        output_terminal_error <= 1'b0;
                        arithmetic_outstanding <= 1'b0;
                        pair_first_row <= '0;
                        collecting_second_row <= 1'b0;
                        pair_has_second_row <= 1'b0;
                        output_second_row <= 1'b0;
                        error <= 1'b0;
                        error_id <= '0;
                        configuration_stage <= 4'd1;
                    end else case (configuration_stage)
                        4'd1: begin
                            if (mode == MODE_PAIRED_GATE ||
                                mode == MODE_PAIRED_UP)
                                required_bytes <=
                                    (25'(stripe_count[13:0]) *
                                     25'(stripe_stride_bytes[9:0])) << 1;
                            else if (mode == MODE_FUSED_PRODUCT)
                                required_bytes <=
                                    25'(stripe_count[13:1]) *
                                    25'(stripe_stride_bytes[9:0]);
                            else
                                required_bytes <=
                                    25'(stripe_count[13:0]) *
                                    25'(stripe_bytes[9:0]);
                            residual_required_bytes <=
                                residual_storage_rows != 0 ?
                                    ((25'(stripe_count[13:0] - 1'b1) *
                                      (25'(residual_storage_rows) << 4)) +
                                     (25'(row_count) << 4)) :
                                    25'(stripe_count[13:0]) *
                                    25'(stripe_bytes[9:0]);
                            configuration_stage <= 4'd2;
                        end

                        4'd2: begin
                            residual_output_offset <= residual_from_ddr ?
                                residual_base - output_base : '0;
                            configured_end <= {1'b0, output_base} +
                                (ADDR_WIDTH+1)'(required_bytes);
                            second_batch_base <= {1'b0, output_base} +
                                (ADDR_WIDTH+1)'(required_bytes);
                            residual_configured_end <=
                                {1'b0, residual_base} +
                                (ADDR_WIDTH+1)'(residual_required_bytes);
                            configuration_row <= '0;
                            configuration_row_address <=
                                {7'd0, output_base[31:0]};
                            if (!configuration_shape_valid) begin
                                error <= 1'b1;
                                error_id <= ERROR_CONFIGURATION;
                                done_pulse <= 1'b1;
                                configuration_stage <= '0;
                            end else if (mode == MODE_LOCAL ||
                                         mode == MODE_RESIDUAL_LOCAL) begin
                                configuration_stage <= 4'd3;
                            end else if (output_batch_count > 3'd1) begin
                                configuration_stage <= 4'd6;
                            end else begin
                                configuration_stage <= 4'd4;
                            end
                        end

                        4'd6: begin
                            required_bytes <= mode == MODE_RESIDUAL_DDR ||
                                mode == MODE_DDR ?
                                25'(stripe_count[13:0]) *
                                    (25'(second_batch_rows) << 4) :
                                mode == MODE_FUSED_PRODUCT ?
                                25'(stripe_count[13:1]) *
                                    ((25'(second_batch_rows) +
                                      25'(second_batch_rows[0])) << 4) :
                                (25'(stripe_count[13:0]) *
                                    ((25'(second_batch_rows) +
                                      25'(second_batch_rows[0])) << 4)) << 1;
                            configuration_stage <= 4'd7;
                        end

                        4'd7: begin
                            if (mode == MODE_RESIDUAL_DDR)
                                residual_configured_end <= residual_configured_end +
                                    (ADDR_WIDTH+1)'(required_bytes);
                            configured_end <= second_batch_base +
                                (ADDR_WIDTH+1)'(required_bytes);
                            third_batch_base <= second_batch_base +
                                (ADDR_WIDTH+1)'(required_bytes);
                            configuration_stage <= output_batch_count > 3'd2 ?
                                4'd8 : 4'd4;
                        end

                        4'd8: begin
                            required_bytes <= mode == MODE_RESIDUAL_DDR ||
                                mode == MODE_DDR ?
                                25'(stripe_count[13:0]) *
                                    (25'(third_batch_rows) << 4) :
                                mode == MODE_FUSED_PRODUCT ?
                                25'(stripe_count[13:1]) *
                                    ((25'(third_batch_rows) +
                                      25'(third_batch_rows[0])) << 4) :
                                (25'(stripe_count[13:0]) *
                                    ((25'(third_batch_rows) +
                                      25'(third_batch_rows[0])) << 4)) << 1;
                            configuration_stage <= 4'd9;
                        end

                        4'd9: begin
                            if (mode == MODE_RESIDUAL_DDR)
                                residual_configured_end <=
                                    residual_configured_end +
                                    (ADDR_WIDTH+1)'(required_bytes);
                            configured_end <= third_batch_base +
                                (ADDR_WIDTH+1)'(required_bytes);
                            fourth_batch_base <= third_batch_base +
                                (ADDR_WIDTH+1)'(required_bytes);
                            configuration_stage <= output_batch_count > 3'd3 ?
                                4'd10 : 4'd4;
                        end

                        4'd10: begin
                            required_bytes <= mode == MODE_RESIDUAL_DDR ||
                                mode == MODE_DDR ?
                                25'(stripe_count[13:0]) *
                                    (25'(fourth_batch_rows) << 4) :
                                mode == MODE_FUSED_PRODUCT ?
                                25'(stripe_count[13:1]) *
                                    ((25'(fourth_batch_rows) +
                                      25'(fourth_batch_rows[0])) << 4) :
                                (25'(stripe_count[13:0]) *
                                    ((25'(fourth_batch_rows) +
                                      25'(fourth_batch_rows[0])) << 4)) << 1;
                            configuration_stage <= 4'd11;
                        end

                        4'd11: begin
                            if (mode == MODE_RESIDUAL_DDR)
                                residual_configured_end <=
                                    residual_configured_end +
                                    (ADDR_WIDTH+1)'(required_bytes);
                            configured_end <= fourth_batch_base +
                                (ADDR_WIDTH+1)'(required_bytes);
                            fifth_batch_base <= fourth_batch_base +
                                (ADDR_WIDTH+1)'(required_bytes);
                            configuration_stage <= output_batch_count > 3'd4 ?
                                4'd12 : 4'd4;
                        end

                        4'd12: begin
                            required_bytes <= mode == MODE_RESIDUAL_DDR ||
                                mode == MODE_DDR ?
                                25'(stripe_count[13:0]) *
                                    (25'(fifth_batch_rows) << 4) :
                                mode == MODE_FUSED_PRODUCT ?
                                25'(stripe_count[13:1]) *
                                    ((25'(fifth_batch_rows) +
                                      25'(fifth_batch_rows[0])) << 4) :
                                (25'(stripe_count[13:0]) *
                                    ((25'(fifth_batch_rows) +
                                      25'(fifth_batch_rows[0])) << 4)) << 1;
                            configuration_stage <= 4'd13;
                        end

                        4'd13: begin
                            if (mode == MODE_RESIDUAL_DDR)
                                residual_configured_end <=
                                    residual_configured_end +
                                    (ADDR_WIDTH+1)'(required_bytes);
                            configured_end <= fifth_batch_base +
                                (ADDR_WIDTH+1)'(required_bytes);
                            sixth_batch_base <= fifth_batch_base +
                                (ADDR_WIDTH+1)'(required_bytes);
                            configuration_stage <= output_batch_count > 3'd5 ?
                                4'd14 : 4'd4;
                        end

                        4'd14: begin
                            required_bytes <= mode == MODE_RESIDUAL_DDR ||
                                mode == MODE_DDR ?
                                25'(stripe_count[13:0]) *
                                    (25'(sixth_batch_rows) << 4) :
                                mode == MODE_FUSED_PRODUCT ?
                                25'(stripe_count[13:1]) *
                                    ((25'(sixth_batch_rows) +
                                      25'(sixth_batch_rows[0])) << 4) :
                                (25'(stripe_count[13:0]) *
                                    ((25'(sixth_batch_rows) +
                                      25'(sixth_batch_rows[0])) << 4)) << 1;
                            configuration_stage <= 4'd15;
                        end

                        4'd15: begin
                            if (mode == MODE_RESIDUAL_DDR)
                                residual_configured_end <=
                                    residual_configured_end +
                                    (ADDR_WIDTH+1)'(required_bytes);
                            configured_end <= sixth_batch_base +
                                (ADDR_WIDTH+1)'(required_bytes);
                            configuration_stage <= 4'd4;
                        end

                        4'd3: begin
                            if (configuration_row == {1'b0, row_base})
                                local_first_address <=
                                    configuration_row_address;
                            if (mode == MODE_LOCAL && output_batch_count == 3'd2 &&
                                configuration_row == {1'b0, first_batch_rows})
                                second_batch_base <= (ADDR_WIDTH+1)'(configuration_row_address);
                            if (configuration_row ==
                                ({1'b0, row_base} + {1'b0, row_count} +
                                 ((mode == MODE_LOCAL && output_batch_count == 3'd2) ?
                                    {1'b0, second_batch_rows} : 7'd0) -
                                 7'd1)) begin
                                local_last_end <=
                                    configuration_row_address +
                                    (39'(output_channels) << 1);
                                configuration_stage <= 4'd4;
                            end else begin
                                configuration_row <= configuration_row + 1'b1;
                                configuration_row_address <=
                                    configuration_row_address +
                                    39'(local_row_stride);
                            end
                        end

                        4'd4: begin
                            configuration_accepted <= configuration_valid;
                            configuration_stage <= 4'd5;
                        end

                        4'd5: begin
                            configuration_stage <= '0;
                            if (!configuration_accepted) begin
                                error <= 1'b1;
                                error_id <= ERROR_CONFIGURATION;
                                done_pulse <= 1'b1;
                            end else begin
                                first_batch_next_address <= output_base +
                                    (mode == MODE_PAIRED_UP ?
                                        ADDR_WIDTH'(stripe_stride_bytes) : '0);
                                second_batch_next_address <= second_batch_base[ADDR_WIDTH-1:0] +
                                    (mode == MODE_PAIRED_UP ?
                                        ((ADDR_WIDTH'(second_batch_rows) + ADDR_WIDTH'(second_batch_rows[0])) << 4) : '0);
                                third_batch_next_address <= third_batch_base[ADDR_WIDTH-1:0] +
                                    (mode == MODE_PAIRED_UP ?
                                        ((ADDR_WIDTH'(third_batch_rows) + ADDR_WIDTH'(third_batch_rows[0])) << 4) : '0);
                                fourth_batch_next_address <= fourth_batch_base[ADDR_WIDTH-1:0] +
                                    (mode == MODE_PAIRED_UP ?
                                        ((ADDR_WIDTH'(fourth_batch_rows) + ADDR_WIDTH'(fourth_batch_rows[0])) << 4) : '0);
                                fifth_batch_next_address <= fifth_batch_base[ADDR_WIDTH-1:0] +
                                    (mode == MODE_PAIRED_UP ?
                                        ((ADDR_WIDTH'(fifth_batch_rows) + ADDR_WIDTH'(fifth_batch_rows[0])) << 4) : '0);
                                sixth_batch_next_address <= sixth_batch_base[ADDR_WIDTH-1:0] +
                                    (mode == MODE_PAIRED_UP ?
                                        ((ADDR_WIDTH'(sixth_batch_rows) + ADDR_WIDTH'(sixth_batch_rows[0])) << 4) : '0);
                                fragment_input_active <= residual_mode;
                                stripe_address_step <=
                                    (mode == MODE_LOCAL ||
                                     mode == MODE_RESIDUAL_LOCAL) ?
                                        ADDR_WIDTH'(16) :
                                    ((mode == MODE_PAIRED_GATE ||
                                      mode == MODE_PAIRED_UP) ?
                                        ADDR_WIDTH'(stripe_stride_bytes << 1) :
                                     mode == MODE_FUSED_PRODUCT ?
                                        ADDR_WIDTH'(stripe_stride_bytes) :
                                        ADDR_WIDTH'(stripe_bytes));
                                residual_address_step <=
                                    residual_from_ddr && residual_storage_rows != 0 ?
                                        (ADDR_WIDTH'(residual_storage_rows) << 4) :
                                    residual_from_ddr ?
                                        ADDR_WIDTH'(stripe_bytes) :
                                    mode == MODE_RESIDUAL_LOCAL ?
                                        ADDR_WIDTH'(16) :
                                        ADDR_WIDTH'(stripe_bytes);
                                residual_current_address <= residual_from_ddr ?
                                    residual_base : output_base;
                                residual_prefetch_address <=
                                    (residual_from_ddr ? residual_base :
                                     output_base) +
                                    (residual_from_ddr ?
                                        (residual_storage_rows != 0 ?
                                            (ADDR_WIDTH'(residual_storage_rows) << 4) :
                                            ADDR_WIDTH'(stripe_bytes)) :
                                     mode == MODE_RESIDUAL_LOCAL ?
                                        ADDR_WIDTH'(16) :
                                        ADDR_WIDTH'(stripe_bytes));
                                if (mode == MODE_LOCAL ||
                                    mode == MODE_RESIDUAL_LOCAL) begin
                                    stripe_address <=
                                        ADDR_WIDTH'(local_first_address);
                                    write_byte_address <=
                                        ADDR_WIDTH'(local_first_address);
                                end else begin
                                    stripe_address <= output_base +
                                        (mode == MODE_PAIRED_UP ?
                                            ADDR_WIDTH'(stripe_stride_bytes) : '0);
                                    write_byte_address <= output_base +
                                        (mode == MODE_PAIRED_UP ?
                                            ADDR_WIDTH'(stripe_stride_bytes) : '0);
                                end
                                state <= (mode == MODE_RESIDUAL_LOCAL ||
                                    mode == MODE_RESIDUAL_DDR) ?
                                    ISSUE_RESIDUAL_READ : ACCEPT_FRAGMENT;
                            end
                        end

                        default: configuration_stage <= '0;
                    endcase
                end

                ISSUE_RESIDUAL_READ:
                    if (residual_read_request_valid &&
                        residual_read_request_ready) begin
                        state <= RECEIVE_RESIDUAL;
                    end

                RECEIVE_RESIDUAL:
                    if (residual_read_data_valid &&
                        residual_read_data_ready) begin
                        if (!residual_data_metadata_valid) begin
                            error <= 1'b1;
                            error_id <= ERROR_READ;
                            fragment_input_active <= 1'b0;
                            state <= ABORT_DRAIN;
                        end else if (residual_receive_row + 1'b1 ==
                            row_count) begin
                            state <= WAIT_RESIDUAL_DONE;
                        end else begin
                            residual_receive_row <=
                                residual_receive_row + 1'b1;
                        end
                    end

                WAIT_RESIDUAL_DONE:
                    if (residual_terminal_seen ||
                        residual_read_request_done ||
                        residual_read_request_error) begin
                        if (residual_terminal_error ||
                            residual_read_request_error) begin
                            error <= 1'b1;
                            error_id <= ERROR_READ;
                            if (mode == MODE_RESIDUAL_DDR &&
                                output_batch_count > 3'd1) begin
                                fragment_input_active <= 1'b0;
                                state <= ABORT_DRAIN;
                            end else begin
                                done_pulse <= 1'b1;
                                state <= IDLE;
                            end
                        end else begin
                            input_row <= '0;
                            if (residual_mode && output_batch_count == 3'd1 &&
                                !final_stripe) begin
                                residual_prefetch_pending <= 1'b1;
                                residual_prefetch_stripe <= stripe_index + 1'b1;
                            end
                            state <= ACCEPT_FRAGMENT;
                        end
                    end

                ACCEPT_FRAGMENT:
                    if ((residual_mode && fragment_fifo_output_fire) ||
                        (!residual_mode && in_valid && in_ready)) begin
                        if (residual_mode) begin
                            if (collecting_second_row)
                                saved_fragment_second <=
                                    fragment_fifo_output_data;
                            else begin
                                saved_fragment <= fragment_fifo_output_data;
                                pair_first_row <= input_row;
                            end
                            state <= READ_RESIDUAL_WORD;
                        end else if (fused_product && !stripe_index[0]) begin
                            if (final_input_row) begin
                                output_batch_index <= switch_batch_index;
                                batch_row_offset <= switch_batch_index == 0 ?
                                    9'd0 : batch_row_offset + {3'd0, row_count};
                                row_count <= switch_batch_rows;
                                input_row <= '0;
                                if (switch_batch_index == 3'd0)
                                    stripe_index <= stripe_index + 1'b1;
                            end else begin
                                input_row <= input_row + 1'b1;
                            end
                        end else if (fused_product) begin
                            saved_up <= in_data;
                            pair_first_row <= input_row;
                            output_row <= input_row;
                            state <= FUSED_WAIT_GATE;
                        end else if (final_input_row) begin
                            output_row <= '0;
                            state <= READ_OUTPUT_WORD;
                        end else begin
                            input_row <= input_row + 1'b1;
                        end
                    end else if (!residual_mode && in_valid &&
                                 !input_metadata_valid) begin
                        error <= 1'b1;
                        error_id <= ERROR_CONFIGURATION;
                        done_pulse <= 1'b1;
                        state <= IDLE;
                    end

                READ_RESIDUAL_WORD:
                    if (staging_read_request_valid &&
                        staging_read_request_ready)
                        state <= WAIT_RESIDUAL_WORD;

                WAIT_RESIDUAL_WORD:
                    if (staging_read_response_valid) begin
                        if (collecting_second_row) begin
                            saved_residual_second <= staging_read_data;
                            pair_has_second_row <= 1'b1;
                            state <= ISSUE_ADD;
                        end else begin
                            saved_residual <= staging_read_data;
                            saved_fragment_second <= '0;
                            saved_residual_second <= '0;
                            if (final_input_row) begin
                                pair_has_second_row <= 1'b0;
                                state <= ISSUE_ADD;
                            end else begin
                                input_row <= input_row + 1'b1;
                                collecting_second_row <= 1'b1;
                                state <= ACCEPT_FRAGMENT;
                            end
                        end
                    end

                ISSUE_ADD:
                    if (arithmetic_req_valid && arithmetic_req_ready) begin
                        arithmetic_outstanding <= 1'b1;
                        state <= WAIT_ADD;
                    end

                WAIT_ADD:
                    if (arithmetic_rsp_valid && arithmetic_rsp_ready) begin
                        arithmetic_outstanding <= 1'b0;
                        if (arithmetic_rsp_tag !=
                                {stripe_index[9:0], pair_first_row} ||
                            arithmetic_rsp_lane_mask !=
                                (pair_has_second_row ? 16'hffff : 16'h00ff)) begin
                            error <= 1'b1;
                            error_id <= ERROR_ARITHMETIC;
                            done_pulse <= 1'b1;
                            fragment_input_active <= 1'b0;
                            state <= IDLE;
                        end else begin
                            saved_output <= arithmetic_rsp_values[127:0];
                            saved_output_second <=
                                arithmetic_rsp_values[255:128];
                            output_row <= pair_first_row;
                            output_second_row <= 1'b0;
                            state <= SEND_OUTPUT_WORD;
                        end
                    end

                READ_OUTPUT_WORD:
                    if (staging_read_request_valid &&
                        staging_read_request_ready)
                        state <= WAIT_OUTPUT_WORD;

                WAIT_OUTPUT_WORD:
                    if (staging_read_response_valid) begin
                        saved_output <= staging_read_data;
                        state <= SEND_OUTPUT_WORD;
                    end

                FUSED_WAIT_GATE:
                    if (staging_read_response_valid) begin
                        saved_gate <= staging_read_data;
                        state <= FUSED_SILU_REQUEST;
                    end

                FUSED_SILU_REQUEST, FUSED_PRODUCT_REQUEST:
                    if (arithmetic_req_valid && arithmetic_req_ready) begin
                        arithmetic_outstanding <= 1'b1;
                        state <= state == FUSED_SILU_REQUEST ?
                            FUSED_SILU_RESPONSE : FUSED_PRODUCT_RESPONSE;
                    end

                FUSED_SILU_RESPONSE, FUSED_PRODUCT_RESPONSE:
                    if (arithmetic_rsp_valid && arithmetic_rsp_ready) begin
                        if (arithmetic_rsp_tag !=
                                {stripe_index[9:0], pair_first_row} ||
                            arithmetic_rsp_lane_mask != fused_lane_mask) begin
                            error <= 1'b1;
                            error_id <= ERROR_ARITHMETIC;
                            state <= ABORT_DRAIN;
                        end else if (state == FUSED_SILU_RESPONSE) begin
                            saved_silu <= arithmetic_rsp_values[127:0];
                            state <= FUSED_PRODUCT_REQUEST;
                        end else begin
                            saved_output <= arithmetic_rsp_values[127:0];
                            output_second_row <= 1'b0;
                            state <= SEND_OUTPUT_WORD;
                        end
                    end

                SEND_OUTPUT_WORD:
                    if (ddr_output_mode &&
                        (output_terminal_error || output_write_error)) begin
                        error <= 1'b1;
                        error_id <= ERROR_WRITE;
                        done_pulse <= 1'b1;
                        fragment_input_active <= 1'b0;
                        state <= IDLE;
                    end else if ((ddr_output_mode && output_write_valid &&
                         output_write_ready) ||
                        (!ddr_output_mode && local_write_valid &&
                         local_write_ready)) begin
                        if (ddr_output_mode && output_write_first) begin
                            output_write_active <= 1'b1;
                            output_terminal_seen <= 1'b0;
                            output_terminal_error <= 1'b0;
                        end
                        if (!final_output_row)
                            write_byte_address <= write_byte_address +
                                (ddr_output_mode ? ADDR_WIDTH'(16) :
                                    ADDR_WIDTH'(local_row_stride));
                        if (residual_mode && pair_has_second_row &&
                            !output_second_row) begin
                            output_row <= output_row + 1'b1;
                            output_second_row <= 1'b1;
                        end else if (final_output_row) begin
                            if (final_stripe && output_batch_index + 3'd1 >= output_batch_count) begin
                                if (ddr_output_mode)
                                    state <= WAIT_OUTPUT_DONE;
                                else begin
                                    done_pulse <= 1'b1;
                                    fragment_input_active <= 1'b0;
                                    state <= IDLE;
                                end
                            end
                            else if (output_batch_count > 3'd1) begin
                                // Each batch has a contiguous image. Batches
                                // alternate within each N8 stripe.
                                case (output_batch_index)
                                    3'd0: first_batch_next_address <=
                                        stripe_address + stripe_address_step;
                                    3'd1: second_batch_next_address <=
                                        stripe_address + stripe_address_step;
                                    3'd2: third_batch_next_address <=
                                        stripe_address + stripe_address_step;
                                    3'd3: fourth_batch_next_address <=
                                        stripe_address + stripe_address_step;
                                    3'd4: fifth_batch_next_address <=
                                        stripe_address + stripe_address_step;
                                    default: sixth_batch_next_address <=
                                        stripe_address + stripe_address_step;
                                endcase
                                output_batch_index <= switch_batch_index;
                                batch_row_offset <= switch_batch_index == 0 ?
                                    9'd0 : batch_row_offset + {3'd0, row_count};
                                input_row <= '0;
                                collecting_second_row <= 1'b0;
                                pair_has_second_row <= 1'b0;
                                output_second_row <= 1'b0;
                                stripe_address <= switch_batch_address;
                                write_byte_address <= switch_batch_address;
                                row_count <= switch_batch_rows;
                                stripe_bytes <= 32'(switch_batch_rows) << 4;
                                stripe_stride_bytes <=
                                    (32'(switch_batch_rows) +
                                     32'(switch_batch_rows[0])) << 4;
                                stripe_address_step <=
                                    mode == MODE_LOCAL ? ADDR_WIDTH'(16) :
                                    mode == MODE_DDR ?
                                    (ADDR_WIDTH'(switch_batch_rows) << 4) : residual_mode ?
                                    (ADDR_WIDTH'(switch_batch_rows) << 4) :
                                    fused_product ?
                                    ((ADDR_WIDTH'(switch_batch_rows) +
                                      ADDR_WIDTH'(switch_batch_rows[0])) << 4) :
                                    ((ADDR_WIDTH'(switch_batch_rows) +
                                      ADDR_WIDTH'(switch_batch_rows[0])) << 5);
                                if (switch_batch_index == 3'd0) begin
                                    stripe_index <= stripe_index + 1'b1;
                                end
                                if (residual_mode) begin
                                    // Batch order changes rows at every stripe,
                                    // so reload the existing residual SRAM.
                                    residual_current_address <=
                                        switch_batch_address + residual_output_offset;
                                    state <= ISSUE_RESIDUAL_READ;
                                end else
                                    state <= ACCEPT_FRAGMENT;
                            end else if (final_stripe) begin
                                done_pulse <= 1'b1;
                                fragment_input_active <= 1'b0;
                                state <= IDLE;
                            end else begin
                                stripe_index <= stripe_index + 1'b1;
                                input_row <= '0;
                                collecting_second_row <= 1'b0;
                                pair_has_second_row <= 1'b0;
                                output_second_row <= 1'b0;
                                stripe_address <= stripe_address +
                                    stripe_address_step;
                                write_byte_address <= stripe_address +
                                    stripe_address_step;
                                if (residual_mode) begin
                                    if (residual_prefetched_valid) begin
                                        residual_buffer_select <=
                                            !residual_buffer_select;
                                        residual_prefetched_valid <= 1'b0;
                                        residual_current_address <=
                                            residual_prefetch_address;
                                        residual_prefetch_address <=
                                            residual_prefetch_address +
                                            residual_address_step;
                                        if (stripe_index + 2 < stripe_count) begin
                                            residual_prefetch_pending <= 1'b1;
                                            residual_prefetch_stripe <=
                                                stripe_index + 2;
                                        end
                                        state <= ACCEPT_FRAGMENT;
                                    end else begin
                                        state <= WAIT_PREFETCHED_RESIDUAL;
                                    end
                                end else
                                    state <= ACCEPT_FRAGMENT;
                            end
                        end else if (residual_mode) begin
                            input_row <= output_row + 1'b1;
                            collecting_second_row <= 1'b0;
                            pair_has_second_row <= 1'b0;
                            output_second_row <= 1'b0;
                            state <= ACCEPT_FRAGMENT;
                        end else if (fused_product) begin
                            input_row <= input_row + 1'b1;
                            state <= ACCEPT_FRAGMENT;
                        end else begin
                            output_row <= output_row + 1'b1;
                            state <= READ_OUTPUT_WORD;
                        end
                    end

                WAIT_OUTPUT_DONE:
                    if (output_terminal_seen || output_write_done ||
                        output_write_error) begin
                        if (output_terminal_error || output_write_error) begin
                            error <= 1'b1;
                            error_id <= ERROR_WRITE;
                            done_pulse <= 1'b1;
                            fragment_input_active <= 1'b0;
                            state <= IDLE;
                        end else if (final_stripe) begin
                            done_pulse <= 1'b1;
                            fragment_input_active <= 1'b0;
                            state <= IDLE;
                        end else begin
                            stripe_index <= stripe_index + 1'b1;
                            input_row <= '0;
                            collecting_second_row <= 1'b0;
                            pair_has_second_row <= 1'b0;
                            output_second_row <= 1'b0;
                            stripe_address <= stripe_address +
                                stripe_address_step;
                            write_byte_address <= stripe_address +
                                stripe_address_step;
                            if (mode == MODE_RESIDUAL_DDR) begin
                                if (residual_prefetched_valid) begin
                                    residual_buffer_select <=
                                        !residual_buffer_select;
                                    residual_prefetched_valid <= 1'b0;
                                    residual_current_address <=
                                        residual_prefetch_address;
                                    residual_prefetch_address <=
                                        residual_prefetch_address +
                                        residual_address_step;
                                    if (stripe_index + 2 < stripe_count) begin
                                        residual_prefetch_pending <= 1'b1;
                                        residual_prefetch_stripe <=
                                            stripe_index + 2;
                                    end
                                    state <= ACCEPT_FRAGMENT;
                                end else begin
                                    state <= WAIT_PREFETCHED_RESIDUAL;
                                end
                            end else begin
                                state <= residual_mode ?
                                    ISSUE_RESIDUAL_READ : ACCEPT_FRAGMENT;
                            end
                        end
                    end

                WAIT_PREFETCHED_RESIDUAL:
                    if (residual_prefetched_valid) begin
                        residual_buffer_select <= !residual_buffer_select;
                        residual_prefetched_valid <= 1'b0;
                        residual_current_address <= residual_prefetch_address;
                        residual_prefetch_address <=
                            residual_prefetch_address + residual_address_step;
                        if (stripe_index + 1 < stripe_count) begin
                            residual_prefetch_pending <= 1'b1;
                            residual_prefetch_stripe <= stripe_index + 1'b1;
                        end
                        state <= ACCEPT_FRAGMENT;
                    end

                ABORT_DRAIN: begin
                    if (arithmetic_rsp_valid && arithmetic_rsp_ready)
                        arithmetic_outstanding <= 1'b0;
                    if ((!residual_read_active ||
                         residual_read_request_done ||
                         residual_read_request_error) &&
                        (!output_write_active ||
                         output_terminal_seen || output_write_done ||
                         output_write_error) &&
                        (!arithmetic_outstanding ||
                         arithmetic_abort_ack) &&
                        (!staging_read_outstanding ||
                         staging_read_response_valid) &&
                        fragment_fifo_occupancy == 4'd0) begin
                        abort_ack <= 1'b1;
                        if (mode == MODE_RESIDUAL_DDR &&
                            output_batch_count > 3'd1 && error)
                            done_pulse <= 1'b1;
                        residual_prefetch_pending <= 1'b0;
                        residual_prefetched_valid <= 1'b0;
                        residual_prefetched_error <= 1'b0;
                        abort_seen <= 1'b1;
                        state <= ABORT_WAIT_LOW;
                    end
                end

                ABORT_WAIT_LOW:
                    if (!abort_request) begin
                        abort_seen <= 1'b0;
                        state <= IDLE;
                    end

                default: state <= IDLE;
            endcase
        end
    end

    initial begin
        if (MAX_ROWS != 48 || ADDR_WIDTH < 32)
            $error("matmul_output_controller requires MAX_ROWS=48 and ADDR_WIDTH>=32");
    end

`ifndef SYNTHESIS
    logic stalled_output;
    logic [ADDR_WIDTH+32+128+16+2-1:0] stalled_output_payload;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_output <= 1'b0;
            stalled_output_payload <= '0;
        end else begin
            if (stalled_output && !abort_request &&
                !output_write_error && !output_terminal_error)
                assert (output_write_valid &&
                    {output_write_byte_address,
                     output_write_transaction_bytes, output_write_data,
                     output_write_byte_enable, output_write_first,
                     output_write_last} == stalled_output_payload)
                    else $error("matmul_output_controller changed a stalled DDR write beat");
            stalled_output <= output_write_valid &&
                !output_write_ready && !abort_request;
            if (output_write_valid && !output_write_ready)
                stalled_output_payload <=
                    {output_write_byte_address,
                     output_write_transaction_bytes, output_write_data,
                     output_write_byte_enable, output_write_first,
                     output_write_last};
            if (in_valid && state == ACCEPT_FRAGMENT && !abort_request)
                assert (residual_mode ? fragment_input_metadata_valid :
                        input_metadata_valid)
                    else $error(
                        "matmul_output_controller received row=%0d channel=%0d, expected row=%0d stripe=%0d, row_base=%0d row_count=%0d output_type=%0d",
                        in_physical_row, in_output_channel,
                        row_base + (residual_mode ? fragment_input_row : input_row),
                        residual_mode ? fragment_input_stripe : stripe_index,
                        row_base, row_count, mode);
            assert (fragment_fifo_occupancy <= 4'd8)
                else $error("matmul_output_controller fragment FIFO occupancy exceeded eight");
            if (done_pulse && residual_mode && !error)
                assert (!fragment_input_active &&
                        fragment_fifo_occupancy == 4'd0)
                    else $error("matmul_output_controller completed before residual fragments drained");
        end
    end
`endif
endmodule

`default_nettype wire
