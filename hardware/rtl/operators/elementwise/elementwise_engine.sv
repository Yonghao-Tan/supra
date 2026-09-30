`default_nettype none

// Executes SiLU(Gate) * Up for one homogeneous row batch at a time.  The
// paired workspace remains N8-stripe-major in DDR; two bounded logical reads
// fetch only the current batch's Gate and Up words for each stripe.
module elementwise_engine #(
    parameter integer FFN_FEATURES = 12288,
    parameter integer MAX_ROWS = 48,
    parameter integer ADDR_WIDTH = 64,
    parameter integer TAG_WIDTH = 16,
    parameter integer PREFETCH_SLOTS = 1,
    parameter bit FIXED_SCHEDULE_INGRESS = 1'b0
) (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       abort_request,
    output logic                       abort_ack,

    input  logic                       start_valid,
    output logic                       start_ready,
    input  logic                       start_r4_enable,
    input  logic                       start_two_token_r4,
    input  logic                       start_precomputed_product,
    input  logic [5:0]                 start_active_rows,
    input  logic [1:0]                 start_segment_count,
    input  logic [5:0]                 start_segment_mode,
    input  logic [17:0]                start_segment_row_base,
    input  logic [17:0]                start_segment_row_count,
    input  logic [95:0]                start_segment_activation_base_byte_offset,
    input  hardware_types_pkg::matmul_row_config_t fixed_row_config,
    input  logic [31:0]                start_activation_limit_byte_offset,
    input  logic [31:0]                start_activation_base_byte_offset,
    input  logic [ADDR_WIDTH-1:0]      start_workspace_base,
    input  logic [ADDR_WIDTH-1:0]      start_workspace_limit,
    output logic                       busy,
    output logic                       done_valid,
    input  logic                       done_ready,
    output hardware_types_pkg::activation_layout_t completed_layout,
    output logic                       read_phase_complete,
    output logic                       down_prefetch_ready,
    output logic                       error,
    output logic [3:0]                 error_id,

    output logic                       read_request_valid,
    input  logic                       read_request_ready,
    output logic [ADDR_WIDTH-1:0]      read_request_address,
    output logic [31:0]                read_request_bytes,
    output logic [7:0]                 read_request_tag,
    output logic                       read_second_span_valid,
    output logic [ADDR_WIDTH-1:0]      read_second_span_address,
    output logic [31:0]                read_pair_stride,
    output logic [10:0]                read_pair_count,
    input  logic                       read_request_done,
    input  logic                       read_request_error,
    input  logic                       read_data_valid,
    output logic                       read_data_ready,
    input  logic [127:0]               read_data,
    input  logic [15:0]                read_byte_enable,
    input  logic                       read_data_last,
    input  logic [7:0]                 read_data_tag,
    input  logic                       read_data_span,

    output logic                       arithmetic_req_valid,
    input  logic                       arithmetic_req_ready,
    output logic [2:0]                 arithmetic_req_operation,
    output logic [1023:0]              arithmetic_req_values,
    output logic [1023:0]              arithmetic_req_paired_values,
    output logic [1023:0]              arithmetic_req_factor0_values,
    output logic [1023:0]              arithmetic_req_factor1_values,
    output logic [63:0]                arithmetic_req_lane_mask,
    output logic [TAG_WIDTH-1:0]       arithmetic_req_tag,
    input  logic                       arithmetic_rsp_valid,
    output logic                       arithmetic_rsp_ready,
    input  logic [1023:0]              arithmetic_rsp_values,
    input  logic [63:0]                arithmetic_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]       arithmetic_rsp_tag,
    input  logic                       arithmetic_abort_ack,

    output logic                       max_req_valid,
    input  logic                       max_req_ready,
    output logic [1023:0]              max_req_values,
    output logic [63:0]                max_req_lane_mask,
    output logic [TAG_WIDTH-1:0]       max_req_tag,
    input  logic                       max_rsp_valid,
    output logic                       max_rsp_ready,
    input  logic [127:0]               max_rsp_values,
    input  logic [7:0]                 max_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0]       max_rsp_tag,

    output logic                       quant_scale_req_valid,
    input  logic                       quant_scale_req_ready,
    output logic [7:0]                 quant_scale_req_a4_row_mask,
    output logic [127:0]               quant_scale_req_row_max_abs,
    output logic [7:0]                 quant_scale_req_row_mask,
    input  logic                       quant_scale_rsp_valid,
    output logic                       quant_scale_rsp_ready,
    input  logic [127:0]               quant_scale_rsp_values_bf16,
    output logic                       quant_values_req_valid,
    input  logic                       quant_values_req_ready,
    output logic [1023:0]              quant_values_req_values_bf16,
    output logic [63:0]                quant_values_req_lane_mask,
    output logic [TAG_WIDTH-1:0]       quant_values_req_tag,
    input  logic                       quant_values_rsp_valid,
    output logic                       quant_values_rsp_ready,
    input  logic [511:0]               quant_values_rsp_values,
    input  logic [63:0]                quant_values_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]       quant_values_rsp_tag,
    input  logic                       quant_abort_ack,

    output logic                       writer_cfg_valid,
    input  logic                       writer_cfg_ready,
    output logic [1:0]                 writer_cfg_mode,
    output logic [5:0]                 writer_cfg_physical_row_base,
    output logic [5:0]                 writer_cfg_row_count,
    output logic [15:0]                writer_cfg_elements_per_row,
    output logic [31:0]                writer_cfg_activation_base_byte_offset,
    output logic [31:0]                writer_cfg_activation_limit_byte_offset,
    output logic                       writer_cfg_mixed_group_enable,
    output logic [5:0]                 writer_cfg_compute_group_count,
    output logic [47:0]                writer_cfg_row_precision_a8,
    output logic [287:0]               writer_cfg_row_compute_group,
    output logic [143:0]               writer_cfg_row_pe_slot,
    output logic [95:0]                writer_cfg_row_phase_mask,
    output logic                       writer_scale_valid,
    input  logic                       writer_scale_ready,
    output logic [5:0]                 writer_scale_row_base,
    output logic [7:0]                 writer_scale_row_mask,
    output logic [127:0]               writer_scale_values_bf16,
    output logic                       writer_quantized_valid,
    input  logic                       writer_quantized_ready,
    output logic [5:0]                 writer_quantized_row_base,
    output logic [15:0]                writer_quantized_element_base,
    output logic [511:0]               writer_quantized_values,
    output logic [63:0]                writer_quantized_lane_mask,
    output logic [TAG_WIDTH-1:0]       writer_quantized_tag,
    input  logic                       writer_done_pulse,
    input  logic                       writer_error,
    output logic                       writer_abort_request,

    output logic                       product_memory_req_valid,
    input  logic                       product_memory_req_ready,
    output hardware_types_pkg::elementwise_product_request_t
                                       product_memory_req,
    input  logic                       product_memory_rsp_valid,
    output logic                       product_memory_rsp_ready,
    input  hardware_types_pkg::elementwise_product_response_t
                                       product_memory_rsp,

    output logic                       trace_sample_valid,
    input  logic                       trace_sample_ready,
    output logic                       trace_sample_pass,
    output logic [10:0]                trace_sample_stripe,
    output logic [5:0]                 trace_sample_row_base,
    output logic [63:0]                trace_sample_lane_mask,
    output logic [1023:0]              trace_sample_silu,
    output logic [1023:0]              trace_sample_product,

    output logic [63:0]                accepted_read_request_count,
    output logic [63:0]                accepted_read_byte_count,
    output logic [63:0]                accepted_arithmetic_request_count,
    output logic [63:0]                completed_product_tile_count,
    output logic [63:0]                accepted_max_request_count,
    output logic [63:0]                accepted_quant_scale_count,
    output logic [63:0]                accepted_quantized_value_count
);
    localparam logic [2:0] BF16_MULTIPLY = 3'd1;
    localparam logic [2:0] BF16_SILU = 3'd3;
    localparam logic [3:0] ERROR_CONFIGURATION = 4'h1;
    localparam logic [3:0] ERROR_DMA = 4'h2;
    localparam logic [3:0] ERROR_ARITHMETIC = 4'h3;
    localparam logic [3:0] ERROR_MAX = 4'h4;
    localparam logic [3:0] ERROR_QUANTIZER = 4'h5;
    localparam logic [3:0] ERROR_WRITER = 4'h6;
    localparam integer STRIPE_COUNT = FFN_FEATURES / 8;

    typedef enum logic [5:0] {
        IDLE,
        GATE_REQUEST, GATE_STREAM, GATE_TERMINAL,
        UP_REQUEST, UP_STREAM, UP_TERMINAL,
        SILU_REQUEST, SILU_RESPONSE,
        PRODUCT_REQUEST, PRODUCT_RESPONSE,
        PRODUCT_STORE_LOW, PRODUCT_STORE_HIGH,
        MAX_REQUEST, MAX_RESPONSE,
        TRACE_SAMPLE,
        WAIT_PREFETCH,
        WRITER_CONFIG,
        SCALE_REQUEST, SCALE_RESPONSE, SCALE_WRITE,
        PRODUCT_LOAD_LOW, PRODUCT_LOAD_HIGH, PRODUCT_LOAD_RESPONSE,
        QUANTIZED_REQUEST, QUANTIZED_RESPONSE, QUANTIZED_WRITE,
        ADVANCE_BATCH, COMPLETE,
        ABORT_DRAIN, ABORT_WAIT_LOW,
        PRODUCT_QUANT_STREAM,
        R4_START, R4_WAIT
    } state_t;

    state_t state;
    typedef enum logic [3:0] {
        PREFETCH_IDLE,
        PREFETCH_GATE_REQUEST, PREFETCH_GATE_STREAM, PREFETCH_GATE_TERMINAL,
        PREFETCH_UP_REQUEST, PREFETCH_UP_STREAM, PREFETCH_UP_TERMINAL,
        PREFETCH_PATTERN_WAIT
    } prefetch_state_t;
    localparam integer PREFETCH_SLOT_WIDTH =
        PREFETCH_SLOTS <= 1 ? 1 : $clog2(PREFETCH_SLOTS);
    prefetch_state_t prefetch_state;
    logic [5:0] active_token_count;
    logic [1:0] segment_count;
    logic [5:0] segment_mode;
    logic [17:0] segment_row_base;
    logic [17:0] segment_row_count;
    logic [95:0] segment_activation_base_byte_offset;
    logic [47:0] row_enable;
    logic mixed_groups;
    logic r4_enable;
    logic two_token_r4;
    logic precomputed_product;
    logic r4_panel_released;
    logic r4_active;
    logic r4_started;
    logic [1:0] r4_quartet_index;
    logic [31:0] activation_limit_byte_offset;
    logic [31:0] activation_base_byte_offset;
    logic [ADDR_WIDTH-1:0] workspace_base;
    logic [1:0] segment_index;
    logic [5:0] batch_row_base;
    logic [3:0] batch_rows;
    logic pass_index;
    logic [10:0] stripe_index;
    logic [4:0] stream_row;
    logic request_active;
    logic request_terminal_seen;
    logic request_terminal_error;
    logic [7:0] accepted_read_tag;
    logic accepted_read_paired;
    logic [10:0] accepted_read_pair_count;
    logic repeated_pair_active;
    logic arithmetic_outstanding;
    logic max_outstanding;
    logic quant_scale_outstanding;
    logic [4:0] quant_outstanding;
    logic [1:0] product_read_outstanding;
    logic product_low_received;
    logic abort_pending;
    // Gate, SiLU, and Product are consecutive states for one tile. Reuse one
    // register image because none of those functional values overlap.
    logic [1023:0] tile_values;
    logic [1023:0] up_values;
    logic [1023:0] second_gate_values;
    logic [1023:0] second_up_values;
    logic [1023:0] prefetch_gate_values [0:PREFETCH_SLOTS-1];
    logic [1023:0] prefetch_up_values [0:PREFETCH_SLOTS-1];
    logic [1023:0] prefetch_second_gate_values [0:PREFETCH_SLOTS-1];
    logic [1023:0] prefetch_second_up_values [0:PREFETCH_SLOTS-1];
    logic [PREFETCH_SLOTS-1:0] prefetch_slot_valid;
    logic [10:0] prefetch_slot_stripe [0:PREFETCH_SLOTS-1];
    logic [ADDR_WIDTH-1:0] prefetch_slot_address [0:PREFETCH_SLOTS-1];
    logic [PREFETCH_SLOT_WIDTH-1:0] prefetch_fill_slot;
    logic [10:0] prefetch_stripe_index;
    logic [4:0] prefetch_stream_row;
    logic [10:0] next_prefetch_stripe;
    logic [ADDR_WIDTH-1:0] next_prefetch_address;
    logic prefetch_window_active;
    logic prefetch_free_found;
    logic [PREFETCH_SLOT_WIDTH-1:0] prefetch_free_slot;
    logic next_stripe_found;
    logic [PREFETCH_SLOT_WIDTH-1:0] next_stripe_slot;
    logic [$clog2(PREFETCH_SLOTS+1)-1:0] prefetch_ready_count;
`ifndef SYNTHESIS
    logic [1023:0] trace_silu_values;
`endif
    logic [127:0] row_max_abs;
    logic [127:0] second_row_max_abs;
    logic [127:0] batch_scales;
    logic [511:0] quantized_values;
    logic [63:0] quantized_lane_mask;
    logic [TAG_WIDTH-1:0] quantized_tag;
    logic product_stream_request_half;
    logic [10:0] product_stream_request_stripe;
    logic [10:0] product_stream_response_stripe;
    logic [10:0] product_stream_quant_response_stripe;
    logic [511:0] product_stream_low_data;
    logic [TAG_WIDTH-1:0] product_stream_low_tag;
    logic product_stream_trace_sent;
    logic product_stream_fifo_input_valid;
    logic product_stream_fifo_input_ready;
    logic [1050:0] product_stream_fifo_input_data;
    logic product_stream_fifo_output_valid;
    logic product_stream_fifo_output_ready;
    logic [1050:0] product_stream_fifo_output_data;
    logic product_stream_entry_valid;
    logic [1050:0] product_stream_entry_data;
    logic [1:0] product_stream_fifo_occupancy;
    logic [1:0] product_stream_reserved_tiles;
    logic [1023:0] product_stream_fifo_values;
    logic [TAG_WIDTH-1:0] product_stream_fifo_tag;
    logic [10:0] product_stream_fifo_stripe;
    logic product_stream_request_fire;
    logic product_stream_response_fire;
    logic product_stream_trace_fire;
    logic product_stream_quant_request_fire;
    logic product_stream_quant_response_fire;
    logic writer_done_seen;
    logic [63:0] lane_mask;
    logic [7:0] row_mask;
    logic [ADDR_WIDTH:0] required_workspace_end;
    logic [31:0] stripe_bytes;
    logic [31:0] stripe_pair_bytes;
    logic [31:0] batch_bytes;
    logic [31:0] read_bytes;
    logic [5:0] read_group_row_base;
    logic [4:0] read_group_rows;
    logic [4:0] read_rows;
    logic [3:0] first_batch_rows;
    logic [3:0] second_batch_rows;
    logic compute_second_batch;
    logic [1:0] current_mode;
    logic [5:0] current_segment_base;
    logic [5:0] current_segment_rows;
    logic [31:0] current_segment_activation_base_byte_offset;
    logic [5:0] next_batch_base;
    logic [6:0] current_segment_end;
    logic final_stripe;
    logic final_batch;
    logic [TAG_WIDTH-1:0] tile_tag;
    logic [1023:0] silu_input_values;
    logic [1023:0] silu_slopes;
    logic [1023:0] silu_intercepts;
    logic [31:0] lane_silu_coefficients [0:63];
    logic main_read_request;
    logic prefetch_read_request;
    logic selected_read_up;
    logic pairing_eligible;
    logic [ADDR_WIDTH-1:0] candidate_second_span_address;
    logic [ADDR_WIDTH-1:0] batch_gate_address;
    logic [ADDR_WIDTH-1:0] current_gate_address;
    logic [ADDR_WIDTH-1:0] prefetch_gate_address;
    logic [1:0] command_segment_count;
    logic [5:0] command_segment_mode;
    logic [17:0] command_segment_row_base;
    logic [17:0] command_segment_row_count;
    logic [95:0] command_segment_activation_base_byte_offset;
    logic [47:0] command_row_enable;
    logic command_mixed_groups;
    logic [31:0] product_group_byte_base;
    logic product_group_recompute;
    logic [15:0] current_batch_product_words;
    logic [31:0] next_group_byte_base;
    logic [31:0] next_segment_group_byte_base;
    logic [4:0] next_group_rows;
    logic [4:0] next_segment_group_rows;
    logic next_group_recompute;
    logic next_segment_group_recompute;
    logic final_read_group;

    logic base_arithmetic_req_valid;
    logic [2:0] base_arithmetic_req_operation;
    logic [1023:0] base_arithmetic_req_values;
    logic [1023:0] base_arithmetic_req_paired_values;
    logic [1023:0] base_arithmetic_req_factor0_values;
    logic [1023:0] base_arithmetic_req_factor1_values;
    logic [63:0] base_arithmetic_req_lane_mask;
    logic [TAG_WIDTH-1:0] base_arithmetic_req_tag;
    logic base_arithmetic_rsp_ready;
    logic base_max_req_valid;
    logic [1023:0] base_max_req_values;
    logic [63:0] base_max_req_lane_mask;
    logic [TAG_WIDTH-1:0] base_max_req_tag;
    logic base_max_rsp_ready;
    logic base_product_memory_req_valid;
    hardware_types_pkg::elementwise_product_request_t base_product_memory_req;
    logic base_product_memory_rsp_ready;

    logic r4_start_ready;
    logic r4_done_valid;
    logic r4_done_error;
    logic [3:0] r4_done_error_id;
    logic [127:0] r4_row_max_abs;
    logic r4_abort_ack;
    logic r4_product_req_valid;
    logic r4_product_req_ready;
    hardware_types_pkg::elementwise_product_request_t r4_product_req;
    logic r4_product_rsp_ready;
    logic r4_arithmetic_req_valid;
    logic r4_arithmetic_req_ready;
    logic [2:0] r4_arithmetic_req_operation;
    logic [1023:0] r4_arithmetic_req_values;
    logic [1023:0] r4_arithmetic_req_paired_values;
    logic [1023:0] r4_arithmetic_req_factor0_values;
    logic [1023:0] r4_arithmetic_req_factor1_values;
    logic [63:0] r4_arithmetic_req_lane_mask;
    logic [TAG_WIDTH-1:0] r4_arithmetic_req_tag;
    logic r4_arithmetic_rsp_ready;
    logic r4_max_req_valid;
    logic r4_max_req_ready;
    logic [1023:0] r4_max_req_values;
    logic [63:0] r4_max_req_lane_mask;
    logic [TAG_WIDTH-1:0] r4_max_req_tag;
    logic r4_max_rsp_ready;
    logic [2:0] r4_start_row_count;
    logic [14:0] r4_start_word_base;
    logic r4_prefetch_valid;
    logic [2:0] r4_prefetch_row_count;
    logic [14:0] r4_prefetch_word_base;
    logic [1:0] r4_next_quartet_index;
    logic [31:0] unused_r4_product_read_count;
    logic [31:0] unused_r4_product_write_count;
    logic [31:0] unused_r4_add_count;
    logic [31:0] unused_r4_multiply_count;
    logic [31:0] unused_r4_max_count;

    always_comb begin : prefetch_slot_select
        prefetch_free_found = 1'b0;
        prefetch_free_slot = '0;
        next_stripe_found = 1'b0;
        next_stripe_slot = '0;
        prefetch_ready_count = '0;
        for (integer slot = 0; slot < PREFETCH_SLOTS; slot = slot + 1) begin
            if (prefetch_slot_valid[slot])
                prefetch_ready_count = prefetch_ready_count + 1'b1;
            if (!prefetch_free_found && !prefetch_slot_valid[slot] &&
                !(prefetch_state != PREFETCH_IDLE &&
                  prefetch_fill_slot == PREFETCH_SLOT_WIDTH'(slot))) begin
                prefetch_free_found = 1'b1;
                prefetch_free_slot = PREFETCH_SLOT_WIDTH'(slot);
            end
            if (!next_stripe_found && prefetch_slot_valid[slot] &&
                prefetch_slot_stripe[slot] == stripe_index + 11'd1) begin
                next_stripe_found = 1'b1;
                next_stripe_slot = PREFETCH_SLOT_WIDTH'(slot);
            end
        end
    end

    function automatic [5:0] segment_base_at(input logic [1:0] index);
        segment_base_at = segment_row_base[index*6 +: 6];
    endfunction

    function automatic [5:0] segment_rows_at(input logic [1:0] index);
        segment_rows_at = segment_row_count[index*6 +: 6];
    endfunction

    function automatic [1:0] segment_mode_at(input logic [1:0] index);
        segment_mode_at = segment_mode[index*2 +: 2];
    endfunction

    function automatic [31:0] segment_activation_at(input logic [1:0] index);
        segment_activation_at = segment_activation_base_byte_offset[index*32 +: 32];
    endfunction

    function automatic [4:0] product_group_rows(
        input logic [5:0] remaining_rows,
        input logic [31:0] output_byte_base,
        input logic use_r4
    );
        logic [32:0] paired_product_end;
        begin
            paired_product_end = {1'b0, output_byte_base} +
                33'(2 * FFN_FEATURES * 16);
            if (use_r4)
                product_group_rows = remaining_rows > 6'd8 ? 5'd8 : 5'(remaining_rows);
            else if (remaining_rows > 6'd8 &&
                paired_product_end <= 33'(480 * 1024))
                product_group_rows = remaining_rows > 6'd16 ?
                    5'd16 : 5'(remaining_rows);
            else
                product_group_rows = remaining_rows > 6'd8 ?
                    5'd8 : 5'(remaining_rows);
        end
    endfunction

    function automatic logic product_group_needs_recompute(
        input logic [5:0] remaining_rows,
        input logic [31:0] output_byte_base
    );
        logic [4:0] selected_rows;
        logic [32:0] product_end;
        begin
            selected_rows = product_group_rows(
                remaining_rows, output_byte_base, 1'b0);
            product_end = {1'b0, output_byte_base} +
                (selected_rows > 5'd8 ?
                    33'(2 * FFN_FEATURES * 16) :
                    33'(FFN_FEATURES * 16));
            product_group_needs_recompute =
                product_end > 33'(480 * 1024);
        end
    endfunction

    always_comb begin : fixed_schedule_segment_derive
        integer compact_segment;
        integer next_activation_base_byte_offset;
        integer segment_row_bytes;

        command_segment_count = start_segment_count;
        command_segment_mode = start_segment_mode;
        command_segment_row_base = start_segment_row_base;
        command_segment_row_count = start_segment_row_count;
        command_segment_activation_base_byte_offset = start_segment_activation_base_byte_offset;
        command_row_enable = 48'd0;
        command_mixed_groups = 1'b0;
        for (integer row = 0; row < 48; row = row + 1)
            command_row_enable[row] = row < start_active_rows;

        if (FIXED_SCHEDULE_INGRESS) begin
            command_segment_count = 2'd0;
            command_segment_mode = 6'd0;
            command_segment_row_base = 18'd0;
            command_segment_row_count = 18'd0;
            command_segment_activation_base_byte_offset = 96'd0;
            command_row_enable = fixed_row_config.row_enable;
            command_mixed_groups = fixed_row_config.mixed_group_enable;
            compact_segment = 0;
            next_activation_base_byte_offset = 0;
            segment_row_bytes = 0;
            for (integer source_segment = 0; source_segment < 3;
                 source_segment = source_segment + 1) begin
                if (fixed_row_config.segment_row_count[
                        source_segment*6 +: 6] != 0) begin
                    command_segment_mode[compact_segment*2 +: 2] =
                        fixed_row_config.segment_mode[
                            source_segment*2 +: 2];
                    command_segment_row_base[compact_segment*6 +: 6] =
                        fixed_row_config.segment_row_base[
                            source_segment*6 +: 6];
                    command_segment_row_count[compact_segment*6 +: 6] =
                        fixed_row_config.segment_row_count[
                            source_segment*6 +: 6];
                    command_segment_activation_base_byte_offset[compact_segment*32 +: 32] =
                        32'(next_activation_base_byte_offset);
                    segment_row_bytes = fixed_row_config.segment_mode[
                        source_segment*2 +: 2] == 2'd0 ?
                        FFN_FEATURES / 2 : FFN_FEATURES;
                    next_activation_base_byte_offset = next_activation_base_byte_offset +
                        integer'(fixed_row_config.segment_row_count[
                            source_segment*6 +: 6]) * segment_row_bytes;
                    compact_segment = compact_segment + 1;
                end
            end
            command_segment_count = 2'(compact_segment);
            if (fixed_row_config.mixed_group_enable) begin
                command_segment_count = 2'd1;
                command_segment_mode = 6'd0;
                command_segment_row_base = 18'd0;
                command_segment_row_count = {12'd0, start_active_rows};
                command_segment_activation_base_byte_offset = 96'd0;
            end
        end
    end

    always_comb begin
        stripe_bytes =
            (32'(active_token_count) + 32'(active_token_count[0])) << 4;
        stripe_pair_bytes = stripe_bytes << 1;
        batch_bytes = {24'd0, batch_rows, 4'b0000};
        read_rows = read_group_rows;
        read_bytes = {23'd0, read_rows, 4'b0000};
        first_batch_rows = read_group_rows > 5'd8 ? 4'd8 :
            4'(read_group_rows);
        second_batch_rows = read_group_rows > 5'd8 ?
            4'(read_group_rows - 5'd8) : 4'd0;
        current_mode = segment_mode_at(segment_index);
        current_segment_base = segment_base_at(segment_index);
        current_segment_rows = segment_rows_at(segment_index);
        current_segment_activation_base_byte_offset = segment_activation_at(segment_index);
        current_segment_end = {1'b0, current_segment_base} +
            {1'b0, current_segment_rows};
        next_batch_base = batch_row_base + {2'd0, batch_rows};
        final_batch = {1'b0, next_batch_base} >= current_segment_end;
        final_stripe = stripe_index + 11'd1 == 11'(STRIPE_COUNT);
        final_read_group = {1'b0, read_group_row_base} +
            {2'd0, read_group_rows} >= current_segment_end;
        row_mask = batch_rows == 8 ? 8'hff :
            8'((9'd1 << batch_rows) - 1'b1);
        lane_mask = '0;
        for (integer row = 0; row < 8; row = row + 1)
            lane_mask[row*8 +: 8] = {8{row_mask[row]}};
        tile_tag = TAG_WIDTH'({pass_index, segment_index, batch_row_base[5:0],
                              stripe_index[6:0]});

        current_batch_product_words = 16'(STRIPE_COUNT * 8);
        next_group_byte_base = r4_enable ? 32'd0 :
            current_segment_activation_base_byte_offset +
                32'(((integer'(next_batch_base) -
                      integer'(current_segment_base)) >> 3) *
                    8 * (current_mode == 2'd0 ?
                        FFN_FEATURES/2 : FFN_FEATURES));
        next_group_rows = product_group_rows(
            6'(current_segment_end - {1'b0, next_batch_base}),
            next_group_byte_base, r4_enable);
        if (two_token_r4 && next_group_rows > 5'd2)
            next_group_rows = 5'd2;
        next_group_recompute = product_group_needs_recompute(
            6'(current_segment_end - {1'b0, next_batch_base}),
            next_group_byte_base);
        next_segment_group_byte_base = '0;
        next_segment_group_rows = '0;
        next_segment_group_recompute = 1'b0;
        if (segment_index + 2'd1 < segment_count) begin
            next_segment_group_byte_base = r4_enable ? 32'd0 :
                segment_activation_at(segment_index + 2'd1);
            next_segment_group_rows = product_group_rows(
                segment_rows_at(segment_index + 2'd1),
                next_segment_group_byte_base, r4_enable);
            if (two_token_r4 && next_segment_group_rows > 5'd2)
                next_segment_group_rows = 5'd2;
            next_segment_group_recompute = product_group_needs_recompute(
                segment_rows_at(segment_index + 2'd1),
                next_segment_group_byte_base);
        end

        required_workspace_end = {1'b0, start_workspace_base} +
            (ADDR_WIDTH+1)'((start_precomputed_product ? 1 : 2) * STRIPE_COUNT *
                (integer'(start_active_rows) +
                 integer'(start_active_rows[0])) * 16);

        r4_start_row_count = read_group_rows -
            {3'd0, r4_quartet_index, 2'b00} >= 5'd4 ? 3'd4 :
            3'(read_group_rows - {3'd0, r4_quartet_index, 2'b00});
        r4_start_word_base = 15'(product_group_byte_base >> 4) +
            (r4_quartet_index[1] ? 15'(current_batch_product_words) : 15'd0) +
            (r4_quartet_index[0] ? 15'd4 : 15'd0);
        r4_next_quartet_index = r4_quartet_index + 2'd1;
        r4_prefetch_valid = {3'd0, r4_quartet_index, 2'b00} + 7'd4 <
            {2'd0, read_group_rows};
        r4_prefetch_row_count = read_group_rows -
            {3'd0, r4_next_quartet_index, 2'b00} >= 5'd4 ? 3'd4 :
            3'(read_group_rows -
                {3'd0, r4_next_quartet_index, 2'b00});
        r4_prefetch_word_base = 15'(product_group_byte_base >> 4) +
            (r4_next_quartet_index[1] ?
                15'(current_batch_product_words) : 15'd0) +
            (r4_next_quartet_index[0] ? 15'd4 : 15'd0);

        silu_input_values = tile_values;
        silu_slopes = '0;
        silu_intercepts = '0;
        for (integer lane = 0; lane < 64; lane = lane + 1) begin
            lane_silu_coefficients[lane] =
                bf16_silu_pwl_pkg::coefficients(
                    silu_input_values[lane*16 +: 16]);
            silu_input_values[lane*16 +: 16] =
                bf16_silu_pwl_pkg::input_value(
                    silu_input_values[lane*16 +: 16]);
            silu_slopes[lane*16 +: 16] =
                lane_silu_coefficients[lane][15:0];
            silu_intercepts[lane*16 +: 16] =
                lane_silu_coefficients[lane][31:16];
        end
    end

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE;
    assign busy = state != IDLE && state != ABORT_WAIT_LOW;
    assign main_read_request = state == GATE_REQUEST || state == UP_REQUEST;
    assign prefetch_read_request = !precomputed_product &&
        (prefetch_state == PREFETCH_GATE_REQUEST ||
         prefetch_state == PREFETCH_UP_REQUEST);
    assign selected_read_up = main_read_request ? state == UP_REQUEST :
        prefetch_state == PREFETCH_UP_REQUEST;
    assign read_request_valid = (main_read_request || prefetch_read_request) &&
        !abort_pending && !abort_request && !read_request_error && !writer_error;
    assign read_request_address =
        (main_read_request ? current_gate_address : prefetch_gate_address) +
        (selected_read_up ? ADDR_WIDTH'(stripe_bytes) : ADDR_WIDTH'(0));
    assign read_request_bytes = read_bytes;
    assign read_request_tag = {1'b0, selected_read_up, segment_index,
                               read_group_row_base[3:0]};
    assign candidate_second_span_address = read_request_address +
        ADDR_WIDTH'(stripe_bytes);
    assign pairing_eligible = (read_bytes == 32'd128 ||
        read_bytes == 32'd256) && read_request_address[4:0] == 0 &&
        candidate_second_span_address[4:0] == 0;
    assign read_second_span_valid = read_request_valid &&
        !precomputed_product &&
        !selected_read_up && pairing_eligible;
    assign read_second_span_address = read_second_span_valid ?
        candidate_second_span_address : '0;
    assign read_pair_stride = read_second_span_valid ?
        stripe_pair_bytes : 32'd0;
    assign read_pair_count = read_second_span_valid ?
        11'(STRIPE_COUNT) - stripe_index : 11'd1;
    assign read_data_ready = ((state == GATE_STREAM || state == UP_STREAM) &&
        !abort_pending && !abort_request) ||
        ((prefetch_state == PREFETCH_GATE_STREAM ||
          prefetch_state == PREFETCH_UP_STREAM) &&
        !abort_pending && !abort_request) ||
        (state == ABORT_DRAIN && request_active);

    assign base_arithmetic_req_valid = (state == SILU_REQUEST ||
        state == PRODUCT_REQUEST) && !abort_pending && !abort_request;
    assign base_arithmetic_req_operation = state == SILU_REQUEST ? BF16_SILU :
        BF16_MULTIPLY;
    assign base_arithmetic_req_values = state == SILU_REQUEST ? silu_input_values :
        tile_values;
    assign base_arithmetic_req_paired_values = '0;
    assign base_arithmetic_req_factor0_values = state == SILU_REQUEST ?
        silu_slopes : up_values;
    assign base_arithmetic_req_factor1_values = state == SILU_REQUEST ?
        silu_intercepts : '0;
    assign base_arithmetic_req_lane_mask = lane_mask;
    assign base_arithmetic_req_tag = tile_tag;
    assign base_arithmetic_rsp_ready = state == SILU_RESPONSE ||
        state == PRODUCT_RESPONSE || state == ABORT_DRAIN;

    assign arithmetic_req_valid = r4_active ? r4_arithmetic_req_valid :
        base_arithmetic_req_valid;
    assign r4_arithmetic_req_ready = r4_active && arithmetic_req_ready;
    assign arithmetic_req_operation = r4_active ? r4_arithmetic_req_operation :
        base_arithmetic_req_operation;
    assign arithmetic_req_values = r4_active ? r4_arithmetic_req_values :
        base_arithmetic_req_values;
    assign arithmetic_req_paired_values = r4_active ?
        r4_arithmetic_req_paired_values : base_arithmetic_req_paired_values;
    assign arithmetic_req_factor0_values = r4_active ?
        r4_arithmetic_req_factor0_values : base_arithmetic_req_factor0_values;
    assign arithmetic_req_factor1_values = r4_active ?
        r4_arithmetic_req_factor1_values : base_arithmetic_req_factor1_values;
    assign arithmetic_req_lane_mask = r4_active ? r4_arithmetic_req_lane_mask :
        base_arithmetic_req_lane_mask;
    assign arithmetic_req_tag = r4_active ? r4_arithmetic_req_tag :
        base_arithmetic_req_tag;
    assign arithmetic_rsp_ready = r4_active ? r4_arithmetic_rsp_ready :
        base_arithmetic_rsp_ready;

    assign base_max_req_valid = state == MAX_REQUEST && !abort_pending &&
        !abort_request;
    assign base_max_req_values = tile_values;
    assign base_max_req_lane_mask = lane_mask;
    assign base_max_req_tag = tile_tag;
    assign base_max_rsp_ready = state == MAX_RESPONSE || state == ABORT_DRAIN;
    assign max_req_valid = r4_active ? r4_max_req_valid : base_max_req_valid;
    assign r4_max_req_ready = r4_active && max_req_ready;
    assign max_req_values = r4_active ? r4_max_req_values : base_max_req_values;
    assign max_req_lane_mask = r4_active ? r4_max_req_lane_mask :
        base_max_req_lane_mask;
    assign max_req_tag = r4_active ? r4_max_req_tag : base_max_req_tag;
    assign max_rsp_ready = r4_active ? r4_max_rsp_ready : base_max_rsp_ready;

    assign writer_cfg_valid = state == WRITER_CONFIG && !abort_pending &&
        !abort_request;
    assign writer_cfg_mode = mixed_groups ? 2'd0 : current_mode;
    assign writer_cfg_physical_row_base = batch_row_base;
    assign writer_cfg_row_count = {2'd0, batch_rows};
    assign writer_cfg_elements_per_row = 16'(FFN_FEATURES);
    assign writer_cfg_activation_base_byte_offset = mixed_groups ? activation_base_byte_offset :
        current_segment_activation_base_byte_offset +
        32'(((integer'(batch_row_base) - integer'(current_segment_base)) >> 3) *
            8 * (current_mode == 2'd0 ? FFN_FEATURES/2 : FFN_FEATURES));
    assign writer_cfg_activation_limit_byte_offset = activation_limit_byte_offset;
    assign writer_cfg_mixed_group_enable = mixed_groups;
    assign writer_cfg_compute_group_count = fixed_row_config.compute_group_count;
    assign writer_cfg_row_precision_a8 = fixed_row_config.row_precision_a8;
    assign writer_cfg_row_compute_group = fixed_row_config.row_compute_group;
    assign writer_cfg_row_pe_slot = fixed_row_config.row_pe_slot;
    assign writer_cfg_row_phase_mask = fixed_row_config.row_phase_mask;

    assign quant_scale_req_valid = state == SCALE_REQUEST && !abort_pending &&
        !abort_request;
    always_comb begin
        quant_scale_req_a4_row_mask = '0;
        for (integer row = 0; row < 8; row++) begin
            if (row < batch_rows)
                quant_scale_req_a4_row_mask[row] = mixed_groups ?
                    !fixed_row_config.row_precision_a8[batch_row_base + row] :
                    current_mode == 2'd0;
        end
    end
    assign quant_scale_req_row_max_abs =
        read_group_rows > 5'd8 && batch_row_base != read_group_row_base ?
        second_row_max_abs : row_max_abs;
    assign quant_scale_req_row_mask = row_mask;
    assign quant_scale_rsp_ready = state == SCALE_RESPONSE ||
        state == ABORT_DRAIN;
    assign writer_scale_valid = state == SCALE_WRITE && !abort_pending &&
        !abort_request;
    assign writer_scale_row_base = batch_row_base;
    assign writer_scale_row_mask = row_mask;
    assign writer_scale_values_bf16 = batch_scales;

    assign quant_values_req_valid =
        (state == QUANTIZED_REQUEST ||
         (state == PRODUCT_QUANT_STREAM &&
          product_stream_fifo_output_valid && product_stream_trace_sent)) &&
        !abort_pending && !abort_request;
    assign quant_values_req_values_bf16 = state == PRODUCT_QUANT_STREAM ?
        product_stream_fifo_values : tile_values;
    assign quant_values_req_lane_mask = lane_mask;
    assign quant_values_req_tag = state == PRODUCT_QUANT_STREAM ?
        product_stream_fifo_tag : tile_tag;
    assign quant_values_rsp_ready = state == PRODUCT_QUANT_STREAM ?
        writer_quantized_ready && !abort_pending && !abort_request :
        state == QUANTIZED_RESPONSE || state == ABORT_DRAIN;
    assign writer_quantized_valid = state == PRODUCT_QUANT_STREAM ?
        quant_values_rsp_valid && !abort_pending && !abort_request :
        state == QUANTIZED_WRITE && !abort_pending && !abort_request;
    assign writer_quantized_row_base = batch_row_base;
    assign writer_quantized_element_base = state == PRODUCT_QUANT_STREAM ?
        {2'b00, product_stream_quant_response_stripe, 3'b000} :
        {2'b00, stripe_index, 3'b000};
    assign writer_quantized_values = state == PRODUCT_QUANT_STREAM ?
        quant_values_rsp_values : quantized_values;
    assign writer_quantized_lane_mask = state == PRODUCT_QUANT_STREAM ?
        quant_values_rsp_lane_mask : quantized_lane_mask;
    assign writer_quantized_tag = state == PRODUCT_QUANT_STREAM ?
        quant_values_rsp_tag : quantized_tag;
    assign writer_abort_request = abort_pending || abort_request || writer_error;

    assign base_product_memory_req_valid =
        (state == PRODUCT_STORE_LOW || state == PRODUCT_STORE_HIGH ||
         state == PRODUCT_LOAD_LOW || state == PRODUCT_LOAD_HIGH ||
         (state == PRODUCT_QUANT_STREAM &&
          product_stream_request_stripe < 11'(STRIPE_COUNT) &&
          (product_stream_request_half ||
           {1'b0, product_stream_fifo_occupancy} +
               {1'b0, product_stream_reserved_tiles} < 3'd2))) &&
        !abort_pending && !abort_request;
    assign base_product_memory_req.write = state == PRODUCT_STORE_LOW ||
        state == PRODUCT_STORE_HIGH;
    assign base_product_memory_req.word_index =
        15'(product_group_byte_base >> 4) +
        (batch_row_base != read_group_row_base ?
            15'(current_batch_product_words) : 15'd0) +
        (state == PRODUCT_QUANT_STREAM ?
            {product_stream_request_stripe, 3'b000} :
            {stripe_index, 3'b000}) +
        ((state == PRODUCT_QUANT_STREAM ? product_stream_request_half :
          state == PRODUCT_STORE_HIGH || state == PRODUCT_LOAD_HIGH) ?
            15'd4 : 15'd0);
    assign base_product_memory_req.data =
        (state == PRODUCT_STORE_HIGH || state == PRODUCT_LOAD_HIGH) ?
            tile_values[1023:512] : tile_values[511:0];
    assign base_product_memory_req.half = state == PRODUCT_QUANT_STREAM ?
        product_stream_request_half :
        state == PRODUCT_STORE_HIGH || state == PRODUCT_LOAD_HIGH;
    assign base_product_memory_req.tag = state == PRODUCT_QUANT_STREAM ?
        TAG_WIDTH'({1'b1, segment_index, batch_row_base[5:0],
                    product_stream_request_stripe[6:0]}) : tile_tag;
    assign base_product_memory_rsp_ready = state == PRODUCT_QUANT_STREAM ?
        ((!product_memory_rsp.half && !two_token_r4) || product_stream_fifo_input_ready) :
        state == PRODUCT_LOAD_RESPONSE || state == ABORT_DRAIN;
    assign product_memory_req_valid = r4_active ? r4_product_req_valid :
        base_product_memory_req_valid;
    assign r4_product_req_ready = r4_active && product_memory_req_ready;
    assign product_memory_req = r4_active ? r4_product_req :
        base_product_memory_req;
    assign product_memory_rsp_ready = r4_active ? r4_product_rsp_ready :
        base_product_memory_rsp_ready;

    assign down_prefetch_ready = read_phase_complete &&
        !two_token_r4 && (!r4_enable || r4_panel_released);
    assign trace_sample_valid =
        (state == TRACE_SAMPLE ||
         (state == PRODUCT_QUANT_STREAM &&
          product_stream_fifo_output_valid && !product_stream_trace_sent)) &&
        !abort_pending && !abort_request;
    assign trace_sample_pass = state == PRODUCT_QUANT_STREAM ?
        1'b1 : pass_index;
    assign trace_sample_stripe = state == PRODUCT_QUANT_STREAM ?
        product_stream_fifo_stripe : stripe_index;
    assign trace_sample_row_base = batch_row_base;
    assign trace_sample_lane_mask = lane_mask;
`ifdef SYNTHESIS
    assign trace_sample_silu = '0;
    assign trace_sample_product = '0;
`else
    assign trace_sample_silu = trace_silu_values;
    assign trace_sample_product = state == PRODUCT_QUANT_STREAM ?
        product_stream_fifo_values : tile_values;
`endif
    assign completed_layout.segment_count = mixed_groups ? 2'd0 : segment_count;
    assign completed_layout.segment_mode = mixed_groups ? 6'd0 : segment_mode;
    assign completed_layout.segment_row_base = mixed_groups ? 18'd0 :
        segment_row_base;
    assign completed_layout.segment_row_count = mixed_groups ? 18'd0 :
        segment_row_count;
    assign completed_layout.segment_activation_base_byte_offset = mixed_groups ? 96'd0 :
        segment_activation_base_byte_offset;
    assign completed_layout.row_enable = row_enable;

    assign product_stream_fifo_input_valid =
        state == PRODUCT_QUANT_STREAM && product_memory_rsp_valid &&
        (product_memory_rsp.half || two_token_r4) && !abort_pending && !abort_request;
    assign product_stream_fifo_input_data = {
        two_token_r4 ? {512'd0, product_memory_rsp.data} :
            {product_memory_rsp.data, product_stream_low_data},
        product_memory_rsp.tag, product_stream_response_stripe};
    assign {product_stream_fifo_values, product_stream_fifo_tag,
            product_stream_fifo_stripe} = product_stream_fifo_output_data;
    assign product_stream_fifo_output_ready =
        state == PRODUCT_QUANT_STREAM && product_stream_fifo_output_valid &&
        product_stream_trace_sent && quant_values_req_ready &&
        !abort_pending && !abort_request;
    assign product_stream_request_fire =
        state == PRODUCT_QUANT_STREAM && product_memory_req_valid &&
        product_memory_req_ready;
    assign product_stream_response_fire =
        state == PRODUCT_QUANT_STREAM && product_memory_rsp_valid &&
        product_memory_rsp_ready;
    assign product_stream_trace_fire =
        state == PRODUCT_QUANT_STREAM && trace_sample_valid &&
        trace_sample_ready;
    assign product_stream_quant_request_fire =
        state == PRODUCT_QUANT_STREAM && quant_values_req_valid &&
        quant_values_req_ready;
    assign product_stream_quant_response_fire =
        state == PRODUCT_QUANT_STREAM && quant_values_rsp_valid &&
        quant_values_rsp_ready;

    assign product_stream_fifo_input_ready = !product_stream_entry_valid ||
        product_stream_fifo_output_ready;
    assign product_stream_fifo_output_valid = product_stream_entry_valid;
    assign product_stream_fifo_output_data = product_stream_entry_data;
    assign product_stream_fifo_occupancy = {1'b0,
        product_stream_entry_valid};

    always_ff @(posedge clk) begin
        if (rst || abort_request || state != PRODUCT_QUANT_STREAM) begin
            product_stream_entry_valid <= 1'b0;
        end else if (product_stream_fifo_input_ready) begin
            product_stream_entry_valid <= product_stream_fifo_input_valid;
            if (product_stream_fifo_input_valid)
                product_stream_entry_data <= product_stream_fifo_input_data;
        end
    end

    ffn_r4_controller #(.TAG_WIDTH(TAG_WIDTH)) r4_controller (
        .clk, .rst,
        .abort_request(abort_pending || abort_request || writer_error),
        .abort_ack(r4_abort_ack), .arithmetic_abort_ack,
        .start_valid(state == R4_START), .start_ready(r4_start_ready),
        .start_row_count(r4_start_row_count),
        .start_word_base(r4_start_word_base),
        .prefetch_valid(r4_prefetch_valid),
        .prefetch_row_count(r4_prefetch_row_count),
        .prefetch_word_base(r4_prefetch_word_base),
        .done_valid(r4_done_valid), .done_ready(1'b1),
        .error(r4_done_error), .error_id(r4_done_error_id),
        .row_max_abs(r4_row_max_abs),
        .product_req_valid(r4_product_req_valid),
        .product_req_ready(r4_product_req_ready),
        .product_req(r4_product_req),
        .product_rsp_valid(product_memory_rsp_valid && r4_active),
        .product_rsp_ready(r4_product_rsp_ready),
        .product_rsp(product_memory_rsp),
        .arithmetic_req_valid(r4_arithmetic_req_valid),
        .arithmetic_req_ready(r4_arithmetic_req_ready),
        .arithmetic_req_operation(r4_arithmetic_req_operation),
        .arithmetic_req_values(r4_arithmetic_req_values),
        .arithmetic_req_paired_values(r4_arithmetic_req_paired_values),
        .arithmetic_req_factor0_values(r4_arithmetic_req_factor0_values),
        .arithmetic_req_factor1_values(r4_arithmetic_req_factor1_values),
        .arithmetic_req_lane_mask(r4_arithmetic_req_lane_mask),
        .arithmetic_req_tag(r4_arithmetic_req_tag),
        .arithmetic_rsp_valid(arithmetic_rsp_valid && r4_active),
        .arithmetic_rsp_ready(r4_arithmetic_rsp_ready),
        .arithmetic_rsp_values, .arithmetic_rsp_lane_mask, .arithmetic_rsp_tag,
        .max_req_valid(r4_max_req_valid), .max_req_ready(r4_max_req_ready),
        .max_req_values(r4_max_req_values),
        .max_req_lane_mask(r4_max_req_lane_mask), .max_req_tag(r4_max_req_tag),
        .max_rsp_valid(max_rsp_valid && r4_active),
        .max_rsp_ready(r4_max_rsp_ready), .max_rsp_values, .max_rsp_row_mask,
        .max_rsp_tag,
        .accepted_product_read_count(unused_r4_product_read_count),
        .accepted_product_write_count(unused_r4_product_write_count),
        .accepted_add_count(unused_r4_add_count),
        .accepted_multiply_count(unused_r4_multiply_count),
        .accepted_max_count(unused_r4_max_count)
    );

`ifdef SYNTHESIS
    always_comb begin
        accepted_read_request_count = '0;
        accepted_read_byte_count = '0;
        accepted_arithmetic_request_count = '0;
        completed_product_tile_count = '0;
        accepted_max_request_count = '0;
        accepted_quant_scale_count = '0;
        accepted_quantized_value_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            prefetch_state <= PREFETCH_IDLE;
            abort_ack <= 1'b0;
            active_token_count <= '0;
            segment_count <= '0;
            segment_mode <= '0;
            segment_row_base <= '0;
            segment_row_count <= '0;
            segment_activation_base_byte_offset <= '0;
            row_enable <= '0;
            mixed_groups <= 1'b0;
            r4_enable <= 1'b0;
            two_token_r4 <= 1'b0;
            activation_base_byte_offset <= '0;
            precomputed_product <= 1'b0;
            r4_panel_released <= 1'b0;
            r4_active <= 1'b0;
            r4_started <= 1'b0;
            r4_quartet_index <= '0;
            activation_limit_byte_offset <= '0;
            workspace_base <= '0;
            batch_gate_address <= '0;
            current_gate_address <= '0;
            prefetch_gate_address <= '0;
            segment_index <= '0;
            batch_row_base <= '0;
            batch_rows <= '0;
            read_group_row_base <= '0;
            read_group_rows <= '0;
            compute_second_batch <= 1'b0;
            pass_index <= 1'b0;
            stripe_index <= '0;
            stream_row <= '0;
            request_active <= 1'b0;
            request_terminal_seen <= 1'b0;
            request_terminal_error <= 1'b0;
            accepted_read_tag <= '0;
            accepted_read_paired <= 1'b0;
            accepted_read_pair_count <= 11'd1;
            repeated_pair_active <= 1'b0;
            arithmetic_outstanding <= 1'b0;
            max_outstanding <= 1'b0;
            quant_scale_outstanding <= 1'b0;
            quant_outstanding <= 1'b0;
            product_read_outstanding <= '0;
            product_low_received <= 1'b0;
            abort_pending <= 1'b0;
            tile_values <= '0;
            up_values <= '0;
            second_gate_values <= '0;
            second_up_values <= '0;
            prefetch_slot_valid <= '0;
            for (integer slot = 0; slot < PREFETCH_SLOTS; slot = slot + 1) begin
                prefetch_gate_values[slot] <= '0;
                prefetch_up_values[slot] <= '0;
                prefetch_second_gate_values[slot] <= '0;
                prefetch_second_up_values[slot] <= '0;
                prefetch_slot_stripe[slot] <= '0;
                prefetch_slot_address[slot] <= '0;
            end
            prefetch_fill_slot <= '0;
            prefetch_stripe_index <= '0;
            prefetch_stream_row <= '0;
            next_prefetch_stripe <= '0;
            next_prefetch_address <= '0;
            prefetch_window_active <= 1'b0;
`ifndef SYNTHESIS
            trace_silu_values <= '0;
`endif
            row_max_abs <= '0;
            second_row_max_abs <= '0;
            batch_scales <= '0;
            quantized_values <= '0;
            quantized_lane_mask <= '0;
            quantized_tag <= '0;
            product_stream_request_half <= 1'b0;
            product_stream_request_stripe <= '0;
            product_stream_response_stripe <= '0;
            product_stream_quant_response_stripe <= '0;
            product_stream_low_data <= '0;
            product_stream_low_tag <= '0;
            product_stream_trace_sent <= 1'b0;
            product_stream_reserved_tiles <= '0;
            writer_done_seen <= 1'b0;
            product_group_byte_base <= '0;
            product_group_recompute <= 1'b0;
            read_phase_complete <= 1'b0;
            error <= 1'b0;
            error_id <= '0;
`ifndef SYNTHESIS
            accepted_read_request_count <= '0;
            accepted_read_byte_count <= '0;
            accepted_arithmetic_request_count <= '0;
            completed_product_tile_count <= '0;
            accepted_max_request_count <= '0;
            accepted_quant_scale_count <= '0;
            accepted_quantized_value_count <= '0;
`endif
        end else begin
            abort_ack <= 1'b0;

            // Accepted returns still retire on the edge that enters error drain.
            // R4 accounts for its own arithmetic, max and product transactions.
            if (!r4_active) begin
                if (arithmetic_req_valid && arithmetic_req_ready)
                    arithmetic_outstanding <= 1'b1;
                if ((arithmetic_rsp_valid && arithmetic_rsp_ready) ||
                    (state == ABORT_DRAIN && arithmetic_abort_ack))
                    arithmetic_outstanding <= 1'b0;
                if (max_req_valid && max_req_ready)
                    max_outstanding <= 1'b1;
                if (max_rsp_valid && max_rsp_ready)
                    max_outstanding <= 1'b0;
                case ({product_memory_req_valid && product_memory_req_ready &&
                           !product_memory_req.write,
                       product_memory_rsp_valid && product_memory_rsp_ready})
                    2'b10: product_read_outstanding <= product_read_outstanding + 2'd1;
                    2'b01: product_read_outstanding <= product_read_outstanding - 2'd1;
                    default: begin end
                endcase
            end
            if (quant_scale_req_valid && quant_scale_req_ready)
                quant_scale_outstanding <= 1'b1;
            if (quant_scale_rsp_valid && quant_scale_rsp_ready)
                quant_scale_outstanding <= 1'b0;
            case ({quant_values_req_valid && quant_values_req_ready,
                   quant_values_rsp_valid && quant_values_rsp_ready})
                2'b10: quant_outstanding <= quant_outstanding + 5'd1;
                2'b01: quant_outstanding <= quant_outstanding - 5'd1;
                default: begin end
            endcase

            if (writer_done_pulse)
                writer_done_seen <= 1'b1;
            if (writer_error) begin
                error <= 1'b1;
                error_id <= ERROR_WRITER;
            end
            if (read_request_done || read_request_error) begin
                request_active <= 1'b0;
                request_terminal_seen <= 1'b1;
                if (read_request_error) begin
                    request_terminal_error <= 1'b1;
                    error <= 1'b1;
                    error_id <= ERROR_DMA;
                end
            end

            if (abort_request && state != IDLE && state != COMPLETE &&
                state != ABORT_DRAIN &&
                state != ABORT_WAIT_LOW) begin
                abort_pending <= 1'b1;
                prefetch_state <= PREFETCH_IDLE;
                prefetch_slot_valid <= '0;
                prefetch_window_active <= 1'b0;
                state <= ABORT_DRAIN;
            end else if (read_request_error && state != IDLE &&
                         state != COMPLETE && state != ABORT_DRAIN &&
                         state != ABORT_WAIT_LOW) begin
                prefetch_state <= PREFETCH_IDLE;
                prefetch_slot_valid <= '0;
                prefetch_window_active <= 1'b0;
                state <= ABORT_DRAIN;
            end else if (writer_error && state != IDLE && state != COMPLETE &&
                         state != ABORT_DRAIN && state != ABORT_WAIT_LOW) begin
                prefetch_state <= PREFETCH_IDLE;
                prefetch_slot_valid <= '0;
                prefetch_window_active <= 1'b0;
                state <= ABORT_DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    if (start_active_rows == 0 ||
                        (start_two_token_r4 && (!start_r4_enable ||
                         !command_mixed_groups || fixed_row_config.row_precision_a8 != 0)) ||
                        (start_activation_base_byte_offset != 0 && !start_two_token_r4) ||
                        start_activation_base_byte_offset[3:0] != 0 ||
                        start_activation_base_byte_offset >= start_activation_limit_byte_offset ||
                        start_active_rows > 6'(MAX_ROWS) ||
                        command_segment_count == 0 ||
                        start_workspace_base[7:0] != 0 ||
                        start_workspace_limit <= start_workspace_base ||
                        required_workspace_end[ADDR_WIDTH] ||
                        required_workspace_end > {1'b0, start_workspace_limit} ||
                        command_segment_activation_base_byte_offset[3:0] != 0 ||
                        product_group_rows(
                            command_segment_row_count[5:0],
                            start_r4_enable ? 32'd0 :
                                command_segment_activation_base_byte_offset[31:0], start_r4_enable) == 0) begin
                        error <= 1'b1;
                        error_id <= ERROR_CONFIGURATION;
                        state <= COMPLETE;
                    end else begin
                        active_token_count <= start_active_rows;
                        segment_count <= command_segment_count;
                        segment_mode <= command_segment_mode;
                        segment_row_base <= command_segment_row_base;
                        segment_row_count <= command_segment_row_count;
                        segment_activation_base_byte_offset <= command_segment_activation_base_byte_offset;
                        row_enable <= command_row_enable;
                        mixed_groups <= command_mixed_groups;
                        r4_enable <= start_r4_enable;
                        two_token_r4 <= start_two_token_r4;
                        precomputed_product <= start_precomputed_product;
                        r4_panel_released <= 1'b0;
                        r4_active <= 1'b0;
                        r4_started <= 1'b0;
                        r4_quartet_index <= '0;
                        activation_limit_byte_offset <= start_activation_limit_byte_offset;
                        activation_base_byte_offset <= start_activation_base_byte_offset;
                        workspace_base <= start_workspace_base;
                        batch_gate_address <= start_workspace_base +
                            ADDR_WIDTH'({command_segment_row_base[5:0], 4'b0000});
                        current_gate_address <= start_workspace_base +
                            ADDR_WIDTH'({command_segment_row_base[5:0], 4'b0000});
                        segment_index <= 2'd0;
                        batch_row_base <= command_segment_row_base[5:0];
                        read_group_row_base <= command_segment_row_base[5:0];
                        read_group_rows <= start_two_token_r4 &&
                            command_segment_row_count[5:0] > 6'd2 ? 5'd2 : product_group_rows(
                            command_segment_row_count[5:0],
                            start_r4_enable ? 32'd0 :
                                command_segment_activation_base_byte_offset[31:0], start_r4_enable);
                        batch_rows <= start_two_token_r4 &&
                            command_segment_row_count[5:0] > 6'd2 ? 4'd2 :
                            command_segment_row_count[5:0] > 6'd8 ?
                            4'd8 : command_segment_row_count[3:0];
                        compute_second_batch <= 1'b0;
                        pass_index <= 1'b0;
                        product_group_byte_base <=
                            start_r4_enable ? 32'd0 :
                                command_segment_activation_base_byte_offset[31:0];
                        product_group_recompute <=
                            start_r4_enable ? 1'b0 :
                                product_group_needs_recompute(
                                command_segment_row_count[5:0],
                                command_segment_activation_base_byte_offset[31:0]);
                        stripe_index <= '0;
                        row_max_abs <= '0;
                        second_row_max_abs <= '0;
                        prefetch_state <= PREFETCH_IDLE;
                        prefetch_slot_valid <= '0;
                        prefetch_window_active <= 1'b0;
                        read_phase_complete <= 1'b0;
                        abort_pending <= 1'b0;
                        error <= 1'b0;
                        error_id <= '0;
                        state <= GATE_REQUEST;
                    end
                end

                GATE_REQUEST: if (read_request_valid && read_request_ready) begin
                    request_active <= 1'b1;
                    request_terminal_seen <= 1'b0;
                    request_terminal_error <= 1'b0;
                    accepted_read_tag <= read_request_tag;
                    accepted_read_paired <= read_second_span_valid;
                    accepted_read_pair_count <= read_pair_count;
                    repeated_pair_active <= read_pair_count > 11'd1;
                    stream_row <= '0;
                    tile_values <= '0;
                    up_values <= '0;
                    second_gate_values <= '0;
                    second_up_values <= '0;
`ifndef SYNTHESIS
                    accepted_read_request_count <=
                        accepted_read_request_count + 64'd1;
`endif
                    state <= GATE_STREAM;
                end
                GATE_STREAM: if (read_data_valid && read_data_ready) begin
                    if (stream_row < 5'd8)
                        tile_values[stream_row*128 +: 128] <= read_data;
                    else
                        second_gate_values[(stream_row-5'd8)*128 +: 128] <=
                            read_data;
`ifndef SYNTHESIS
                    accepted_read_byte_count <= accepted_read_byte_count + 64'd16;
`endif
                    if (read_data_tag != accepted_read_tag ||
                        read_data_span ||
                        read_byte_enable != 16'hffff ||
                        read_data_last != (!accepted_read_paired &&
                            stream_row + 5'd1 == read_rows)) begin
                        error <= 1'b1;
                        error_id <= ERROR_DMA;
                    end
                    if (stream_row + 5'd1 == read_rows &&
                        accepted_read_paired) begin
                        stream_row <= '0;
                        state <= UP_STREAM;
                    end else if (stream_row + 5'd1 == read_rows)
                        state <= GATE_TERMINAL;
                    else
                        stream_row <= stream_row + 5'd1;
                end
                GATE_TERMINAL: if (request_terminal_seen ||
                    read_request_done || read_request_error) begin
                    if (request_terminal_error || read_request_error)
                        state <= ABORT_DRAIN;
                    else if (precomputed_product) begin
`ifndef SYNTHESIS
                        completed_product_tile_count <=
                            completed_product_tile_count + 64'd1;
`endif
                        if (final_stripe && final_read_group &&
                            segment_index + 2'd1 >= segment_count)
                            read_phase_complete <= 1'b1;
                        state <= PRODUCT_STORE_LOW;
                    end else
                        state <= UP_REQUEST;
                end
                UP_REQUEST: if (read_request_valid && read_request_ready) begin
                    request_active <= 1'b1;
                    request_terminal_seen <= 1'b0;
                    request_terminal_error <= 1'b0;
                    accepted_read_tag <= read_request_tag;
                    accepted_read_paired <= 1'b0;
                    stream_row <= '0;
                    up_values <= '0;
                    second_up_values <= '0;
`ifndef SYNTHESIS
                    accepted_read_request_count <=
                        accepted_read_request_count + 64'd1;
`endif
                    state <= UP_STREAM;
                end
                UP_STREAM: if (read_data_valid && read_data_ready) begin
                    if (stream_row < 5'd8)
                        up_values[stream_row*128 +: 128] <= read_data;
                    else
                        second_up_values[(stream_row-5'd8)*128 +: 128] <=
                            read_data;
`ifndef SYNTHESIS
                    accepted_read_byte_count <= accepted_read_byte_count + 64'd16;
`endif
                    if (read_data_tag != accepted_read_tag ||
                        read_data_span != accepted_read_paired ||
                        read_byte_enable != 16'hffff ||
                        read_data_last !=
                            (stream_row + 5'd1 == read_rows &&
                             (!repeated_pair_active ||
                              accepted_read_pair_count == 11'd1))) begin
                        error <= 1'b1;
                        error_id <= ERROR_DMA;
                    end
                    if (stream_row + 5'd1 == read_rows) begin
                        if (repeated_pair_active) begin
                            prefetch_fill_slot <= '0;
                            prefetch_stripe_index <= stripe_index + 11'd1;
                            prefetch_gate_address <= current_gate_address +
                                ADDR_WIDTH'(stripe_pair_bytes);
                            prefetch_stream_row <= '0;
                            prefetch_gate_values[0] <= '0;
                            prefetch_up_values[0] <= '0;
                            prefetch_second_gate_values[0] <= '0;
                            prefetch_second_up_values[0] <= '0;
                            prefetch_slot_valid[0] <= 1'b0;
                            prefetch_state <= PREFETCH_GATE_STREAM;
                            compute_second_batch <= 1'b0;
                            state <= SILU_REQUEST;
                        end else begin
                            state <= UP_TERMINAL;
                        end
                    end else begin
                        stream_row <= stream_row + 5'd1;
                    end
                end
                UP_TERMINAL: if (request_terminal_seen ||
                    read_request_done || read_request_error) begin
                    if (request_terminal_error || read_request_error)
                        state <= ABORT_DRAIN;
                    else begin
                        if ((!product_group_recompute || pass_index) &&
                            final_stripe && final_read_group &&
                            segment_index + 2'd1 >= segment_count)
                            read_phase_complete <= 1'b1;
                        if (!final_stripe) begin
                            prefetch_fill_slot <= '0;
                            prefetch_stripe_index <= stripe_index + 11'd1;
                            prefetch_gate_address <= current_gate_address +
                                ADDR_WIDTH'(stripe_pair_bytes);
                            prefetch_gate_values[0] <= '0;
                            prefetch_up_values[0] <= '0;
                            prefetch_second_gate_values[0] <= '0;
                            prefetch_second_up_values[0] <= '0;
                            prefetch_slot_valid[0] <= 1'b0;
                            next_prefetch_stripe <= stripe_index + 11'd2;
                            next_prefetch_address <= current_gate_address +
                                ADDR_WIDTH'(stripe_pair_bytes) +
                                ADDR_WIDTH'(stripe_pair_bytes);
                            prefetch_window_active <= 1'b1;
                            prefetch_state <= PREFETCH_GATE_REQUEST;
                        end else begin
                            prefetch_window_active <= 1'b0;
                        end
                        compute_second_batch <= 1'b0;
                        state <= SILU_REQUEST;
                    end
                end
                SILU_REQUEST: if (arithmetic_req_valid && arithmetic_req_ready) begin
`ifndef SYNTHESIS
                    accepted_arithmetic_request_count <=
                        accepted_arithmetic_request_count + 64'd1;
`endif
                    state <= SILU_RESPONSE;
                end
                SILU_RESPONSE: if (arithmetic_rsp_valid && arithmetic_rsp_ready) begin
                    if (arithmetic_rsp_tag != tile_tag ||
                        arithmetic_rsp_lane_mask != lane_mask) begin
                        error <= 1'b1;
                        error_id <= ERROR_ARITHMETIC;
                    end
                    tile_values <= arithmetic_rsp_values;
`ifndef SYNTHESIS
                    trace_silu_values <= arithmetic_rsp_values;
`endif
                    state <= PRODUCT_REQUEST;
                end
                PRODUCT_REQUEST: if (arithmetic_req_valid && arithmetic_req_ready) begin
`ifndef SYNTHESIS
                    accepted_arithmetic_request_count <=
                        accepted_arithmetic_request_count + 64'd1;
`endif
                    state <= PRODUCT_RESPONSE;
                end
                PRODUCT_RESPONSE: if (arithmetic_rsp_valid && arithmetic_rsp_ready) begin
                    if (arithmetic_rsp_tag != tile_tag ||
                        arithmetic_rsp_lane_mask != lane_mask) begin
                        error <= 1'b1;
                        error_id <= ERROR_ARITHMETIC;
                    end
                    tile_values <= arithmetic_rsp_values;
`ifndef SYNTHESIS
                    completed_product_tile_count <=
                        completed_product_tile_count + 64'd1;
`endif
                    if (product_group_recompute)
                        state <= pass_index ? QUANTIZED_REQUEST : MAX_REQUEST;
                    else
                        state <= PRODUCT_STORE_LOW;
                end
                PRODUCT_STORE_LOW: if (product_memory_req_valid &&
                    product_memory_req_ready) begin
                    state <= two_token_r4 ? TRACE_SAMPLE : PRODUCT_STORE_HIGH;
                end
                PRODUCT_STORE_HIGH: if (product_memory_req_valid &&
                    product_memory_req_ready) begin
                    state <= r4_enable ? TRACE_SAMPLE : MAX_REQUEST;
                end
                MAX_REQUEST: if (max_req_valid && max_req_ready) begin
`ifndef SYNTHESIS
                    accepted_max_request_count <=
                        accepted_max_request_count + 64'd1;
`endif
                    state <= MAX_RESPONSE;
                end
                MAX_RESPONSE: if (max_rsp_valid && max_rsp_ready) begin
                    if (max_rsp_tag != tile_tag || max_rsp_row_mask != row_mask) begin
                        error <= 1'b1;
                        error_id <= ERROR_MAX;
                    end
                    for (integer row = 0; row < 8; row = row + 1) begin
                        if (compute_second_batch) begin
                            if (row_mask[row] &&
                                max_rsp_values[row*16 +: 15] >
                                    second_row_max_abs[row*16 +: 15])
                                second_row_max_abs[row*16 +: 16] <=
                                    {1'b0, max_rsp_values[row*16 +: 15]};
                        end else if (row_mask[row] &&
                            max_rsp_values[row*16 +: 15] >
                                row_max_abs[row*16 +: 15]) begin
                            row_max_abs[row*16 +: 16] <=
                                {1'b0, max_rsp_values[row*16 +: 15]};
                        end
                    end
                    state <= TRACE_SAMPLE;
                end
                TRACE_SAMPLE: if (trace_sample_valid && trace_sample_ready) begin
                    if (!pass_index && !compute_second_batch &&
                        read_group_rows > 5'd8) begin
                        compute_second_batch <= 1'b1;
                        batch_row_base <= read_group_row_base + 6'd8;
                        batch_rows <= second_batch_rows;
                        tile_values <= second_gate_values;
                        up_values <= second_up_values;
                        state <= precomputed_product ? PRODUCT_STORE_LOW :
                            SILU_REQUEST;
                    end else if (final_stripe) begin
                        stripe_index <= '0;
                        compute_second_batch <= 1'b0;
                        if (!pass_index) begin
                            batch_row_base <= read_group_row_base;
                            batch_rows <= first_batch_rows;
                            if (r4_enable) begin
                                row_max_abs <= '0;
                                second_row_max_abs <= '0;
                                r4_quartet_index <= '0;
                                r4_active <= 1'b1;
                                r4_started <= 1'b0;
                                state <= R4_START;
                            end else begin
                                state <= WRITER_CONFIG;
                            end
                        end else begin
                            state <= ADVANCE_BATCH;
                        end
                    end else if (pass_index && !product_group_recompute) begin
                        stripe_index <= stripe_index + 11'd1;
                        product_low_received <= 1'b0;
                        state <= PRODUCT_LOAD_LOW;
                    end else if (precomputed_product && !pass_index) begin
                        stripe_index <= stripe_index + 11'd1;
                        batch_row_base <= read_group_row_base;
                        batch_rows <= first_batch_rows;
                        current_gate_address <= current_gate_address +
                            ADDR_WIDTH'(stripe_bytes);
                        state <= GATE_REQUEST;
                    end else if (next_stripe_found) begin
                        stripe_index <= stripe_index + 11'd1;
                        compute_second_batch <= 1'b0;
                        if (!pass_index) begin
                            batch_row_base <= read_group_row_base;
                            batch_rows <= first_batch_rows;
                        end
                        current_gate_address <=
                            prefetch_slot_address[next_stripe_slot];
                        tile_values <= prefetch_gate_values[next_stripe_slot];
                        up_values <= prefetch_up_values[next_stripe_slot];
                        second_gate_values <=
                            prefetch_second_gate_values[next_stripe_slot];
                        second_up_values <=
                            prefetch_second_up_values[next_stripe_slot];
                        prefetch_slot_valid[next_stripe_slot] <= 1'b0;
                        state <= SILU_REQUEST;
                    end else begin
                        state <= WAIT_PREFETCH;
                    end
                end
                WAIT_PREFETCH: if (next_stripe_found) begin
                    stripe_index <= stripe_index + 11'd1;
                    compute_second_batch <= 1'b0;
                    if (!pass_index) begin
                        batch_row_base <= read_group_row_base;
                        batch_rows <= first_batch_rows;
                    end
                    current_gate_address <=
                        prefetch_slot_address[next_stripe_slot];
                    tile_values <= prefetch_gate_values[next_stripe_slot];
                    up_values <= prefetch_up_values[next_stripe_slot];
                    second_gate_values <=
                        prefetch_second_gate_values[next_stripe_slot];
                    second_up_values <=
                        prefetch_second_up_values[next_stripe_slot];
                    prefetch_slot_valid[next_stripe_slot] <= 1'b0;
                    state <= SILU_REQUEST;
                end
                R4_START: if (r4_start_ready) begin
                    r4_started <= 1'b1;
                    state <= R4_WAIT;
                end
                R4_WAIT: if (r4_done_valid) begin
                    r4_active <= 1'b0;
                    r4_started <= 1'b0;
                    if (r4_done_error) begin
                        error <= 1'b1;
                        error_id <= r4_done_error_id == 0 ?
                            ERROR_ARITHMETIC : r4_done_error_id;
                        state <= COMPLETE;
                    end else begin
                        for (integer row = 0; row < 4; row++) begin
                            if (row < r4_start_row_count) begin
                                if (!r4_quartet_index[1])
                                    row_max_abs[(r4_quartet_index[0]*4+row)*16
                                        +: 16] <= r4_row_max_abs[row*16 +: 16];
                                else
                                    second_row_max_abs[
                                        (r4_quartet_index[0]*4+row)*16 +: 16]
                                        <= r4_row_max_abs[row*16 +: 16];
                            end
                        end
                        if ({3'd0, r4_quartet_index, 2'b00} + 7'd4 <
                            {2'd0, read_group_rows}) begin
                            r4_quartet_index <= r4_quartet_index + 2'd1;
                            r4_active <= 1'b1;
                            r4_started <= 1'b0;
                            state <= R4_START;
                        end else begin
                            batch_row_base <= read_group_row_base;
                            batch_rows <= first_batch_rows;
                            state <= WRITER_CONFIG;
                        end
                    end
                end
                WRITER_CONFIG: if (writer_cfg_valid && writer_cfg_ready) begin
                    writer_done_seen <= 1'b0;
                    state <= SCALE_REQUEST;
                end
                SCALE_REQUEST: if (quant_scale_req_valid &&
                    quant_scale_req_ready) begin
`ifndef SYNTHESIS
                    accepted_quant_scale_count <=
                        accepted_quant_scale_count + 64'd1;
`endif
                    state <= SCALE_RESPONSE;
                end
                SCALE_RESPONSE: if (quant_scale_rsp_valid &&
                    quant_scale_rsp_ready) begin
                    batch_scales <= quant_scale_rsp_values_bf16;
                    state <= SCALE_WRITE;
                end
                SCALE_WRITE: if (writer_scale_valid && writer_scale_ready) begin
                    pass_index <= 1'b1;
                    stripe_index <= '0;
                    prefetch_state <= PREFETCH_IDLE;
                    prefetch_slot_valid <= '0;
                    prefetch_window_active <= 1'b0;
                    product_low_received <= 1'b0;
                    if (product_group_recompute) begin
                        current_gate_address <= batch_gate_address;
                        state <= GATE_REQUEST;
                    end else begin
                        product_stream_request_half <= 1'b0;
                        product_stream_request_stripe <= '0;
                        product_stream_response_stripe <= '0;
                        product_stream_quant_response_stripe <= '0;
                        product_stream_low_data <= '0;
                        product_stream_low_tag <= '0;
                        product_stream_trace_sent <= 1'b0;
                        product_stream_reserved_tiles <= '0;
                        product_read_outstanding <= '0;
                        quant_outstanding <= '0;
                        state <= PRODUCT_QUANT_STREAM;
                    end
                end
                PRODUCT_LOAD_LOW: if (product_memory_req_valid &&
                    product_memory_req_ready) begin
                    state <= two_token_r4 ? PRODUCT_LOAD_RESPONSE : PRODUCT_LOAD_HIGH;
                end
                PRODUCT_LOAD_HIGH: if (product_memory_req_valid &&
                    product_memory_req_ready) begin
                    state <= PRODUCT_LOAD_RESPONSE;
                end
                PRODUCT_LOAD_RESPONSE: if (product_memory_rsp_valid &&
                    product_memory_rsp_ready) begin
                    if (product_memory_rsp.tag != tile_tag ||
                        product_memory_rsp.half != product_low_received) begin
                        error <= 1'b1;
                        error_id <= ERROR_CONFIGURATION;
                    end
                    if (!product_memory_rsp.half) begin
                        tile_values[511:0] <= product_memory_rsp.data;
                        product_low_received <= !two_token_r4;
                        if (two_token_r4) begin
                            tile_values[1023:512] <= '0;
                            state <= QUANTIZED_REQUEST;
                        end
                    end else begin
                        tile_values[1023:512] <= product_memory_rsp.data;
                        product_low_received <= 1'b0;
                        state <= QUANTIZED_REQUEST;
                    end
                end
                QUANTIZED_REQUEST: if (quant_values_req_valid && quant_values_req_ready) begin
`ifndef SYNTHESIS
                    accepted_quantized_value_count <=
                        accepted_quantized_value_count + 64'd1;
`endif
                    state <= QUANTIZED_RESPONSE;
                end
                QUANTIZED_RESPONSE: if (quant_values_rsp_valid &&
                    quant_values_rsp_ready) begin
                    if (quant_values_rsp_tag != tile_tag ||
                        quant_values_rsp_lane_mask != lane_mask) begin
                        error <= 1'b1;
                        error_id <= ERROR_QUANTIZER;
                    end
                    quantized_values <= quant_values_rsp_values;
                    quantized_lane_mask <= quant_values_rsp_lane_mask;
                    quantized_tag <= quant_values_rsp_tag;
                    state <= QUANTIZED_WRITE;
                end
                QUANTIZED_WRITE: if (writer_quantized_valid && writer_quantized_ready) begin
                    state <= TRACE_SAMPLE;
                end
                PRODUCT_QUANT_STREAM: begin
                    case ({product_stream_request_fire &&
                               !product_stream_request_half,
                           product_stream_fifo_input_valid &&
                               product_stream_fifo_input_ready})
                        2'b10: product_stream_reserved_tiles <=
                            product_stream_reserved_tiles + 2'd1;
                        2'b01: product_stream_reserved_tiles <=
                            product_stream_reserved_tiles - 2'd1;
                        default: begin end
                    endcase
                    if (product_stream_request_fire) begin
                        if (product_stream_request_half || two_token_r4) begin
                            product_stream_request_half <= 1'b0;
                            product_stream_request_stripe <=
                                product_stream_request_stripe + 11'd1;
                        end else begin
                            product_stream_request_half <= 1'b1;
                        end
                    end
                    if (product_stream_response_fire) begin
                        if (product_memory_rsp.half != product_low_received ||
                            product_memory_rsp.tag != TAG_WIDTH'({
                                1'b1, segment_index, batch_row_base[5:0],
                                product_stream_response_stripe[6:0]})) begin
                            error <= 1'b1;
                            error_id <= ERROR_CONFIGURATION;
                        end
                        if (two_token_r4) begin
                            product_low_received <= 1'b0;
                            product_stream_response_stripe <=
                                product_stream_response_stripe + 11'd1;
                        end else if (!product_memory_rsp.half) begin
                            product_stream_low_data <= product_memory_rsp.data;
                            product_stream_low_tag <= product_memory_rsp.tag;
                            product_low_received <= 1'b1;
                        end else begin
                            if (product_memory_rsp.tag != product_stream_low_tag) begin
                                error <= 1'b1;
                                error_id <= ERROR_CONFIGURATION;
                            end
                            product_low_received <= 1'b0;
                            product_stream_response_stripe <=
                                product_stream_response_stripe + 11'd1;
                        end
                    end
                    if (product_stream_trace_fire)
                        product_stream_trace_sent <= 1'b1;
                    if (product_stream_quant_request_fire) begin
                        product_stream_trace_sent <= 1'b0;
`ifndef SYNTHESIS
                        accepted_quantized_value_count <=
                            accepted_quantized_value_count + 64'd1;
`endif
                    end
                    if (product_stream_quant_response_fire) begin
                        // The first eight scratch banks are dead after 1024 N8 stripes.
                        if (r4_enable && final_batch && final_read_group &&
                            segment_index + 2'd1 >= segment_count &&
                            product_stream_quant_response_stripe >=
                                11'(STRIPE_COUNT < 1024 ? STRIPE_COUNT - 1 : 1023))
                            r4_panel_released <= 1'b1;
                        if (quant_values_rsp_tag != TAG_WIDTH'({
                                1'b1, segment_index, batch_row_base[5:0],
                                product_stream_quant_response_stripe[6:0]}) ||
                            quant_values_rsp_lane_mask != lane_mask) begin
                            error <= 1'b1;
                            error_id <= ERROR_QUANTIZER;
                        end
                        if (product_stream_quant_response_stripe + 11'd1 ==
                            11'(STRIPE_COUNT)) begin
                            state <= ADVANCE_BATCH;
                        end else begin
                            product_stream_quant_response_stripe <=
                                product_stream_quant_response_stripe + 11'd1;
                        end
                    end
                end
                ADVANCE_BATCH: begin
                    if (!writer_done_seen && !writer_done_pulse) begin
                        state <= ADVANCE_BATCH;
                    end else if (read_group_rows > 5'd8 &&
                        batch_row_base == read_group_row_base) begin
                        batch_row_base <= read_group_row_base + 6'd8;
                        batch_gate_address <= workspace_base + ADDR_WIDTH'(
                            {(read_group_row_base + 6'd8), 4'b0000});
                        current_gate_address <= workspace_base + ADDR_WIDTH'(
                            {(read_group_row_base + 6'd8), 4'b0000});
                        batch_rows <= second_batch_rows;
                        pass_index <= 1'b1;
                        stripe_index <= '0;
                        prefetch_state <= PREFETCH_IDLE;
                        prefetch_slot_valid <= '0;
                        prefetch_window_active <= 1'b0;
                        state <= WRITER_CONFIG;
                    end else if (!final_batch) begin
                        batch_row_base <= next_batch_base;
                        read_group_row_base <= next_batch_base;
                        read_group_rows <= next_group_rows;
                        product_group_byte_base <= next_group_byte_base;
                        product_group_recompute <= next_group_recompute;
                        batch_gate_address <= workspace_base +
                            ADDR_WIDTH'({next_batch_base, 4'b0000});
                        current_gate_address <= workspace_base +
                            ADDR_WIDTH'({next_batch_base, 4'b0000});
                        batch_rows <= two_token_r4 ? 4'(next_group_rows) : current_segment_end -
                            {1'b0, next_batch_base} > 7'd8 ?
                            4'd8 : 4'(current_segment_end - next_batch_base);
                        pass_index <= 1'b0;
                        stripe_index <= '0;
                        row_max_abs <= '0;
                        second_row_max_abs <= '0;
                        compute_second_batch <= 1'b0;
                        prefetch_state <= PREFETCH_IDLE;
                        prefetch_slot_valid <= '0;
                        prefetch_window_active <= 1'b0;
                        state <= GATE_REQUEST;
                    end else if (segment_index + 2'd1 < segment_count) begin
                        segment_index <= segment_index + 2'd1;
                        batch_row_base <= segment_base_at(segment_index + 2'd1);
                        read_group_row_base <=
                            segment_base_at(segment_index + 2'd1);
                        read_group_rows <= next_segment_group_rows;
                        product_group_byte_base <=
                            next_segment_group_byte_base;
                        product_group_recompute <=
                            next_segment_group_recompute;
                        batch_gate_address <= workspace_base + ADDR_WIDTH'(
                            {segment_base_at(segment_index + 2'd1), 4'b0000});
                        current_gate_address <= workspace_base + ADDR_WIDTH'(
                            {segment_base_at(segment_index + 2'd1), 4'b0000});
                        batch_rows <= two_token_r4 ? 4'(next_segment_group_rows) :
                            segment_rows_at(segment_index + 2'd1) > 8 ?
                            4'd8 : 4'(segment_rows_at(segment_index + 2'd1));
                        pass_index <= 1'b0;
                        stripe_index <= '0;
                        row_max_abs <= '0;
                        second_row_max_abs <= '0;
                        compute_second_batch <= 1'b0;
                        prefetch_state <= PREFETCH_IDLE;
                        prefetch_slot_valid <= '0;
                        prefetch_window_active <= 1'b0;
                        state <= GATE_REQUEST;
                    end else begin
                        state <= COMPLETE;
                    end
                end
                COMPLETE: if (done_ready)
                    state <= IDLE;
                ABORT_DRAIN: begin
                    if (r4_active && !r4_started) begin
                        r4_active <= 1'b0;
                        r4_started <= 1'b0;
                    end
                    if (r4_abort_ack) begin
                        r4_active <= 1'b0;
                        r4_started <= 1'b0;
                    end
                    if ((!request_active || request_terminal_seen ||
                         read_request_done || read_request_error) &&
                        (!arithmetic_outstanding || arithmetic_abort_ack) &&
                        !max_outstanding &&
                        product_read_outstanding == 0 &&
                        (!r4_active || !r4_started || r4_abort_ack) &&
                        (!(quant_scale_outstanding || quant_outstanding != 0) ||
                         quant_abort_ack)) begin
                        quant_scale_outstanding <= 1'b0;
                        quant_outstanding <= 1'b0;
                        if (abort_pending || abort_request) begin
                            abort_ack <= 1'b1;
                            state <= ABORT_WAIT_LOW;
                        end else begin
                            state <= COMPLETE;
                        end
                    end
                end
                ABORT_WAIT_LOW: if (!abort_request) begin
                    abort_pending <= 1'b0;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase

            if (state != ABORT_DRAIN && state != ABORT_WAIT_LOW &&
                !abort_request && !writer_error && !read_request_error) begin
                case (prefetch_state)
                    PREFETCH_GATE_REQUEST: if (read_request_valid &&
                        read_request_ready && !main_read_request) begin
                        request_active <= 1'b1;
                        request_terminal_seen <= 1'b0;
                        request_terminal_error <= 1'b0;
                        accepted_read_tag <= read_request_tag;
                        accepted_read_paired <= read_second_span_valid;
                        prefetch_stream_row <= '0;
`ifndef SYNTHESIS
                        accepted_read_request_count <=
                            accepted_read_request_count + 64'd1;
`endif
                        prefetch_state <= PREFETCH_GATE_STREAM;
                    end
                    PREFETCH_GATE_STREAM: if (read_data_valid && read_data_ready) begin
                        if (prefetch_stream_row < 5'd8)
                            prefetch_gate_values[prefetch_fill_slot]
                                [prefetch_stream_row*128 +: 128] <= read_data;
                        else
                            prefetch_second_gate_values[prefetch_fill_slot]
                                [(prefetch_stream_row-5'd8)*128 +: 128] <=
                                read_data;
`ifndef SYNTHESIS
                        accepted_read_byte_count <= accepted_read_byte_count + 64'd16;
`endif
                        if (read_data_tag != accepted_read_tag ||
                            read_data_span ||
                            read_byte_enable != 16'hffff ||
                            read_data_last != (!accepted_read_paired &&
                                prefetch_stream_row + 5'd1 == read_rows)) begin
                            error <= 1'b1;
                            error_id <= ERROR_DMA;
                        end
                        if (prefetch_stream_row + 5'd1 == read_rows &&
                            accepted_read_paired) begin
                            prefetch_stream_row <= '0;
                            prefetch_state <= PREFETCH_UP_STREAM;
                        end else if (prefetch_stream_row + 5'd1 == read_rows)
                            prefetch_state <= PREFETCH_GATE_TERMINAL;
                        else
                            prefetch_stream_row <= prefetch_stream_row + 5'd1;
                    end
                    PREFETCH_GATE_TERMINAL: if (request_terminal_seen ||
                        read_request_done || read_request_error) begin
                        if (request_terminal_error || read_request_error) begin
                            prefetch_state <= PREFETCH_IDLE;
                            state <= ABORT_DRAIN;
                        end else begin
                            prefetch_state <= PREFETCH_UP_REQUEST;
                        end
                    end
                    PREFETCH_UP_REQUEST: if (read_request_valid &&
                        read_request_ready && !main_read_request) begin
                        request_active <= 1'b1;
                        request_terminal_seen <= 1'b0;
                        request_terminal_error <= 1'b0;
                        accepted_read_tag <= read_request_tag;
                        accepted_read_paired <= 1'b0;
                        prefetch_stream_row <= '0;
`ifndef SYNTHESIS
                        accepted_read_request_count <=
                            accepted_read_request_count + 64'd1;
`endif
                        prefetch_state <= PREFETCH_UP_STREAM;
                    end
                    PREFETCH_UP_STREAM: if (read_data_valid && read_data_ready) begin
                        if (prefetch_stream_row < 5'd8)
                            prefetch_up_values[prefetch_fill_slot]
                                [prefetch_stream_row*128 +: 128] <= read_data;
                        else
                            prefetch_second_up_values[prefetch_fill_slot]
                                [(prefetch_stream_row-5'd8)*128 +: 128] <=
                                read_data;
`ifndef SYNTHESIS
                        accepted_read_byte_count <= accepted_read_byte_count + 64'd16;
`endif
                        if (read_data_tag != accepted_read_tag ||
                            read_data_span != accepted_read_paired ||
                            read_byte_enable != 16'hffff ||
                            read_data_last !=
                                (prefetch_stream_row + 5'd1 == read_rows &&
                                 (!repeated_pair_active ||
                                  prefetch_stripe_index + 11'd1 ==
                                      accepted_read_pair_count))) begin
                            error <= 1'b1;
                            error_id <= ERROR_DMA;
                        end
                        if (prefetch_stream_row + 5'd1 == read_rows) begin
                            if (repeated_pair_active &&
                                prefetch_stripe_index + 11'd1 !=
                                    accepted_read_pair_count) begin
                                prefetch_slot_valid[prefetch_fill_slot] <= 1'b1;
                                prefetch_slot_stripe[prefetch_fill_slot] <=
                                    prefetch_stripe_index;
                                prefetch_slot_address[prefetch_fill_slot] <=
                                    prefetch_gate_address;
                                prefetch_stripe_index <=
                                    prefetch_stripe_index + 11'd1;
                                prefetch_state <= PREFETCH_PATTERN_WAIT;
                            end else begin
                                prefetch_state <= PREFETCH_UP_TERMINAL;
                            end
                        end else begin
                            prefetch_stream_row <= prefetch_stream_row + 5'd1;
                        end
                    end
                    PREFETCH_UP_TERMINAL: if (request_terminal_seen ||
                        read_request_done || read_request_error) begin
                        if (request_terminal_error || read_request_error) begin
                            prefetch_state <= PREFETCH_IDLE;
                            state <= ABORT_DRAIN;
                        end else begin
                            if ((!product_group_recompute || pass_index) &&
                                prefetch_stripe_index + 11'd1 ==
                                    11'(STRIPE_COUNT) &&
                                final_read_group &&
                                segment_index + 2'd1 >= segment_count)
                                read_phase_complete <= 1'b1;
                            prefetch_slot_valid[prefetch_fill_slot] <= 1'b1;
                            prefetch_slot_stripe[prefetch_fill_slot] <=
                                prefetch_stripe_index;
                            prefetch_slot_address[prefetch_fill_slot] <=
                                prefetch_gate_address;
                            prefetch_state <= PREFETCH_IDLE;
                        end
                    end
                    PREFETCH_PATTERN_WAIT: if (
                        !prefetch_slot_valid[prefetch_fill_slot]) begin
                        prefetch_gate_address <= prefetch_gate_address +
                            ADDR_WIDTH'(stripe_pair_bytes);
                        prefetch_stream_row <= '0;
                        prefetch_gate_values[prefetch_fill_slot] <= '0;
                        prefetch_up_values[prefetch_fill_slot] <= '0;
                        prefetch_second_gate_values[prefetch_fill_slot] <= '0;
                        prefetch_second_up_values[prefetch_fill_slot] <= '0;
                        prefetch_state <= PREFETCH_GATE_STREAM;
                    end
                    default: begin end
                endcase

                if (prefetch_window_active &&
                    prefetch_state == PREFETCH_IDLE &&
                    next_prefetch_stripe < 11'(STRIPE_COUNT) &&
                    prefetch_free_found) begin
                    prefetch_fill_slot <= prefetch_free_slot;
                    prefetch_stripe_index <= next_prefetch_stripe;
                    prefetch_gate_address <= next_prefetch_address;
                    prefetch_gate_values[prefetch_free_slot] <= '0;
                    prefetch_up_values[prefetch_free_slot] <= '0;
                    prefetch_second_gate_values[prefetch_free_slot] <= '0;
                    prefetch_second_up_values[prefetch_free_slot] <= '0;
                    prefetch_slot_valid[prefetch_free_slot] <= 1'b0;
                    next_prefetch_stripe <= next_prefetch_stripe + 11'd1;
                    next_prefetch_address <= next_prefetch_address +
                        ADDR_WIDTH'(stripe_pair_bytes);
                    prefetch_state <= PREFETCH_GATE_REQUEST;
                end
            end

        end
    end

    initial begin
        if (FFN_FEATURES <= 0 || FFN_FEATURES % 8 != 0 ||
            FFN_FEATURES > 12288 || MAX_ROWS != 48 || ADDR_WIDTH < 32)
            $error("elementwise_engine requires FFN_FEATURES multiple of 8 and MAX_ROWS=48");
        if (16 * FFN_FEATURES * 2 > 480 * 1024)
            $error("elementwise_engine 16-row product group exceeds product scratch");
        if (PREFETCH_SLOTS < 1 || PREFETCH_SLOTS > 3)
            $error("elementwise_engine requires PREFETCH_SLOTS in [1,3]");
    end

`ifndef SYNTHESIS
    logic check_stalled_outputs;
    assign check_stalled_outputs = !abort_request && !abort_pending &&
        !read_request_error && !writer_error &&
        state != ABORT_DRAIN && state != ABORT_WAIT_LOW;
    logic stalled_read_request;
    logic [2*ADDR_WIDTH+83:0] held_read_request;
    logic stalled_writer_quantized;
    logic [5+16+512+64+TAG_WIDTH:0] held_writer_quantized;
    logic stalled_product_stream_entry;
    logic [1050:0] held_product_stream_entry;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_writer_quantized <= 1'b0;
            stalled_read_request <= 1'b0;
            stalled_product_stream_entry <= 1'b0;
            held_writer_quantized <= '0;
            held_product_stream_entry <= '0;
        end else begin
            if (stalled_product_stream_entry && check_stalled_outputs)
                assert (product_stream_fifo_output_valid &&
                        product_stream_fifo_output_data ===
                            held_product_stream_entry)
                    else $error("elementwise_engine changed a stalled product stream entry");
            stalled_product_stream_entry <= check_stalled_outputs &&
                product_stream_fifo_output_valid &&
                !product_stream_fifo_output_ready;
            if (product_stream_fifo_output_valid &&
                !product_stream_fifo_output_ready)
                held_product_stream_entry <=
                    product_stream_fifo_output_data;
            if (stalled_read_request && check_stalled_outputs)
                assert (read_request_valid &&
                    {read_request_address, read_request_bytes,
                     read_request_tag, read_second_span_valid,
                     read_second_span_address, read_pair_stride,
                     read_pair_count} == held_read_request)
                    else $error("elementwise_engine changed a stalled read request");
            stalled_read_request <= check_stalled_outputs &&
                read_request_valid && !read_request_ready;
            held_read_request <= {read_request_address, read_request_bytes,
                read_request_tag, read_second_span_valid,
                read_second_span_address, read_pair_stride,
                read_pair_count};
            if (read_request_valid)
                assert (read_group_rows != 0 && read_request_bytes != 0)
                    else $error("elementwise_engine issued a zero-row read group");
            if (product_group_recompute)
                assert (read_group_rows <= 5'd8 &&
                        !product_memory_req_valid)
                    else $error("elementwise_engine recompute group accessed product scratch");
            if (state == ADVANCE_BATCH &&
                (writer_done_seen || writer_done_pulse))
                assert (product_read_outstanding == 0)
                    else $error("elementwise_engine changed product group with a pending scratch read");
            if (stalled_writer_quantized && check_stalled_outputs)
                assert (writer_quantized_valid &&
                    {writer_quantized_row_base, writer_quantized_element_base,
                     writer_quantized_values, writer_quantized_lane_mask,
                     writer_quantized_tag} == held_writer_quantized)
                    else $error("elementwise_engine changed a stalled quantized tile");
            stalled_writer_quantized <= check_stalled_outputs &&
                writer_quantized_valid && !writer_quantized_ready;
            if (writer_quantized_valid && !writer_quantized_ready)
                held_writer_quantized <= {writer_quantized_row_base,
                    writer_quantized_element_base, writer_quantized_values,
                    writer_quantized_lane_mask, writer_quantized_tag};
            if ($past(done_valid && !done_ready))
                assert (done_valid &&
                    $stable({error, error_id, completed_layout}))
                    else $error("elementwise_engine changed a stalled completion response");
        end
    end
`endif
endmodule

`default_nettype wire
