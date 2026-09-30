`default_nettype none
module rope_engine #(
    parameter integer PAIR_LANES = 8,
    parameter integer HEAD_DIM = 128,
    parameter integer MAX_ROWS = 48,
    parameter integer MAX_HEADS = 32,
    parameter integer MAX_SEQUENCE_LENGTH = 2048,
    parameter integer TENSOR_ADDR_WIDTH = 18,
    parameter integer TENSOR_CAPACITY_WORDS = 131072,
    parameter integer DMA_ADDR_WIDTH = 64,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                             clk,
    input  logic                             rst,
    input  logic                             abort_request,
    output logic                             abort_ack,
    input  logic                             start_valid,
    output logic                             start_ready,
    input  logic [5:0]                       start_row_count,
    input  logic [5:0]                       start_head_count,
    input  logic [8:0]                       start_head_dim,
    input  logic                             start_source_is_k,
    input  logic [MAX_ROWS*11-1:0]           start_token_position,
    input  logic [TENSOR_ADDR_WIDTH-1:0]     start_source_base,
    input  logic [TENSOR_ADDR_WIDTH-1:0]     start_source_row_stride,
    input  logic [TENSOR_ADDR_WIDTH-1:0]     start_destination_base,
    input  logic [TENSOR_ADDR_WIDTH-1:0]     start_destination_row_stride,
    input  logic [DMA_ADDR_WIDTH-1:0]        start_cos_lut_base,
    input  logic [DMA_ADDR_WIDTH-1:0]        start_sin_lut_base,
    output logic                             busy,
    output logic                             done_pulse,
    output logic                             error,
    output logic [3:0]                       error_id,

    output logic                             source_read_req_valid,
    input  logic                             source_read_req_ready,
    output logic                             source_read_is_k,
    output logic [2*PAIR_LANES-1:0]          source_read_lane_mask,
    output logic [2*PAIR_LANES*TENSOR_ADDR_WIDTH-1:0]
                                               source_read_lane_address,
    input  logic                             source_read_rsp_valid,
    output logic                             source_read_rsp_ready,
    input  logic [2*PAIR_LANES-1:0]          source_read_rsp_lane_mask,
    input  logic [2*PAIR_LANES*16-1:0]       source_read_rsp_lane_data,

    output logic                             cos_read_req_valid,
    input  logic                             cos_read_req_ready,
    output logic [DMA_ADDR_WIDTH-1:0]        cos_read_req_address,
    output logic [15:0]                      cos_read_req_bytes,
    output logic [5:0]                       constant_read_physical_row,
    output logic [6:0]                       constant_read_pair_base,
    input  logic                             cos_read_rsp_valid,
    output logic                             cos_read_rsp_ready,
    input  logic [PAIR_LANES-1:0]            cos_read_rsp_lane_mask,
    input  logic [PAIR_LANES*16-1:0]         cos_read_rsp_lane_data,

    output logic                             sin_read_req_valid,
    input  logic                             sin_read_req_ready,
    output logic [DMA_ADDR_WIDTH-1:0]        sin_read_req_address,
    output logic [15:0]                      sin_read_req_bytes,
    input  logic                             sin_read_rsp_valid,
    output logic                             sin_read_rsp_ready,
    input  logic [PAIR_LANES-1:0]            sin_read_rsp_lane_mask,
    input  logic [PAIR_LANES*16-1:0]         sin_read_rsp_lane_data,

    output logic                             destination_write_valid,
    input  logic                             destination_write_ready,
    output logic                             destination_write_is_k,
    output logic [2*PAIR_LANES-1:0]          destination_write_lane_mask,
    output logic [2*PAIR_LANES*TENSOR_ADDR_WIDTH-1:0]
                                               destination_write_lane_address,
    output logic [2*PAIR_LANES*16-1:0]       destination_write_lane_data,

    output logic                             trace_sample_valid,
    input  logic                             trace_sample_ready,
    output logic [5:0]                       trace_sample_physical_row,
    output logic [5:0]                       trace_sample_head,
    output logic [6:0]                       trace_sample_pair_base,
    output logic [10:0]                      trace_sample_token_position,
    output logic [2*PAIR_LANES*16-1:0]       trace_sample_data,
    output logic [4*PAIR_LANES-1:0]          trace_sample_byte_enable,

    output logic                             arithmetic_req_valid,
    input  logic                             arithmetic_req_ready,
    output logic [2:0]                       arithmetic_req_operation,
    output logic [2*PAIR_LANES*16-1:0]       arithmetic_req_values,
    output logic [2*PAIR_LANES*16-1:0]       arithmetic_req_paired_values,
    output logic [2*PAIR_LANES*16-1:0]       arithmetic_req_factor0_values,
    output logic [2*PAIR_LANES*16-1:0]       arithmetic_req_factor1_values,
    output logic [2*PAIR_LANES-1:0]          arithmetic_req_lane_mask,
    output logic [TAG_WIDTH-1:0]             arithmetic_req_tag,
    input  logic                             arithmetic_rsp_valid,
    output logic                             arithmetic_rsp_ready,
    input  logic [2*PAIR_LANES*16-1:0]       arithmetic_rsp_values,
    input  logic [2*PAIR_LANES-1:0]          arithmetic_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]             arithmetic_rsp_tag,

    output logic [31:0]                      accepted_source_request_count,
    output logic [31:0]                      accepted_cos_request_count,
    output logic [31:0]                      accepted_sin_request_count,
    output logic [31:0]                      accepted_arithmetic_request_count,
    output logic [31:0]                      accepted_destination_write_count,
    output logic [31:0]                      accepted_trace_sample_count,
    output logic [31:0]                      completed_chunk_count,
    output logic [31:0]                      completed_command_count
);
    localparam logic [3:0] ERR_NONE = 4'd0;
    localparam logic [3:0] ERR_CONFIG = 4'd1;
    localparam logic [3:0] ERR_POSITION = 4'd2;
    localparam logic [3:0] ERR_SOURCE_MASK = 4'd3;
    localparam logic [3:0] ERR_LUT_MASK = 4'd4;
    localparam logic [3:0] ERR_ARITHMETIC = 4'd5;
    localparam integer HALF_DIM = HEAD_DIM / 2;
    localparam logic [DMA_ADDR_WIDTH-1:0] LUT_TABLE_BYTES =
        DMA_ADDR_WIDTH'(MAX_SEQUENCE_LENGTH * HEAD_DIM * 2);

    localparam integer CHUNK_META_WIDTH =
        2 + TENSOR_ADDR_WIDTH + 6 + 6 + 7 + 11 + PAIR_LANES + TAG_WIDTH;
    localparam integer OPERAND_FIFO_WIDTH =
        CHUNK_META_WIDTH + 4 * PAIR_LANES * 16;
    localparam integer RESULT_FIFO_WIDTH =
        CHUNK_META_WIDTH + 2 * PAIR_LANES * 16;
    localparam integer OPERAND_FIFO_DEPTH = 2;
    localparam integer ARITHMETIC_META_FIFO_DEPTH = 8;
    localparam integer RESULT_FIFO_DEPTH = 2;

    typedef enum logic [2:0] {
        IDLE, CHECK_INPUT_RANGE, RUN_PIPELINE, ERROR_QUIESCE, DRAIN_PIPELINE
    } state_t;
    state_t state;

    typedef struct packed {
        logic                             last_chunk;
        logic                             source_is_k;
        logic [TENSOR_ADDR_WIDTH-1:0]     destination_head_base;
        logic [5:0]                       physical_row;
        logic [5:0]                       head_index;
        logic [6:0]                       pair_base;
        logic [10:0]                      token_position;
        logic [PAIR_LANES-1:0]            pair_mask;
        logic [TAG_WIDTH-1:0]             tag;
    } chunk_meta_t;

    logic [5:0] saved_row_count, saved_head_count;
    logic saved_source_is_k;
    logic [MAX_ROWS*11-1:0] validation_position_shift;
    logic [MAX_ROWS*11-1:0] execution_position_shift;
    logic [TENSOR_ADDR_WIDTH-1:0] saved_source_row_stride;
    logic [TENSOR_ADDR_WIDTH-1:0] saved_destination_row_stride;
    logic [TENSOR_ADDR_WIDTH-1:0] source_row_base, destination_row_base;
    logic [TENSOR_ADDR_WIDTH-1:0] source_head_base, destination_head_base;
    logic [63:0] range_check_source_base, range_check_destination_base;
    logic [DMA_ADDR_WIDTH-1:0] saved_cos_lut_base, saved_sin_lut_base;
    logic [5:0] physical_row, head_index, range_check_row;
    logic [6:0] pair_base;
    logic [10:0] token_position;
    logic [PAIR_LANES-1:0] pair_mask;
    logic [2*PAIR_LANES-1:0] element_mask;
    logic [TAG_WIDTH-1:0] chunk_tag;
    logic issue_exhausted;
    logic request_outstanding;
    logic source_sent, cos_sent, sin_sent;
    logic source_received, cos_received, sin_received;
    logic load_error_pending;
    logic [3:0] load_error_id;
    logic [2*PAIR_LANES*16-1:0] source_data;
    logic [PAIR_LANES*16-1:0] cos_data, sin_data;
    logic [15:0] arithmetic_pending_count;
    logic destination_accepted, trace_sample_accepted;
    logic terminal_abort, terminal_error;
    logic [3:0] terminal_error_id;
    logic quiesce_source_request, quiesce_cos_request;
    logic quiesce_sin_request, quiesce_arithmetic_request;
    logic quiesce_result_output;

    chunk_meta_t issue_meta;
    chunk_meta_t operand_meta;
    chunk_meta_t arithmetic_meta;
    chunk_meta_t result_meta;

    logic operand_fifo_input_valid, operand_fifo_input_ready;
    logic [OPERAND_FIFO_WIDTH-1:0] operand_fifo_input_data;
    logic operand_fifo_output_valid, operand_fifo_output_ready;
    logic [OPERAND_FIFO_WIDTH-1:0] operand_fifo_output_data;
    logic [1:0] operand_fifo_occupancy;
    logic [2*PAIR_LANES*16-1:0] operand_source_data;
    logic [PAIR_LANES*16-1:0] operand_cos_data, operand_sin_data;

    logic arithmetic_meta_input_valid, arithmetic_meta_input_ready;
    logic [CHUNK_META_WIDTH-1:0] arithmetic_meta_input_data;
    logic arithmetic_meta_output_valid, arithmetic_meta_output_ready;
    logic [CHUNK_META_WIDTH-1:0] arithmetic_meta_output_data;
    logic [3:0] arithmetic_meta_occupancy;

    logic result_fifo_input_valid, result_fifo_input_ready;
    logic [RESULT_FIFO_WIDTH-1:0] result_fifo_input_data;
    logic result_fifo_output_valid, result_fifo_output_ready;
    logic [RESULT_FIFO_WIDTH-1:0] result_fifo_output_data;
    logic [1:0] result_fifo_occupancy;
    logic [2*PAIR_LANES*16-1:0] result_data;
    logic [2*PAIR_LANES-1:0] result_mask;

    logic adapter_req_valid, adapter_req_ready;
    logic adapter_rsp_valid, adapter_rsp_ready;
    logic [PAIR_LANES*16-1:0] adapter_rsp_first, adapter_rsp_second;
    logic [PAIR_LANES-1:0] adapter_rsp_pair_mask;
    logic [TAG_WIDTH-1:0] adapter_rsp_tag;

    wire source_request_fire = source_read_req_valid && source_read_req_ready;
    wire cos_request_fire = cos_read_req_valid && cos_read_req_ready;
    wire sin_request_fire = sin_read_req_valid && sin_read_req_ready;
    wire source_response_fire = source_read_rsp_valid && source_read_rsp_ready;
    wire cos_response_fire = cos_read_rsp_valid && cos_read_rsp_ready;
    wire sin_response_fire = sin_read_rsp_valid && sin_read_rsp_ready;
    wire arithmetic_request_fire = adapter_req_valid && adapter_req_ready;
    wire arithmetic_response_fire = adapter_rsp_valid && adapter_rsp_ready;
    wire destination_write_fire = destination_write_valid && destination_write_ready;
    wire trace_sample_fire = trace_sample_valid && trace_sample_ready;

    wire source_complete = source_received || source_response_fire;
    wire cos_complete = cos_received || cos_response_fire;
    wire sin_complete = sin_received || sin_response_fire;
    wire source_mask_bad = source_response_fire &&
        source_read_rsp_lane_mask != element_mask;
    wire cos_mask_bad = cos_response_fire && cos_read_rsp_lane_mask != pair_mask;
    wire sin_mask_bad = sin_response_fire && sin_read_rsp_lane_mask != pair_mask;
    wire request_channels_sent = (source_sent || source_request_fire) &&
        (cos_sent || cos_request_fire) && (sin_sent || sin_request_fire);
    wire request_responses_complete = request_channels_sent &&
        source_complete && cos_complete && sin_complete;
    wire load_response_bad = load_error_pending || source_mask_bad ||
        cos_mask_bad || sin_mask_bad;
    wire [3:0] current_load_error_id = source_mask_bad ? ERR_SOURCE_MASK :
        ((cos_mask_bad || sin_mask_bad) ? ERR_LUT_MASK : load_error_id);
    wire operand_fifo_push = operand_fifo_input_valid && operand_fifo_input_ready;
    wire arithmetic_response_bad = adapter_rsp_valid &&
        (!arithmetic_meta_output_valid ||
         adapter_rsp_pair_mask != arithmetic_meta.pair_mask ||
         adapter_rsp_tag != arithmetic_meta.tag);
    wire result_fifo_pop = result_fifo_output_valid && result_fifo_output_ready;
    wire request_drained = !request_outstanding ||
        ((!source_sent || source_complete) && (!cos_sent || cos_complete) &&
         (!sin_sent || sin_complete));
    wire arithmetic_drained = arithmetic_pending_count == 0 ||
        (arithmetic_pending_count == 1 && arithmetic_response_fire);

    wire [63:0] range_check_source_end = range_check_source_base +
        64'(saved_head_count) * HEAD_DIM;
    wire [63:0] range_check_destination_end = range_check_destination_base +
        64'(saved_head_count) * HEAD_DIM;

    assign start_ready = state == IDLE && !abort_request;
    assign busy = state != IDLE;
    assign source_read_req_valid = !abort_request &&
        ((state == RUN_PIPELINE && !issue_exhausted && !source_sent) ||
         (state == ERROR_QUIESCE && quiesce_source_request));
    assign cos_read_req_valid = !abort_request &&
        ((state == RUN_PIPELINE && !issue_exhausted && !cos_sent) ||
         (state == ERROR_QUIESCE && quiesce_cos_request));
    assign sin_read_req_valid = !abort_request &&
        ((state == RUN_PIPELINE && !issue_exhausted && !sin_sent) ||
         (state == ERROR_QUIESCE && quiesce_sin_request));
    assign source_read_rsp_ready = (state == RUN_PIPELINE ||
        state == ERROR_QUIESCE || state == DRAIN_PIPELINE) &&
        (source_sent || source_request_fire) && !source_received;
    assign cos_read_rsp_ready = (state == RUN_PIPELINE ||
        state == ERROR_QUIESCE || state == DRAIN_PIPELINE) &&
        (cos_sent || cos_request_fire) && !cos_received;
    assign sin_read_rsp_ready = (state == RUN_PIPELINE ||
        state == ERROR_QUIESCE || state == DRAIN_PIPELINE) &&
        (sin_sent || sin_request_fire) && !sin_received;
    assign source_read_is_k = saved_source_is_k;
    assign source_read_lane_mask = element_mask;
    assign destination_write_is_k = saved_source_is_k;

    assign adapter_req_valid = !abort_request && operand_fifo_output_valid &&
        arithmetic_meta_input_ready &&
        (state == RUN_PIPELINE ||
         (state == ERROR_QUIESCE && quiesce_arithmetic_request));
    assign operand_fifo_output_ready = !abort_request && adapter_req_ready &&
        arithmetic_meta_input_ready &&
        (state == RUN_PIPELINE ||
         (state == ERROR_QUIESCE && quiesce_arithmetic_request));
    assign arithmetic_meta_input_valid = arithmetic_request_fire;
    assign arithmetic_meta_input_data = operand_meta;
    assign adapter_rsp_ready = (state == ERROR_QUIESCE ||
        state == DRAIN_PIPELINE) ? 1'b1 :
        (state == RUN_PIPELINE && !abort_request &&
         (arithmetic_response_bad ||
          (arithmetic_meta_output_valid && result_fifo_input_ready)));
    assign arithmetic_meta_output_ready = arithmetic_response_fire &&
        arithmetic_meta_output_valid;
    assign destination_write_valid = !abort_request && result_fifo_output_valid &&
        !destination_accepted &&
        (state == RUN_PIPELINE ||
         (state == ERROR_QUIESCE && quiesce_result_output));
    assign trace_sample_valid = !abort_request && result_fifo_output_valid &&
        !trace_sample_accepted &&
        (state == RUN_PIPELINE ||
         (state == ERROR_QUIESCE && quiesce_result_output));

    assign cos_read_req_address = saved_cos_lut_base +
        (DMA_ADDR_WIDTH'(token_position) * HEAD_DIM + pair_base) * 2;
    assign sin_read_req_address = saved_sin_lut_base +
        (DMA_ADDR_WIDTH'(token_position) * HEAD_DIM + pair_base) * 2;
    assign cos_read_req_bytes = 16'(PAIR_LANES * 2);
    assign sin_read_req_bytes = 16'(PAIR_LANES * 2);
    assign constant_read_physical_row = physical_row;
    assign constant_read_pair_base = pair_base;
    assign trace_sample_physical_row = result_meta.physical_row;
    assign trace_sample_head = result_meta.head_index;
    assign trace_sample_pair_base = result_meta.pair_base;
    assign trace_sample_token_position = result_meta.token_position;
    assign trace_sample_data = result_data;
    assign destination_write_lane_data = result_data;
    assign result_mask = {result_meta.pair_mask, result_meta.pair_mask};
    assign destination_write_lane_mask = result_mask;

    always_comb begin
        pair_mask = '0;
        element_mask = '0;
        source_read_lane_address = '0;
        destination_write_lane_address = '0;
        trace_sample_byte_enable = '0;
        for (integer lane = 0; lane < PAIR_LANES; lane = lane + 1) begin
            pair_mask[lane] = pair_base + lane < HALF_DIM;
            element_mask[lane] = pair_mask[lane];
            element_mask[PAIR_LANES+lane] = pair_mask[lane];
            source_read_lane_address[lane*TENSOR_ADDR_WIDTH +: TENSOR_ADDR_WIDTH] =
                source_head_base + pair_base + lane;
            source_read_lane_address[(PAIR_LANES+lane)*TENSOR_ADDR_WIDTH +:
                                     TENSOR_ADDR_WIDTH] =
                source_head_base + HALF_DIM + pair_base + lane;
            destination_write_lane_address[lane*TENSOR_ADDR_WIDTH +: TENSOR_ADDR_WIDTH] =
                result_meta.destination_head_base + result_meta.pair_base + lane;
            destination_write_lane_address[(PAIR_LANES+lane)*TENSOR_ADDR_WIDTH +:
                                            TENSOR_ADDR_WIDTH] =
                result_meta.destination_head_base + HALF_DIM + result_meta.pair_base + lane;
            trace_sample_byte_enable[lane*2 +: 2] = {2{result_mask[lane]}};
            trace_sample_byte_enable[(PAIR_LANES+lane)*2 +: 2] =
                {2{result_mask[PAIR_LANES+lane]}};
        end
    end

    assign chunk_tag = TAG_WIDTH'({saved_source_is_k, physical_row[5:0],
                                   head_index[4:0], pair_base[6:3]});

    always_comb begin
        issue_meta = '0;
        issue_meta.last_chunk = pair_base + PAIR_LANES >= HALF_DIM &&
            head_index + 1'b1 >= saved_head_count &&
            physical_row + 1'b1 >= saved_row_count;
        issue_meta.source_is_k = saved_source_is_k;
        issue_meta.destination_head_base = destination_head_base;
        issue_meta.physical_row = physical_row;
        issue_meta.head_index = head_index;
        issue_meta.pair_base = pair_base;
        issue_meta.token_position = token_position;
        issue_meta.pair_mask = pair_mask;
        issue_meta.tag = chunk_tag;
    end

    assign operand_fifo_input_valid = state == RUN_PIPELINE &&
        request_outstanding && request_responses_complete && !load_response_bad &&
        !abort_request;
    assign operand_fifo_input_data = {
        issue_meta,
        source_response_fire ? source_read_rsp_lane_data : source_data,
        cos_response_fire ? cos_read_rsp_lane_data : cos_data,
        sin_response_fire ? sin_read_rsp_lane_data : sin_data
    };
    assign {operand_meta, operand_source_data, operand_cos_data, operand_sin_data} =
        operand_fifo_output_data;
    assign {arithmetic_meta} = arithmetic_meta_output_data;
    assign {result_meta, result_data} = result_fifo_output_data;

    assign result_fifo_input_valid = state == RUN_PIPELINE && adapter_rsp_valid &&
        arithmetic_meta_output_valid && !arithmetic_response_bad && !abort_request;
    assign result_fifo_input_data = {
        arithmetic_meta, {adapter_rsp_second, adapter_rsp_first}
    };
    assign result_fifo_output_ready =
        (destination_accepted || destination_write_fire) &&
        (trace_sample_accepted || trace_sample_fire);

    ready_valid_fifo #(
        .DATA_WIDTH(OPERAND_FIFO_WIDTH), .DEPTH(OPERAND_FIFO_DEPTH)
    ) operand_fifo (
        .clk(clk), .rst(rst || state == DRAIN_PIPELINE),
        .input_valid(operand_fifo_input_valid), .input_ready(operand_fifo_input_ready),
        .input_data(operand_fifo_input_data), .output_valid(operand_fifo_output_valid),
        .output_ready(operand_fifo_output_ready), .output_data(operand_fifo_output_data),
        .occupancy(operand_fifo_occupancy));

    ready_valid_fifo #(
        .DATA_WIDTH(CHUNK_META_WIDTH), .DEPTH(ARITHMETIC_META_FIFO_DEPTH)
    ) arithmetic_meta_fifo (
        .clk(clk), .rst(rst || state == DRAIN_PIPELINE),
        .input_valid(arithmetic_meta_input_valid),
        .input_ready(arithmetic_meta_input_ready), .input_data(arithmetic_meta_input_data),
        .output_valid(arithmetic_meta_output_valid),
        .output_ready(arithmetic_meta_output_ready),
        .output_data(arithmetic_meta_output_data),
        .occupancy(arithmetic_meta_occupancy));

    ready_valid_fifo #(
        .DATA_WIDTH(RESULT_FIFO_WIDTH), .DEPTH(RESULT_FIFO_DEPTH)
    ) result_fifo (
        .clk(clk), .rst(rst || state == DRAIN_PIPELINE),
        .input_valid(result_fifo_input_valid), .input_ready(result_fifo_input_ready),
        .input_data(result_fifo_input_data), .output_valid(result_fifo_output_valid),
        .output_ready(result_fifo_output_ready), .output_data(result_fifo_output_data),
        .occupancy(result_fifo_occupancy));

    rope_vector_adapter #(
        .PAIR_LANES(PAIR_LANES), .TAG_WIDTH(TAG_WIDTH)
    ) adapter (
        .clk(clk), .rst(rst), .req_valid(adapter_req_valid),
        .req_ready(adapter_req_ready),
        .req_first_values(operand_source_data[0 +: PAIR_LANES*16]),
        .req_second_values(operand_source_data[PAIR_LANES*16 +: PAIR_LANES*16]),
        .req_cos_values(operand_cos_data), .req_sin_values(operand_sin_data),
        .req_pair_mask(operand_meta.pair_mask), .req_tag(operand_meta.tag),
        .rsp_valid(adapter_rsp_valid), .rsp_ready(adapter_rsp_ready),
        .rsp_first_values(adapter_rsp_first), .rsp_second_values(adapter_rsp_second),
        .rsp_pair_mask(adapter_rsp_pair_mask), .rsp_tag(adapter_rsp_tag),
        .arithmetic_req_valid(arithmetic_req_valid),
        .arithmetic_req_ready(arithmetic_req_ready),
        .arithmetic_req_operation(arithmetic_req_operation),
        .arithmetic_req_values(arithmetic_req_values),
        .arithmetic_req_paired_values(arithmetic_req_paired_values),
        .arithmetic_req_factor0_values(arithmetic_req_factor0_values),
        .arithmetic_req_factor1_values(arithmetic_req_factor1_values),
        .arithmetic_req_lane_mask(arithmetic_req_lane_mask),
        .arithmetic_req_tag(arithmetic_req_tag),
        .arithmetic_rsp_valid(arithmetic_rsp_valid),
        .arithmetic_rsp_ready(arithmetic_rsp_ready),
        .arithmetic_rsp_values(arithmetic_rsp_values),
        .arithmetic_rsp_lane_mask(arithmetic_rsp_lane_mask),
        .arithmetic_rsp_tag(arithmetic_rsp_tag)
    );

`ifdef SYNTHESIS
    always_comb begin
        accepted_source_request_count = '0;
        accepted_cos_request_count = '0;
        accepted_sin_request_count = '0;
        accepted_arithmetic_request_count = '0;
        accepted_destination_write_count = '0;
        accepted_trace_sample_count = '0;
        completed_chunk_count = '0;
        completed_command_count = '0;
    end
`endif

    always_ff @(posedge clk) begin : pipeline_control
        if (rst) begin
            state <= IDLE;
            saved_row_count <= '0;
            saved_head_count <= '0;
            saved_source_is_k <= 1'b0;
            validation_position_shift <= '0;
            execution_position_shift <= '0;
            saved_source_row_stride <= '0;
            saved_destination_row_stride <= '0;
            source_row_base <= '0;
            destination_row_base <= '0;
            source_head_base <= '0;
            destination_head_base <= '0;
            range_check_source_base <= '0;
            range_check_destination_base <= '0;
            saved_cos_lut_base <= '0;
            saved_sin_lut_base <= '0;
            physical_row <= '0;
            head_index <= '0;
            range_check_row <= '0;
            pair_base <= '0;
            token_position <= '0;
            issue_exhausted <= 1'b0;
            request_outstanding <= 1'b0;
            source_sent <= 1'b0;
            cos_sent <= 1'b0;
            sin_sent <= 1'b0;
            source_received <= 1'b0;
            cos_received <= 1'b0;
            sin_received <= 1'b0;
            load_error_pending <= 1'b0;
            load_error_id <= ERR_NONE;
            source_data <= '0;
            cos_data <= '0;
            sin_data <= '0;
            arithmetic_pending_count <= '0;
            destination_accepted <= 1'b0;
            trace_sample_accepted <= 1'b0;
            terminal_abort <= 1'b0;
            terminal_error <= 1'b0;
            terminal_error_id <= ERR_NONE;
            quiesce_source_request <= 1'b0;
            quiesce_cos_request <= 1'b0;
            quiesce_sin_request <= 1'b0;
            quiesce_arithmetic_request <= 1'b0;
            quiesce_result_output <= 1'b0;
            done_pulse <= 1'b0;
            error <= 1'b0;
            error_id <= ERR_NONE;
            abort_ack <= 1'b0;
`ifndef SYNTHESIS
            accepted_source_request_count <= '0;
            accepted_cos_request_count <= '0;
            accepted_sin_request_count <= '0;
            accepted_arithmetic_request_count <= '0;
            accepted_destination_write_count <= '0;
            accepted_trace_sample_count <= '0;
            completed_chunk_count <= '0;
            completed_command_count <= '0;
`endif
        end else begin
            done_pulse <= 1'b0;
            error <= 1'b0;
            error_id <= ERR_NONE;
            abort_ack <= 1'b0;

            if (source_request_fire || cos_request_fire || sin_request_fire)
                request_outstanding <= 1'b1;
            if (source_request_fire) begin
                source_sent <= 1'b1;
                source_received <= 1'b0;
`ifndef SYNTHESIS
                accepted_source_request_count <= accepted_source_request_count + 1'b1;
`endif
            end
            if (cos_request_fire) begin
                cos_sent <= 1'b1;
                cos_received <= 1'b0;
`ifndef SYNTHESIS
                accepted_cos_request_count <= accepted_cos_request_count + 1'b1;
`endif
            end
            if (sin_request_fire) begin
                sin_sent <= 1'b1;
                sin_received <= 1'b0;
`ifndef SYNTHESIS
                accepted_sin_request_count <= accepted_sin_request_count + 1'b1;
`endif
            end
            if (!request_outstanding &&
                (source_request_fire || cos_request_fire || sin_request_fire)) begin
                load_error_pending <= 1'b0;
                load_error_id <= ERR_NONE;
            end
            if (source_response_fire) begin
                source_received <= 1'b1;
                source_data <= source_read_rsp_lane_data;
                if (source_read_rsp_lane_mask != element_mask) begin
                    load_error_pending <= 1'b1;
                    load_error_id <= ERR_SOURCE_MASK;
                end
            end
            if (cos_response_fire) begin
                cos_received <= 1'b1;
                cos_data <= cos_read_rsp_lane_data;
                if (cos_read_rsp_lane_mask != pair_mask) begin
                    load_error_pending <= 1'b1;
                    if (!load_error_pending && !source_mask_bad)
                        load_error_id <= ERR_LUT_MASK;
                end
            end
            if (sin_response_fire) begin
                sin_received <= 1'b1;
                sin_data <= sin_read_rsp_lane_data;
                if (sin_read_rsp_lane_mask != pair_mask) begin
                    load_error_pending <= 1'b1;
                    if (!load_error_pending && !source_mask_bad && !cos_mask_bad)
                        load_error_id <= ERR_LUT_MASK;
                end
            end
`ifndef SYNTHESIS
            if (arithmetic_request_fire) begin
                accepted_arithmetic_request_count <=
                    accepted_arithmetic_request_count + 1'b1;
            end
`endif
            case ({arithmetic_request_fire,
                   arithmetic_response_fire && arithmetic_pending_count != 0})
                2'b10: arithmetic_pending_count <= arithmetic_pending_count + 1'b1;
                2'b01: arithmetic_pending_count <= arithmetic_pending_count - 1'b1;
                default: begin end
            endcase
            if (destination_write_fire) begin
                destination_accepted <= 1'b1;
`ifndef SYNTHESIS
                accepted_destination_write_count <=
                    accepted_destination_write_count + 1'b1;
`endif
            end
            if (trace_sample_fire) begin
                trace_sample_accepted <= 1'b1;
`ifndef SYNTHESIS
                accepted_trace_sample_count <= accepted_trace_sample_count + 1'b1;
`endif
            end
            if (result_fifo_pop) begin
                destination_accepted <= 1'b0;
                trace_sample_accepted <= 1'b0;
`ifndef SYNTHESIS
                completed_chunk_count <= completed_chunk_count + 1'b1;
`endif
            end

            if ((state == RUN_PIPELINE || state == ERROR_QUIESCE) &&
                abort_request) begin
                terminal_abort <= 1'b1;
                terminal_error <= 1'b0;
                destination_accepted <= 1'b0;
                trace_sample_accepted <= 1'b0;
                quiesce_source_request <= 1'b0;
                quiesce_cos_request <= 1'b0;
                quiesce_sin_request <= 1'b0;
                quiesce_arithmetic_request <= 1'b0;
                quiesce_result_output <= 1'b0;
                state <= DRAIN_PIPELINE;
            end else begin
                case (state)
                    IDLE: begin
                        request_outstanding <= 1'b0;
                        source_sent <= 1'b0;
                        cos_sent <= 1'b0;
                        sin_sent <= 1'b0;
                        source_received <= 1'b0;
                        cos_received <= 1'b0;
                        sin_received <= 1'b0;
                        arithmetic_pending_count <= '0;
                        destination_accepted <= 1'b0;
                        trace_sample_accepted <= 1'b0;
                        terminal_abort <= 1'b0;
                        terminal_error <= 1'b0;
                        terminal_error_id <= ERR_NONE;
                        quiesce_source_request <= 1'b0;
                        quiesce_cos_request <= 1'b0;
                        quiesce_sin_request <= 1'b0;
                        quiesce_arithmetic_request <= 1'b0;
                        quiesce_result_output <= 1'b0;
                        if (abort_request) begin
                            abort_ack <= 1'b1;
                        end else if (start_valid && start_ready) begin
                            if (start_row_count == 0 || start_row_count > MAX_ROWS ||
                                start_head_count == 0 || start_head_count > MAX_HEADS ||
                                start_head_dim != HEAD_DIM ||
                                start_source_row_stride < start_head_count * HEAD_DIM ||
                                start_destination_row_stride < start_head_count * HEAD_DIM ||
                                start_cos_lut_base > {DMA_ADDR_WIDTH{1'b1}} - LUT_TABLE_BYTES ||
                                start_sin_lut_base > {DMA_ADDR_WIDTH{1'b1}} - LUT_TABLE_BYTES) begin
                                done_pulse <= 1'b1;
                                error <= 1'b1;
                                error_id <= ERR_CONFIG;
                            end else begin
                                saved_row_count <= start_row_count;
                                saved_head_count <= start_head_count;
                                saved_source_is_k <= start_source_is_k;
                                validation_position_shift <= start_token_position;
                                execution_position_shift <= start_token_position;
                                saved_source_row_stride <= start_source_row_stride;
                                saved_destination_row_stride <= start_destination_row_stride;
                                source_row_base <= start_source_base;
                                destination_row_base <= start_destination_base;
                                source_head_base <= start_source_base;
                                destination_head_base <= start_destination_base;
                                range_check_source_base <= 64'(start_source_base);
                                range_check_destination_base <= 64'(start_destination_base);
                                saved_cos_lut_base <= start_cos_lut_base;
                                saved_sin_lut_base <= start_sin_lut_base;
                                physical_row <= '0;
                                head_index <= '0;
                                pair_base <= '0;
                                range_check_row <= '0;
                                issue_exhausted <= 1'b0;
                                state <= CHECK_INPUT_RANGE;
                            end
                        end
                    end
                    CHECK_INPUT_RANGE: begin
                        if (abort_request) begin
                            abort_ack <= 1'b1;
                            state <= IDLE;
                        end else if (range_check_source_end > TENSOR_CAPACITY_WORDS ||
                                     range_check_destination_end > TENSOR_CAPACITY_WORDS) begin
                            done_pulse <= 1'b1;
                            error <= 1'b1;
                            error_id <= ERR_CONFIG;
                            state <= IDLE;
                        end else if (validation_position_shift[10:0] >=
                                     MAX_SEQUENCE_LENGTH) begin
                            done_pulse <= 1'b1;
                            error <= 1'b1;
                            error_id <= ERR_POSITION;
                            state <= IDLE;
                        end else if (range_check_row + 1'b1 == saved_row_count) begin
                            token_position <= execution_position_shift[10:0];
                            state <= RUN_PIPELINE;
                        end else begin
                            range_check_row <= range_check_row + 1'b1;
                            validation_position_shift <= validation_position_shift >> 11;
                            range_check_source_base <= range_check_source_base +
                                64'(saved_source_row_stride);
                            range_check_destination_base <= range_check_destination_base +
                                64'(saved_destination_row_stride);
                        end
                    end
                    RUN_PIPELINE: begin
                        if (request_outstanding && request_responses_complete &&
                            load_response_bad) begin
                            request_outstanding <= 1'b0;
                            terminal_error <= 1'b1;
                            terminal_error_id <= current_load_error_id;
                            quiesce_source_request <= source_read_req_valid &&
                                !source_request_fire;
                            quiesce_cos_request <= cos_read_req_valid &&
                                !cos_request_fire;
                            quiesce_sin_request <= sin_read_req_valid &&
                                !sin_request_fire;
                            quiesce_arithmetic_request <= adapter_req_valid &&
                                !arithmetic_request_fire;
                            quiesce_result_output <= result_fifo_output_valid &&
                                !result_fifo_pop;
                            state <= ERROR_QUIESCE;
                        end else if (arithmetic_response_fire &&
                                     arithmetic_response_bad) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERR_ARITHMETIC;
                            quiesce_source_request <= source_read_req_valid &&
                                !source_request_fire;
                            quiesce_cos_request <= cos_read_req_valid &&
                                !cos_request_fire;
                            quiesce_sin_request <= sin_read_req_valid &&
                                !sin_request_fire;
                            quiesce_arithmetic_request <= adapter_req_valid &&
                                !arithmetic_request_fire;
                            quiesce_result_output <= result_fifo_output_valid &&
                                !result_fifo_pop;
                            state <= ERROR_QUIESCE;
                        end else if (operand_fifo_push) begin
                            request_outstanding <= 1'b0;
                            source_sent <= 1'b0;
                            cos_sent <= 1'b0;
                            sin_sent <= 1'b0;
                            source_received <= 1'b0;
                            cos_received <= 1'b0;
                            sin_received <= 1'b0;
                            if (issue_meta.last_chunk) begin
                                issue_exhausted <= 1'b1;
                            end else if (pair_base + PAIR_LANES < HALF_DIM) begin
                                pair_base <= pair_base + PAIR_LANES;
                            end else if (head_index + 1'b1 < saved_head_count) begin
                                pair_base <= '0;
                                head_index <= head_index + 1'b1;
                                source_head_base <= source_head_base + HEAD_DIM;
                                destination_head_base <= destination_head_base + HEAD_DIM;
                            end else begin
                                pair_base <= '0;
                                head_index <= '0;
                                physical_row <= physical_row + 1'b1;
                                execution_position_shift <= execution_position_shift >> 11;
                                token_position <= execution_position_shift[21:11];
                                source_row_base <= source_row_base + saved_source_row_stride;
                                destination_row_base <= destination_row_base +
                                    saved_destination_row_stride;
                                source_head_base <= source_row_base + saved_source_row_stride;
                                destination_head_base <= destination_row_base +
                                    saved_destination_row_stride;
                            end
                        end
                        if (result_fifo_pop && result_meta.last_chunk) begin
                            done_pulse <= 1'b1;
`ifndef SYNTHESIS
                            completed_command_count <= completed_command_count + 1'b1;
`endif
                            state <= IDLE;
                        end
                    end
                    ERROR_QUIESCE: begin
                        if (source_request_fire)
                            quiesce_source_request <= 1'b0;
                        if (cos_request_fire)
                            quiesce_cos_request <= 1'b0;
                        if (sin_request_fire)
                            quiesce_sin_request <= 1'b0;
                        if (arithmetic_request_fire)
                            quiesce_arithmetic_request <= 1'b0;
                        if (result_fifo_pop)
                            quiesce_result_output <= 1'b0;
                        if (request_outstanding && request_responses_complete) begin
                            request_outstanding <= 1'b0;
                            source_sent <= 1'b0;
                            cos_sent <= 1'b0;
                            sin_sent <= 1'b0;
                            source_received <= 1'b0;
                            cos_received <= 1'b0;
                            sin_received <= 1'b0;
                        end
                        if ((!quiesce_source_request || source_request_fire) &&
                            (!quiesce_cos_request || cos_request_fire) &&
                            (!quiesce_sin_request || sin_request_fire) &&
                            (!quiesce_arithmetic_request ||
                                arithmetic_request_fire) &&
                            (!quiesce_result_output || result_fifo_pop)) begin
                            state <= DRAIN_PIPELINE;
                        end
                    end
                    DRAIN_PIPELINE: begin
                        if (request_outstanding && request_responses_complete) begin
                            request_outstanding <= 1'b0;
                            source_sent <= 1'b0;
                            cos_sent <= 1'b0;
                            sin_sent <= 1'b0;
                            source_received <= 1'b0;
                            cos_received <= 1'b0;
                            sin_received <= 1'b0;
                        end
                        if (request_drained && arithmetic_drained) begin
                            if (terminal_abort) begin
                                abort_ack <= 1'b1;
                            end else if (terminal_error) begin
                                done_pulse <= 1'b1;
                                error <= 1'b1;
                                error_id <= terminal_error_id;
                            end
                            state <= IDLE;
                        end
                    end
                    default: state <= IDLE;
                endcase
            end
        end
    end


    initial begin
        if (PAIR_LANES < 1 || (HEAD_DIM & 1) != 0 || HALF_DIM % PAIR_LANES != 0 ||
            MAX_ROWS < 1 || MAX_ROWS > 48 || MAX_HEADS < 1 || MAX_HEADS > 32 ||
            MAX_SEQUENCE_LENGTH < 1 || MAX_SEQUENCE_LENGTH > 2048 ||
            TENSOR_ADDR_WIDTH < 1 || TENSOR_CAPACITY_WORDS > (1 << TENSOR_ADDR_WIDTH) ||
            TAG_WIDTH < 16)
            $error("rope_engine parameter configuration is invalid");
    end

`ifndef SYNTHESIS
    logic source_stalled, cos_stalled, sin_stalled, destination_stalled, trace_sample_stalled;
    logic [1+2*PAIR_LANES+2*PAIR_LANES*TENSOR_ADDR_WIDTH-1:0] source_held;
    logic [DMA_ADDR_WIDTH+16-1:0] cos_held, sin_held;
    logic [1+2*PAIR_LANES+2*PAIR_LANES*TENSOR_ADDR_WIDTH+2*PAIR_LANES*16-1:0]
        destination_held;
    logic [6+6+7+11+2*PAIR_LANES*16+4*PAIR_LANES-1:0] trace_sample_held;
    always_ff @(posedge clk) begin
        if (rst) begin
            source_stalled <= 1'b0;
            cos_stalled <= 1'b0;
            sin_stalled <= 1'b0;
            destination_stalled <= 1'b0;
            trace_sample_stalled <= 1'b0;
            source_held <= '0;
            cos_held <= '0;
            sin_held <= '0;
            destination_held <= '0;
            trace_sample_held <= '0;
        end else begin
            if (source_stalled && !abort_request)
                assert (source_read_req_valid &&
                    {source_read_is_k, source_read_lane_mask, source_read_lane_address} ==
                    source_held)
                    else $error("rope_engine changed a stalled source request");
            if (cos_stalled && !abort_request)
                assert (cos_read_req_valid && {cos_read_req_address, cos_read_req_bytes} == cos_held)
                    else $error("rope_engine changed a stalled cosine request: valid=%0d address=%h bytes=%0d held_address=%h held_bytes=%0d",
                                cos_read_req_valid, cos_read_req_address, cos_read_req_bytes,
                                cos_held[DMA_ADDR_WIDTH+16-1:16], cos_held[15:0]);
            if (sin_stalled && !abort_request)
                assert (sin_read_req_valid && {sin_read_req_address, sin_read_req_bytes} == sin_held)
                    else $error("rope_engine changed a stalled sine request");
            if (destination_stalled && !abort_request)
                assert (destination_write_valid &&
                    {destination_write_is_k, destination_write_lane_mask,
                     destination_write_lane_address, destination_write_lane_data} == destination_held)
                    else $error("rope_engine changed a stalled destination write");
            if (trace_sample_stalled && !abort_request)
                assert (trace_sample_valid &&
                    {trace_sample_physical_row, trace_sample_head, trace_sample_pair_base,
                     trace_sample_token_position, trace_sample_data, trace_sample_byte_enable} ==
                    trace_sample_held)
                    else $error("rope_engine changed a stalled trace_sample");
            if (source_response_fire)
                assert (source_sent || source_request_fire)
                    else $error("rope_engine accepted a source response without a request");
            if (cos_response_fire)
                assert (cos_sent || cos_request_fire)
                    else $error("rope_engine accepted a cosine response without a request");
            if (sin_response_fire)
                assert (sin_sent || sin_request_fire)
                    else $error("rope_engine accepted a sine response without a request");

            source_stalled <= source_read_req_valid && !source_read_req_ready && !abort_request;
            cos_stalled <= cos_read_req_valid && !cos_read_req_ready && !abort_request;
            sin_stalled <= sin_read_req_valid && !sin_read_req_ready && !abort_request;
            destination_stalled <= destination_write_valid && !destination_write_ready &&
                !abort_request;
            trace_sample_stalled <= trace_sample_valid && !trace_sample_ready && !abort_request;
            source_held <= {source_read_is_k, source_read_lane_mask, source_read_lane_address};
            cos_held <= {cos_read_req_address, cos_read_req_bytes};
            sin_held <= {sin_read_req_address, sin_read_req_bytes};
            destination_held <= {destination_write_is_k, destination_write_lane_mask,
                                 destination_write_lane_address, destination_write_lane_data};
            trace_sample_held <= {trace_sample_physical_row, trace_sample_head,
                                trace_sample_pair_base, trace_sample_token_position,
                                trace_sample_data, trace_sample_byte_enable};
        end
    end
`endif
endmodule

`default_nettype wire
