`default_nettype none

// Runs RMSNorm in batches of eight physical rows. Address-to-bank mapping is
// owned by the fixed local-memory logical bundle.
module rmsnorm_engine #(
    parameter integer MAX_ELEMENTS = 4096,
    parameter integer MAX_ROWS = 48,
    parameter integer TAG_WIDTH = 16,
    parameter bit LOAD_GAMMA_FROM_DDR = 1'b0
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         abort_request,
    output logic                         abort_ack,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [5:0]                   start_row_count,
    input  logic [15:0]                  start_element_count,
    input  logic [15:0]                  start_epsilon_bf16,
    input  logic [63:0]                  start_gamma_ddr_address,
    input  logic                         start_gamma_bypass,

    output logic                         tile_read_req_valid,
    input  logic                         tile_read_req_ready,
    output logic                         tile_read_output_pass,
    output logic                         tile_read_gamma_bypass,
    output logic [5:0]                   tile_read_row_base,
    output logic [15:0]                  tile_read_element,
    output logic [TAG_WIDTH-1:0]         tile_read_tag,
    input  logic                         tile_read_rsp_valid,
    output logic                         tile_read_rsp_ready,
    input  logic [1023:0]                tile_read_values,
    input  logic [127:0]                 tile_read_gamma,
    input  logic [63:0]                  tile_read_lane_mask,
    input  logic [TAG_WIDTH-1:0]         tile_read_rsp_tag,

    output logic                         gamma_dma_request_valid,
    input  logic                         gamma_dma_request_ready,
    output logic [63:0]                  gamma_dma_request_address,
    output logic [31:0]                  gamma_dma_request_bytes,
    input  logic                         gamma_dma_request_done,
    input  logic                         gamma_dma_request_error,
    input  logic                         gamma_dma_data_valid,
    output logic                         gamma_dma_data_ready,
    input  logic [127:0]                 gamma_dma_data,
    input  logic [15:0]                  gamma_dma_byte_enable,
    output logic                         gamma_stage_write_valid,
    input  logic                         gamma_stage_write_ready,
    output logic [9:0]                   gamma_stage_write_word,
    output logic [127:0]                 gamma_stage_write_data,
    output logic [15:0]                  gamma_stage_write_byte_enable,

    output logic                         scratch_read_valid,
    input  logic                         scratch_read_ready,
    output logic                         scratch_read_bank,
    output logic [9:0]                   scratch_read_left_address,
    output logic [9:0]                   scratch_read_right_address,
    output logic [TAG_WIDTH-1:0]         scratch_read_tag,
    input  logic                         scratch_read_rsp_valid,
    output logic                         scratch_read_rsp_ready,
    input  logic [127:0]                 scratch_read_left_data,
    input  logic [127:0]                 scratch_read_right_data,
    input  logic [TAG_WIDTH-1:0]         scratch_read_rsp_tag,
    output logic                         scratch_write_valid,
    input  logic                         scratch_write_ready,
    output logic                         scratch_write_bank,
    output logic [9:0]                   scratch_write_address,
    output logic [127:0]                 scratch_write_data,
    output logic [15:0]                  scratch_write_byte_enable,

    output logic                         norm_write_valid,
    input  logic                         norm_write_ready,
    output logic [5:0]                   norm_write_row_base,
    output logic [15:0]                  norm_write_element,
    output logic [1023:0]                norm_write_data,
    output logic [63:0]                  norm_write_lane_mask,
    output logic [TAG_WIDTH-1:0]         norm_write_tag,
    output logic                         trace_sample_valid,
    input  logic                         trace_sample_ready,
    output logic [5:0]                   trace_sample_row_base,
    output logic [15:0]                  trace_sample_element,
    output logic [1023:0]                trace_sample_data,
    output logic [127:0]                 trace_sample_byte_enable,

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

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic                         error,
    output logic [3:0]                   error_id,
    output logic                         gamma_loaded,
    output logic                         execution_active,
    output logic [63:0]                  accepted_tile_read_count,
    output logic [63:0]                  completed_tile_read_count,
    output logic [63:0]                  accepted_norm_write_count,
    output logic [63:0]                  completed_row_count,
    output logic [63:0]                  completed_command_count
);
`ifdef SYNTHESIS
    always_comb begin
        accepted_tile_read_count = '0;
        completed_tile_read_count = '0;
        accepted_norm_write_count = '0;
        completed_row_count = '0;
        completed_command_count = '0;
    end
`endif
    localparam logic [3:0] ERR_NONE = 4'd0;
    localparam logic [3:0] ERR_CONFIG = 4'd1;
    localparam logic [3:0] ERR_PIPELINE = 4'd2;
    localparam logic [3:0] ERR_GAMMA_DMA = 4'd3;

    typedef enum logic [2:0] {
        IDLE, GAMMA_REQUEST, GAMMA_STREAM, START_BATCH, RUN_BATCH, COMPLETE,
        ABORT_DRAIN
    } state_t;
    state_t state;

    logic [5:0] saved_row_count;
    logic [15:0] saved_element_count;
    logic [15:0] saved_epsilon;
    logic [63:0] saved_gamma_ddr_address;
    logic saved_gamma_bypass;
    logic [5:0] current_row_base;
    logic [3:0] current_batch_rows;
    logic [9:0] gamma_word_count;
    logic gamma_request_active;
    logic abort_seen;

    logic pipeline_start_valid;
    logic pipeline_start_ready;
    logic pipeline_source_request_valid;
    logic pipeline_source_request_ready;
    logic pipeline_source_output_pass;
    logic [15:0] pipeline_source_element;
    logic [TAG_WIDTH-1:0] pipeline_source_tag;
    logic pipeline_source_response_valid;
    logic pipeline_source_response_ready;
    logic pipeline_output_valid;
    logic pipeline_output_ready;
    logic [15:0] pipeline_output_element;
    logic [1023:0] pipeline_output_values;
    logic [63:0] pipeline_output_mask;
    logic [TAG_WIDTH-1:0] pipeline_output_tag;
    logic pipeline_done_pulse;
    logic pipeline_error;
    logic pipeline_reset;
    logic pipeline_scratch_read_rsp_ready;
    logic pipeline_vector_rsp_ready;
    logic pipeline_reduction_rsp_ready;

    logic [15:0] tile_read_outstanding;
    logic [15:0] scratch_read_outstanding;
    logic [15:0] vector_outstanding;
    logic [15:0] reduction_outstanding;
    logic tile_read_request_fire;
    logic tile_read_response_fire;
    logic scratch_read_request_fire;
    logic scratch_read_response_fire;
    logic vector_request_fire;
    logic vector_response_fire;
    logic reduction_request_fire;
    logic reduction_response_fire;
    logic norm_output_fire;
    logic abort_drained;

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE;
    assign execution_active = state != IDLE || tile_read_outstanding != 0 ||
        scratch_read_outstanding != 0 || vector_outstanding != 0 ||
        reduction_outstanding != 0 || gamma_request_active;
    assign pipeline_reset = rst || state == ABORT_DRAIN;
    assign pipeline_start_valid = state == START_BATCH;
    assign current_batch_rows = saved_row_count - current_row_base >= 8 ? 4'd8 :
        4'(saved_row_count - current_row_base);

    assign gamma_dma_request_valid = LOAD_GAMMA_FROM_DDR && state == GAMMA_REQUEST;
    assign gamma_dma_request_address = saved_gamma_ddr_address;
    assign gamma_dma_request_bytes = {15'd0, saved_element_count, 1'b0};
    assign gamma_stage_write_valid = LOAD_GAMMA_FROM_DDR && state == GAMMA_STREAM &&
        gamma_dma_data_valid && !abort_request;
    assign gamma_dma_data_ready = state == ABORT_DRAIN ? gamma_request_active :
        gamma_stage_write_valid && gamma_stage_write_ready;
    assign gamma_stage_write_word = gamma_word_count;
    assign gamma_stage_write_data = gamma_dma_data;
    assign gamma_stage_write_byte_enable = gamma_dma_byte_enable;

    assign tile_read_req_valid = state == RUN_BATCH && pipeline_source_request_valid;
    assign pipeline_source_request_ready = state == RUN_BATCH && tile_read_req_ready;
    assign tile_read_output_pass = pipeline_source_output_pass;
    assign tile_read_gamma_bypass = saved_gamma_bypass;
    assign tile_read_row_base = current_row_base;
    assign tile_read_element = pipeline_source_element;
    assign tile_read_tag = pipeline_source_tag;
    assign pipeline_source_response_valid = state == RUN_BATCH && tile_read_rsp_valid;
    assign tile_read_rsp_ready = state == ABORT_DRAIN ? tile_read_outstanding != 0 :
        state == RUN_BATCH && pipeline_source_response_ready;
    assign scratch_read_rsp_ready = state == ABORT_DRAIN ?
        scratch_read_outstanding != 0 : pipeline_scratch_read_rsp_ready;
    assign vector_rsp_ready = state == ABORT_DRAIN ?
        vector_outstanding != 0 : pipeline_vector_rsp_ready;
    assign reduction_rsp_ready = state == ABORT_DRAIN ?
        reduction_outstanding != 0 : pipeline_reduction_rsp_ready;
    assign tile_read_request_fire = tile_read_req_valid && tile_read_req_ready;
    assign tile_read_response_fire = tile_read_rsp_valid && tile_read_rsp_ready;

    assign norm_write_valid = state == RUN_BATCH && pipeline_output_valid && trace_sample_ready;
    assign trace_sample_valid = state == RUN_BATCH && pipeline_output_valid && norm_write_ready;
    assign pipeline_output_ready = state == RUN_BATCH && norm_write_ready && trace_sample_ready;
    assign norm_write_row_base = current_row_base;
    assign norm_write_element = pipeline_output_element;
    assign norm_write_data = pipeline_output_values;
    assign norm_write_lane_mask = pipeline_output_mask;
    assign norm_write_tag = pipeline_output_tag;
    assign trace_sample_row_base = current_row_base;
    assign trace_sample_element = pipeline_output_element;
`ifdef SYNTHESIS
    assign trace_sample_data = '0;
    assign trace_sample_byte_enable = '0;
`else
    assign trace_sample_data = pipeline_output_values;
    for (genvar trace_sample_lane = 0; trace_sample_lane < 64;
         trace_sample_lane = trace_sample_lane + 1) begin : g_trace_sample_enable
        assign trace_sample_byte_enable[trace_sample_lane*2 +: 2] =
            {2{pipeline_output_mask[trace_sample_lane]}};
    end
`endif
    assign norm_output_fire = pipeline_output_valid && pipeline_output_ready;

    assign scratch_read_request_fire = scratch_read_valid && scratch_read_ready;
    assign scratch_read_response_fire = scratch_read_rsp_valid && scratch_read_rsp_ready;
    assign vector_request_fire = vector_req_valid && vector_req_ready;
    assign vector_response_fire = vector_rsp_valid && vector_rsp_ready;
    assign reduction_request_fire = reduction_req_valid && reduction_req_ready;
    assign reduction_response_fire = reduction_rsp_valid && reduction_rsp_ready;
    assign abort_drained = tile_read_outstanding == 0 && scratch_read_outstanding == 0 &&
        vector_outstanding == 0 && reduction_outstanding == 0 && !gamma_request_active;

    rmsnorm_row_pipeline #(.MAX_ELEMENTS(MAX_ELEMENTS), .TAG_WIDTH(TAG_WIDTH)) pipeline (
        .clk(clk), .rst(pipeline_reset), .start_valid(pipeline_start_valid),
        .start_ready(pipeline_start_ready), .row_count(current_batch_rows),
        .element_count(saved_element_count), .epsilon_bf16(saved_epsilon),
        .gamma_bypass(saved_gamma_bypass),
        .source_request_valid(pipeline_source_request_valid),
        .source_request_ready(pipeline_source_request_ready),
        .source_request_output_pass(pipeline_source_output_pass),
        .source_request_element(pipeline_source_element),
        .source_request_tag(pipeline_source_tag),
        .source_response_valid(pipeline_source_response_valid),
        .source_response_ready(pipeline_source_response_ready),
        .source_values(tile_read_values), .source_gamma(tile_read_gamma),
        .source_lane_mask(tile_read_lane_mask), .source_response_tag(tile_read_rsp_tag),
        .scratch_read_valid(scratch_read_valid), .scratch_read_ready(scratch_read_ready),
        .scratch_read_bank(scratch_read_bank),
        .scratch_read_left_address(scratch_read_left_address),
        .scratch_read_right_address(scratch_read_right_address),
        .scratch_read_tag(scratch_read_tag),
        .scratch_read_response_valid(state == RUN_BATCH && scratch_read_rsp_valid),
        .scratch_read_response_ready(pipeline_scratch_read_rsp_ready),
        .scratch_read_left_data(scratch_read_left_data),
        .scratch_read_right_data(scratch_read_right_data),
        .scratch_read_response_tag(scratch_read_rsp_tag),
        .scratch_write_valid(scratch_write_valid),
        .scratch_write_ready(scratch_write_ready), .scratch_write_bank(scratch_write_bank),
        .scratch_write_address(scratch_write_address),
        .scratch_write_data(scratch_write_data),
        .scratch_write_byte_enable(scratch_write_byte_enable),
        .output_valid(pipeline_output_valid), .output_ready(pipeline_output_ready),
        .output_element(pipeline_output_element), .output_values(pipeline_output_values),
        .output_lane_mask(pipeline_output_mask), .output_tag(pipeline_output_tag),
        .vector_req_valid(vector_req_valid), .vector_req_ready(vector_req_ready),
        .vector_req_operation(vector_req_operation), .vector_req_values(vector_req_values),
        .vector_req_paired_values(vector_req_paired_values),
        .vector_req_factor0_values(vector_req_factor0_values),
        .vector_req_factor1_values(vector_req_factor1_values),
        .vector_req_lane_mask(vector_req_lane_mask), .vector_req_tag(vector_req_tag),
        .vector_rsp_valid(state == RUN_BATCH && vector_rsp_valid),
        .vector_rsp_ready(pipeline_vector_rsp_ready), .vector_rsp_values(vector_rsp_values),
        .vector_rsp_lane_mask(vector_rsp_lane_mask), .vector_rsp_tag(vector_rsp_tag),
        .reduction_req_valid(reduction_req_valid),
        .reduction_req_ready(reduction_req_ready),
        .reduction_req_values(reduction_req_values),
        .reduction_req_lane_mask(reduction_req_lane_mask),
        .reduction_req_tag(reduction_req_tag),
        .reduction_rsp_valid(state == RUN_BATCH && reduction_rsp_valid),
        .reduction_rsp_ready(pipeline_reduction_rsp_ready),
        .reduction_rsp_values(reduction_rsp_values),
        .reduction_rsp_row_mask(reduction_rsp_row_mask),
        .reduction_rsp_tag(reduction_rsp_tag),
        .done_pulse(pipeline_done_pulse),
        .error(pipeline_error), .accepted_source_tile_count(),
        .accepted_output_tile_count(), .accepted_scratch_read_count(),
        .accepted_scratch_write_count(), .accepted_vector_request_count(),
        .accepted_reduction_request_count());

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            saved_row_count <= '0;
            saved_element_count <= '0;
            saved_epsilon <= '0;
            saved_gamma_ddr_address <= '0;
            saved_gamma_bypass <= 1'b0;
            current_row_base <= '0;
            gamma_word_count <= '0;
            gamma_request_active <= 1'b0;
            abort_seen <= 1'b0;
            abort_ack <= 1'b0;
            tile_read_outstanding <= '0;
            scratch_read_outstanding <= '0;
            vector_outstanding <= '0;
            reduction_outstanding <= '0;
            error <= 1'b0;
            error_id <= ERR_NONE;
            gamma_loaded <= 1'b0;
`ifndef SYNTHESIS
            accepted_tile_read_count <= '0;
            completed_tile_read_count <= '0;
            accepted_norm_write_count <= '0;
            completed_row_count <= '0;
            completed_command_count <= '0;
`endif
        end else begin
            abort_ack <= 1'b0;
            if (!abort_request)
                abort_seen <= 1'b0;

            case ({tile_read_request_fire, tile_read_response_fire})
                2'b10: tile_read_outstanding <= tile_read_outstanding + 1'b1;
                2'b01: tile_read_outstanding <= tile_read_outstanding - 1'b1;
                default: begin end
            endcase
            case ({scratch_read_request_fire, scratch_read_response_fire})
                2'b10: scratch_read_outstanding <= scratch_read_outstanding + 1'b1;
                2'b01: scratch_read_outstanding <= scratch_read_outstanding - 1'b1;
                default: begin end
            endcase
            case ({vector_request_fire, vector_response_fire})
                2'b10: vector_outstanding <= vector_outstanding + 1'b1;
                2'b01: vector_outstanding <= vector_outstanding - 1'b1;
                default: begin end
            endcase
            case ({reduction_request_fire, reduction_response_fire})
                2'b10: reduction_outstanding <= reduction_outstanding + 1'b1;
                2'b01: reduction_outstanding <= reduction_outstanding - 1'b1;
                default: begin end
            endcase
`ifndef SYNTHESIS
            if (tile_read_request_fire)
                accepted_tile_read_count <= accepted_tile_read_count + 1'b1;
            if (tile_read_response_fire)
                completed_tile_read_count <= completed_tile_read_count + 1'b1;
            if (norm_output_fire)
                accepted_norm_write_count <= accepted_norm_write_count + 1'b1;
`endif

            if (abort_request && state != IDLE && state != COMPLETE &&
                state != ABORT_DRAIN)
                state <= ABORT_DRAIN;
            else begin
                case (state)
                    IDLE: if (start_valid && start_ready) begin
                        error <= 1'b0;
                        error_id <= ERR_NONE;
                        gamma_loaded <= start_gamma_bypass || !LOAD_GAMMA_FROM_DDR;
                        if (start_row_count == 0 || start_row_count > MAX_ROWS ||
                            start_element_count == 0 || start_element_count > MAX_ELEMENTS) begin
                            error <= 1'b1;
                            error_id <= ERR_CONFIG;
                            state <= COMPLETE;
                        end else begin
                            saved_row_count <= start_row_count;
                            saved_element_count <= start_element_count;
                            saved_epsilon <= start_epsilon_bf16;
                            saved_gamma_ddr_address <= start_gamma_ddr_address;
                            saved_gamma_bypass <= start_gamma_bypass;
                            current_row_base <= '0;
                            gamma_word_count <= '0;
                            state <= LOAD_GAMMA_FROM_DDR && !start_gamma_bypass ?
                                GAMMA_REQUEST : START_BATCH;
                        end
                    end
                    GAMMA_REQUEST: if (gamma_dma_request_valid && gamma_dma_request_ready) begin
                        gamma_request_active <= 1'b1;
                        state <= GAMMA_STREAM;
                    end
                    GAMMA_STREAM: begin
                        if (gamma_dma_data_valid && gamma_dma_data_ready)
                            gamma_word_count <= gamma_word_count + 1'b1;
                        if (gamma_dma_request_done || gamma_dma_request_error) begin
                            gamma_request_active <= 1'b0;
                            if (gamma_dma_request_error || gamma_word_count +
                                (gamma_dma_data_valid && gamma_dma_data_ready) !=
                                ((saved_element_count + 7) >> 3)) begin
                                error <= 1'b1;
                                error_id <= ERR_GAMMA_DMA;
                                state <= COMPLETE;
                            end else begin
                                gamma_loaded <= 1'b1;
                                state <= START_BATCH;
                            end
                        end
                    end
                    START_BATCH: if (pipeline_start_ready)
                        state <= RUN_BATCH;
                    RUN_BATCH: begin
                        if (pipeline_error) begin
                            error <= 1'b1;
                            error_id <= ERR_PIPELINE;
                            state <= COMPLETE;
                        end else if (pipeline_done_pulse) begin
`ifndef SYNTHESIS
                            completed_row_count <= completed_row_count + current_batch_rows;
`endif
                            if (current_row_base + current_batch_rows == saved_row_count) begin
`ifndef SYNTHESIS
                                completed_command_count <= completed_command_count + 1'b1;
`endif
                                state <= COMPLETE;
                            end else begin
                                current_row_base <= current_row_base + current_batch_rows;
                                state <= START_BATCH;
                            end
                        end
                    end
                    COMPLETE: if (done_ready)
                        state <= IDLE;
                    ABORT_DRAIN: begin
                        if (gamma_request_active &&
                            (gamma_dma_request_done || gamma_dma_request_error))
                            gamma_request_active <= 1'b0;
                        if (abort_drained && !abort_seen) begin
                            abort_ack <= 1'b1;
                            abort_seen <= 1'b1;
                            state <= IDLE;
                        end
                    end
                    default: state <= IDLE;
                endcase
            end
        end
    end

    initial begin
        if (MAX_ELEMENTS < 8 || MAX_ELEMENTS > 4096 || MAX_ROWS < 1 ||
            MAX_ROWS > 48 || TAG_WIDTH < 10)
            $error("rmsnorm_engine parameter configuration is invalid");
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (completed_tile_read_count <= accepted_tile_read_count)
                else $error("rmsnorm_engine completed an unrequested tile read");
            if (state == START_BATCH)
                assert (tile_read_outstanding == 0 && scratch_read_outstanding == 0 &&
                    vector_outstanding == 0 && reduction_outstanding == 0)
                    else $error("rmsnorm_engine started a row batch before drain");
            if (pipeline_done_pulse)
                assert (state == RUN_BATCH)
                    else $error("RMSNorm row completion pulse had no active parent wait state");
            if ($past(done_valid && !done_ready))
                assert (done_valid && $stable({error, error_id}))
                    else $error("rmsnorm_engine changed a stalled completion response");
        end
    end
`endif
endmodule

`default_nettype wire
