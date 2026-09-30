`default_nettype none

// Iterates Softmax heads and 8-row batches. The row pipeline owns the four
// numeric passes; this module only adds head/row addressing and command
// completion. Score/probability SRAM and shared arithmetic remain external.
module softmax_engine #(
    parameter integer MAX_ROWS = 48,
    parameter integer MAX_HEADS = 32,
    parameter integer MAX_LENGTH = 2048,
    parameter integer ADDR_WIDTH = 24,
    parameter integer TAG_WIDTH = 16
) (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    output logic          abort_ack,
    input  logic          start_valid,
    output logic          start_ready,
    input  logic [5:0]    start_row_count,
    input  logic [5:0]    start_head_count,
    input  logic [11:0]   start_row_length,
    input  logic [ADDR_WIDTH-1:0] start_score_base,
    input  logic [ADDR_WIDTH-1:0] start_score_head_stride,
    input  logic [ADDR_WIDTH-1:0] start_score_row_stride,
    input  logic [ADDR_WIDTH-1:0] start_probability_base,
    input  logic [ADDR_WIDTH-1:0] start_probability_head_stride,
    input  logic [ADDR_WIDTH-1:0] start_probability_row_stride,
    input  logic          start_capture_probability_enable,
    output logic          capture_probability_valid,
    input  logic          capture_probability_ready,
    output logic [5:0]    capture_probability_head,
    output logic [5:0]    capture_probability_row_base,
    output logic [7:0]    capture_probability_row_mask,
    output logic          busy,
    output logic          done_pulse,
    output logic          error,
    output logic [3:0]    error_id,

    output logic          tile_read_req_valid,
    input  logic          tile_read_req_ready,
    output logic [1:0]    tile_read_pass,
    output logic [ADDR_WIDTH-1:0] tile_read_group_base,
    output logic [5:0]    tile_read_head,
    output logic [5:0]    tile_read_row_base,
    output logic [11:0]   tile_read_key,
    output logic [15:0]   tile_read_tag,
    input  logic          tile_read_rsp_valid,
    output logic          tile_read_rsp_ready,
    input  logic [1023:0] tile_read_rsp_values,
    input  logic [63:0]   tile_read_rsp_lane_mask,
    input  logic [15:0]   tile_read_rsp_tag,

    output logic          bf16_write_valid,
    input  logic          bf16_write_ready,
    output logic          bf16_write_probability,
    output logic [ADDR_WIDTH-1:0] bf16_write_group_base,
    output logic [5:0]    bf16_write_head,
    output logic [5:0]    bf16_write_row_base,
    output logic [11:0]   bf16_write_key,
    output logic [1023:0] bf16_write_values,
    output logic [63:0]   bf16_write_lane_mask,
    output logic [15:0]   bf16_write_tag,
    output logic          quantized_write_valid,
    input  logic          quantized_write_ready,
    output logic [5:0]    quantized_write_head,
    output logic [5:0]    quantized_write_row_base,
    output logic [11:0]   quantized_write_key,
    output logic [511:0]  quantized_write_values,
    output logic [63:0]   quantized_write_lane_mask,
    output logic [15:0]   quantized_write_tag,
    output logic          scale_write_valid,
    input  logic          scale_write_ready,
    output logic [5:0]    scale_write_head,
    output logic [5:0]    scale_write_row_base,
    output logic [127:0]  scale_write_values,
    output logic [7:0]    scale_write_row_mask,

    output logic          scratch_read_valid,
    input  logic          scratch_read_ready,
    output logic          scratch_read_bank,
    output logic [9:0]    scratch_read_left_address,
    output logic [9:0]    scratch_read_right_address,
    output logic [15:0]   scratch_read_tag,
    input  logic          scratch_rsp_valid,
    output logic          scratch_rsp_ready,
    input  logic [127:0]  scratch_rsp_left_data,
    input  logic [127:0]  scratch_rsp_right_data,
    input  logic [15:0]   scratch_rsp_tag,
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
    output logic [15:0]   vector_req_tag,
    input  logic          vector_rsp_valid,
    output logic          vector_rsp_ready,
    input  logic [1023:0] vector_rsp_values,
    input  logic [63:0]   vector_rsp_lane_mask,
    input  logic [15:0]   vector_rsp_tag,
    output logic          max_req_valid,
    input  logic          max_req_ready,
    output logic          max_req_magnitude,
    output logic [1023:0] max_req_values,
    output logic [63:0]   max_req_lane_mask,
    output logic [15:0]   max_req_tag,
    input  logic          max_rsp_valid,
    output logic          max_rsp_ready,
    input  logic [127:0]  max_rsp_values,
    input  logic [7:0]    max_rsp_row_mask,
    input  logic [15:0]   max_rsp_tag,
    output logic          reduction_req_valid,
    input  logic          reduction_req_ready,
    output logic [1023:0] reduction_req_values,
    output logic [63:0]   reduction_req_lane_mask,
    output logic [15:0]   reduction_req_tag,
    input  logic          reduction_rsp_valid,
    output logic          reduction_rsp_ready,
    input  logic [127:0]  reduction_rsp_values,
    input  logic [7:0]    reduction_rsp_row_mask,
    input  logic [15:0]   reduction_rsp_tag,
    output logic          exp_req_valid,
    input  logic          exp_req_ready,
    output logic [1023:0] exp_req_delta_values,
    output logic [63:0]   exp_req_lane_mask,
    output logic [15:0]   exp_req_tag,
    input  logic          exp_rsp_valid,
    output logic          exp_rsp_ready,
    input  logic [1023:0] exp_rsp_values,
    input  logic [63:0]   exp_rsp_lane_mask,
    input  logic [15:0]   exp_rsp_tag,
    output logic          reciprocal_req_valid,
    input  logic          reciprocal_req_ready,
    output logic [127:0]  reciprocal_req_sum_values,
    output logic [7:0]    reciprocal_req_row_mask,
    output logic [15:0]   reciprocal_req_tag,
    input  logic          reciprocal_rsp_valid,
    output logic          reciprocal_rsp_ready,
    input  logic [127:0]  reciprocal_rsp_values,
    input  logic [7:0]    reciprocal_rsp_row_mask,
    input  logic [15:0]   reciprocal_rsp_tag,
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
    output logic [15:0]   quant_values_req_tag,
    input  logic          quant_values_rsp_valid,
    output logic          quant_values_rsp_ready,
    input  logic [511:0]  quant_values_rsp_values,
    input  logic [63:0]   quant_values_rsp_lane_mask,
    input  logic [15:0]   quant_values_rsp_tag,

    output logic [63:0]   accepted_source_tile_count,
    output logic [63:0]   accepted_bf16_write_count,
    output logic [63:0]   accepted_quantized_write_count,
    output logic [63:0]   accepted_scratch_read_count,
    output logic [63:0]   accepted_scratch_write_count,
    output logic [63:0]   accepted_vector_request_count,
    output logic [63:0]   accepted_max_request_count,
    output logic [63:0]   accepted_reduction_request_count,
    output logic [63:0]   accepted_quantized_value_count,
    output logic [63:0]   completed_command_count
);
`ifdef SYNTHESIS
    always_comb completed_command_count = '0;
`endif
    localparam logic [3:0] ERROR_CONFIGURATION = 4'h1;
    localparam logic [3:0] ERROR_PIPELINE = 4'h2;
    typedef enum logic [2:0] {
        IDLE, START_BATCH, WAIT_BATCH, COMPLETE, ABORT_WAIT, ABORT_WAIT_LOW
    } state_t;
    state_t state;

    logic [5:0] row_count;
    logic [5:0] head_count;
    logic [11:0] row_length;
    logic [ADDR_WIDTH-1:0] score_head_stride, score_row_stride;
    logic [ADDR_WIDTH-1:0] probability_head_stride;
    logic [ADDR_WIDTH-1:0] probability_row_stride;
    logic [ADDR_WIDTH-1:0] current_score_head_base;
    logic [ADDR_WIDTH-1:0] current_probability_head_base;
    logic [ADDR_WIDTH-1:0] current_score_batch_base;
    logic [ADDR_WIDTH-1:0] current_probability_batch_base;
    logic [5:0] current_head;
    logic [5:0] current_row_base;
    logic [3:0] current_batch_rows;
    logic pipeline_start_valid, pipeline_start_ready;
    logic pipeline_done_pulse, pipeline_error, pipeline_abort_ack;
    logic [11:0] pipeline_write_key;
    logic capture_probability_enable;

    wire configuration_valid = start_row_count != 0 &&
        start_row_count <= 6'(MAX_ROWS) && start_head_count != 0 &&
        start_head_count <= 6'(MAX_HEADS) && start_row_length != 0 &&
        start_row_length <= 12'(MAX_LENGTH);
    wire start_fire = start_valid && start_ready;
    wire final_batch = current_row_base + 6'(current_batch_rows) >= row_count;
    wire final_head = current_head + 1'b1 >= head_count;

    assign start_ready = state == IDLE && !abort_request;
    // The parent holds abort until every child has drained. After the row
    // pipeline acknowledges, no work remains; start_ready still waits for IDLE
    // and abort deassertion before another command can be accepted.
    assign busy = state != IDLE && state != ABORT_WAIT_LOW;
    assign pipeline_start_valid = state == START_BATCH && !abort_request;
    assign tile_read_head = current_head;
    assign tile_read_row_base = current_row_base;
    assign tile_read_group_base = tile_read_pass < 2 ?
        current_score_batch_base + ADDR_WIDTH'({tile_read_key, 1'b0}) :
        current_probability_batch_base + ADDR_WIDTH'({tile_read_key, 1'b0});
    assign bf16_write_head = current_head;
    assign bf16_write_row_base = current_row_base;
    assign bf16_write_group_base = current_probability_batch_base +
        ADDR_WIDTH'({pipeline_write_key, 1'b0});
    assign bf16_write_key = pipeline_write_key;
    assign quantized_write_head = current_head;
    assign quantized_write_row_base = current_row_base;
    assign scale_write_head = current_head;
    assign scale_write_row_base = current_row_base;
    assign capture_probability_head = current_head;
    assign capture_probability_row_base = current_row_base;
    assign capture_probability_row_mask = 8'hff >> (8-current_batch_rows);

    softmax_row_pipeline row_pipeline (
        .capture_probability_enable, .capture_probability_valid, .capture_probability_ready,
        .clk(clk), .rst(rst), .abort_request(abort_request),
        .abort_ack(pipeline_abort_ack), .start_valid(pipeline_start_valid),
        .start_ready(pipeline_start_ready), .row_count(current_batch_rows),
        .row_length(row_length), .source_req_valid(tile_read_req_valid),
        .source_req_ready(tile_read_req_ready), .source_req_pass(tile_read_pass),
        .source_req_key(tile_read_key), .source_req_tag(tile_read_tag),
        .source_rsp_valid(tile_read_rsp_valid),
        .source_rsp_ready(tile_read_rsp_ready),
        .source_rsp_values(tile_read_rsp_values),
        .source_rsp_lane_mask(tile_read_rsp_lane_mask),
        .source_rsp_tag(tile_read_rsp_tag), .bf16_write_valid(bf16_write_valid),
        .bf16_write_ready(bf16_write_ready),
        .bf16_write_probability(bf16_write_probability),
        .bf16_write_key(pipeline_write_key), .bf16_write_values(bf16_write_values),
        .bf16_write_lane_mask(bf16_write_lane_mask),
        .bf16_write_tag(bf16_write_tag), .quantized_write_valid(quantized_write_valid),
        .quantized_write_ready(quantized_write_ready), .quantized_write_key(quantized_write_key),
        .quantized_write_values(quantized_write_values),
        .quantized_write_lane_mask(quantized_write_lane_mask), .quantized_write_tag(quantized_write_tag),
        .scale_write_valid(scale_write_valid), .scale_write_ready(scale_write_ready),
        .scale_write_values(scale_write_values),
        .scale_write_row_mask(scale_write_row_mask),
        .scratch_read_valid(scratch_read_valid),
        .scratch_read_ready(scratch_read_ready), .scratch_read_bank(scratch_read_bank),
        .scratch_read_left_address(scratch_read_left_address),
        .scratch_read_right_address(scratch_read_right_address),
        .scratch_read_tag(scratch_read_tag), .scratch_rsp_valid(scratch_rsp_valid),
        .scratch_rsp_ready(scratch_rsp_ready),
        .scratch_rsp_left_data(scratch_rsp_left_data),
        .scratch_rsp_right_data(scratch_rsp_right_data),
        .scratch_rsp_tag(scratch_rsp_tag),
        .scratch_write_valid(scratch_write_valid),
        .scratch_write_ready(scratch_write_ready),
        .scratch_write_bank(scratch_write_bank),
        .scratch_write_address(scratch_write_address),
        .scratch_write_data(scratch_write_data),
        .scratch_write_byte_enable(scratch_write_byte_enable),
        .vector_req_valid(vector_req_valid), .vector_req_ready(vector_req_ready),
        .vector_req_operation(vector_req_operation),
        .vector_req_values(vector_req_values),
        .vector_req_paired_values(vector_req_paired_values),
        .vector_req_factor0_values(vector_req_factor0_values),
        .vector_req_factor1_values(vector_req_factor1_values),
        .vector_req_lane_mask(vector_req_lane_mask), .vector_req_tag(vector_req_tag),
        .vector_rsp_valid(vector_rsp_valid), .vector_rsp_ready(vector_rsp_ready),
        .vector_rsp_values(vector_rsp_values),
        .vector_rsp_lane_mask(vector_rsp_lane_mask), .vector_rsp_tag(vector_rsp_tag),
        .max_req_valid(max_req_valid), .max_req_ready(max_req_ready),
        .max_req_magnitude(max_req_magnitude), .max_req_values(max_req_values),
        .max_req_lane_mask(max_req_lane_mask), .max_req_tag(max_req_tag),
        .max_rsp_valid(max_rsp_valid), .max_rsp_ready(max_rsp_ready),
        .max_rsp_values(max_rsp_values), .max_rsp_row_mask(max_rsp_row_mask),
        .max_rsp_tag(max_rsp_tag), .reduction_req_valid(reduction_req_valid),
        .reduction_req_ready(reduction_req_ready),
        .reduction_req_values(reduction_req_values),
        .reduction_req_lane_mask(reduction_req_lane_mask),
        .reduction_req_tag(reduction_req_tag),
        .reduction_rsp_valid(reduction_rsp_valid),
        .reduction_rsp_ready(reduction_rsp_ready),
        .reduction_rsp_values(reduction_rsp_values),
        .reduction_rsp_row_mask(reduction_rsp_row_mask),
        .reduction_rsp_tag(reduction_rsp_tag),
        .exp_req_valid(exp_req_valid), .exp_req_ready(exp_req_ready),
        .exp_req_delta_values(exp_req_delta_values),
        .exp_req_lane_mask(exp_req_lane_mask), .exp_req_tag(exp_req_tag),
        .exp_rsp_valid(exp_rsp_valid), .exp_rsp_ready(exp_rsp_ready),
        .exp_rsp_values(exp_rsp_values),
        .exp_rsp_lane_mask(exp_rsp_lane_mask), .exp_rsp_tag(exp_rsp_tag),
        .reciprocal_req_valid(reciprocal_req_valid),
        .reciprocal_req_ready(reciprocal_req_ready),
        .reciprocal_req_sum_values(reciprocal_req_sum_values),
        .reciprocal_req_row_mask(reciprocal_req_row_mask),
        .reciprocal_req_tag(reciprocal_req_tag),
        .reciprocal_rsp_valid(reciprocal_rsp_valid),
        .reciprocal_rsp_ready(reciprocal_rsp_ready),
        .reciprocal_rsp_values(reciprocal_rsp_values),
        .reciprocal_rsp_row_mask(reciprocal_rsp_row_mask),
        .reciprocal_rsp_tag(reciprocal_rsp_tag),
        .quant_scale_req_valid(quant_scale_req_valid),
        .quant_scale_req_ready(quant_scale_req_ready),
        .quant_scale_req_row_max_abs(quant_scale_req_row_max_abs),
        .quant_scale_req_row_mask(quant_scale_req_row_mask),
        .quant_scale_rsp_valid(quant_scale_rsp_valid),
        .quant_scale_rsp_ready(quant_scale_rsp_ready),
        .quant_scale_rsp_values(quant_scale_rsp_values),
        .quant_values_req_valid(quant_values_req_valid),
        .quant_values_req_ready(quant_values_req_ready),
        .quant_values_req_values(quant_values_req_values),
        .quant_values_req_lane_mask(quant_values_req_lane_mask),
        .quant_values_req_tag(quant_values_req_tag),
        .quant_values_rsp_valid(quant_values_rsp_valid),
        .quant_values_rsp_ready(quant_values_rsp_ready),
        .quant_values_rsp_values(quant_values_rsp_values),
        .quant_values_rsp_lane_mask(quant_values_rsp_lane_mask),
        .quant_values_rsp_tag(quant_values_rsp_tag),
        .done_pulse(pipeline_done_pulse),
        .error(pipeline_error),
        .accepted_source_tile_count(accepted_source_tile_count),
        .accepted_bf16_write_count(accepted_bf16_write_count),
        .accepted_quantized_write_count(accepted_quantized_write_count),
        .accepted_scratch_read_count(accepted_scratch_read_count),
        .accepted_scratch_write_count(accepted_scratch_write_count),
        .accepted_vector_request_count(accepted_vector_request_count),
        .accepted_max_request_count(accepted_max_request_count),
        .accepted_reduction_request_count(accepted_reduction_request_count),
        .accepted_quantized_value_count(accepted_quantized_value_count));

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            capture_probability_enable <= 1'b0;
            row_count <= '0;
            head_count <= '0;
            row_length <= '0;
            score_head_stride <= '0;
            score_row_stride <= '0;
            probability_head_stride <= '0;
            probability_row_stride <= '0;
            current_score_head_base <= '0;
            current_probability_head_base <= '0;
            current_score_batch_base <= '0;
            current_probability_batch_base <= '0;
            current_head <= '0;
            current_row_base <= '0;
            current_batch_rows <= '0;
            abort_ack <= 1'b0;
            done_pulse <= 1'b0;
            error <= 1'b0;
            error_id <= '0;
`ifndef SYNTHESIS
            completed_command_count <= '0;
`endif
        end else begin
            abort_ack <= 1'b0;
            done_pulse <= 1'b0;
            if (abort_request && state != IDLE && state != ABORT_WAIT &&
                state != ABORT_WAIT_LOW) begin
                if (state == START_BATCH || state == COMPLETE ||
                    (state == WAIT_BATCH && pipeline_done_pulse)) begin
                    abort_ack <= 1'b1;
                    state <= ABORT_WAIT_LOW;
                end else
                    state <= ABORT_WAIT;
            end
            else begin
                case (state)
                    IDLE: if (start_fire) begin
                        error <= 1'b0;
                        error_id <= '0;
                        if (!configuration_valid) begin
                            error <= 1'b1;
                            error_id <= ERROR_CONFIGURATION;
                            done_pulse <= 1'b1;
                        end else begin
                            row_count <= start_row_count;
                            head_count <= start_head_count;
                            row_length <= start_row_length;
                            score_head_stride <= start_score_head_stride;
                            score_row_stride <= start_score_row_stride;
                            probability_head_stride <=
                                start_probability_head_stride;
                            probability_row_stride <=
                                start_probability_row_stride;
                            capture_probability_enable <= start_capture_probability_enable;
                            current_score_head_base <= start_score_base;
                            current_probability_head_base <=
                                start_probability_base;
                            current_score_batch_base <= start_score_base;
                            current_probability_batch_base <=
                                start_probability_base;
                            current_head <= '0;
                            current_row_base <= '0;
                            current_batch_rows <= start_row_count >= 8 ?
                                4'd8 : start_row_count[3:0];
                            state <= START_BATCH;
                        end
                    end
                    START_BATCH: if (pipeline_start_valid && pipeline_start_ready)
                        state <= WAIT_BATCH;
                    WAIT_BATCH: if (pipeline_done_pulse) begin
                        if (pipeline_error) begin
                            error <= 1'b1;
                            error_id <= ERROR_PIPELINE;
                            state <= COMPLETE;
                        end else if (final_batch && final_head) begin
                            state <= COMPLETE;
                        end else if (final_batch) begin
                            current_head <= current_head + 1'b1;
                            current_row_base <= '0;
                            current_score_head_base <= current_score_head_base +
                                score_head_stride;
                            current_probability_head_base <=
                                current_probability_head_base +
                                probability_head_stride;
                            current_score_batch_base <= current_score_head_base +
                                score_head_stride;
                            current_probability_batch_base <=
                                current_probability_head_base +
                                probability_head_stride;
                            current_batch_rows <= row_count >= 8 ? 4'd8 :
                                row_count[3:0];
                            state <= START_BATCH;
                        end else begin
                            current_row_base <= current_row_base +
                                6'(current_batch_rows);
                            current_score_batch_base <= current_score_batch_base +
                                (score_row_stride << 3);
                            current_probability_batch_base <=
                                current_probability_batch_base +
                                (probability_row_stride << 3);
                            current_batch_rows <= row_count -
                                (current_row_base + 6'(current_batch_rows)) >= 8 ?
                                4'd8 : 4'(row_count -
                                    (current_row_base + 6'(current_batch_rows)));
                            state <= START_BATCH;
                        end
                    end
                    COMPLETE: begin
                        done_pulse <= 1'b1;
`ifndef SYNTHESIS
                        completed_command_count <=
                            completed_command_count + 1'b1;
`endif
                        state <= IDLE;
                    end
                    ABORT_WAIT: if (pipeline_abort_ack) begin
                        abort_ack <= 1'b1;
                        state <= ABORT_WAIT_LOW;
                    end
                    ABORT_WAIT_LOW: if (!abort_request)
                        state <= IDLE;
                    default: begin
                        error <= 1'b1;
                        error_id <= ERROR_PIPELINE;
                        done_pulse <= 1'b1;
                        state <= IDLE;
                    end
                endcase
            end
        end
    end

    initial begin
        if (MAX_ROWS != 48 || MAX_HEADS != 32 || MAX_LENGTH != 2048)
            $error("softmax_engine production dimensions are fixed at 48x32x2048");
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst && pipeline_done_pulse)
            assert (state == WAIT_BATCH)
                else $error("Softmax row completion pulse had no active parent wait state");
    end

    assert property (@(posedge clk) disable iff (rst)
        (state == IDLE && start_valid && !start_ready) |=>
            (state == IDLE &&
             $stable({row_count, head_count, row_length,
                      score_head_stride, score_row_stride,
                      probability_head_stride, probability_row_stride,
                      current_score_head_base, current_probability_head_base,
                      current_score_batch_base,
                      current_probability_batch_base, current_head,
                      current_row_base, current_batch_rows})))
        else $error("softmax_engine changed accepted command state without a start handshake");
`endif
endmodule

`default_nettype wire
