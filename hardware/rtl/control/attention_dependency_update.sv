`default_nettype none

// A 64-byte job names eight query/key pairs across 32 captured heads.
// kind 0 replaces BF16 dependency entries, kind 1 takes their layer maximum,
// kind 2 updates eight scout records, and kind 3 updates one scout record.
// The existing ordered DMA interfaces carry descriptors, samples and results.
module attention_dependency_update (
    input logic clk, rst, abort_request,
    output logic abort_ack,
    input logic start_valid,
    output logic start_ready,
    input logic [63:0] start_job_base, start_job_limit,
    input logic [15:0] start_job_count,
    input logic start_layer_update,
    input logic start_first_layer,
    output logic done_valid,
    input logic done_ready,
    output logic error,
    output logic [7:0] error_id,
    output logic read_request_valid,
    input logic read_request_ready,
    output logic [63:0] read_request_address,
    output logic [31:0] read_request_bytes,
    output logic [7:0] read_request_tag,
    input logic read_valid,
    output logic read_ready,
    input logic [127:0] read_data,
    input logic [15:0] read_byte_enable,
    input logic read_last,
    input logic [7:0] read_tag,
    input logic read_error, read_abort_ack,
    output logic write_request_valid,
    input logic write_request_ready,
    output logic [63:0] write_request_address,
    output logic [31:0] write_request_bytes,
    output logic [7:0] write_request_tag,
    output logic write_valid,
    input logic write_ready,
    output logic [127:0] write_data,
    output logic [15:0] write_byte_enable,
    output logic write_last,
    input logic write_done, write_error,
    output logic dequant_valid,
    input logic dequant_ready,
    output logic [127:0] dequant_values,
    output logic [15:0] dequant_scale,
    output logic [7:0] dequant_lane_mask,
    input logic dequant_result_valid,
    output logic dequant_result_ready,
    input logic [127:0] dequant_result_values,
    input logic dequant_error,
    output logic dequant_abort,
    input logic dequant_abort_ack
);
    typedef enum logic [4:0] {
        IDLE, JOB_START, JOB_WAIT, JOB_SPAN, JOB_RANGE, JOB_CHECK, MEAN_START, HEAD_START,
        HEAD_WAIT, MEAN_WAIT, OUTPUT_START, OUTPUT_WAIT, UPDATE_OUTPUT,
        WRITE_START, WRITE_STREAM, WRITE_WAIT, NEXT_JOB, COMPLETE,
        ABORT_DRAIN, ABORT_LOW
    } state_t;
    state_t state;
    logic [63:0] job_base, job_address, job_limit;
    logic [15:0] remaining_jobs;
    logic [64:0] start_jobs_end, source_end, output_end;
    logic [36:0] source_span;
    logic [63:0] source_base, source_limit, source_address, output_address, output_limit;
    logic [31:0] head_stride;
    logic [7:0] lane_mask;
    logic [1:0] output_kind;
    logic replace_score;
    logic source_p8;
    logic layer_update, first_layer, merge_initial;
    logic descriptor_reserved_error;
    logic [5:0] head_index;
    logic [127:0] result_bf16;
    logic [63:0] result_score;
    logic [127:0] output_words [0:3];
    logic [127:0] output_data;
    logic [1:0] output_word;
    logic [31:0] output_bytes;
    logic terminal_error, aborting;
    logic [7:0] terminal_error_id;
    logic reader_start_valid, reader_start_ready, reader_valid, reader_ready, reader_done, reader_error, reader_abort_ack;
    logic [63:0] reader_address;
    logic [31:0] reader_bytes, reader_offset;
    logic [127:0] reader_data;
    logic writer_start_ready, writer_ready, writer_done, writer_error, writer_abort_ack;
    logic mean_start_ready, mean_values_ready, mean_valid;
    logic [127:0] mean_bf16;
    logic [63:0] mean_score;
    logic [7:0] mean_mask;
    logic [15:0] mean_tag;
    logic bad_probability, stop_children;
    logic bad_dequant;
    logic [7:0] maximum_l0 [0:3], maximum_l1 [0:1], maximum_score;

    assign start_ready = state == IDLE && !abort_request;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;
    assign start_jobs_end = {1'b0, start_job_base} + {43'd0, start_job_count, 6'd0};
    assign output_bytes = output_kind == 2 ? 32'd64 : output_kind == 3 ? 32'd8 : 32'd16;
    assign stop_children = abort_request || state == ABORT_DRAIN;
    assign reader_start_valid = !stop_children &&
        (state == JOB_START || state == HEAD_START || state == OUTPUT_START);
    assign reader_address = state == JOB_START ? job_address :
        state == HEAD_START ? source_address : output_address;
    assign reader_bytes = state == JOB_START ? 32'd64 : state == HEAD_START ?
        (head_stride == 16 ? 32'd512 : 32'd16) : output_bytes;
    assign reader_ready = state == HEAD_WAIT ? (source_p8 ? dequant_ready : mean_values_ready) : 1'b1;
    assign dequant_valid = state == HEAD_WAIT && source_p8 && reader_valid && !bad_probability && !stop_children;
    assign dequant_scale = reader_data[79:64];
    assign dequant_lane_mask = lane_mask;
    assign dequant_result_ready = source_p8 && (mean_values_ready || stop_children);
    assign dequant_abort = source_p8 && stop_children;
    function automatic logic [15:0] positive_code_bf16(input logic [6:0] code);
        casez (code)
            7'b1??????: positive_code_bf16 = {1'b0, 8'd133, code[5:0], 1'b0};
            7'b01?????: positive_code_bf16 = {1'b0, 8'd132, code[4:0], 2'd0};
            7'b001????: positive_code_bf16 = {1'b0, 8'd131, code[3:0], 3'd0};
            7'b0001???: positive_code_bf16 = {1'b0, 8'd130, code[2:0], 4'd0};
            7'b00001??: positive_code_bf16 = {1'b0, 8'd129, code[1:0], 5'd0};
            7'b000001?: positive_code_bf16 = {1'b0, 8'd128, code[0], 6'd0};
            7'b0000001: positive_code_bf16 = 16'h3f80;
            default: positive_code_bf16 = '0;
        endcase
    endfunction
    always_comb begin
        bad_probability = source_p8 && (reader_data[127:80] != 0 || reader_data[79:64] > 16'h3f80);
        bad_dequant = dequant_error;
        dequant_values = '0;
        for (integer lane = 0; lane < 8; lane++) begin
            dequant_values[lane*16 +: 16] = positive_code_bf16(reader_data[lane*8 +: 7]);
            if (lane_mask[lane]) begin
                if (source_p8 ? reader_data[lane*8+7] : reader_data[lane*16 +: 16] > 16'h3f80)
                    bad_probability = 1'b1;
                if (dequant_result_values[lane*16 +: 16] > 16'h3f80) bad_dequant = 1'b1;
            end
        end
    end
    for (genvar pair = 0; pair < 4; pair++) begin : scout_max_l0
        assign maximum_l0[pair] = result_score[pair*16 +: 8] > result_score[pair*16+8 +: 8] ?
            result_score[pair*16 +: 8] : result_score[pair*16+8 +: 8];
    end
    for (genvar pair = 0; pair < 2; pair++) begin : scout_max_l1
        assign maximum_l1[pair] = maximum_l0[pair*2] > maximum_l0[pair*2+1] ?
            maximum_l0[pair*2] : maximum_l0[pair*2+1];
    end
    assign maximum_score = maximum_l1[0] > maximum_l1[1] ? maximum_l1[0] : maximum_l1[1];

    operator_dma_reader #(.DATA_WIDTH(128), .REQUEST_TAG(8'hb7)) reader (
        .clk, .rst, .abort_request(stop_children), .start_valid(reader_start_valid), .start_ready(reader_start_ready),
        .start_address(reader_address), .start_bytes(reader_bytes), .start_total_bytes(reader_bytes),
        .request_valid(read_request_valid), .request_ready(read_request_ready), .request_address(read_request_address),
        .request_bytes(read_request_bytes), .request_tag(read_request_tag), .read_valid, .read_ready, .read_data,
        .read_byte_enable, .read_last, .read_tag, .response_error(read_error), .upstream_abort_ack(read_abort_ack),
        .data_valid(reader_valid), .data_ready(reader_ready), .data(reader_data), .data_byte_enable(),
        .data_last(), .data_byte_offset(reader_offset), .done_pulse(reader_done), .error(reader_error), .abort_ack(reader_abort_ack)
    );
    operator_dma_writer #(.REQUEST_TAG(8'hb8)) writer (
        .clk, .rst, .abort_request(stop_children), .start_valid(state == WRITE_START && !stop_children),
        .start_ready(writer_start_ready), .start_address(output_address), .start_bytes(output_bytes),
        .request_valid(write_request_valid), .request_ready(write_request_ready), .request_address(write_request_address),
        .request_bytes(write_request_bytes), .request_tag(write_request_tag),
        .data_valid(state == WRITE_STREAM), .data_ready(writer_ready), .data(output_data),
        .data_byte_enable(output_kind == 3 ? 16'h00ff : 16'hffff),
        .data_last(output_kind != 2 || output_word == 2'd3),
        .write_valid, .write_ready, .write_data, .write_byte_enable, .write_last,
        .transaction_done(write_done), .transaction_error(write_error),
        .done_pulse(writer_done), .error(writer_error), .abort_ack(writer_abort_ack)
    );
    attention_head_mean mean (
        .clk, .rst, .abort_request(stop_children),
        .start_valid(state == MEAN_START && !stop_children), .start_ready(mean_start_ready),
        .start_lane_mask(lane_mask), .start_tag(remaining_jobs),
        .values_valid(!stop_children && (source_p8 ? (dequant_result_valid && !bad_dequant) :
            (state == HEAD_WAIT && reader_valid && !bad_probability))),
        .values_ready(mean_values_ready), .values_bf16(source_p8 ? dequant_result_values : reader_data),
        .result_valid(mean_valid), .result_ready(state == MEAN_WAIT),
        .result_mean_fp32(), .result_mean_bf16(mean_bf16), .result_score_u8(mean_score),
        .result_lane_mask(mean_mask), .result_tag(mean_tag)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE; job_base <= '0; job_address <= '0; job_limit <= '0; remaining_jobs <= '0;
            source_base <= '0; source_limit <= '0; source_address <= '0; output_address <= '0; output_limit <= '0;
            head_stride <= '0; lane_mask <= '0; output_kind <= '0; replace_score <= 1'b0; source_p8 <= 1'b0; descriptor_reserved_error <= 1'b0;
            layer_update <= 1'b0; first_layer <= 1'b0; merge_initial <= 1'b0;
            head_index <= '0; result_bf16 <= '0; result_score <= '0; output_word <= '0;
            source_span <= '0; source_end <= '0; output_end <= '0; output_data <= '0;
            terminal_error <= 1'b0; terminal_error_id <= '0; aborting <= 1'b0; abort_ack <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            // Invalid P8 suppresses dequant_valid, so the shared arbiter may
            // never grant ready. Detect the bad read before waiting for it.
            if ((abort_request || reader_error || writer_error ||
                 (source_p8 && dequant_result_valid && bad_dequant) ||
                 (state == HEAD_WAIT && reader_valid && bad_probability)) &&
                state != ABORT_DRAIN && state != ABORT_LOW && state != COMPLETE) begin
                aborting <= abort_request;
                terminal_error <= !abort_request;
                terminal_error_id <= reader_error ? 8'h03 : writer_error ? 8'h04 : 8'h05;
                state <= ABORT_DRAIN;
            end else case (state)
                IDLE: if (start_valid && start_ready) begin
                    layer_update <= start_layer_update;
                    first_layer <= start_first_layer;
                    terminal_error <= 1'b0; terminal_error_id <= '0;
                    job_base <= start_job_base; job_address <= start_job_base;
                    job_limit <= start_job_limit; remaining_jobs <= start_job_count;
                    if (start_job_base[3:0] != 0 || start_jobs_end[64] || start_jobs_end[63:0] > start_job_limit) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h01; state <= COMPLETE;
                    end else state <= start_job_count == 0 ? COMPLETE : JOB_START;
                end
                JOB_START: if (reader_start_ready) begin descriptor_reserved_error <= 1'b0; state <= JOB_WAIT; end
                JOB_WAIT: begin
                    if (reader_valid && reader_ready) begin
                        case (reader_offset[5:4])
                            2'd0: begin
                                source_base <= reader_data[63:0]; head_stride <= reader_data[95:64];
                                lane_mask <= reader_data[103:96]; output_kind <= reader_data[105:104];
                                replace_score <= reader_data[106];
                                source_p8 <= reader_data[107];
                                merge_initial <= reader_data[108];
                                if (reader_data[127:109] != 0) descriptor_reserved_error <= 1'b1;
                            end
                            2'd1: begin output_address <= reader_data[63:0]; source_limit <= reader_data[127:64]; end
                            2'd2: begin
                                output_limit <= reader_data[63:0];
                                if (reader_data[127:64] != 0) descriptor_reserved_error <= 1'b1;
                            end
                            default: if (reader_data != 0) descriptor_reserved_error <= 1'b1;
                        endcase
                    end
                    if (reader_done) state <= JOB_SPAN;
                end
                JOB_SPAN: begin
                    source_span <= ({5'd0, head_stride} << 5) - {5'd0, head_stride} + 37'd16;
                    output_end <= {1'b0, output_address} + {33'd0, output_bytes};
                    state <= JOB_RANGE;
                end
                JOB_RANGE: begin
                    source_end <= {1'b0, source_base} + {28'd0, source_span};
                    state <= JOB_CHECK;
                end
                JOB_CHECK: begin
                    if (descriptor_reserved_error || (layer_update && output_kind >= 2) ||
                        (merge_initial && output_kind != 1) ||
                        source_base[3:0] != 0 || head_stride[3:0] != 0 || head_stride < 16 ||
                        (output_kind < 2 ? output_address[0] != 0 : output_address[2:0] != 0) ||
                        source_end[64] || source_end[63:0] > source_limit ||
                        output_end[64] || output_end[63:0] > output_limit ||
                        (output_address < source_end[63:0] && source_base < output_end[63:0]) ||
                        (output_address < job_limit && job_base < output_end[63:0])) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h02; state <= COMPLETE;
                    end else begin
                        head_index <= '0; source_address <= source_base;
                        if (lane_mask == 0) begin
                            result_bf16 <= '0; result_score <= '0;
                            state <= replace_score && output_kind >= 2 ? OUTPUT_START : NEXT_JOB;
                        end else state <= MEAN_START;
                    end
                end
                MEAN_START: if (mean_start_ready) state <= HEAD_START;
                HEAD_START: if (reader_start_ready) state <= HEAD_WAIT;
                HEAD_WAIT: if (reader_done) begin
                    if (head_stride == 16 || head_index == 6'd31) state <= MEAN_WAIT;
                    else begin head_index <= head_index + 6'd1; source_address <= source_address + {32'd0, head_stride}; state <= HEAD_START; end
                end
                MEAN_WAIT: if (mean_valid) begin
                    result_bf16 <= mean_bf16; result_score <= mean_score;
                    if (mean_tag != remaining_jobs || mean_mask != lane_mask) begin
                        terminal_error <= 1'b1; terminal_error_id <= 8'h05; state <= COMPLETE;
                    end else state <= OUTPUT_START;
                end
                OUTPUT_START: if (reader_start_ready) state <= OUTPUT_WAIT;
                OUTPUT_WAIT: begin
                    if (reader_valid && reader_ready) output_words[reader_offset[5:4]] <= reader_data;
                    if (reader_done) state <= UPDATE_OUTPUT;
                end
                UPDATE_OUTPUT: begin
                    if (output_kind < 2) begin
                        for (integer lane = 0; lane < 8; lane++)
                            if (lane_mask[lane] && ((layer_update ? first_layer && !merge_initial : output_kind == 0) ||
                                result_bf16[lane*16 +: 16] > output_words[0][lane*16 +: 16]))
                                output_words[0][lane*16 +: 16] <= result_bf16[lane*16 +: 16];
                    end else if (output_kind == 2) begin
                        for (integer lane = 0; lane < 8; lane++)
                            if (replace_score || (lane_mask[lane] && result_score[lane*8 +: 8] > output_words[lane/2][(lane%2)*64 +: 8]))
                                output_words[lane/2][(lane%2)*64 +: 8] <= result_score[lane*8 +: 8];
                    end else if (replace_score || maximum_score > output_words[0][7:0]) output_words[0][7:0] <= maximum_score;
                    output_word <= '0; state <= WRITE_START;
                end
                WRITE_START: if (writer_start_ready) begin output_data <= output_words[0]; state <= WRITE_STREAM; end
                WRITE_STREAM: if (writer_ready) begin
                    output_word <= output_word + 2'd1;
                    if (output_kind != 2 || output_word == 2'd3) state <= WRITE_WAIT;
                    else output_data <= output_words[output_word+2'd1];
                end
                WRITE_WAIT: if (writer_done) state <= NEXT_JOB;
                NEXT_JOB: begin
                    remaining_jobs <= remaining_jobs - 16'd1; job_address <= job_address + 64'd64;
                    state <= remaining_jobs == 16'd1 ? COMPLETE : JOB_START;
                end
                COMPLETE: if (done_ready) state <= IDLE;
                ABORT_DRAIN: if ((reader_start_ready || reader_abort_ack) && writer_abort_ack && (!source_p8 || dequant_abort_ack)) begin
                    abort_ack <= aborting; state <= aborting ? ABORT_LOW : COMPLETE;
                end
                ABORT_LOW: if (!abort_request) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
