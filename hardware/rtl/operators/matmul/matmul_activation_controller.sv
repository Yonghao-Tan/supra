`default_nettype none

// Quantizes up to eight physical rows using 8-row x 8-element tiles.
module matmul_activation_controller #(
    parameter integer MAX_ELEMENTS = 12288,
    parameter integer ADDR_WIDTH = 14,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         abort_request,
    output logic                         abort_ack,
    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [7:0]                   activation_a4_rows,
    input  logic                         precomputed_max_valid,
    input  logic [8*16-1:0]              precomputed_row_max_abs,
    input  logic [5:0]                   row_base,
    input  logic [3:0]                   row_count,
    input  logic [ADDR_WIDTH-1:0]        element_count,

    output logic                         source_request_valid,
    input  logic                         source_request_ready,
    output logic [5:0]                   source_request_row_base,
    output logic [3:0]                   source_request_row_count,
    output logic [ADDR_WIDTH-1:0]        source_request_element,
    input  logic                         source_response_valid,
    output logic                         source_response_ready,
    input  logic [64*16-1:0]             source_response_values,
    input  logic [63:0]                  source_response_lane_mask,

    output logic                         max_req_valid,
    input  logic                         max_req_ready,
    output logic [64*16-1:0]             max_req_values,
    output logic [63:0]                  max_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         max_req_tag,
    input  logic                         max_rsp_valid,
    output logic                         max_rsp_ready,
    input  logic [8*16-1:0]              max_rsp_values,
    input  logic [7:0]                   max_rsp_row_mask,
    input  logic [TAG_WIDTH-1:0]         max_rsp_tag,

    output logic                         quant_scale_req_valid,
    input  logic                         quant_scale_req_ready,
    output logic [7:0]                   quant_scale_req_a4_row_mask,
    output logic [8*16-1:0]              quant_scale_req_row_max_abs,
    output logic [7:0]                   quant_scale_req_row_mask,
    input  logic                         quant_scale_rsp_valid,
    output logic                         quant_scale_rsp_ready,
    input  logic [8*16-1:0]              quant_scale_rsp_values_bf16,

    output logic                         quant_values_req_valid,
    input  logic                         quant_values_req_ready,
    output logic [64*16-1:0]             quant_values_req_values_bf16,
    output logic [63:0]                  quant_values_req_lane_mask,
    output logic [TAG_WIDTH-1:0]         quant_values_req_tag,
    input  logic                         quant_values_rsp_valid,
    output logic                         quant_values_rsp_ready,
    input  logic [64*8-1:0]              quant_values_rsp_values,
    input  logic [63:0]                  quant_values_rsp_lane_mask,
    input  logic [TAG_WIDTH-1:0]         quant_values_rsp_tag,

    output logic                         scale_valid,
    input  logic                         scale_ready,
    output logic [5:0]                   scale_row_base,
    output logic [7:0]                   scale_row_mask,
    output logic [8*16-1:0]              scale_values_bf16,
    output logic                         quantized_valid,
    input  logic                         quantized_ready,
    output logic [5:0]                   quantized_row_base,
    output logic [ADDR_WIDTH-1:0]        quantized_element,
    output logic [64*8-1:0]              quantized_values,
    output logic [63:0]                  quantized_lane_mask,
    output logic [TAG_WIDTH-1:0]         quantized_tag,

    output logic                         done_pulse,
    output logic                         error,
    output logic [63:0]                  accepted_source_tiles,
    output logic [63:0]                  accepted_max_tiles,
    output logic [63:0]                  accepted_quantized_tiles
);
    typedef enum logic [2:0] {
        IDLE,
        MAX_STREAM,
        SCALE_REQUEST,
        SCALE_FORWARD,
        QUANTIZED_STREAM,
        COMPLETE,
        ABORT_DRAIN
    } state_t;

    state_t state;
    logic [7:0] saved_a4_rows;
    logic [5:0] saved_row_base;
    logic [3:0] saved_row_count;
    logic [ADDR_WIDTH-1:0] saved_element_count;
    logic [7:0] saved_row_mask;


    logic [ADDR_WIDTH-1:0] stream_request_element;
    logic [15:0] stream_request_count;
    logic [15:0] stream_response_count;
    logic source_metadata_valid;
    logic [ADDR_WIDTH-1:0] source_metadata_element;
    logic source_fifo_input_valid;
    logic source_fifo_input_ready;
    logic [64*16+64+ADDR_WIDTH-1:0] source_fifo_input_data;
    logic source_tile_valid;
    logic source_tile_ready;
    logic [64*16-1:0] source_tile_values;
    logic [63:0] source_tile_lane_mask;
    logic [ADDR_WIDTH-1:0] source_tile_element;
    logic [1:0] source_fifo_occupancy;
    logic source_tile_fire;
    logic [15:0] max_response_count;
    localparam integer MAX_OUTSTANDING_WIDTH = $clog2((MAX_ELEMENTS + 7) / 8 + 1);
    logic [MAX_OUTSTANDING_WIDTH-1:0] max_outstanding;
    logic [8*16-1:0] running_row_max;
    logic source_request_fire;
    logic source_response_fire;
    logic max_response_fire;
    logic quantized_response_fire;
    logic quant_values_response_fire;
    logic stream_requests_complete;
    logic stream_responses_complete;
    logic max_responses_complete;

    function automatic [7:0] row_mask_for_count(input logic [3:0] count);
        row_mask_for_count = count == 4'd8 ? 8'hff :
            8'((9'd1 << count) - 1'b1);
    endfunction

    function automatic [63:0] tile_lane_mask(
        input logic [7:0] rows_active,
        input logic [ADDR_WIDTH-1:0] element_base,
        input logic [ADDR_WIDTH-1:0] elements
    );
        logic [63:0] mask;
        begin
            mask = '0;
            for (integer row = 0; row < 8; row = row + 1)
                for (integer column = 0; column < 8; column = column + 1)
                    if (rows_active[row] &&
                        element_base + ADDR_WIDTH'(column) < elements)
                        mask[row*8 + column] = 1'b1;
            tile_lane_mask = mask;
        end
    endfunction

    assign start_ready = state == IDLE && !abort_request;
    assign done_pulse = state == COMPLETE;
    assign source_request_row_base = saved_row_base;
    assign source_request_row_count = saved_row_count;
    assign source_request_element = stream_request_element;
    assign source_request_valid = !abort_request &&
        (state == MAX_STREAM || state == QUANTIZED_STREAM) &&
        !stream_requests_complete &&
        (!source_metadata_valid || source_response_fire) &&
        (source_fifo_occupancy + source_metadata_valid < 2 || source_tile_fire);
    assign source_request_fire = source_request_valid && source_request_ready;

    always_comb begin
        max_req_valid = 1'b0;
        max_req_values = source_tile_values;
        max_req_lane_mask = source_tile_lane_mask;
        max_req_tag = TAG_WIDTH'(source_tile_element >> 3);
        quant_values_req_valid = 1'b0;
        quant_values_req_values_bf16 = source_tile_values;
        quant_values_req_lane_mask = source_tile_lane_mask;
        quant_values_req_tag = TAG_WIDTH'(source_tile_element >> 3);
        if (state == MAX_STREAM) begin
            max_req_valid = source_tile_valid && !abort_request;
        end else if (state == QUANTIZED_STREAM) begin
            quant_values_req_valid = source_tile_valid && !abort_request;
        end
    end

    assign source_fifo_input_valid = source_response_valid && source_response_ready &&
        state != ABORT_DRAIN;
    assign source_fifo_input_data = {source_metadata_element,
        source_response_lane_mask,
        source_response_values};
    assign source_response_ready = state == ABORT_DRAIN ? 1'b1 :
        source_metadata_valid && source_fifo_occupancy < 2;
    assign source_response_fire = source_response_valid && source_response_ready;
    assign source_tile_ready =
        state == MAX_STREAM ? max_req_ready :
        state == QUANTIZED_STREAM ? quant_values_req_ready : 1'b0;
    assign source_tile_fire = source_tile_valid && source_tile_ready;

    ready_valid_fifo #(
        .DATA_WIDTH(64*16+64+ADDR_WIDTH),
        .DEPTH(2)
    ) source_response_fifo (
        .clk(clk), .rst(rst || abort_request),
        .input_valid(source_fifo_input_valid),
        .input_ready(source_fifo_input_ready),
        .input_data(source_fifo_input_data),
        .output_valid(source_tile_valid),
        .output_ready(source_tile_ready),
        .output_data({source_tile_element,
            source_tile_lane_mask, source_tile_values}),
        .occupancy(source_fifo_occupancy)
    );

    assign max_rsp_ready = state == MAX_STREAM ||
        (state == ABORT_DRAIN && max_outstanding != 0);
    assign max_response_fire = max_rsp_valid && max_rsp_ready;
    assign quant_scale_req_valid = state == SCALE_REQUEST && !abort_request;
    assign quant_scale_req_a4_row_mask = saved_a4_rows;
    assign quant_scale_req_row_max_abs = running_row_max;
    assign quant_scale_req_row_mask = saved_row_mask;
    assign scale_valid = state == SCALE_FORWARD && quant_scale_rsp_valid &&
        !abort_request;
    assign scale_row_base = saved_row_base;
    assign scale_row_mask = saved_row_mask;
    assign scale_values_bf16 = quant_scale_rsp_values_bf16;
    assign quant_scale_rsp_ready = state == SCALE_FORWARD && scale_ready;

    assign quantized_valid = state == QUANTIZED_STREAM &&
        quant_values_rsp_valid && !abort_request;
    assign quant_values_rsp_ready = state == QUANTIZED_STREAM && quantized_ready;
    assign quantized_row_base = saved_row_base;
    assign quantized_element = ADDR_WIDTH'(quant_values_rsp_tag << 3);
    assign quantized_values = quant_values_rsp_values;
    assign quantized_lane_mask = quant_values_rsp_lane_mask;
    assign quantized_tag = quant_values_rsp_tag;
    assign quantized_response_fire = quantized_valid && quantized_ready;
    assign quant_values_response_fire = quant_values_rsp_valid &&
        quant_values_rsp_ready;

    assign stream_requests_complete = stream_request_element >= saved_element_count;
    assign stream_responses_complete = stream_response_count ==
        (({2'b00, saved_element_count} + 16'd7) >> 3);
    assign max_responses_complete = max_response_count ==
        (({2'b00, saved_element_count} + 16'd7) >> 3);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            saved_a4_rows <= '0;
            saved_row_base <= '0;
            saved_row_count <= '0;
            saved_element_count <= '0;
            saved_row_mask <= '0;
            stream_request_element <= '0;
            stream_request_count <= '0;
            stream_response_count <= '0;
            source_metadata_valid <= 1'b0;
            source_metadata_element <= '0;
            max_response_count <= '0;
            max_outstanding <= '0;
            running_row_max <= '0;
            abort_ack <= 1'b0;

            error <= 1'b0;
            accepted_source_tiles <= '0;
            accepted_max_tiles <= '0;
            accepted_quantized_tiles <= '0;
        end else begin
            abort_ack <= 1'b0;

            // The shared max pipeline is not reset by a Matmul abort.
            case ({max_req_valid && max_req_ready, max_response_fire})
                2'b10: max_outstanding <= max_outstanding + 1'b1;
                2'b01: max_outstanding <= max_outstanding - 1'b1;
                default: begin end
            endcase

            if (abort_request && state != IDLE && state != ABORT_DRAIN) begin
                if (source_response_fire)
                    source_metadata_valid <= 1'b0;
                state <= ABORT_DRAIN;
            end else case (state)
                IDLE: begin
                    source_metadata_valid <= 1'b0;
                    if (start_valid && start_ready) begin
                        if (row_count == 0 || row_count > 8 ||
                            row_base + row_count > 48 || element_count == 0 ||
                            element_count > MAX_ELEMENTS) begin
                            error <= 1'b1;
                            state <= COMPLETE;
                        end else begin
                            saved_a4_rows <= activation_a4_rows;
                            saved_row_base <= row_base;
                            saved_row_count <= row_count;
                            saved_element_count <= element_count;
                            saved_row_mask <= row_mask_for_count(row_count);
                            stream_request_element <= '0;
                            stream_request_count <= '0;
                            stream_response_count <= '0;
                            max_response_count <= '0;
                            running_row_max <= precomputed_max_valid ?
                                precomputed_row_max_abs : '0;
                            error <= 1'b0;
                            accepted_source_tiles <= '0;
                            accepted_max_tiles <= '0;
                            accepted_quantized_tiles <= '0;
                            state <= precomputed_max_valid ? SCALE_REQUEST : MAX_STREAM;
                        end
                    end
                end

                MAX_STREAM: begin
                    if (source_tile_fire) begin
                        stream_response_count <= stream_response_count + 1'b1;
                    end
                    if (source_response_fire)
                        source_metadata_valid <= 1'b0;
                    if (source_request_fire) begin
                        source_metadata_valid <= 1'b1;
                        source_metadata_element <= stream_request_element;
                        stream_request_element <= stream_request_element + ADDR_WIDTH'(8);
                        stream_request_count <= stream_request_count + 1'b1;
                        accepted_source_tiles <= accepted_source_tiles + 1'b1;
                    end
                    if (max_response_fire) begin
                        max_response_count <= max_response_count + 1'b1;
                        accepted_max_tiles <= accepted_max_tiles + 1'b1;
                        for (integer row = 0; row < 8; row = row + 1)
                            if (max_rsp_row_mask[row] &&
                                max_rsp_values[row*16 +: 15] >
                                running_row_max[row*16 +: 15])
                                running_row_max[row*16 +: 16] <=
                                    {1'b0, max_rsp_values[row*16 +: 15]};
                    end
                    if (stream_responses_complete && max_responses_complete &&
                        !source_metadata_valid && !source_tile_valid) begin
                        state <= SCALE_REQUEST;
                    end
                end

                SCALE_REQUEST: if (quant_scale_req_valid && quant_scale_req_ready)
                    state <= SCALE_FORWARD;

                SCALE_FORWARD: if (quant_scale_rsp_valid &&
                                      quant_scale_rsp_ready) begin
                    source_metadata_valid <= 1'b0;
                    stream_request_element <= '0;
                    stream_request_count <= '0;
                    stream_response_count <= '0;
                    state <= QUANTIZED_STREAM;
                end

                QUANTIZED_STREAM: begin
                    if (source_tile_fire) begin
                        stream_response_count <= stream_response_count + 1'b1;
                    end
                    if (source_response_fire)
                        source_metadata_valid <= 1'b0;
                    if (source_request_fire) begin
                        source_metadata_valid <= 1'b1;
                        source_metadata_element <= stream_request_element;
                        stream_request_element <= stream_request_element + ADDR_WIDTH'(8);
                        stream_request_count <= stream_request_count + 1'b1;
                        accepted_source_tiles <= accepted_source_tiles + 1'b1;
                    end
                    if (quantized_response_fire)
                        accepted_quantized_tiles <= accepted_quantized_tiles + 1'b1;
                    if (stream_responses_complete &&
                        accepted_quantized_tiles ==
                            (({2'b00, saved_element_count} + 16'd7) >> 3) &&
                        !source_metadata_valid && !source_tile_valid &&
                        !quant_values_rsp_valid)
                        state <= COMPLETE;
                end

                COMPLETE: state <= IDLE;

                ABORT_DRAIN: begin
                    if (source_response_fire)
                        source_metadata_valid <= 1'b0;
                    if ((!source_metadata_valid || source_response_fire) &&
                        (max_outstanding == 0 ||
                         (max_outstanding == 1 && max_response_fire))) begin
                        abort_ack <= 1'b1;
                        state <= IDLE;
                    end
                end

                default: begin
                    error <= 1'b1;
                    state <= COMPLETE;
                end
            endcase
        end
    end

    initial begin
        if (MAX_ELEMENTS > (1 << ADDR_WIDTH) || TAG_WIDTH < 12)
            $error("matmul_activation_controller parameters cannot represent the production shape");
    end

`ifndef SYNTHESIS
    logic previous_quantized_stalled;
    logic [5+ADDR_WIDTH+512+64+TAG_WIDTH:0] held_quantized;
    always_ff @(posedge clk) begin
        if (rst) begin
            previous_quantized_stalled <= 1'b0;
            held_quantized <= '0;
        end else begin
            if (previous_quantized_stalled && !abort_request)
                assert (quantized_valid &&
                    {quantized_row_base, quantized_element, quantized_values,
                     quantized_lane_mask, quantized_tag} == held_quantized)
                    else $error("matmul_activation_controller changed a stalled quantized tile");
            previous_quantized_stalled <= quantized_valid && !quantized_ready && !abort_request;
            if (quantized_valid && !quantized_ready)
                held_quantized <= {quantized_row_base, quantized_element, quantized_values,
                              quantized_lane_mask, quantized_tag};
            if (!rst && max_rsp_valid && max_rsp_ready)
                assert (max_rsp_tag < ((saved_element_count + 7) >> 3))
                    else $error("matmul_activation_controller max response tag is outside the row");
        end
    end
`endif
endmodule

`default_nettype wire
