`default_nettype none

module shared_compute_resources (
    input  logic          clk,
    input  logic          rst,

    input  logic [1:0]    reduction_req_valid,
    output logic [1:0]    reduction_req_ready,
    input  hardware_types_pkg::reduction_request_t reduction_req [0:1],
    output logic [1:0]    reduction_rsp_valid,
    input  logic [1:0]    reduction_rsp_ready,
    output hardware_types_pkg::reduction_response_t reduction_rsp [0:1],

    input  logic          softmax_exp_req_valid,
    output logic          softmax_exp_req_ready,
    input  hardware_types_pkg::softmax_exp_request_t softmax_exp_req,
    output logic          softmax_exp_rsp_valid,
    input  logic          softmax_exp_rsp_ready,
    output hardware_types_pkg::softmax_exp_response_t softmax_exp_rsp,
    input  logic          softmax_reciprocal_req_valid,
    output logic          softmax_reciprocal_req_ready,
    input  hardware_types_pkg::softmax_reciprocal_request_t softmax_reciprocal_req,
    output logic          softmax_reciprocal_rsp_valid,
    input  logic          softmax_reciprocal_rsp_ready,
    output hardware_types_pkg::softmax_reciprocal_response_t softmax_reciprocal_rsp,

    input  logic [4:0]    bf16_abort_request,
    output logic [4:0]    bf16_abort_ack,
    input  logic [4:0]    bf16_req_valid,
    output logic [4:0]    bf16_req_ready,
    input  hardware_types_pkg::bf16_request_t bf16_req [0:4],
    output logic [4:0]    bf16_rsp_valid,
    input  logic [4:0]    bf16_rsp_ready,
    output hardware_types_pkg::bf16_response_t bf16_rsp [0:4],

    input  logic [3:0]    max_req_valid,
    output logic [3:0]    max_req_ready,
    input  hardware_types_pkg::maximum_request_t max_req [0:3],
    output logic [3:0]    max_rsp_valid,
    input  logic [3:0]    max_rsp_ready,
    output hardware_types_pkg::maximum_response_t max_rsp [0:3],

    input  logic [1:0]    pe_abort_request,
    output logic [1:0]    pe_abort_ack,
    input  logic [1:0]    pe_req_valid,
    output logic [1:0]    pe_req_ready,
    input  hardware_types_pkg::pe_request_t pe_req [0:1],
    output logic [1:0]    pe_accum_result_valid,
    input  logic [1:0]    pe_accum_result_ready,
    output hardware_types_pkg::pe_response_t pe_accum_result [0:1],
    output logic          pe_idle,

    input  logic [1:0]    rescale_claim_valid,
    output logic [1:0]    rescale_claim_ready,
    input  logic [1:0]    rescale_abort_request,
    output logic [1:0]    rescale_abort_ack,
    input  logic [1:0]    rescale_req_valid,
    output logic [1:0]    rescale_req_ready,
    input  hardware_types_pkg::rescale_request_t rescale_req [0:1],
    output logic [1:0]    rescale_rsp_valid,
    input  logic [1:0]    rescale_rsp_ready,
    output hardware_types_pkg::rescale_response_t rescale_rsp [0:1],
    output logic          rescale_idle,

    input  logic [3:0]    quant_abort_request,
    output logic [3:0]    quant_abort_ack,
    input  logic [3:0]    quant_scale_req_valid,
    output logic [3:0]    quant_scale_req_ready,
    input  hardware_types_pkg::quant_scale_request_t quant_scale_req [0:3],
    output logic [3:0]    quant_scale_rsp_valid,
    input  logic [3:0]    quant_scale_rsp_ready,
    output hardware_types_pkg::quant_scale_response_t quant_scale_rsp [0:3],
    input  logic [3:0]    quant_values_req_valid,
    output logic [3:0]    quant_values_req_ready,
    input  hardware_types_pkg::quant_values_request_t quant_values_req [0:3],
    output logic [3:0]    quant_values_rsp_valid,
    input  logic [3:0]    quant_values_rsp_ready,
    output hardware_types_pkg::quant_values_response_t quant_values_rsp [0:3],

    input  logic [1:0]    writer_abort_request,
    input  logic [1:0]    writer_cfg_valid,
    output logic [1:0]    writer_cfg_ready,
    input  hardware_types_pkg::activation_writer_config_t writer_cfg [0:1],
    input  logic [1:0]    writer_scale_valid,
    output logic [1:0]    writer_scale_ready,
    input  hardware_types_pkg::activation_writer_scale_t writer_scale [0:1],
    input  logic [1:0]    writer_quantized_valid,
    output logic [1:0]    writer_quantized_ready,
    input  hardware_types_pkg::activation_writer_values_t writer_quantized [0:1],
    output logic [1:0]    writer_done_pulse,
    output logic [1:0]    writer_error,
    output logic          activation_scale_valid,
    input  logic          activation_scale_ready,
    output logic [5:0]    activation_scale_row_base,
    output logic [7:0]    activation_scale_row_mask,
    output logic [127:0]  activation_scale_values,
    output logic          activation_write_valid,
    input  logic          activation_write_ready,
    output logic [7:0]    activation_write_slot_valid,
    output logic [255:0]  activation_write_address,
    output logic [1023:0] activation_write_data,
    output logic [127:0]  activation_write_byte_enable,
    output logic [15:0]   activation_write_tag
);
    localparam logic [1:0] CLIENT2_NONE = 2'd3;
    localparam logic [2:0] CLIENT5_NONE = 3'd7;
    localparam logic [2:0] CLIENT4_NONE = 3'd7;

    function automatic logic [1:0] select_client2(
        input logic [1:0] request_valid
    );
        case (request_valid)
            2'b01: select_client2 = 2'd0;
            2'b10: select_client2 = 2'd1;
            default: select_client2 = CLIENT2_NONE;
        endcase
    endfunction

    function automatic logic [2:0] select_client4(
        input logic [3:0] request_valid
    );
        case (request_valid)
            4'b0001: select_client4 = 3'd0;
            4'b0010: select_client4 = 3'd1;
            4'b0100: select_client4 = 3'd2;
            4'b1000: select_client4 = 3'd3;
            default: select_client4 = CLIENT4_NONE;
        endcase
    endfunction

    function automatic logic [2:0] select_client5(
        input logic [4:0] request_valid
    );
        case (request_valid)
            5'b00001: select_client5 = 3'd0;
            5'b00010: select_client5 = 3'd1;
            5'b00100: select_client5 = 3'd2;
            5'b01000: select_client5 = 3'd3;
            5'b10000: select_client5 = 3'd4;
            default: select_client5 = CLIENT5_NONE;
        endcase
    endfunction

    logic [1:0] reduction_active_client;
    logic [1:0] reduction_route_client;
    logic [7:0] reduction_outstanding;
    logic shared_reduction_req_valid;
    logic shared_reduction_req_ready;
    hardware_types_pkg::reduction_request_t shared_reduction_req;
    hardware_types_pkg::reduction_response_t shared_reduction_rsp;
    logic shared_reduction_rsp_valid;
    logic shared_reduction_rsp_ready;
    logic reduction_request_fire;
    logic reduction_response_fire;

    assign reduction_route_client = reduction_outstanding == 0 ?
        select_client2(reduction_req_valid) : reduction_active_client;
    always_comb begin
        shared_reduction_req_valid = 1'b0;
        shared_reduction_req = '0;
        shared_reduction_rsp_ready = 1'b0;
        reduction_req_ready = '0;
        reduction_rsp_valid = '0;
        reduction_rsp[0] = '0;
        reduction_rsp[1] = '0;
        case (reduction_route_client)
            2'd0: begin
                shared_reduction_req_valid = reduction_req_valid[0];
                shared_reduction_req = reduction_req[0];
                reduction_req_ready[0] = shared_reduction_req_ready;
            end
            2'd1: begin
                shared_reduction_req_valid = reduction_req_valid[1];
                shared_reduction_req = reduction_req[1];
                reduction_req_ready[1] = shared_reduction_req_ready;
            end
            default: begin end
        endcase
        case (reduction_active_client)
            2'd0: begin
                reduction_rsp_valid[0] = shared_reduction_rsp_valid;
                reduction_rsp[0] = shared_reduction_rsp;
                shared_reduction_rsp_ready = reduction_rsp_ready[0];
            end
            2'd1: begin
                reduction_rsp_valid[1] = shared_reduction_rsp_valid;
                reduction_rsp[1] = shared_reduction_rsp;
                shared_reduction_rsp_ready = reduction_rsp_ready[1];
            end
            default: begin end
        endcase
    end
    assign reduction_request_fire = shared_reduction_req_valid &&
        shared_reduction_req_ready;
    assign reduction_response_fire = shared_reduction_rsp_valid &&
        shared_reduction_rsp_ready;
    always_ff @(posedge clk) begin
        if (rst) begin
            reduction_active_client <= CLIENT2_NONE;
            reduction_outstanding <= '0;
        end else begin
            case ({reduction_request_fire, reduction_response_fire})
                2'b10: begin
                    if (reduction_outstanding == 0)
                        reduction_active_client <= reduction_route_client;
                    reduction_outstanding <= reduction_outstanding + 8'd1;
                end
                2'b01: begin
                    reduction_outstanding <= reduction_outstanding - 8'd1;
                    if (reduction_outstanding == 8'd1)
                        reduction_active_client <= CLIENT2_NONE;
                end
                default: begin end
            endcase
        end
    end
    bf16_tile_reduction_pipe #(.TAG_WIDTH(16)) shared_reduction_pipe (
        .clk(clk), .rst(rst), .req_valid(shared_reduction_req_valid),
        .req_ready(shared_reduction_req_ready),
        .req_values(shared_reduction_req.values),
        .req_lane_mask(shared_reduction_req.lane_mask),
        .req_tag(shared_reduction_req.tag),
        .rsp_valid(shared_reduction_rsp_valid),
        .rsp_ready(shared_reduction_rsp_ready),
        .rsp_values(shared_reduction_rsp.values),
        .rsp_row_mask(shared_reduction_rsp.row_mask),
        .rsp_tag(shared_reduction_rsp.tag));

    softmax_lut_resource shared_softmax_lut (
        .clk(clk), .rst(rst),
        .exp_req_valid(softmax_exp_req_valid),
        .exp_req_ready(softmax_exp_req_ready), .exp_req(softmax_exp_req),
        .exp_rsp_valid(softmax_exp_rsp_valid),
        .exp_rsp_ready(softmax_exp_rsp_ready), .exp_rsp(softmax_exp_rsp),
        .reciprocal_req_valid(softmax_reciprocal_req_valid),
        .reciprocal_req_ready(softmax_reciprocal_req_ready),
        .reciprocal_req(softmax_reciprocal_req),
        .reciprocal_rsp_valid(softmax_reciprocal_rsp_valid),
        .reciprocal_rsp_ready(softmax_reciprocal_rsp_ready),
        .reciprocal_rsp(softmax_reciprocal_rsp));

    logic [2:0] bf16_active_client;
    logic [2:0] bf16_route_client;
    logic [7:0] bf16_outstanding;
    logic shared_bf16_abort_request;
    logic shared_bf16_abort_ack;
    logic shared_bf16_req_valid;
    logic shared_bf16_req_ready;
    hardware_types_pkg::bf16_request_t shared_bf16_req;
    hardware_types_pkg::bf16_response_t shared_bf16_rsp;
    logic shared_bf16_rsp_valid;
    logic shared_bf16_rsp_ready;
    logic shared_bf16_pipe_rsp_valid;
    logic shared_bf16_pipe_rsp_ready;
    logic [1023:0] shared_bf16_pipe_rsp_values;
    logic [63:0] shared_bf16_pipe_rsp_lane_mask;
    logic [15:0] shared_bf16_pipe_rsp_tag;
    logic shared_bf16_response_valid;
    logic [1103:0] shared_bf16_response_data;
    logic shared_bf16_idle;
    logic bf16_request_fire;
    logic bf16_response_fire;

    assign shared_bf16_abort_request = |bf16_abort_request;
    assign bf16_abort_ack = {5{shared_bf16_abort_ack}};
    assign bf16_route_client = bf16_outstanding == 0 ?
        select_client5(bf16_req_valid) : bf16_active_client;
    always_comb begin
        shared_bf16_req_valid = 1'b0;
        shared_bf16_req = '0;
        shared_bf16_rsp_ready = 1'b0;
        bf16_req_ready = '0;
        bf16_rsp_valid = '0;
        bf16_rsp[0] = '0;
        bf16_rsp[1] = '0;
        bf16_rsp[2] = '0;
        bf16_rsp[3] = '0;
        bf16_rsp[4] = '0;
        case (bf16_route_client)
            3'd0: begin
                shared_bf16_req_valid = bf16_req_valid[0];
                shared_bf16_req = bf16_req[0];
                bf16_req_ready[0] = shared_bf16_req_ready;
            end
            3'd1: begin
                shared_bf16_req_valid = bf16_req_valid[1];
                shared_bf16_req = bf16_req[1];
                bf16_req_ready[1] = shared_bf16_req_ready;
            end
            3'd2: begin
                shared_bf16_req_valid = bf16_req_valid[2];
                shared_bf16_req = bf16_req[2];
                bf16_req_ready[2] = shared_bf16_req_ready;
            end
            3'd3: begin
                shared_bf16_req_valid = bf16_req_valid[3];
                shared_bf16_req = bf16_req[3];
                bf16_req_ready[3] = shared_bf16_req_ready;
            end
            3'd4: begin
                shared_bf16_req_valid = bf16_req_valid[4];
                shared_bf16_req = bf16_req[4];
                bf16_req_ready[4] = shared_bf16_req_ready;
            end
            default: begin end
        endcase
        case (bf16_active_client)
            3'd0: begin
                bf16_rsp_valid[0] = shared_bf16_rsp_valid;
                bf16_rsp[0] = shared_bf16_rsp;
                shared_bf16_rsp_ready = bf16_rsp_ready[0];
            end
            3'd1: begin
                bf16_rsp_valid[1] = shared_bf16_rsp_valid;
                bf16_rsp[1] = shared_bf16_rsp;
                shared_bf16_rsp_ready = bf16_rsp_ready[1];
            end
            3'd2: begin
                bf16_rsp_valid[2] = shared_bf16_rsp_valid;
                bf16_rsp[2] = shared_bf16_rsp;
                shared_bf16_rsp_ready = bf16_rsp_ready[2];
            end
            3'd3: begin
                bf16_rsp_valid[3] = shared_bf16_rsp_valid;
                bf16_rsp[3] = shared_bf16_rsp;
                shared_bf16_rsp_ready = bf16_rsp_ready[3];
            end
            3'd4: begin
                bf16_rsp_valid[4] = shared_bf16_rsp_valid;
                bf16_rsp[4] = shared_bf16_rsp;
                shared_bf16_rsp_ready = bf16_rsp_ready[4];
            end
            default: begin end
        endcase
    end
    assign bf16_request_fire = shared_bf16_req_valid && shared_bf16_req_ready;
    assign bf16_response_fire = shared_bf16_rsp_valid && shared_bf16_rsp_ready;
    always_ff @(posedge clk) begin
        if (rst || shared_bf16_abort_ack) begin
            bf16_active_client <= CLIENT5_NONE;
            bf16_outstanding <= '0;
        end else begin
            case ({bf16_request_fire, bf16_response_fire})
                2'b10: begin
                    if (bf16_outstanding == 0)
                        bf16_active_client <= bf16_route_client;
                    bf16_outstanding <= bf16_outstanding + 8'd1;
                end
                2'b01: begin
                    bf16_outstanding <= bf16_outstanding - 8'd1;
                    if (bf16_outstanding == 8'd1)
                        bf16_active_client <= CLIENT5_NONE;
                end
                default: begin end
            endcase
        end
    end
    bf16_vector_pipe #(.LANES(64), .TAG_WIDTH(16)) shared_bf16 (
        .clk(clk), .rst(rst), .abort_request(shared_bf16_abort_request),
        .abort_ack(shared_bf16_abort_ack), .req_valid(shared_bf16_req_valid),
        .req_ready(shared_bf16_req_ready),
        .req_operation(shared_bf16_req.operation),
        .req_values(shared_bf16_req.values),
        .req_paired_values(shared_bf16_req.paired_values),
        .req_factor0_values(shared_bf16_req.factor0_values),
        .req_factor1_values(shared_bf16_req.factor1_values),
        .req_lane_mask(shared_bf16_req.lane_mask),
        .req_tag(shared_bf16_req.tag),
        .rsp_valid(shared_bf16_pipe_rsp_valid),
        .rsp_ready(shared_bf16_pipe_rsp_ready),
        .rsp_values(shared_bf16_pipe_rsp_values),
        .rsp_lane_mask(shared_bf16_pipe_rsp_lane_mask),
        .rsp_tag(shared_bf16_pipe_rsp_tag), .idle(shared_bf16_idle),
        .trace_sample_valid(), .trace_sample_stage(), .trace_sample_values(),
        .trace_sample_lane_mask(), .trace_sample_tag());
    assign shared_bf16_pipe_rsp_ready = !shared_bf16_response_valid ||
        shared_bf16_rsp_ready;
    assign shared_bf16_rsp_valid = shared_bf16_response_valid;
    assign {shared_bf16_rsp.values, shared_bf16_rsp.lane_mask,
            shared_bf16_rsp.tag} = shared_bf16_response_data;
    always_ff @(posedge clk) begin
        if (rst || shared_bf16_abort_request) begin
            shared_bf16_response_valid <= 1'b0;
        end else if (shared_bf16_pipe_rsp_ready) begin
            shared_bf16_response_valid <= shared_bf16_pipe_rsp_valid;
            if (shared_bf16_pipe_rsp_valid)
                shared_bf16_response_data <= {
                    shared_bf16_pipe_rsp_values,
                    shared_bf16_pipe_rsp_lane_mask,
                    shared_bf16_pipe_rsp_tag};
        end
    end

    logic [2:0] max_active_client;
    logic [2:0] max_route_client;
    logic [7:0] max_outstanding;
    logic shared_max_req_valid;
    logic shared_max_req_ready;
    hardware_types_pkg::maximum_request_t shared_max_req;
    hardware_types_pkg::maximum_response_t shared_max_rsp;
    logic shared_max_rsp_valid;
    logic shared_max_rsp_ready;
    logic max_request_fire;
    logic max_response_fire;

    assign max_route_client = max_outstanding == 0 ?
        select_client4(max_req_valid) : max_active_client;
    always_comb begin
        shared_max_req_valid = 1'b0;
        shared_max_req = '0;
        shared_max_rsp_ready = 1'b0;
        max_req_ready = '0;
        max_rsp_valid = '0;
        max_rsp[0] = '0;
        max_rsp[1] = '0;
        max_rsp[2] = '0;
        max_rsp[3] = '0;
        case (max_route_client)
            3'd0, 3'd1, 3'd2, 3'd3: begin
                case (max_route_client)
                    3'd0: begin
                        shared_max_req_valid = max_req_valid[0];
                        shared_max_req = max_req[0];
                        max_req_ready[0] = shared_max_req_ready;
                    end
                    3'd1: begin
                        shared_max_req_valid = max_req_valid[1];
                        shared_max_req = max_req[1];
                        max_req_ready[1] = shared_max_req_ready;
                    end
                    3'd2: begin
                        shared_max_req_valid = max_req_valid[2];
                        shared_max_req = max_req[2];
                        max_req_ready[2] = shared_max_req_ready;
                    end
                    default: begin
                        shared_max_req_valid = max_req_valid[3];
                        shared_max_req = max_req[3];
                        max_req_ready[3] = shared_max_req_ready;
                    end
                endcase
            end
            default: begin end
        endcase
        case (max_active_client)
            3'd0: begin
                max_rsp_valid[0] = shared_max_rsp_valid;
                max_rsp[0] = shared_max_rsp;
                shared_max_rsp_ready = max_rsp_ready[0];
            end
            3'd1: begin
                max_rsp_valid[1] = shared_max_rsp_valid;
                max_rsp[1] = shared_max_rsp;
                shared_max_rsp_ready = max_rsp_ready[1];
            end
            3'd2: begin
                max_rsp_valid[2] = shared_max_rsp_valid;
                max_rsp[2] = shared_max_rsp;
                shared_max_rsp_ready = max_rsp_ready[2];
            end
            3'd3: begin
                max_rsp_valid[3] = shared_max_rsp_valid;
                max_rsp[3] = shared_max_rsp;
                shared_max_rsp_ready = max_rsp_ready[3];
            end
            default: begin end
        endcase
    end
    assign max_request_fire = shared_max_req_valid && shared_max_req_ready;
    assign max_response_fire = shared_max_rsp_valid && shared_max_rsp_ready;
    always_ff @(posedge clk) begin
        if (rst) begin
            max_active_client <= CLIENT4_NONE;
            max_outstanding <= '0;
        end else begin
            case ({max_request_fire, max_response_fire})
                2'b10: begin
                    if (max_outstanding == 0)
                        max_active_client <= max_route_client;
                    max_outstanding <= max_outstanding + 8'd1;
                end
                2'b01: begin
                    max_outstanding <= max_outstanding - 8'd1;
                    if (max_outstanding == 8'd1)
                        max_active_client <= CLIENT4_NONE;
                end
                default: begin end
            endcase
        end
    end
    bf16_tile_max_pipe #(.TAG_WIDTH(16)) shared_tile_max (
        .clk(clk), .rst(rst), .req_valid(shared_max_req_valid),
        .req_ready(shared_max_req_ready),
        .req_magnitude(shared_max_req.magnitude),
        .req_values(shared_max_req.values),
        .req_lane_mask(shared_max_req.lane_mask),
        .req_tag(shared_max_req.tag),
        .rsp_valid(shared_max_rsp_valid), .rsp_ready(shared_max_rsp_ready),
        .rsp_values(shared_max_rsp.values),
        .rsp_row_mask(shared_max_rsp.row_mask),
        .rsp_tag(shared_max_rsp.tag));

    logic [1:0] pe_active_client;
    logic [1:0] pe_route_client;
    logic shared_pe_abort_request;
    logic shared_pe_abort_ack;
    logic shared_pe_req_valid;
    logic shared_pe_req_ready;
    hardware_types_pkg::pe_request_t shared_pe_req;
    hardware_types_pkg::pe_response_t shared_pe_accum_result;
    logic shared_pe_accum_result_valid;
    logic shared_pe_accum_result_ready;
    logic shared_pe_idle;
    assign pe_idle = shared_pe_idle;
    logic pe_request_fire;

    assign shared_pe_abort_request = |pe_abort_request;
    assign pe_abort_ack = {2{shared_pe_abort_ack}};
    assign pe_route_client = shared_pe_idle ?
        select_client2(pe_req_valid) : pe_active_client;
    always_comb begin
        shared_pe_req_valid = 1'b0;
        shared_pe_req = '0;
        shared_pe_accum_result_ready = 1'b0;
        pe_req_ready = '0;
        pe_accum_result_valid = '0;
        pe_accum_result[0] = '0;
        pe_accum_result[1] = '0;
        case (pe_route_client)
            2'd0: begin
                shared_pe_req_valid = pe_req_valid[0];
                shared_pe_req = pe_req[0];
                pe_req_ready[0] = shared_pe_req_ready;
            end
            2'd1: begin
                shared_pe_req_valid = pe_req_valid[1];
                shared_pe_req = pe_req[1];
                pe_req_ready[1] = shared_pe_req_ready;
            end
            default: begin end
        endcase
        case (pe_active_client)
            2'd0: begin
                pe_accum_result_valid[0] = shared_pe_accum_result_valid;
                pe_accum_result[0] = shared_pe_accum_result;
                shared_pe_accum_result_ready = pe_accum_result_ready[0];
            end
            2'd1: begin
                pe_accum_result_valid[1] = shared_pe_accum_result_valid;
                pe_accum_result[1] = shared_pe_accum_result;
                shared_pe_accum_result_ready = pe_accum_result_ready[1];
            end
            default: begin end
        endcase
    end
    assign pe_request_fire = shared_pe_req_valid && shared_pe_req_ready;
    always_ff @(posedge clk) begin
        if (rst || shared_pe_abort_ack) begin
            pe_active_client <= CLIENT2_NONE;
        end else begin
            if (shared_pe_idle && !pe_request_fire)
                pe_active_client <= CLIENT2_NONE;
            if (pe_request_fire && shared_pe_idle)
                pe_active_client <= pe_route_client;
        end
    end
    mixed_precision_pe shared_pe (
        .clk(clk), .rst(rst), .abort_request(shared_pe_abort_request),
        .abort_ack(shared_pe_abort_ack), .req_valid(shared_pe_req_valid),
        .req_ready(shared_pe_req_ready), .req_mode(shared_pe_req.mode),
        .req_activation_payload(shared_pe_req.activation_payload),
        .req_weight_payload(shared_pe_req.weight_payload),
        .req_row_mask(shared_pe_req.row_mask), .req_k_mask(shared_pe_req.k_mask),
        .req_col_mask(shared_pe_req.column_mask),
        .req_first_k_step(shared_pe_req.first_k_step),
        .req_last_k_step(shared_pe_req.last_k_step),
        .req_mixed_phase(shared_pe_req.mixed_phase),
        .req_mixed_phase_first(shared_pe_req.mixed_phase_first),
        .req_mixed_a8_rows(shared_pe_req.mixed_a8_rows),
        .req_activation_scales(shared_pe_req.activation_scales),
        .req_weight_scales(shared_pe_req.weight_scales),
        .req_tag(shared_pe_req.tag),
        .accum_result_valid(shared_pe_accum_result_valid),
        .accum_result_ready(shared_pe_accum_result_ready),
        .accum_result_accumulators(shared_pe_accum_result.accumulators),
        .accum_result_mask(shared_pe_accum_result.accumulator_mask),
        .accum_result_mode(shared_pe_accum_result.mode),
        .accum_result_last_k_step(shared_pe_accum_result.last_k_step),
        .accum_result_activation_scales(
            shared_pe_accum_result.activation_scales),
        .accum_result_weight_scales(shared_pe_accum_result.weight_scales),
        .accum_result_tag(shared_pe_accum_result.tag), .idle(shared_pe_idle),
        .accepted_operand_count(), .accepted_accum_result_count());

    logic [1:0] rescale_active_client;
    logic [1:0] rescale_route_client;
    logic [7:0] rescale_outstanding;
    logic shared_rescale_abort_request;
    logic shared_rescale_abort_pending;
    logic shared_rescale_abort_ack;
    logic shared_rescale_req_valid;
    logic shared_rescale_req_ready;
    hardware_types_pkg::rescale_request_t shared_rescale_req;
    hardware_types_pkg::rescale_response_t shared_rescale_rsp;
    logic shared_rescale_rsp_valid;
    logic shared_rescale_rsp_ready;
    logic shared_rescale_idle;
    assign rescale_idle = shared_rescale_idle;
    logic rescale_claim_fire;
    logic rescale_request_fire;
    logic rescale_response_fire;

    assign rescale_abort_ack = {2{shared_rescale_abort_ack}};
    assign shared_rescale_abort_request = shared_rescale_abort_pending;
    assign rescale_route_client = rescale_active_client != CLIENT2_NONE ?
        rescale_active_client : select_client2(rescale_req_valid | rescale_claim_valid);
    always_comb begin
        shared_rescale_req_valid = 1'b0;
        shared_rescale_req = '0;
        shared_rescale_rsp_ready = 1'b0;
        rescale_claim_ready = '0;
        rescale_req_ready = '0;
        rescale_rsp_valid = '0;
        rescale_rsp[0] = '0;
        rescale_rsp[1] = '0;
        case (rescale_route_client)
            2'd0: begin
                rescale_claim_ready[0] = rescale_active_client != 2'd1 &&
                    shared_rescale_idle;
                shared_rescale_req_valid = rescale_req_valid[0];
                shared_rescale_req = rescale_req[0];
                rescale_req_ready[0] = shared_rescale_req_ready;
            end
            2'd1: begin
                rescale_claim_ready[1] = rescale_active_client != 2'd0 &&
                    shared_rescale_idle;
                shared_rescale_req_valid = rescale_req_valid[1];
                shared_rescale_req = rescale_req[1];
                rescale_req_ready[1] = shared_rescale_req_ready;
            end
            default: begin end
        endcase
        case (rescale_active_client)
            2'd0: begin
                rescale_rsp_valid[0] = shared_rescale_rsp_valid;
                rescale_rsp[0] = shared_rescale_rsp;
                shared_rescale_rsp_ready = rescale_rsp_ready[0];
            end
            2'd1: begin
                rescale_rsp_valid[1] = shared_rescale_rsp_valid;
                rescale_rsp[1] = shared_rescale_rsp;
                shared_rescale_rsp_ready = rescale_rsp_ready[1];
            end
            default: begin end
        endcase
    end
    assign rescale_claim_fire = |(rescale_claim_valid & rescale_claim_ready);
    assign rescale_request_fire = shared_rescale_req_valid &&
        shared_rescale_req_ready;
    assign rescale_response_fire = shared_rescale_rsp_valid &&
        shared_rescale_rsp_ready;
    always_ff @(posedge clk) begin
        if (rst) begin
            shared_rescale_abort_pending <= 1'b0;
        end else begin
            if (shared_rescale_abort_ack)
                shared_rescale_abort_pending <= 1'b0;
            else if (|rescale_abort_request)
                shared_rescale_abort_pending <= 1'b1;
        end
    end
    always_ff @(posedge clk) begin
        if (rst || shared_rescale_abort_ack) begin
            rescale_active_client <= CLIENT2_NONE;
            rescale_outstanding <= '0;
        end else begin
            if (rescale_claim_fire)
                rescale_active_client <= rescale_route_client;
            case ({rescale_request_fire, rescale_response_fire})
                2'b10: begin
                    if (rescale_outstanding == 0)
                        rescale_active_client <= rescale_route_client;
                    rescale_outstanding <= rescale_outstanding + 8'd1;
                end
                2'b01: begin
                    rescale_outstanding <= rescale_outstanding - 8'd1;
                    if (rescale_outstanding == 8'd1)
                        rescale_active_client <= CLIENT2_NONE;
                end
                default: begin end
            endcase
        end
    end
    int32_bf16_rescale #(.LANES(64)) shared_rescale (
        .clk(clk), .rst(rst), .abort_request(shared_rescale_abort_request),
        .abort_ack(shared_rescale_abort_ack), .req_valid(shared_rescale_req_valid),
        .req_ready(shared_rescale_req_ready),
        .req_accumulators(shared_rescale_req.accumulators),
        .req_activation_scales(shared_rescale_req.activation_scales),
        .req_weight_scales(shared_rescale_req.weight_scales),
        .req_qk_scale_bf16(shared_rescale_req.qk_scale_bf16),
        .req_rescale_mode(shared_rescale_req.rescale_mode),
        .req_lane_mask(shared_rescale_req.lane_mask),
        .req_tag(shared_rescale_req.tag), .rsp_valid(shared_rescale_rsp_valid),
        .rsp_ready(shared_rescale_rsp_ready),
        .rsp_values(shared_rescale_rsp.values),
        .rsp_lane_mask(shared_rescale_rsp.lane_mask),
        .rsp_tag(shared_rescale_rsp.tag),
        .idle(shared_rescale_idle));

    logic [2:0] quant_active_client;
    logic [2:0] quant_route_client;
    logic [7:0] quant_scale_outstanding;
    logic [7:0] quant_values_outstanding;
    logic shared_quant_abort_request;
    logic shared_quant_abort_ack;
    logic shared_quant_scale_req_valid;
    logic shared_quant_scale_req_ready;
    hardware_types_pkg::quant_scale_request_t shared_quant_scale_req;
    hardware_types_pkg::quant_scale_response_t shared_quant_scale_rsp;
    logic shared_quant_scale_rsp_valid;
    logic shared_quant_scale_rsp_ready;
    logic shared_quant_req_valid;
    logic shared_quant_req_ready;
    hardware_types_pkg::quant_values_request_t shared_quant_values_req;
    hardware_types_pkg::quant_values_response_t shared_quant_values_rsp;
    logic shared_quant_rsp_valid;
    logic shared_quant_rsp_ready;
    logic shared_quant_idle;
    logic quant_scale_request_fire;
    logic quant_scale_response_fire;
    logic quant_values_request_fire;
    logic quant_values_response_fire;

    assign shared_quant_abort_request = |quant_abort_request;
    assign quant_abort_ack = {4{shared_quant_abort_ack}};
    always_comb begin
        quant_route_client = quant_active_client;
        if (quant_active_client == CLIENT4_NONE) begin
            quant_route_client =
                select_client4(quant_scale_req_valid | quant_values_req_valid);
        end else if (quant_scale_outstanding == 0 &&
                     quant_values_outstanding == 0 && shared_quant_idle) begin
            if (quant_scale_req_valid[quant_active_client[1:0]] ||
                quant_values_req_valid[quant_active_client[1:0]]) begin
                quant_route_client = quant_active_client;
            end else if (|quant_scale_req_valid) begin
                // A new scale request replaces the prior client's quantizer
                // configuration only after its response pipeline has drained.
                quant_route_client = select_client4(quant_scale_req_valid);
            end
        end
    end
    always_comb begin
        shared_quant_scale_req_valid = 1'b0;
        shared_quant_scale_req = '0;
        shared_quant_scale_rsp_ready = 1'b0;
        shared_quant_req_valid = 1'b0;
        shared_quant_values_req = '0;
        shared_quant_rsp_ready = 1'b0;
        quant_scale_req_ready = '0;
        quant_scale_rsp_valid = '0;
        quant_values_req_ready = '0;
        quant_values_rsp_valid = '0;
        quant_scale_rsp[0] = '0;
        quant_scale_rsp[1] = '0;
        quant_scale_rsp[2] = '0;
        quant_scale_rsp[3] = '0;
        quant_values_rsp[0] = '0;
        quant_values_rsp[1] = '0;
        quant_values_rsp[2] = '0;
        quant_values_rsp[3] = '0;
        case (quant_route_client)
            3'd0, 3'd1, 3'd2, 3'd3: begin
                case (quant_route_client)
                    3'd0: begin
                        shared_quant_scale_req_valid = quant_scale_req_valid[0];
                        shared_quant_scale_req = quant_scale_req[0];
                        shared_quant_req_valid = quant_values_req_valid[0];
                        shared_quant_values_req = quant_values_req[0];
                    end
                    3'd1: begin
                        shared_quant_scale_req_valid = quant_scale_req_valid[1];
                        shared_quant_scale_req = quant_scale_req[1];
                        shared_quant_req_valid = quant_values_req_valid[1];
                        shared_quant_values_req = quant_values_req[1];
                    end
                    3'd2: begin
                        shared_quant_scale_req_valid = quant_scale_req_valid[2];
                        shared_quant_scale_req = quant_scale_req[2];
                        shared_quant_req_valid = quant_values_req_valid[2];
                        shared_quant_values_req = quant_values_req[2];
                    end
                    default: begin
                        shared_quant_scale_req_valid = quant_scale_req_valid[3];
                        shared_quant_scale_req = quant_scale_req[3];
                        shared_quant_req_valid = quant_values_req_valid[3];
                        shared_quant_values_req = quant_values_req[3];
                    end
                endcase
                quant_scale_req_ready[quant_route_client[1:0]] =
                    shared_quant_scale_req_ready;
                quant_values_req_ready[quant_route_client[1:0]] =
                    shared_quant_req_ready;
            end
            default: begin end
        endcase
        case (quant_active_client)
            3'd0: begin
                quant_scale_rsp_valid[0] = shared_quant_scale_rsp_valid;
                quant_scale_rsp[0] = shared_quant_scale_rsp;
                shared_quant_scale_rsp_ready = quant_scale_rsp_ready[0];
                quant_values_rsp_valid[0] = shared_quant_rsp_valid;
                quant_values_rsp[0] = shared_quant_values_rsp;
                shared_quant_rsp_ready = quant_values_rsp_ready[0];
            end
            3'd1: begin
                quant_scale_rsp_valid[1] = shared_quant_scale_rsp_valid;
                quant_scale_rsp[1] = shared_quant_scale_rsp;
                shared_quant_scale_rsp_ready = quant_scale_rsp_ready[1];
                quant_values_rsp_valid[1] = shared_quant_rsp_valid;
                quant_values_rsp[1] = shared_quant_values_rsp;
                shared_quant_rsp_ready = quant_values_rsp_ready[1];
            end
            3'd2: begin
                quant_scale_rsp_valid[2] = shared_quant_scale_rsp_valid;
                quant_scale_rsp[2] = shared_quant_scale_rsp;
                shared_quant_scale_rsp_ready = quant_scale_rsp_ready[2];
                quant_values_rsp_valid[2] = shared_quant_rsp_valid;
                quant_values_rsp[2] = shared_quant_values_rsp;
                shared_quant_rsp_ready = quant_values_rsp_ready[2];
            end
            3'd3: begin
                quant_scale_rsp_valid[3] = shared_quant_scale_rsp_valid;
                quant_scale_rsp[3] = shared_quant_scale_rsp;
                shared_quant_scale_rsp_ready = quant_scale_rsp_ready[3];
                quant_values_rsp_valid[3] = shared_quant_rsp_valid;
                quant_values_rsp[3] = shared_quant_values_rsp;
                shared_quant_rsp_ready = quant_values_rsp_ready[3];
            end
            default: begin end
        endcase
    end
    assign quant_scale_request_fire = shared_quant_scale_req_valid &&
        shared_quant_scale_req_ready;
    assign quant_scale_response_fire = shared_quant_scale_rsp_valid &&
        shared_quant_scale_rsp_ready;
    assign quant_values_request_fire = shared_quant_req_valid && shared_quant_req_ready;
    assign quant_values_response_fire = shared_quant_rsp_valid && shared_quant_rsp_ready;
    always_ff @(posedge clk) begin
        if (rst || shared_quant_abort_ack) begin
            quant_active_client <= CLIENT4_NONE;
            quant_scale_outstanding <= '0;
            quant_values_outstanding <= '0;
        end else begin
            case ({quant_scale_request_fire, quant_scale_response_fire})
                2'b10: quant_scale_outstanding <=
                    quant_scale_outstanding + 8'd1;
                2'b01: quant_scale_outstanding <=
                    quant_scale_outstanding - 8'd1;
                default: begin end
            endcase
            case ({quant_values_request_fire, quant_values_response_fire})
                2'b10: quant_values_outstanding <= quant_values_outstanding + 8'd1;
                2'b01: quant_values_outstanding <= quant_values_outstanding - 8'd1;
                default: begin end
            endcase
            if (quant_scale_request_fire ||
                (quant_active_client == CLIENT4_NONE && quant_values_request_fire))
                quant_active_client <= quant_route_client;
        end
    end
    activation_quantizer #(.LANES(64)) shared_quantizer (
        .clk(clk), .rst(rst), .abort_request(shared_quant_abort_request),
        .abort_ack(shared_quant_abort_ack),
        .scale_req_valid(shared_quant_scale_req_valid),
        .scale_req_ready(shared_quant_scale_req_ready),
        .scale_req_a4_row_mask(shared_quant_scale_req.a4_row_mask),
        .scale_req_use_static_scale(shared_quant_scale_req.use_static_scale),
        .scale_req_static_scales_bf16(shared_quant_scale_req.static_scales_bf16),
        .scale_req_row_max_abs(shared_quant_scale_req.row_max_abs),
        .scale_req_row_mask(shared_quant_scale_req.row_mask),
        .scale_req_clip_ratio_bf16(shared_quant_scale_req.clip_ratio_bf16),
        .scale_rsp_valid(shared_quant_scale_rsp_valid),
        .scale_rsp_ready(shared_quant_scale_rsp_ready),
        .scale_rsp_values_bf16(shared_quant_scale_rsp.values_bf16),
        .quantized_req_valid(shared_quant_req_valid),
        .quantized_req_ready(shared_quant_req_ready),
        .quantized_req_values_bf16(shared_quant_values_req.values_bf16),
        .quantized_req_lane_mask(shared_quant_values_req.lane_mask),
        .quantized_req_tag(shared_quant_values_req.tag),
        .quantized_rsp_valid(shared_quant_rsp_valid),
        .quantized_rsp_ready(shared_quant_rsp_ready),
        .quantized_rsp_values(shared_quant_values_rsp.values),
        .quantized_rsp_lane_mask(shared_quant_values_rsp.lane_mask),
        .quantized_rsp_tag(shared_quant_values_rsp.tag),
        .idle(shared_quant_idle));

    logic [1:0] writer_active_client;
    logic [1:0] writer_route_client;
    logic shared_writer_cfg_valid;
    logic shared_writer_cfg_ready;
    hardware_types_pkg::activation_writer_config_t shared_writer_cfg;
    logic shared_writer_scale_valid;
    logic shared_writer_scale_ready;
    hardware_types_pkg::activation_writer_scale_t shared_writer_scale;
    logic shared_writer_quantized_valid;
    logic shared_writer_quantized_ready;
    hardware_types_pkg::activation_writer_values_t shared_writer_quantized;
    logic shared_writer_done_pulse;
    logic shared_writer_error;
    logic writer_cfg_fire;

    assign writer_route_client = writer_active_client == CLIENT2_NONE ?
        select_client2(writer_cfg_valid) : writer_active_client;
    always_comb begin
        shared_writer_cfg_valid = 1'b0;
        shared_writer_cfg = '0;
        shared_writer_scale_valid = 1'b0;
        shared_writer_scale = '0;
        shared_writer_quantized_valid = 1'b0;
        shared_writer_quantized = '0;
        writer_cfg_ready = '0;
        writer_scale_ready = '0;
        writer_quantized_ready = '0;
        writer_done_pulse = '0;
        writer_error = '0;
        case (writer_route_client)
            2'd0: begin
                shared_writer_cfg_valid = writer_cfg_valid[0];
                shared_writer_cfg = writer_cfg[0];
                shared_writer_scale_valid = writer_scale_valid[0];
                shared_writer_scale = writer_scale[0];
                shared_writer_quantized_valid = writer_quantized_valid[0];
                shared_writer_quantized = writer_quantized[0];
                writer_cfg_ready[0] = shared_writer_cfg_ready;
                writer_scale_ready[0] = shared_writer_scale_ready;
                writer_quantized_ready[0] = shared_writer_quantized_ready;
            end
            2'd1: begin
                shared_writer_cfg_valid = writer_cfg_valid[1];
                shared_writer_cfg = writer_cfg[1];
                shared_writer_scale_valid = writer_scale_valid[1];
                shared_writer_scale = writer_scale[1];
                shared_writer_quantized_valid = writer_quantized_valid[1];
                shared_writer_quantized = writer_quantized[1];
                writer_cfg_ready[1] = shared_writer_cfg_ready;
                writer_scale_ready[1] = shared_writer_scale_ready;
                writer_quantized_ready[1] = shared_writer_quantized_ready;
            end
            default: begin end
        endcase
        case (writer_active_client)
            2'd0: begin
                writer_done_pulse[0] = shared_writer_done_pulse;
                writer_error[0] = shared_writer_error;
            end
            2'd1: begin
                writer_done_pulse[1] = shared_writer_done_pulse;
                writer_error[1] = shared_writer_error;
            end
            default: begin end
        endcase
    end
    assign writer_cfg_fire = shared_writer_cfg_valid && shared_writer_cfg_ready;
    always_ff @(posedge clk) begin
        if (rst || |writer_abort_request) begin
            writer_active_client <= CLIENT2_NONE;
        end else begin
            if (writer_cfg_fire)
                writer_active_client <= writer_route_client;
            if (shared_writer_done_pulse || shared_writer_error)
                writer_active_client <= CLIENT2_NONE;
        end
    end
    activation_quantized_writer shared_activation_writer (
        .clk(clk), .rst(rst), .abort_request(|writer_abort_request),
        .cfg_valid(shared_writer_cfg_valid), .cfg_ready(shared_writer_cfg_ready),
        .cfg_mode(shared_writer_cfg.mode),
        .cfg_physical_row_base(shared_writer_cfg.physical_row_base),
        .cfg_row_count(shared_writer_cfg.row_count),
        .cfg_elements_per_row(shared_writer_cfg.elements_per_row),
        .cfg_activation_base_byte_offset(shared_writer_cfg.activation_base_byte_offset),
        .cfg_activation_limit_byte_offset(shared_writer_cfg.activation_limit_byte_offset),
        .cfg_mixed_rows(shared_writer_cfg.mixed_rows),
        .cfg_row_partial(shared_writer_cfg.row_partial),
        .cfg_segment_count(shared_writer_cfg.segment_count),
        .cfg_segment_mode(shared_writer_cfg.segment_mode),
        .cfg_segment_row_base(shared_writer_cfg.segment_row_base),
        .cfg_segment_row_count(shared_writer_cfg.segment_row_count),
        .cfg_mixed_group_enable(shared_writer_cfg.mixed_group_enable),
        .cfg_compute_group_count(shared_writer_cfg.compute_group_count),
        .cfg_row_precision_a8(shared_writer_cfg.row_precision_a8),
        .cfg_row_compute_group(shared_writer_cfg.row_compute_group),
        .cfg_row_pe_slot(shared_writer_cfg.row_pe_slot),
        .cfg_row_phase_mask(shared_writer_cfg.row_phase_mask),
        .scale_valid(shared_writer_scale_valid),
        .scale_ready(shared_writer_scale_ready),
        .scale_row_base(shared_writer_scale.row_base),
        .scale_row_mask(shared_writer_scale.row_mask),
        .scale_values_bf16(shared_writer_scale.values_bf16),
        .scale_write_valid(activation_scale_valid),
        .scale_write_ready(activation_scale_ready),
        .scale_write_row_base(activation_scale_row_base),
        .scale_write_row_mask(activation_scale_row_mask),
        .scale_write_values_bf16(activation_scale_values),
        .in_valid(shared_writer_quantized_valid), .in_ready(shared_writer_quantized_ready),
        .in_physical_row_base(shared_writer_quantized.physical_row_base),
        .in_element_base(shared_writer_quantized.element_base),
        .in_values(shared_writer_quantized.values),
        .in_lane_mask(shared_writer_quantized.lane_mask),
        .in_tag(shared_writer_quantized.tag),
        .write_valid(activation_write_valid),
        .write_ready(activation_write_ready),
        .write_slot_valid(activation_write_slot_valid),
        .write_byte_address(activation_write_address),
        .write_data(activation_write_data),
        .write_byte_enable(activation_write_byte_enable),
        .write_tag(activation_write_tag),
        .done_pulse(shared_writer_done_pulse),
        .error(shared_writer_error));

`ifndef SYNTHESIS
    logic stalled_bf16_response;
    logic [1103:0] stalled_bf16_response_data;
    always_ff @(posedge clk) begin
        if (rst || shared_bf16_abort_request) begin
            stalled_bf16_response <= 1'b0;
            stalled_bf16_response_data <= '0;
        end else begin
            if (stalled_bf16_response)
                assert (shared_bf16_rsp_valid &&
                        shared_bf16_response_data ===
                            stalled_bf16_response_data)
                    else $error("shared BF16 response changed while stalled");
            stalled_bf16_response <= shared_bf16_rsp_valid &&
                !shared_bf16_rsp_ready;
            if (shared_bf16_rsp_valid && !shared_bf16_rsp_ready)
                stalled_bf16_response_data <= shared_bf16_response_data;
            assert ($onehot0(reduction_req_valid))
                else $error("shared reduction received simultaneous client requests");
            assert ($onehot0(bf16_req_valid))
                else $error("shared BF16 received simultaneous client requests");
            assert ($onehot0(max_req_valid))
                else $error("shared max received simultaneous client requests");
            assert ($onehot0(pe_req_valid))
                else $error("shared PE received simultaneous client requests");
            assert ($onehot0(rescale_req_valid))
                else $error("shared rescale received simultaneous client requests");
            assert ($onehot0(rescale_req_valid | rescale_claim_valid))
                else $error("shared rescale received simultaneous client claims or requests");
            assert ($onehot0(quant_scale_req_valid | quant_values_req_valid))
                else $error("shared quantizer received simultaneous client requests");
            assert ($onehot0(writer_cfg_valid))
                else $error("shared activation writer received simultaneous configurations");
            assert (!(reduction_response_fire && reduction_outstanding == 0))
                else $error("shared reduction response retired without an accepted request");
            assert (!(bf16_response_fire && bf16_outstanding == 0))
                else $error("shared BF16 response retired without an accepted request");
            if (!shared_bf16_abort_request && shared_bf16_rsp_valid) begin
                assert (!$isunknown({shared_bf16_rsp.lane_mask, shared_bf16_rsp.tag}))
                    else $error("shared BF16 response mask or tag contains X/Z");
                for (integer lane = 0; lane < hardware_types_pkg::BF16_LANES; lane = lane + 1)
                    if (shared_bf16_rsp.lane_mask[lane] === 1'b1)
                        assert (!$isunknown(shared_bf16_rsp.values[lane*16 +: 16]))
                            else $error("shared BF16 active response lane contains X/Z lane=%0d tag=%h",
                                lane, shared_bf16_rsp.tag);
            end
            assert (!(max_response_fire && max_outstanding == 0))
                else $error("shared max response retired without an accepted request");
            assert (!(rescale_response_fire && rescale_outstanding == 0))
                else $error("shared rescale response retired without an accepted request");
            assert (!(quant_scale_response_fire && quant_scale_outstanding == 0))
                else $error("shared quant scale response retired without an accepted request");
            assert (!(quant_values_response_fire && quant_values_outstanding == 0))
                else $error("shared quant quantized response retired without an accepted request");
            if (quant_active_client != CLIENT4_NONE && !shared_quant_idle)
                assert (quant_route_client == quant_active_client)
                    else $error("shared quantizer changed active client before pipeline drain");
            if (quant_values_request_fire && quant_active_client != CLIENT4_NONE)
                assert (quant_route_client == quant_active_client)
                    else $error("shared quantizer accepted input from a client other than the active client");
        end
    end
`endif
endmodule

`default_nettype wire
