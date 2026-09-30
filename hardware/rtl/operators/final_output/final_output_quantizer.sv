`default_nettype none

module final_output_quantizer #(
    parameter integer ELEMENTS = 4096
) (
    input  logic          clk,
    input  logic          rst,
    input  logic          start_valid,
    output logic          start_ready,
    input  logic [3:0]    start_group,
    input  logic [3:0]    start_rows,

    output logic          source_req_valid,
    input  logic          source_req_ready,
    output hardware_types_pkg::rms_tile_read_request_t source_req,
    input  logic          source_rsp_valid,
    output logic          source_rsp_ready,
    input  hardware_types_pkg::rms_tile_read_response_t source_rsp,

    output logic          max_req_valid,
    input  logic          max_req_ready,
    output hardware_types_pkg::maximum_request_t max_req,
    input  logic          max_rsp_valid,
    output logic          max_rsp_ready,
    input  hardware_types_pkg::maximum_response_t max_rsp,

    output logic          quant_scale_req_valid,
    input  logic          quant_scale_req_ready,
    output hardware_types_pkg::quant_scale_request_t quant_scale_req,
    input  logic          quant_scale_rsp_valid,
    output logic          quant_scale_rsp_ready,
    input  hardware_types_pkg::quant_scale_response_t quant_scale_rsp,
    output logic          quant_values_req_valid,
    input  logic          quant_values_req_ready,
    output hardware_types_pkg::quant_values_request_t quant_values_req,
    input  logic          quant_values_rsp_valid,
    output logic          quant_values_rsp_ready,
    input  hardware_types_pkg::quant_values_response_t quant_values_rsp,

    output logic          writer_cfg_valid,
    input  logic          writer_cfg_ready,
    output hardware_types_pkg::activation_writer_config_t writer_cfg,
    output logic          writer_scale_valid,
    input  logic          writer_scale_ready,
    output hardware_types_pkg::activation_writer_scale_t writer_scale,
    output logic          writer_values_valid,
    input  logic          writer_values_ready,
    output hardware_types_pkg::activation_writer_values_t writer_values,
    input  logic          writer_done_pulse,
    input  logic          writer_error,

    output logic          done_valid,
    input  logic          done_ready,
    output logic          error,
    output logic [3:0]    error_id
);
    localparam integer TILE_COUNT = ELEMENTS / 8;
    localparam logic [3:0] ERROR_START = 4'h1;
    localparam logic [3:0] ERROR_SOURCE = 4'h2;
    localparam logic [3:0] ERROR_MAXIMUM = 4'h3;
    localparam logic [3:0] ERROR_QUANTIZER = 4'h4;
    localparam logic [3:0] ERROR_WRITER = 4'h5;
    localparam logic [1:0] MODE_W8A8 = 2'd2;

    typedef enum logic [3:0] {
        IDLE,
        WRITER_CONFIG,
        MAX_STREAM,
        SCALE_REQUEST,
        SCALE_RESPONSE,
        QUANT_STREAM,
        WRITER_WAIT,
        COMPLETE
    } state_t;

    state_t state;
    logic [3:0] saved_group;
    logic [3:0] saved_rows;
    logic [7:0] row_mask;
    logic [127:0] row_max_abs;
    logic [9:0] source_issue_count;
    logic [9:0] source_response_count;
    logic [9:0] arithmetic_response_count;
    logic terminal_error;
    logic [3:0] terminal_error_id;
    logic source_req_fire;
    logic source_rsp_fire;
    logic max_rsp_fire;
    logic quant_values_rsp_fire;
    logic [15:0] expected_source_tag;
    logic [15:0] expected_arithmetic_tag;

    function automatic logic [15:0] max_abs2(
        input logic [15:0] lhs,
        input logic [15:0] rhs
    );
        max_abs2 = lhs[14:0] >= rhs[14:0] ?
            {1'b0, lhs[14:0]} : {1'b0, rhs[14:0]};
    endfunction

    function automatic logic [15:0] make_tag(
        input logic pass,
        input logic [3:0] group,
        input logic [8:0] tile
    );
        make_tag = {pass, group, tile, 2'b00};
    endfunction

    function automatic logic [63:0] active_lane_mask(
        input logic [7:0] active_token_count
    );
        for (integer row = 0; row < 8; row++)
            active_lane_mask[row*8 +: 8] = {8{active_token_count[row]}};
    endfunction

    assign start_ready = state == IDLE;
    assign done_valid = state == COMPLETE;
    assign error = done_valid && terminal_error;
    assign error_id = terminal_error_id;

    assign writer_cfg_valid = state == WRITER_CONFIG;
    always_comb begin
        writer_cfg = '0;
        writer_cfg.mode = MODE_W8A8;
        writer_cfg.physical_row_base = 6'd0;
        writer_cfg.row_count = {2'b00, saved_rows};
        writer_cfg.elements_per_row = 16'(ELEMENTS);
        writer_cfg.activation_base_byte_offset = 32'(saved_group) * 32'd32768;
        writer_cfg.activation_limit_byte_offset = writer_cfg.activation_base_byte_offset + 32'd32768;
    end

    assign source_req_valid =
        (state == MAX_STREAM || state == QUANT_STREAM) &&
        source_issue_count < 10'(TILE_COUNT) && !terminal_error;
    assign source_req.row_base = 6'd0;
    assign source_req.element = 16'(source_issue_count) * 16'd8;
    assign source_req.tag = make_tag(state == QUANT_STREAM, saved_group,
        source_issue_count[8:0]);
    assign source_req_fire = source_req_valid && source_req_ready;
    assign expected_source_tag = make_tag(state == QUANT_STREAM, saved_group,
        source_response_count[8:0]);

    assign max_req_valid = state == MAX_STREAM && source_rsp_valid &&
        source_rsp.tag == expected_source_tag &&
        source_rsp.lane_mask == active_lane_mask(row_mask);
    assign max_req.magnitude = 1'b1;
    assign max_req.values = source_rsp.values;
    assign max_req.lane_mask = source_rsp.lane_mask;
    assign max_req.tag = source_rsp.tag;

    assign quant_values_req_valid = state == QUANT_STREAM && source_rsp_valid &&
        source_rsp.tag == expected_source_tag &&
        source_rsp.lane_mask == active_lane_mask(row_mask);
    assign quant_values_req.values_bf16 = source_rsp.values;
    assign quant_values_req.lane_mask = source_rsp.lane_mask;
    assign quant_values_req.tag = source_rsp.tag;
    assign source_rsp_ready = state == MAX_STREAM ?
        ((source_rsp.tag == expected_source_tag &&
          source_rsp.lane_mask == active_lane_mask(row_mask)) ?
            max_req_ready : 1'b1) :
        state == QUANT_STREAM ?
        ((source_rsp.tag == expected_source_tag &&
          source_rsp.lane_mask == active_lane_mask(row_mask)) ?
            quant_values_req_ready : 1'b1) : 1'b0;
    assign source_rsp_fire = source_rsp_valid && source_rsp_ready;

    assign max_rsp_ready = state == MAX_STREAM;
    assign max_rsp_fire = max_rsp_valid && max_rsp_ready;
    assign expected_arithmetic_tag = make_tag(state == QUANT_STREAM,
        saved_group, arithmetic_response_count[8:0]);

    assign quant_scale_req_valid = state == SCALE_REQUEST && !terminal_error;
    assign quant_scale_req.a4_row_mask = 8'd0;
    assign quant_scale_req.clip_ratio_bf16 = 16'd0;
    assign quant_scale_req.use_static_scale = 1'b0;
    assign quant_scale_req.static_scales_bf16 = '0;
    assign quant_scale_req.row_max_abs = row_max_abs;
    assign quant_scale_req.row_mask = row_mask;
    assign quant_scale_rsp_ready = state == SCALE_RESPONSE &&
        writer_scale_ready;
    assign writer_scale_valid = state == SCALE_RESPONSE &&
        quant_scale_rsp_valid;
    assign writer_scale.row_base = 6'd0;
    assign writer_scale.row_mask = row_mask;
    assign writer_scale.values_bf16 = quant_scale_rsp.values_bf16;

    assign writer_values_valid = state == QUANT_STREAM &&
        quant_values_rsp_valid &&
        quant_values_rsp.tag == expected_arithmetic_tag &&
        quant_values_rsp.lane_mask == active_lane_mask(row_mask);
    assign writer_values.physical_row_base = 6'd0;
    assign writer_values.element_base =
        16'(arithmetic_response_count) * 16'd8;
    assign writer_values.values = quant_values_rsp.values;
    assign writer_values.lane_mask = quant_values_rsp.lane_mask;
    assign writer_values.tag = quant_values_rsp.tag;
    assign quant_values_rsp_ready = state == QUANT_STREAM ?
        ((quant_values_rsp.tag == expected_arithmetic_tag &&
          quant_values_rsp.lane_mask == active_lane_mask(row_mask)) ?
            writer_values_ready : 1'b1) : 1'b0;
    assign quant_values_rsp_fire = quant_values_rsp_valid &&
        quant_values_rsp_ready;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            saved_group <= '0;
            saved_rows <= '0;
            row_mask <= '0;
            row_max_abs <= '0;
            source_issue_count <= '0;
            source_response_count <= '0;
            arithmetic_response_count <= '0;
            terminal_error <= 1'b0;
            terminal_error_id <= '0;
        end else begin
            case (state)
                IDLE: if (start_valid && start_ready) begin
                    saved_group <= start_group;
                    saved_rows <= start_rows;
                    row_mask <= start_rows == 8 ? 8'hff :
                        8'((9'd1 << start_rows) - 1'b1);
                    row_max_abs <= '0;
                    terminal_error <= 1'b0;
                    terminal_error_id <= '0;
                    if (start_rows == 0 || start_rows > 8) begin
                        terminal_error <= 1'b1;
                        terminal_error_id <= ERROR_START;
                        state <= COMPLETE;
                    end else begin
                        state <= WRITER_CONFIG;
                    end
                end

                WRITER_CONFIG: if (writer_cfg_valid && writer_cfg_ready) begin
                    source_issue_count <= '0;
                    source_response_count <= '0;
                    arithmetic_response_count <= '0;
                    state <= MAX_STREAM;
                end

                MAX_STREAM: begin
                    if (source_req_fire)
                        source_issue_count <= source_issue_count + 10'd1;
                    if (source_rsp_fire) begin
                        if (source_rsp.tag != expected_source_tag ||
                            source_rsp.lane_mask != active_lane_mask(row_mask)) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_SOURCE;
                            state <= COMPLETE;
                        end else begin
                            source_response_count <= source_response_count + 10'd1;
                        end
                    end
                    if (max_rsp_fire) begin
                        if (max_rsp.tag != make_tag(1'b0, saved_group,
                                arithmetic_response_count[8:0]) ||
                            max_rsp.row_mask != row_mask) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_MAXIMUM;
                            state <= COMPLETE;
                        end else begin
                            for (integer row = 0; row < 8; row++) begin
                                if (row_mask[row])
                                    row_max_abs[row*16 +: 16] <= max_abs2(
                                        row_max_abs[row*16 +: 16],
                                        max_rsp.values[row*16 +: 16]);
                            end
                            arithmetic_response_count <=
                                arithmetic_response_count + 10'd1;
                            if (arithmetic_response_count == 10'(TILE_COUNT-1))
                                state <= SCALE_REQUEST;
                        end
                    end
                end

                SCALE_REQUEST: if (quant_scale_req_valid &&
                                          quant_scale_req_ready)
                    state <= SCALE_RESPONSE;

                SCALE_RESPONSE: if (quant_scale_rsp_valid &&
                                           quant_scale_rsp_ready) begin
                    source_issue_count <= '0;
                    source_response_count <= '0;
                    arithmetic_response_count <= '0;
                    state <= QUANT_STREAM;
                end

                QUANT_STREAM: begin
                    if (source_req_fire)
                        source_issue_count <= source_issue_count + 10'd1;
                    if (source_rsp_fire) begin
                        if (source_rsp.tag != expected_source_tag ||
                            source_rsp.lane_mask != active_lane_mask(row_mask)) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_SOURCE;
                            state <= COMPLETE;
                        end else begin
                            source_response_count <= source_response_count + 10'd1;
                        end
                    end
                    if (quant_values_rsp_fire) begin
                        if (quant_values_rsp.tag != expected_arithmetic_tag ||
                            quant_values_rsp.lane_mask !=
                                active_lane_mask(row_mask)) begin
                            terminal_error <= 1'b1;
                            terminal_error_id <= ERROR_QUANTIZER;
                            state <= COMPLETE;
                        end else begin
                            arithmetic_response_count <=
                                arithmetic_response_count + 10'd1;
                            if (arithmetic_response_count == 10'(TILE_COUNT-1))
                                state <= WRITER_WAIT;
                        end
                    end
                end

                WRITER_WAIT: if (writer_done_pulse || writer_error) begin
                    terminal_error <= writer_error;
                    terminal_error_id <= writer_error ? ERROR_WRITER : 4'd0;
                    state <= COMPLETE;
                end

                COMPLETE: if (done_valid && done_ready)
                    state <= IDLE;

                default: state <= IDLE;
            endcase
        end
    end

    initial begin
        if (ELEMENTS != 4096 || TILE_COUNT != 512)
            $error("final_output_quantizer parameter configuration is invalid");
    end
endmodule

`default_nettype wire
