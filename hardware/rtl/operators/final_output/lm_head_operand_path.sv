`default_nettype none

module lm_head_operand_path (
    input  logic          clk,
    input  logic          rst,
    input  logic          abort_request,
    output logic          abort_ack,

    input  logic          req_valid,
    output logic          req_ready,
    input  logic          req_panel,
    input  logic [3:0]    req_group,
    input  logic [8:0]    req_k_tile,
    input  logic [7:0]    req_row_mask,
    input  logic [15:0]   req_tag,
    input  logic [1535:0] activation_scales_bf16,

    output logic          local_read_valid,
    input  logic          local_read_ready,
    output hardware_types_pkg::matmul_read_request_t local_read,
    input  logic          local_rsp_valid,
    input  hardware_types_pkg::matmul_read_response_t local_rsp,

    output logic          rsp_valid,
    input  logic          rsp_ready,
    output logic [511:0]  rsp_activation,
    output logic [511:0]  rsp_weight,
    output logic [127:0]  rsp_activation_scales_bf16,
    output logic [7:0]    rsp_row_mask,
    output logic [15:0]   rsp_tag,

    output logic [63:0]   accepted_request_count,
    output logic [63:0]   returned_response_count
);
    localparam integer META_WIDTH = 1 + 128 + 8 + 16;
    logic [1:0] metadata_pending;
    logic [META_WIDTH-1:0] metadata_pipe [0:1];
    logic [127:0] selected_activation_scales;
    logic [3:0] selected_row_count;
    logic selected_start_high_half;
    logic request_shape_valid;
    logic [10:0] selected_group_word_offset;
    logic [3:0] selected_word_count;
    logic [511:0] aligned_activation;
    logic request_fire;
    logic abort_pending;

    function automatic logic [127:0] group_scales(
        input logic [3:0] group,
        input logic [1535:0] scales
    );
        case (group)
            4'd0: group_scales = scales[0*128 +: 128];
            4'd1: group_scales = scales[1*128 +: 128];
            4'd2: group_scales = scales[2*128 +: 128];
            4'd3: group_scales = scales[3*128 +: 128];
            4'd4: group_scales = scales[4*128 +: 128];
            4'd5: group_scales = scales[5*128 +: 128];
            4'd6: group_scales = scales[6*128 +: 128];
            4'd7: group_scales = scales[7*128 +: 128];
            4'd8: group_scales = scales[8*128 +: 128];
            4'd9: group_scales = scales[9*128 +: 128];
            4'd10: group_scales = scales[10*128 +: 128];
            default: group_scales = scales[11*128 +: 128];
        endcase
    endfunction

    function automatic logic [3:0] row_count_from_mask(
        input logic [7:0] row_mask
    );
        case (row_mask)
            8'h01: row_count_from_mask = 4'd1;
            8'h03: row_count_from_mask = 4'd2;
            8'h07: row_count_from_mask = 4'd3;
            8'h0f: row_count_from_mask = 4'd4;
            8'h1f: row_count_from_mask = 4'd5;
            8'h3f: row_count_from_mask = 4'd6;
            8'h7f: row_count_from_mask = 4'd7;
            8'hff: row_count_from_mask = 4'd8;
            default: row_count_from_mask = 4'd0;
        endcase
    endfunction

    function automatic logic [10:0] group_word_offset(
        input logic [8:0] k_tile,
        input logic [3:0] row_count
    );
        logic [10:0] extended_k;
        begin
            extended_k = {2'b00, k_tile};
            case (row_count)
                4'd1: group_word_offset = extended_k >> 1;
                4'd2: group_word_offset = extended_k;
                4'd3: group_word_offset = extended_k + (extended_k >> 1);
                4'd4: group_word_offset = extended_k << 1;
                4'd5: group_word_offset = (extended_k << 1) +
                    (extended_k >> 1);
                4'd6: group_word_offset = extended_k +
                    (extended_k << 1);
                4'd7: group_word_offset = (extended_k +
                    (extended_k << 1)) + (extended_k >> 1);
                default: group_word_offset = extended_k << 2;
            endcase
        end
    endfunction

    assign selected_activation_scales = group_scales(
        req_group, activation_scales_bf16);
    assign selected_row_count = row_count_from_mask(req_row_mask);
    assign selected_start_high_half = selected_row_count[0] && req_k_tile[0];
    assign request_shape_valid = req_group < 4'd12 &&
        selected_row_count != 0;
    assign selected_group_word_offset = group_word_offset(
        req_k_tile, selected_row_count);
    assign selected_word_count = {1'b0, selected_row_count[3:1]} +
        {3'd0, selected_row_count[0] || selected_start_high_half};
    assign local_read_valid = req_valid && !abort_pending && !abort_request &&
        request_shape_valid;
    assign req_ready = local_read_ready && !abort_pending && !abort_request &&
        request_shape_valid;
    assign request_fire = req_valid && req_ready;

    always_comb begin
        local_read = '0;
        local_read.activation_word_index = {req_group, 11'd0} +
            {4'd0, selected_group_word_offset};
        local_read.activation_word_count = selected_word_count;
        local_read.qkv_padded_layout = 1'b0;
        local_read.physical_row_base = 6'd0;
        local_read.panel_select = req_panel;
        local_read.panel_port_mask = 12'h00f;
        for (integer stripe = 0; stripe < 4; stripe++)
            local_read.panel_address[stripe*10 +: 10] =
                {1'b0, req_k_tile};
    end

    always_comb begin
        aligned_activation = '0;
        for (integer row = 0; row < 8; row++) begin
            if (metadata_pipe[1][16 + row]) begin
                if (metadata_pipe[1][META_WIDTH-1])
                    aligned_activation[row*64 +: 64] =
                        local_rsp.activation_data[(row+1)*64 +: 64];
                else
                    aligned_activation[row*64 +: 64] =
                        local_rsp.activation_data[row*64 +: 64];
            end
        end
    end
    assign rsp_valid = metadata_pending[1] && local_rsp_valid &&
        !abort_pending;
    assign rsp_activation = aligned_activation;
    assign rsp_weight = local_rsp.panel_data[511:0];
    assign rsp_activation_scales_bf16 =
        metadata_pipe[1][16+8 +: 128];
    assign rsp_row_mask = metadata_pipe[1][16 +: 8];
    assign rsp_tag = metadata_pipe[1][15:0];

`ifdef SYNTHESIS
    always_comb begin
        accepted_request_count = '0;
        returned_response_count = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            metadata_pending <= '0;
            metadata_pipe[0] <= '0;
            metadata_pipe[1] <= '0;
            abort_pending <= 1'b0;
            abort_ack <= 1'b0;
`ifndef SYNTHESIS
            accepted_request_count <= '0;
            returned_response_count <= '0;
`endif
        end else begin
            abort_ack <= 1'b0;
            metadata_pending <= {metadata_pending[0], request_fire};
            metadata_pipe[1] <= metadata_pipe[0];
            if (request_fire) begin
                metadata_pipe[0] <= {
                    selected_start_high_half, selected_activation_scales,
                    req_row_mask, req_tag
                };
`ifndef SYNTHESIS
                accepted_request_count <= accepted_request_count + 64'd1;
`endif
            end
`ifndef SYNTHESIS
            if (rsp_valid && rsp_ready)
                returned_response_count <= returned_response_count + 64'd1;
`endif
            if (abort_request)
                abort_pending <= 1'b1;
            if (abort_pending && metadata_pending == 0) begin
                abort_pending <= 1'b0;
                abort_ack <= 1'b1;
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (!(metadata_pending[1] && !local_rsp_valid))
                else $error("LM-head local SRAM response missing for accepted read");
            assert (!(rsp_valid && !rsp_ready))
                else $error("LM-head compute did not reserve its SRAM response slot");
        end
    end
`endif
endmodule

`default_nettype wire
