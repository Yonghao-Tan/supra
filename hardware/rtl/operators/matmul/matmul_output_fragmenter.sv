`default_nettype none

// Converts one row-major M8xN8 BF16 result into one 16-byte fragment per
// active physical row. The accepted tile is held until every fragment is
// accepted, so output metadata and data remain stable through backpressure.
module matmul_output_fragmenter #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                     clk,
    input  logic                     rst,
    input  logic                     abort_request,
    output logic                     abort_ack,

    input  logic                     tile_valid,
    output logic                     tile_ready,
    input  logic [64*16-1:0]         tile_bf16,
    input  logic [63:0]              tile_mask,
    input  logic [5:0]               tile_physical_row_base,
    input  logic [15:0]              tile_output_channel_base,
    input  logic [7:0]               tile_row_mask,
    input  logic [7:0]               tile_column_mask,
    input  logic [ADDR_WIDTH-1:0]    tile_row_byte_base,
    input  logic [31:0]              tile_row_stride,
    input  logic [TAG_WIDTH-1:0]     tile_tag,

    output logic                     fragment_valid,
    input  logic                     fragment_ready,
    output logic [5:0]               fragment_physical_row,
    output logic [15:0]              fragment_output_channel,
    output logic [ADDR_WIDTH-1:0]    fragment_row_byte_base,
    output logic [127:0]             fragment_data,
    output logic [15:0]              fragment_byte_enable,
    output logic [TAG_WIDTH-1:0]     fragment_tag,
    output logic                     error
);
    logic active;
    logic [64*16-1:0] saved_bf16;
    logic [63:0] saved_mask;
    logic [5:0] saved_row_base;
    logic [15:0] saved_channel_base;
    logic [7:0] saved_row_mask;
    logic [7:0] saved_column_mask;
    logic [31:0] saved_row_stride;
    logic [ADDR_WIDTH-1:0] current_fragment_row_byte_base;
    logic [TAG_WIDTH-1:0] saved_tag;
    logic [2:0] row_lane;
    logic [2:0] fragment_row_offset;
    logic [2:0] first_active_row;
    logic [2:0] next_active_row;
    logic next_active_found;
    logic [5:0] active_column_count;
    logic [63:0] expected_mask;
    logic input_metadata_valid;
    logic abort_seen;

    always_comb begin
        first_active_row = 3'd0;
        for (integer row = 7; row >= 0; row = row - 1)
            if (tile_row_mask[row])
                first_active_row = 3'(row);

        next_active_row = row_lane;
        next_active_found = 1'b0;
        for (integer row = 7; row >= 0; row = row - 1)
            if (row > row_lane && saved_row_mask[row]) begin
                next_active_row = 3'(row);
                next_active_found = 1'b1;
            end

        active_column_count = 6'd0;
        for (integer column = 0; column < 8; column = column + 1)
            if (tile_column_mask[column])
                active_column_count = active_column_count + 1'b1;

        expected_mask = '0;
        for (integer row = 0; row < 8; row = row + 1)
            for (integer column = 0; column < 8; column = column + 1)
                expected_mask[row*8+column] =
                    tile_row_mask[row] && tile_column_mask[column];

        input_metadata_valid = tile_row_mask != 0 && tile_column_mask != 0 &&
            tile_column_mask == (8'hff >> (8 - active_column_count)) &&
            tile_mask == expected_mask && tile_output_channel_base[2:0] == 0 &&
            tile_physical_row_base + $countones(tile_row_mask) <= 48 &&
            tile_row_stride != 0;
    end

    assign tile_ready = !active && !abort_request;
    assign fragment_valid = active && !abort_request;
    assign fragment_physical_row =
        saved_row_base + {3'd0, fragment_row_offset};
    assign fragment_output_channel = saved_channel_base;
    assign fragment_row_byte_base = current_fragment_row_byte_base;
    assign fragment_data = saved_bf16[row_lane*128 +: 128];
    assign fragment_byte_enable = saved_column_mask == 8'hff ? 16'hffff :
        (16'hffff >> (16 - 2*$countones(saved_column_mask)));
    assign fragment_tag = saved_tag;

    always_ff @(posedge clk) begin
        if (rst) begin
            active <= 1'b0;
            saved_bf16 <= '0;
            saved_mask <= '0;
            saved_row_base <= '0;
            saved_channel_base <= '0;
            saved_row_mask <= '0;
            saved_column_mask <= '0;
            saved_row_stride <= '0;
            current_fragment_row_byte_base <= '0;
            saved_tag <= '0;
            row_lane <= '0;
            fragment_row_offset <= '0;
            abort_ack <= 1'b0;
            abort_seen <= 1'b0;
            error <= 1'b0;
        end else if (abort_request) begin
            active <= 1'b0;
            abort_ack <= !abort_seen;
            abort_seen <= 1'b1;
            error <= 1'b0;
        end else begin
            abort_ack <= 1'b0;
            abort_seen <= 1'b0;
            if (tile_valid && tile_ready) begin
                if (!input_metadata_valid) begin
                    error <= 1'b1;
                end else begin
                    active <= 1'b1;
                    saved_bf16 <= tile_bf16;
                    saved_mask <= tile_mask;
                    saved_row_base <= tile_physical_row_base;
                    saved_channel_base <= tile_output_channel_base;
                    saved_row_mask <= tile_row_mask;
                    saved_column_mask <= tile_column_mask;
                    saved_row_stride <= tile_row_stride;
                    current_fragment_row_byte_base <= tile_row_byte_base;
                    saved_tag <= tile_tag;
                    row_lane <= first_active_row;
                    fragment_row_offset <= '0;
                    error <= 1'b0;
                end
            end
            if (fragment_valid && fragment_ready) begin
                if (next_active_found) begin
                    row_lane <= next_active_row;
                    fragment_row_offset <= fragment_row_offset + 1'b1;
                    current_fragment_row_byte_base <=
                        current_fragment_row_byte_base +
                        ADDR_WIDTH'(saved_row_stride);
                end else
                    active <= 1'b0;
            end
        end
    end

`ifndef SYNTHESIS
    logic stalled_fragment;
    logic [6+16+ADDR_WIDTH+128+16+TAG_WIDTH-1:0] held_fragment;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_fragment <= 1'b0;
            held_fragment <= '0;
        end else begin
            if (stalled_fragment && !abort_request)
                assert (fragment_valid && {fragment_physical_row,
                    fragment_output_channel, fragment_row_byte_base,
                    fragment_data, fragment_byte_enable, fragment_tag} == held_fragment)
                    else $error("matmul_output_fragmenter changed a stalled fragment");
            stalled_fragment <= fragment_valid && !fragment_ready && !abort_request;
            if (fragment_valid && !fragment_ready)
                held_fragment <= {fragment_physical_row, fragment_output_channel,
                    fragment_row_byte_base, fragment_data,
                    fragment_byte_enable, fragment_tag};
        end
    end
`endif
endmodule

`default_nettype wire
