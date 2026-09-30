`default_nettype none
// Converts the 128-bit logical operator stream at the memory boundary to the
// 256-bit AXI stream. Matmul weight reads may consume both 128-bit words in
// one cycle; all other reads are returned as two ordered 128-bit words.
module memory_stream_width_adapter #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer TAG_WIDTH = 8,
    parameter integer LOGICAL_DATA_WIDTH = 128,
    parameter integer AXI_DATA_WIDTH = 256
) (
    input  logic                             clk,
    input  logic                             rst,

    input  logic                             read_request_valid,
    output logic                             read_request_ready,
    input  logic [ADDR_WIDTH-1:0]            read_request_address,
    input  logic [31:0]                      read_request_bytes,
    input  logic [TAG_WIDTH-1:0]             read_request_tag,
    input  logic                             read_request_wide,
    input  logic                             read_second_span_valid,
    input  logic [ADDR_WIDTH-1:0]            read_second_span_address,
    input  logic [31:0]                      read_pair_stride,
    input  logic [10:0]                      read_pair_count,
    output logic                             read_request_done,
    output logic                             read_request_error,
    output logic                             read_data_valid,
    input  logic                             read_data_ready,
    output logic [LOGICAL_DATA_WIDTH-1:0]    read_data,
    output logic [LOGICAL_DATA_WIDTH/8-1:0]  read_byte_enable,
    output logic                             read_data_last,
    output logic [TAG_WIDTH-1:0]             read_data_tag,
    output logic                             read_data_span,
    output logic                             read_wide_data_valid,
    input  logic                             read_wide_data_ready,
    output logic [AXI_DATA_WIDTH-1:0]        read_wide_data,
    output logic [AXI_DATA_WIDTH/8-1:0]      read_wide_byte_enable,
    output logic                             read_wide_data_last,
    output logic [TAG_WIDTH-1:0]             read_wide_data_tag,

    output logic                             wide_read_request_valid,
    input  logic                             wide_read_request_ready,
    output logic [ADDR_WIDTH-1:0]            wide_read_request_address,
    output logic [31:0]                      wide_read_request_bytes,
    output logic [TAG_WIDTH-1:0]             wide_read_request_tag,
    output logic                             wide_read_second_span_valid,
    output logic [ADDR_WIDTH-1:0]            wide_read_second_span_address,
    output logic [31:0]                      wide_read_pair_stride,
    output logic [10:0]                      wide_read_pair_count,
    input  logic                             wide_read_request_done,
    input  logic                             wide_read_request_error,
    input  logic                             wide_read_data_valid,
    output logic                             wide_read_data_ready,
    input  logic [AXI_DATA_WIDTH-1:0]        wide_read_data,
    input  logic [AXI_DATA_WIDTH/8-1:0]      wide_read_byte_enable,
    input  logic                             wide_read_data_last,
    input  logic [TAG_WIDTH-1:0]             wide_read_data_tag,
    input  logic                             wide_read_data_span,

    input  logic                             write_request_valid,
    output logic                             write_request_ready,
    input  logic [ADDR_WIDTH-1:0]            write_request_address,
    input  logic [31:0]                      write_request_bytes,
    input  logic [TAG_WIDTH-1:0]             write_request_tag,
    output logic                             write_request_done,
    output logic                             write_request_error,
    input  logic                             write_data_valid,
    output logic                             write_data_ready,
    input  logic [LOGICAL_DATA_WIDTH-1:0]    write_data,
    input  logic [LOGICAL_DATA_WIDTH/8-1:0]  write_byte_enable,
    input  logic                             write_data_last,

    output logic                             wide_write_request_valid,
    input  logic                             wide_write_request_ready,
    output logic [ADDR_WIDTH-1:0]            wide_write_request_address,
    output logic [31:0]                      wide_write_request_bytes,
    output logic [TAG_WIDTH-1:0]             wide_write_request_tag,
    input  logic                             wide_write_request_done,
    input  logic                             wide_write_request_error,
    output logic                             wide_write_data_valid,
    input  logic                             wide_write_data_ready,
    output logic [AXI_DATA_WIDTH-1:0]        wide_write_data,
    output logic [AXI_DATA_WIDTH/8-1:0]      wide_write_byte_enable,
    output logic                             wide_write_data_last,
    output logic                             idle
);
    localparam integer LOGICAL_BYTES = LOGICAL_DATA_WIDTH / 8;
    localparam integer AXI_BYTES = AXI_DATA_WIDTH / 8;

    logic read_active;
    logic read_mode_wide;
    logic read_done_pending;
    logic read_error_pending;
    logic split_valid;
    logic split_high;
    logic split_has_high;
    logic [AXI_DATA_WIDTH-1:0] split_data;
    logic [AXI_BYTES-1:0] split_byte_enable;
    logic split_last;
    logic [TAG_WIDTH-1:0] split_tag;
    logic split_span;
    logic read_request_fire;
    logic wide_read_data_fire;
    logic narrow_read_data_fire;

    logic write_half_valid;
    logic [LOGICAL_DATA_WIDTH-1:0] write_half_data;
    logic [LOGICAL_BYTES-1:0] write_half_byte_enable;
    logic write_pack_valid;
    logic [AXI_DATA_WIDTH-1:0] write_pack_data;
    logic [AXI_BYTES-1:0] write_pack_byte_enable;
    logic write_pack_last;
    logic write_input_fire;
    logic write_pack_fire;

    assign read_request_ready = !read_active && wide_read_request_ready;
    assign read_request_fire = read_request_valid && read_request_ready;
    assign wide_read_request_valid = read_request_valid && !read_active;
    assign wide_read_request_address = read_request_address;
    assign wide_read_request_bytes = read_request_bytes;
    assign wide_read_request_tag = read_request_tag;
    assign wide_read_second_span_valid = read_second_span_valid;
    assign wide_read_second_span_address = read_second_span_address;
    assign wide_read_pair_stride = read_pair_stride;
    assign wide_read_pair_count = read_pair_count;

    assign read_wide_data_valid = read_active && read_mode_wide &&
        wide_read_data_valid;
    assign read_wide_data = wide_read_data;
    assign read_wide_byte_enable = wide_read_byte_enable;
    assign read_wide_data_last = wide_read_data_last;
    assign read_wide_data_tag = wide_read_data_tag;
    assign wide_read_data_ready = read_active &&
        (read_mode_wide ? read_wide_data_ready :
         (!split_valid || (read_data_ready &&
                           (split_high || !split_has_high))));
    assign wide_read_data_fire = wide_read_data_valid && wide_read_data_ready;

    assign read_data_valid = split_valid;
    assign read_data = split_high ?
        split_data[AXI_DATA_WIDTH-1 -: LOGICAL_DATA_WIDTH] :
        split_data[LOGICAL_DATA_WIDTH-1:0];
    assign read_byte_enable = split_high ?
        split_byte_enable[AXI_BYTES-1 -: LOGICAL_BYTES] :
        split_byte_enable[LOGICAL_BYTES-1:0];
    assign read_data_last = split_last && (split_high || !split_has_high);
    assign read_data_tag = split_tag;
    assign read_data_span = split_span;
    assign narrow_read_data_fire = read_data_valid && read_data_ready;

    assign write_request_ready = wide_write_request_ready;
    assign wide_write_request_valid = write_request_valid;
    assign wide_write_request_address = write_request_address;
    assign wide_write_request_bytes = write_request_bytes;
    assign wide_write_request_tag = write_request_tag;
    assign write_request_done = wide_write_request_done;
    assign write_request_error = wide_write_request_error;

    assign write_data_ready = !write_pack_valid || wide_write_data_ready;
    assign write_input_fire = write_data_valid && write_data_ready;
    assign wide_write_data_valid = write_pack_valid;
    assign wide_write_data = write_pack_data;
    assign wide_write_byte_enable = write_pack_byte_enable;
    assign wide_write_data_last = write_pack_last;
    assign write_pack_fire = wide_write_data_valid && wide_write_data_ready;
    assign idle = !read_active && !read_done_pending && !split_valid &&
        !write_half_valid && !write_pack_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            read_active <= 1'b0;
            read_mode_wide <= 1'b0;
            read_done_pending <= 1'b0;
            read_error_pending <= 1'b0;
            read_request_done <= 1'b0;
            read_request_error <= 1'b0;
            split_valid <= 1'b0;
            split_high <= 1'b0;
            split_has_high <= 1'b0;
            split_data <= '0;
            split_byte_enable <= '0;
            split_last <= 1'b0;
            split_tag <= '0;
            split_span <= 1'b0;
            write_half_valid <= 1'b0;
            write_half_data <= '0;
            write_half_byte_enable <= '0;
            write_pack_valid <= 1'b0;
            write_pack_data <= '0;
            write_pack_byte_enable <= '0;
            write_pack_last <= 1'b0;
        end else begin
            read_request_done <= 1'b0;
            read_request_error <= 1'b0;

            if (read_request_fire) begin
                read_active <= 1'b1;
                read_mode_wide <= read_request_wide;
                read_done_pending <= 1'b0;
                read_error_pending <= 1'b0;
            end
            // The consumed final half can be replaced in the same register.
            if (wide_read_data_fire && !read_mode_wide) begin
                split_valid <= 1'b1;
                split_high <= 1'b0;
                split_has_high <=
                    |wide_read_byte_enable[AXI_BYTES-1:LOGICAL_BYTES];
                split_data <= wide_read_data;
                split_byte_enable <= wide_read_byte_enable;
                split_last <= wide_read_data_last;
                split_tag <= wide_read_data_tag;
                split_span <= wide_read_data_span;
            end else if (narrow_read_data_fire) begin
                if (!split_high && split_has_high)
                    split_high <= 1'b1;
                else begin
                    split_valid <= 1'b0;
                    split_high <= 1'b0;
                    split_has_high <= 1'b0;
                end
            end
            if (wide_read_request_done) begin
                read_done_pending <= 1'b1;
                read_error_pending <= wide_read_request_error;
            end
            if (read_done_pending && !split_valid && !wide_read_data_valid) begin
                read_request_done <= 1'b1;
                read_request_error <= read_error_pending;
                read_done_pending <= 1'b0;
                read_error_pending <= 1'b0;
                read_active <= 1'b0;
            end

            if (write_pack_fire)
                write_pack_valid <= 1'b0;
            if (write_input_fire) begin
                if (!write_half_valid) begin
                    if (write_data_last) begin
                        write_pack_valid <= 1'b1;
                        write_pack_data <= {{LOGICAL_DATA_WIDTH{1'b0}},
                                            write_data};
                        write_pack_byte_enable <=
                            {{LOGICAL_BYTES{1'b0}}, write_byte_enable};
                        write_pack_last <= 1'b1;
                    end else begin
                        write_half_valid <= 1'b1;
                        write_half_data <= write_data;
                        write_half_byte_enable <= write_byte_enable;
                    end
                end else begin
                    write_pack_valid <= 1'b1;
                    write_pack_data <= {write_data, write_half_data};
                    write_pack_byte_enable <=
                        {write_byte_enable, write_half_byte_enable};
                    write_pack_last <= write_data_last;
                    write_half_valid <= 1'b0;
                end
            end
            if (wide_write_request_done && wide_write_request_error) begin
                write_half_valid <= 1'b0;
                write_pack_valid <= 1'b0;
            end
        end
    end

    initial begin
        if (LOGICAL_DATA_WIDTH != 128 || AXI_DATA_WIDTH != 256 ||
            AXI_DATA_WIDTH != 2 * LOGICAL_DATA_WIDTH)
            $error("memory_stream_width_adapter requires a 128-to-256 relation");
    end

`ifndef SYNTHESIS
    logic held_read_request;
    logic [2*ADDR_WIDTH+TAG_WIDTH+75:0] held_read_request_payload;
    logic held_narrow_read;
    logic [LOGICAL_DATA_WIDTH+LOGICAL_BYTES+TAG_WIDTH+1:0]
        held_narrow_payload;
    logic held_wide_write;
    logic [AXI_DATA_WIDTH+AXI_BYTES:0] held_wide_write_payload;
    always_ff @(posedge clk) begin
        if (rst) begin
            held_narrow_read <= 1'b0;
            held_read_request <= 1'b0;
            held_wide_write <= 1'b0;
        end else begin
            if (held_read_request)
                assert (wide_read_request_valid &&
                    {wide_read_request_address, wide_read_request_bytes,
                     wide_read_request_tag, wide_read_second_span_valid,
                     wide_read_second_span_address, wide_read_pair_stride,
                     wide_read_pair_count} ==
                        held_read_request_payload)
                    else $error("memory width adapter changed a stalled read request");
            if (held_narrow_read)
                assert (read_data_valid &&
                    {read_data, read_byte_enable, read_data_last, read_data_tag,
                     read_data_span} ==
                        held_narrow_payload)
                    else $error("memory width adapter changed a stalled narrow read");
            if (held_wide_write)
                assert (wide_write_data_valid &&
                    {wide_write_data, wide_write_byte_enable,
                     wide_write_data_last} == held_wide_write_payload)
                    else $error("memory width adapter changed a stalled AXI write");
            held_narrow_read <= read_data_valid && !read_data_ready;
            held_read_request <= wide_read_request_valid &&
                !wide_read_request_ready;
            held_read_request_payload <= {wide_read_request_address,
                wide_read_request_bytes, wide_read_request_tag,
                wide_read_second_span_valid,
                wide_read_second_span_address, wide_read_pair_stride,
                wide_read_pair_count};
            held_narrow_payload <=
                {read_data, read_byte_enable, read_data_last, read_data_tag,
                 read_data_span};
            if (read_request_valid)
                assert ((!read_second_span_valid || !read_request_wide) &&
                        (read_second_span_valid ||
                         (read_pair_count == 11'd1 && read_pair_stride == 0)))
                    else $error("memory width adapter received a paired wide read");
            held_wide_write <= wide_write_data_valid && !wide_write_data_ready &&
                !(wide_write_request_done && wide_write_request_error);
            held_wide_write_payload <=
                {wide_write_data, wide_write_byte_enable, wide_write_data_last};
        end
    end
`endif
endmodule

`default_nettype wire
