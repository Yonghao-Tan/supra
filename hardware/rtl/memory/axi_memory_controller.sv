`default_nettype none
// One ordered logical read source and one ordered logical write source. Each
// direction has independent alignment and AXI state; the external AXI ID is
// fixed and responses are drained before memory_drained is asserted.
module axi_memory_controller #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer DATA_WIDTH = 128,
    parameter integer ID_WIDTH = 4,
    parameter integer TAG_WIDTH = 8,
    parameter integer MAX_BURST_BYTES = 256,
    parameter integer MAX_READ_OUTSTANDING = 12,
    parameter integer MAX_WRITE_OUTSTANDING = 8
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         abort_request,

    input  logic                         read_request_valid,
    output logic                         read_request_ready,
    input  logic [ADDR_WIDTH-1:0]        read_request_address,
    input  logic [31:0]                  read_request_bytes,
    input  logic [TAG_WIDTH-1:0]         read_request_tag,
    input  logic                         read_second_span_valid,
    input  logic [ADDR_WIDTH-1:0]        read_second_span_address,
    input  logic [31:0]                  read_pair_stride,
    input  logic [10:0]                  read_pair_count,
    output logic                         read_request_done,
    output logic                         read_request_error,
    output logic                         read_data_valid,
    input  logic                         read_data_ready,
    output logic [DATA_WIDTH-1:0]        read_data,
    output logic [DATA_WIDTH/8-1:0]      read_byte_enable,
    output logic                         read_data_last,
    output logic [TAG_WIDTH-1:0]         read_data_tag,
    output logic                         read_data_span,

    input  logic                         write_request_valid,
    output logic                         write_request_ready,
    input  logic [ADDR_WIDTH-1:0]        write_request_address,
    input  logic [31:0]                  write_request_bytes,
    input  logic [TAG_WIDTH-1:0]         write_request_tag,
    output logic                         write_request_done,
    output logic                         write_request_error,
    input  logic                         write_data_valid,
    output logic                         write_data_ready,
    input  logic [DATA_WIDTH-1:0]        write_data,
    input  logic [DATA_WIDTH/8-1:0]      write_byte_enable,
    input  logic                         write_data_last,

    output logic                         read_busy,
    output logic                         write_busy,
    output logic                         memory_drained,
    output logic [4:0]                   read_outstanding,
    output logic [4:0]                   write_outstanding,
    output logic [4:0]                   read_outstanding_high_water,
    output logic [4:0]                   write_outstanding_high_water,
    output logic [63:0]                  read_credit_stall_cycles,
    output logic [63:0]                  write_credit_stall_cycles,
    output logic [63:0]                  read_burst_count,
    output logic [63:0]                  write_burst_count,
    output logic [63:0]                  actual_read_bytes,
    output logic [63:0]                  actual_write_bytes,
    output logic [63:0]                  useful_read_bytes,
    output logic [63:0]                  useful_write_bytes,

    output logic [ID_WIDTH-1:0]          m_axi_awid,
    output logic [ADDR_WIDTH-1:0]        m_axi_awaddr,
    output logic [7:0]                   m_axi_awlen,
    output logic [2:0]                   m_axi_awsize,
    output logic [1:0]                   m_axi_awburst,
    output logic                         m_axi_awvalid,
    input  logic                         m_axi_awready,
    output logic [DATA_WIDTH-1:0]        m_axi_wdata,
    output logic [DATA_WIDTH/8-1:0]      m_axi_wstrb,
    output logic                         m_axi_wlast,
    output logic                         m_axi_wvalid,
    input  logic                         m_axi_wready,
    input  logic [ID_WIDTH-1:0]          m_axi_bid,
    input  logic [1:0]                   m_axi_bresp,
    input  logic                         m_axi_bvalid,
    output logic                         m_axi_bready,
    output logic [ID_WIDTH-1:0]          m_axi_arid,
    output logic [ADDR_WIDTH-1:0]        m_axi_araddr,
    output logic [7:0]                   m_axi_arlen,
    output logic [2:0]                   m_axi_arsize,
    output logic [1:0]                   m_axi_arburst,
    output logic                         m_axi_arvalid,
    input  logic                         m_axi_arready,
    input  logic [ID_WIDTH-1:0]          m_axi_rid,
    input  logic [DATA_WIDTH-1:0]        m_axi_rdata,
    input  logic [1:0]                   m_axi_rresp,
    input  logic                         m_axi_rlast,
    input  logic                         m_axi_rvalid,
    output logic                         m_axi_rready
);
    logic read_alignment_ready;
    logic read_aligned_request_valid;
    logic read_aligned_request_ready;
    logic [ADDR_WIDTH-1:0] read_aligned_address;
    logic [31:0] read_aligned_bytes;
    logic [TAG_WIDTH-1:0] read_aligned_tag;
    logic read_aligned_is_write;
    logic read_aligned_done;
    logic read_aligned_error;
    logic read_aligned_data_valid;
    logic read_aligned_data_ready;
    logic [DATA_WIDTH-1:0] read_aligned_data;
    logic [DATA_WIDTH/8-1:0] read_aligned_byte_enable;
    logic read_aligned_data_last;
    logic [TAG_WIDTH-1:0] read_aligned_data_tag;
    logic read_alignment_done;
    logic read_alignment_error;
    logic read_alignment_data_valid;
    logic read_alignment_data_ready;
    logic [DATA_WIDTH-1:0] read_alignment_data;
    logic [DATA_WIDTH/8-1:0] read_alignment_byte_enable;
    logic read_alignment_data_last;
    logic [TAG_WIDTH-1:0] read_alignment_data_tag;

    logic paired_read_active;
    logic paired_read_fire;
    logic master_read_request_valid;
    logic master_read_request_ready;
    logic [ADDR_WIDTH-1:0] master_read_request_address;
    logic [31:0] master_read_request_bytes;
    logic [TAG_WIDTH-1:0] master_read_request_tag;
    logic master_read_second_span_valid;
    logic [ADDR_WIDTH-1:0] master_read_second_span_address;
    logic [31:0] master_read_pair_stride;
    logic [10:0] master_read_pair_count;
    logic master_read_done;
    logic master_read_error;
    logic master_read_data_valid;
    logic master_read_data_ready;
    logic [DATA_WIDTH-1:0] master_read_data;
    logic [DATA_WIDTH/8-1:0] master_read_byte_enable;
    logic master_read_data_last;
    logic [TAG_WIDTH-1:0] master_read_data_tag;
    logic master_read_data_span;
    logic master_select_paired;

    logic write_alignment_ready;
    logic write_aligned_request_valid;
    logic write_aligned_request_ready;
    logic [ADDR_WIDTH-1:0] write_aligned_address;
    logic [31:0] write_aligned_bytes;
    logic [TAG_WIDTH-1:0] write_aligned_tag;
    logic write_aligned_is_write;
    logic write_aligned_done;
    logic write_aligned_error;
    logic write_aligned_data_valid;
    logic write_aligned_data_ready;
    logic [DATA_WIDTH-1:0] write_aligned_data;
    logic [DATA_WIDTH/8-1:0] write_aligned_byte_enable;
    logic write_aligned_data_last;

    logic master_read_idle;
    logic master_write_idle;
    logic unused_read_write_ready;
    logic unused_read_write_valid;
    logic [DATA_WIDTH-1:0] unused_read_write_data;
    logic [DATA_WIDTH/8-1:0] unused_read_write_enable;
    logic unused_read_write_last;
    logic unused_write_read_valid;
    logic [DATA_WIDTH-1:0] unused_write_read_data;
    logic [DATA_WIDTH/8-1:0] unused_write_read_enable;
    logic unused_write_read_last;
    logic [TAG_WIDTH-1:0] unused_write_read_tag;
    logic unused_write_aligned_read_ready;

    assign read_request_ready = !abort_request && !paired_read_active &&
        (read_second_span_valid ?
            (read_alignment_ready && !read_aligned_request_valid &&
             master_read_request_ready) :
            read_alignment_ready);
    assign paired_read_fire = read_request_valid && read_request_ready &&
        read_second_span_valid;
    assign master_select_paired = !read_aligned_request_valid &&
        read_request_valid && read_second_span_valid && read_alignment_ready;
    assign master_read_request_valid = !paired_read_active &&
        (read_aligned_request_valid ||
         (master_select_paired && !abort_request));
    assign master_read_request_address = master_select_paired ?
        read_request_address : read_aligned_address;
    assign master_read_request_bytes = master_select_paired ?
        read_request_bytes : read_aligned_bytes;
    assign master_read_request_tag = master_select_paired ?
        read_request_tag : read_aligned_tag;
    assign master_read_second_span_valid = master_select_paired &&
        !paired_read_active;
    assign master_read_second_span_address = read_second_span_address;
    assign master_read_pair_stride = master_select_paired ?
        read_pair_stride : 32'd0;
    assign master_read_pair_count = master_select_paired ?
        read_pair_count : 11'd1;
    assign read_aligned_request_ready = !paired_read_active &&
        master_read_request_ready;
    assign read_aligned_done = !paired_read_active && master_read_done;
    assign read_aligned_error = master_read_error;
    assign read_aligned_data_valid = !paired_read_active &&
        master_read_data_valid;
    assign read_aligned_data = master_read_data;
    assign read_aligned_byte_enable = master_read_byte_enable;
    assign read_aligned_data_last = master_read_data_last;
    assign read_aligned_data_tag = master_read_data_tag;

    assign read_request_done = paired_read_active ? master_read_done :
        read_alignment_done;
    assign read_request_error = paired_read_active ? master_read_error :
        read_alignment_error;
    assign read_data_valid = paired_read_active ? master_read_data_valid :
        read_alignment_data_valid;
    assign master_read_data_ready = paired_read_active ? read_data_ready :
        read_aligned_data_ready;
    assign read_alignment_data_ready = read_data_ready;
    assign read_data = paired_read_active ? master_read_data :
        read_alignment_data;
    assign read_byte_enable = paired_read_active ? master_read_byte_enable :
        read_alignment_byte_enable;
    assign read_data_last = paired_read_active ? master_read_data_last :
        read_alignment_data_last;
    assign read_data_tag = paired_read_active ? master_read_data_tag :
        read_alignment_data_tag;
    assign read_data_span = paired_read_active && master_read_data_span;

    always_ff @(posedge clk) begin
        if (rst)
            paired_read_active <= 1'b0;
        else begin
            if (paired_read_fire)
                paired_read_active <= 1'b1;
            if (paired_read_active && master_read_done)
                paired_read_active <= 1'b0;
        end
    end
    assign write_request_ready = !abort_request && write_alignment_ready;
    assign read_busy = paired_read_active || !read_alignment_ready ||
        !master_read_idle;
    assign write_busy = !write_alignment_ready || !master_write_idle;
    assign memory_drained = !read_request_valid && !write_request_valid &&
        !paired_read_active && read_alignment_ready && write_alignment_ready &&
        master_read_idle && master_write_idle && read_outstanding == 0 &&
        write_outstanding == 0;

    dma_alignment #(
        .DATA_WIDTH(DATA_WIDTH),
        .TAG_WIDTH(TAG_WIDTH)
    ) read_alignment (
        .clk(clk),
        .rst(rst),
        .abort_request(abort_request),
        .request_valid(read_request_valid && !abort_request &&
            !read_second_span_valid),
        .request_ready(read_alignment_ready),
        .request_write(1'b0),
        .request_address(read_request_address),
        .request_bytes(read_request_bytes),
        .request_tag(read_request_tag),
        .request_done(read_alignment_done),
        .request_error(read_alignment_error),
        .read_data_valid(read_alignment_data_valid),
        .read_data_ready(read_alignment_data_ready),
        .read_data(read_alignment_data),
        .read_byte_enable(read_alignment_byte_enable),
        .read_data_last(read_alignment_data_last),
        .read_data_tag(read_alignment_data_tag),
        .write_data_valid(1'b0),
        .write_data_ready(unused_read_write_ready),
        .write_data('0),
        .write_byte_enable('0),
        .write_data_last(1'b0),
        .aligned_request_valid(read_aligned_request_valid),
        .aligned_request_ready(read_aligned_request_ready),
        .aligned_request_write(read_aligned_is_write),
        .aligned_request_address(read_aligned_address),
        .aligned_request_bytes(read_aligned_bytes),
        .aligned_request_tag(read_aligned_tag),
        .aligned_request_done(read_aligned_done),
        .aligned_request_error(read_aligned_error),
        .aligned_read_valid(read_aligned_data_valid),
        .aligned_read_ready(read_aligned_data_ready),
        .aligned_read_data(read_aligned_data),
        .aligned_read_byte_enable(read_aligned_byte_enable),
        .aligned_read_last(read_aligned_data_last),
        .aligned_read_tag(read_aligned_data_tag),
        .aligned_write_valid(unused_read_write_valid),
        .aligned_write_ready(1'b0),
        .aligned_write_data(unused_read_write_data),
        .aligned_write_byte_enable(unused_read_write_enable),
        .aligned_write_last(unused_read_write_last)
    );

    dma_alignment #(
        .DATA_WIDTH(DATA_WIDTH),
        .TAG_WIDTH(TAG_WIDTH)
    ) write_alignment (
        .clk(clk),
        .rst(rst),
        .abort_request(abort_request),
        .request_valid(write_request_valid && !abort_request),
        .request_ready(write_alignment_ready),
        .request_write(1'b1),
        .request_address(write_request_address),
        .request_bytes(write_request_bytes),
        .request_tag(write_request_tag),
        .request_done(write_request_done),
        .request_error(write_request_error),
        .read_data_valid(unused_write_read_valid),
        .read_data_ready(1'b0),
        .read_data(unused_write_read_data),
        .read_byte_enable(unused_write_read_enable),
        .read_data_last(unused_write_read_last),
        .read_data_tag(unused_write_read_tag),
        .write_data_valid(write_data_valid),
        .write_data_ready(write_data_ready),
        .write_data(write_data),
        .write_byte_enable(write_byte_enable),
        .write_data_last(write_data_last),
        .aligned_request_valid(write_aligned_request_valid),
        .aligned_request_ready(write_aligned_request_ready),
        .aligned_request_write(write_aligned_is_write),
        .aligned_request_address(write_aligned_address),
        .aligned_request_bytes(write_aligned_bytes),
        .aligned_request_tag(write_aligned_tag),
        .aligned_request_done(write_aligned_done),
        .aligned_request_error(write_aligned_error),
        .aligned_read_valid(1'b0),
        .aligned_read_ready(unused_write_aligned_read_ready),
        .aligned_read_data('0),
        .aligned_read_byte_enable('0),
        .aligned_read_last(1'b0),
        .aligned_read_tag('0),
        .aligned_write_valid(write_aligned_data_valid),
        .aligned_write_ready(write_aligned_data_ready),
        .aligned_write_data(write_aligned_data),
        .aligned_write_byte_enable(write_aligned_byte_enable),
        .aligned_write_last(write_aligned_data_last)
    );

    axi_master #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .ID_WIDTH(ID_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .MAX_BURST_BYTES(MAX_BURST_BYTES),
        .MAX_READ_OUTSTANDING(MAX_READ_OUTSTANDING),
        .MAX_WRITE_OUTSTANDING(MAX_WRITE_OUTSTANDING),
        .ALLOW_SPARSE_WRITE_STROBE(1)
    ) axi (
        .clk(clk),
        .rst(rst),
        .abort_request(abort_request),
        .read_request_valid(master_read_request_valid),
        .read_request_ready(master_read_request_ready),
        .read_request_address(master_read_request_address),
        .read_request_bytes(master_read_request_bytes),
        .read_request_tag(master_read_request_tag),
        .read_second_span_valid(master_read_second_span_valid),
        .read_second_span_address(master_read_second_span_address),
        .read_pair_stride(master_read_pair_stride),
        .read_pair_count(master_read_pair_count),
        .read_request_done(master_read_done),
        .read_request_error(master_read_error),
        .read_data_valid(master_read_data_valid),
        .read_data_ready(master_read_data_ready),
        .read_data(master_read_data),
        .read_byte_enable(master_read_byte_enable),
        .read_data_last(master_read_data_last),
        .read_data_tag(master_read_data_tag),
        .read_data_span(master_read_data_span),
        .write_request_valid(write_aligned_request_valid),
        .write_request_ready(write_aligned_request_ready),
        .write_request_address(write_aligned_address),
        .write_request_bytes(write_aligned_bytes),
        .write_request_done(write_aligned_done),
        .write_request_error(write_aligned_error),
        .write_data_valid(write_aligned_data_valid),
        .write_data_ready(write_aligned_data_ready),
        .write_data(write_aligned_data),
        .write_byte_enable(write_aligned_byte_enable),
        .write_data_last(write_aligned_data_last),
        .read_outstanding(read_outstanding),
        .write_outstanding(write_outstanding),
        .read_outstanding_high_water(read_outstanding_high_water),
        .write_outstanding_high_water(write_outstanding_high_water),
        .read_credit_stall_cycles(read_credit_stall_cycles),
        .write_credit_stall_cycles(write_credit_stall_cycles),
        .read_burst_count(read_burst_count),
        .write_burst_count(write_burst_count),
        .actual_read_bytes(actual_read_bytes),
        .actual_write_bytes(actual_write_bytes),
        .useful_read_bytes(useful_read_bytes),
        .useful_write_bytes(useful_write_bytes),
        .read_idle(master_read_idle),
        .write_idle(master_write_idle),
        .m_axi_awid(m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            assert (!(read_request_valid && read_request_ready && abort_request))
                else $error("AXI memory controller accepted a read while aborting");
            if (paired_read_fire)
                assert (read_alignment_ready)
                    else $error("AXI memory controller accepted a paired read while alignment was active");
            if (read_request_valid && !read_second_span_valid)
                assert (read_pair_count == 11'd1 && read_pair_stride == 0)
                    else $error("AXI memory controller ordinary read carried pair repetition");
            assert (!(write_request_valid && write_request_ready && abort_request))
                else $error("AXI memory controller accepted a write while aborting");
            assert (!read_aligned_request_valid || !read_aligned_is_write)
                else $error("AXI memory controller read alignment emitted a write request");
            assert (!write_aligned_request_valid || write_aligned_is_write)
                else $error("AXI memory controller write alignment emitted a read request");
            assert (!write_aligned_request_valid || !$isunknown(write_aligned_tag))
                else $error("AXI memory controller write alignment tag contains X or Z");
            assert (!unused_write_read_valid && !unused_write_aligned_read_ready)
                else $error("AXI memory controller write alignment entered a read state");
            if (memory_drained) begin
                assert (!read_busy && !write_busy && read_outstanding == 0 &&
                        write_outstanding == 0)
                    else $error("AXI memory controller asserted drained with pending work");
            end
        end
    end
`endif

    initial begin
        if (ADDR_WIDTH < 32 ||
            (DATA_WIDTH != 128 && DATA_WIDTH != 256) ||
            ID_WIDTH < 1 || TAG_WIDTH < 1 ||
            MAX_BURST_BYTES != 256)
            $error("AXI memory controller parameter relation is invalid");
    end
endmodule

`default_nettype wire
