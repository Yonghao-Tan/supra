`default_nettype none
// Owns the 128-bit logical stream boundary, 128-to-256 width conversion,
// alignment, outstanding tracking and the external AXI4 channels.
module memory_controller #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer ID_WIDTH = 4,
    parameter integer TAG_WIDTH = 8,
    parameter integer LOGICAL_DATA_WIDTH = 128,
    parameter integer AXI_DATA_WIDTH = 256,
    parameter integer MAX_BURST_BYTES = 256,
    parameter integer MAX_READ_OUTSTANDING = 12,
    parameter integer MAX_WRITE_OUTSTANDING = 8
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         abort_request,

    input  logic                         read_request_valid,
    output logic                         read_request_ready,
    input  hardware_types_pkg::dma_read_request_t read_request,
    input  logic                         read_second_span_valid,
    input  logic [ADDR_WIDTH-1:0]        read_second_span_address,
    input  logic [31:0]                  read_pair_stride,
    input  logic [10:0]                  read_pair_count,
    output hardware_types_pkg::dma_completion_t read_completion,
    output logic                         read_data_valid,
    input  logic                         read_data_ready,
    output hardware_types_pkg::dma_read_beat_t read_data,
    output logic                         read_data_span,
    output logic                         read_wide_data_valid,
    input  logic                         read_wide_data_ready,
    output hardware_types_pkg::dma_wide_read_beat_t read_wide_data,

    input  logic                         write_request_valid,
    output logic                         write_request_ready,
    input  hardware_types_pkg::dma_write_request_t write_request,
    output hardware_types_pkg::dma_completion_t write_completion,
    input  logic                         write_data_valid,
    output logic                         write_data_ready,
    input  hardware_types_pkg::dma_write_beat_t write_data,

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
    output logic [AXI_DATA_WIDTH-1:0]    m_axi_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0]  m_axi_wstrb,
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
    input  logic [AXI_DATA_WIDTH-1:0]    m_axi_rdata,
    input  logic [1:0]                   m_axi_rresp,
    input  logic                         m_axi_rlast,
    input  logic                         m_axi_rvalid,
    output logic                         m_axi_rready
);
    logic wide_read_request_valid;
    logic wide_read_request_ready;
    logic [ADDR_WIDTH-1:0] wide_read_request_address;
    logic [31:0] wide_read_request_bytes;
    logic [TAG_WIDTH-1:0] wide_read_request_tag;
    logic wide_read_second_span_valid;
    logic [ADDR_WIDTH-1:0] wide_read_second_span_address;
    logic [31:0] wide_read_pair_stride;
    logic [10:0] wide_read_pair_count;
    logic wide_read_request_done;
    logic wide_read_request_error;
    logic wide_read_data_valid;
    logic wide_read_data_ready;
    logic [AXI_DATA_WIDTH-1:0] wide_read_data;
    logic [AXI_DATA_WIDTH/8-1:0] wide_read_byte_enable;
    logic wide_read_data_last;
    logic [TAG_WIDTH-1:0] wide_read_data_tag;
    logic wide_read_data_span;
    logic wide_write_request_valid;
    logic wide_write_request_ready;
    logic [ADDR_WIDTH-1:0] wide_write_request_address;
    logic [31:0] wide_write_request_bytes;
    logic [TAG_WIDTH-1:0] wide_write_request_tag;
    logic wide_write_request_done;
    logic wide_write_request_error;
    logic wide_write_data_valid;
    logic wide_write_data_ready;
    logic [AXI_DATA_WIDTH-1:0] wide_write_data;
    logic [AXI_DATA_WIDTH/8-1:0] wide_write_byte_enable;
    logic wide_write_data_last;
    logic width_adapter_idle;
    logic axi_read_busy;
    logic axi_write_busy;
    logic axi_drained;

    memory_stream_width_adapter #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .LOGICAL_DATA_WIDTH(LOGICAL_DATA_WIDTH),
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH)
    ) width_adapter (
        .clk(clk), .rst(rst),
        .read_request_valid(read_request_valid),
        .read_request_ready(read_request_ready),
        .read_request_address(read_request.byte_address),
        .read_request_bytes(read_request.byte_count),
        .read_request_tag(read_request.tag),
        .read_request_wide(read_request.wide_data),
        .read_second_span_valid(read_second_span_valid),
        .read_second_span_address(read_second_span_address),
        .read_pair_stride(read_pair_stride),
        .read_pair_count(read_pair_count),
        .read_request_done(read_completion.done_pulse),
        .read_request_error(read_completion.error),
        .read_data_valid(read_data_valid),
        .read_data_ready(read_data_ready),
        .read_data(read_data.data),
        .read_byte_enable(read_data.byte_enable),
        .read_data_last(read_data.last),
        .read_data_tag(read_data.tag),
        .read_data_span(read_data_span),
        .read_wide_data_valid(read_wide_data_valid),
        .read_wide_data_ready(read_wide_data_ready),
        .read_wide_data(read_wide_data.data),
        .read_wide_byte_enable(read_wide_data.byte_enable),
        .read_wide_data_last(read_wide_data.last),
        .read_wide_data_tag(read_wide_data.tag),
        .wide_read_request_valid(wide_read_request_valid),
        .wide_read_request_ready(wide_read_request_ready),
        .wide_read_request_address(wide_read_request_address),
        .wide_read_request_bytes(wide_read_request_bytes),
        .wide_read_request_tag(wide_read_request_tag),
        .wide_read_second_span_valid(wide_read_second_span_valid),
        .wide_read_second_span_address(wide_read_second_span_address),
        .wide_read_pair_stride(wide_read_pair_stride),
        .wide_read_pair_count(wide_read_pair_count),
        .wide_read_request_done(wide_read_request_done),
        .wide_read_request_error(wide_read_request_error),
        .wide_read_data_valid(wide_read_data_valid),
        .wide_read_data_ready(wide_read_data_ready),
        .wide_read_data(wide_read_data),
        .wide_read_byte_enable(wide_read_byte_enable),
        .wide_read_data_last(wide_read_data_last),
        .wide_read_data_tag(wide_read_data_tag),
        .wide_read_data_span(wide_read_data_span),
        .write_request_valid(write_request_valid),
        .write_request_ready(write_request_ready),
        .write_request_address(write_request.byte_address),
        .write_request_bytes(write_request.byte_count),
        .write_request_tag(write_request.tag),
        .write_request_done(write_completion.done_pulse),
        .write_request_error(write_completion.error),
        .write_data_valid(write_data_valid),
        .write_data_ready(write_data_ready),
        .write_data(write_data.data),
        .write_byte_enable(write_data.byte_enable),
        .write_data_last(write_data.last),
        .wide_write_request_valid(wide_write_request_valid),
        .wide_write_request_ready(wide_write_request_ready),
        .wide_write_request_address(wide_write_request_address),
        .wide_write_request_bytes(wide_write_request_bytes),
        .wide_write_request_tag(wide_write_request_tag),
        .wide_write_request_done(wide_write_request_done),
        .wide_write_request_error(wide_write_request_error),
        .wide_write_data_valid(wide_write_data_valid),
        .wide_write_data_ready(wide_write_data_ready),
        .wide_write_data(wide_write_data),
        .wide_write_byte_enable(wide_write_byte_enable),
        .wide_write_data_last(wide_write_data_last),
        .idle(width_adapter_idle)
    );

    assign read_busy = axi_read_busy || !width_adapter_idle;
    assign write_busy = axi_write_busy || !width_adapter_idle;
    assign memory_drained = axi_drained && width_adapter_idle;

    axi_memory_controller #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(AXI_DATA_WIDTH),
        .ID_WIDTH(ID_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .MAX_BURST_BYTES(MAX_BURST_BYTES),
        .MAX_READ_OUTSTANDING(MAX_READ_OUTSTANDING),
        .MAX_WRITE_OUTSTANDING(MAX_WRITE_OUTSTANDING)
    ) axi_transport (
        .clk(clk), .rst(rst), .abort_request(abort_request),
        .read_request_valid(wide_read_request_valid),
        .read_request_ready(wide_read_request_ready),
        .read_request_address(wide_read_request_address),
        .read_request_bytes(wide_read_request_bytes),
        .read_request_tag(wide_read_request_tag),
        .read_second_span_valid(wide_read_second_span_valid),
        .read_second_span_address(wide_read_second_span_address),
        .read_pair_stride(wide_read_pair_stride),
        .read_pair_count(wide_read_pair_count),
        .read_request_done(wide_read_request_done),
        .read_request_error(wide_read_request_error),
        .read_data_valid(wide_read_data_valid),
        .read_data_ready(wide_read_data_ready),
        .read_data(wide_read_data),
        .read_byte_enable(wide_read_byte_enable),
        .read_data_last(wide_read_data_last),
        .read_data_tag(wide_read_data_tag),
        .read_data_span(wide_read_data_span),
        .write_request_valid(wide_write_request_valid),
        .write_request_ready(wide_write_request_ready),
        .write_request_address(wide_write_request_address),
        .write_request_bytes(wide_write_request_bytes),
        .write_request_tag(wide_write_request_tag),
        .write_request_done(wide_write_request_done),
        .write_request_error(wide_write_request_error),
        .write_data_valid(wide_write_data_valid),
        .write_data_ready(wide_write_data_ready),
        .write_data(wide_write_data),
        .write_byte_enable(wide_write_byte_enable),
        .write_data_last(wide_write_data_last),
        .read_busy(axi_read_busy), .write_busy(axi_write_busy),
        .memory_drained(axi_drained),
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
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready), .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready), .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    initial begin
        if (LOGICAL_DATA_WIDTH != 128 || AXI_DATA_WIDTH != 256 ||
            AXI_DATA_WIDTH != 2 * LOGICAL_DATA_WIDTH)
            $error("memory_controller requires a 128-bit logical stream and 256-bit AXI");
    end
endmodule

`default_nettype wire
