`default_nettype none

// AXI4 master with independent ordered read and write streams.
// Reads use one fixed ID and a bounded burst-metadata FIFO. Write data is not
// interleaved, but B responses may remain pending while later bursts are sent.
module axi_master #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer DATA_WIDTH = 128,
    parameter integer ID_WIDTH = 4,
    parameter integer TAG_WIDTH = 8,
    parameter integer MAX_BURST_BYTES = 256,
    parameter integer MAX_READ_OUTSTANDING = 8,
    parameter integer MAX_WRITE_OUTSTANDING = 8,
    parameter integer ALLOW_SPARSE_WRITE_STROBE = 0
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
    output logic                         write_request_done,
    output logic                         write_request_error,
    input  logic                         write_data_valid,
    output logic                         write_data_ready,
    input  logic [DATA_WIDTH-1:0]        write_data,
    input  logic [DATA_WIDTH/8-1:0]      write_byte_enable,
    input  logic                         write_data_last,

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
    output logic                         read_idle,
    output logic                         write_idle,

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
    localparam integer BEAT_BYTES = DATA_WIDTH / 8;
    localparam integer BEAT_SHIFT = $clog2(BEAT_BYTES);
    localparam integer READ_PTR_WIDTH = MAX_READ_OUTSTANDING <= 2 ?
        1 : $clog2(MAX_READ_OUTSTANDING);
    localparam integer WRITE_PTR_WIDTH = MAX_WRITE_OUTSTANDING <= 2 ?
        1 : $clog2(MAX_WRITE_OUTSTANDING);
    localparam integer WRITE_COUNT_WIDTH = $clog2(MAX_WRITE_OUTSTANDING + 1);
    localparam logic [ID_WIDTH-1:0] AXI_ID = '0;
    localparam logic [31:0] READ_CREDIT_LIMIT = MAX_READ_OUTSTANDING[31:0];
    localparam logic [31:0] WRITE_CREDIT_LIMIT = MAX_WRITE_OUTSTANDING[31:0];
    localparam logic [63:0] BEAT_BYTES_64 = {32'd0, BEAT_BYTES[31:0]};

    function automatic logic [READ_PTR_WIDTH-1:0] next_read_fifo_pointer(
        input logic [READ_PTR_WIDTH-1:0] pointer
    );
        if (pointer == READ_PTR_WIDTH'(MAX_READ_OUTSTANDING - 1))
            next_read_fifo_pointer = '0;
        else
            next_read_fifo_pointer = pointer + 1'b1;
    endfunction

    logic read_active, read_stopping, read_stop_pending, read_error_sticky;
    logic [ADDR_WIDTH-1:0] read_second_base_address;
    logic [ADDR_WIDTH-1:0] read_first_base_address;
    logic [31:0] read_total_bytes, read_issued_bytes, read_delivered_bytes;
    logic [31:0] read_span_bytes;
    logic [31:0] saved_read_pair_stride;
    logic [10:0] read_pairs_remaining;
    logic read_has_second_span;
    logic read_issue_span;
    logic [ADDR_WIDTH-1:0] read_issue_base_address;
    logic [31:0] read_span_issued_bytes;
    logic [31:0] read_issued_bursts, read_completed_bursts;
    logic [TAG_WIDTH-1:0] saved_read_tag;
    logic [8:0] read_burst_beats [0:MAX_READ_OUTSTANDING-1];
    logic read_burst_span [0:MAX_READ_OUTSTANDING-1];
    logic [READ_PTR_WIDTH-1:0] read_fifo_write, read_fifo_read;
    logic [8:0] read_response_beat;

    logic write_active, write_stopping, write_stop_pending, write_error_sticky;
    logic [ADDR_WIDTH-1:0] write_base_address;
    logic [31:0] write_total_bytes, write_issued_bytes, write_delivered_bytes;
    logic [31:0] write_issued_bursts, write_completed_bursts;
    logic write_burst_active;
    logic [8:0] write_burst_total_beats, write_beats_remaining;
    logic [31:0] write_burst_useful_bytes;
    logic [31:0] write_completion_issued [0:MAX_WRITE_OUTSTANDING-1];
    logic [31:0] write_completion_completed [0:MAX_WRITE_OUTSTANDING-1];
    logic write_completion_payload_done [0:MAX_WRITE_OUTSTANDING-1];
    logic write_completion_error [0:MAX_WRITE_OUTSTANDING-1];
    logic [WRITE_PTR_WIDTH-1:0] write_completion_write;
    logic [WRITE_PTR_WIDTH-1:0] write_completion_read;
    logic [WRITE_PTR_WIDTH-1:0] write_active_entry;
    logic [WRITE_COUNT_WIDTH-1:0] write_completion_count;

    logic [31:0] read_outstanding_wide, write_outstanding_wide;
    logic [ADDR_WIDTH-1:0] next_read_address, next_write_address;
    logic [31:0] read_bytes_remaining, write_bytes_remaining_to_issue;
    logic [31:0] next_read_useful_bytes, next_read_actual_bytes;
    logic [31:0] next_write_useful_bytes, next_write_actual_bytes;
    logic [8:0] next_read_beats, next_write_beats;
    logic [31:0] read_boundary_bytes, write_boundary_bytes;
    logic read_issue, read_accept, read_expected_last, read_metadata_end;
    logic read_protocol_good, read_final_beat;
    logic write_aw_accept, write_accept, write_response;
    logic write_request_fire, write_request_invalid;
    logic write_completion_pop, write_front_has_response;
    logic write_drain_active;
    logic write_source_held;
    logic write_beat_error;
    logic [31:0] write_bytes_before_beat, write_bytes_remaining;
    logic [31:0] write_beat_useful_bytes;
    logic [BEAT_BYTES-1:0] write_required_enable;
    logic [$clog2(BEAT_BYTES+1)-1:0] read_accepted_bytes;
    logic [$clog2(BEAT_BYTES+1)-1:0] write_accepted_bytes;
    logic read_request_invalid;
    logic [31:0] read_request_total_bytes;
    logic [ADDR_WIDTH:0] read_first_span_end;
    logic [ADDR_WIDTH:0] read_second_span_end;
    logic [ADDR_WIDTH:0] read_pair_extent;
    logic [ADDR_WIDTH:0] read_pair_stride_wide;
    logic [ADDR_WIDTH:0] read_address_guard;

    assign read_outstanding_wide = read_issued_bursts - read_completed_bursts;
    assign write_outstanding_wide = write_issued_bursts - write_completed_bursts;
    assign read_outstanding = read_outstanding_wide[4:0];
    assign write_outstanding = write_outstanding_wide[4:0];
    assign read_request_ready = !read_active;
    assign write_request_ready = !abort_request && !write_active &&
        write_completion_count < WRITE_COUNT_WIDTH'(MAX_WRITE_OUTSTANDING);
    assign write_request_fire = write_request_valid && write_request_ready;
    assign write_request_invalid = write_request_bytes == 0 ||
        write_request_address[BEAT_SHIFT-1:0] != 0 ||
        write_request_address + {{(ADDR_WIDTH-32){1'b0}}, write_request_bytes} <
            write_request_address;
    assign write_completion_pop = write_completion_count != 0 &&
        write_completion_payload_done[write_completion_read] &&
        write_completion_completed[write_completion_read] ==
            write_completion_issued[write_completion_read];
    assign write_front_has_response = write_completion_count != 0 &&
        write_completion_completed[write_completion_read] <
            write_completion_issued[write_completion_read];
    assign read_idle = !read_active && read_outstanding_wide == 0 && !m_axi_arvalid;
    assign write_idle = !write_active && write_completion_count == 0 &&
        write_outstanding_wide == 0 && !write_burst_active &&
        !m_axi_awvalid && !m_axi_wvalid;

    assign next_read_address = read_issue_base_address +
        {{(ADDR_WIDTH-32){1'b0}}, read_span_issued_bytes};
    assign next_write_address = write_base_address +
        {{(ADDR_WIDTH-32){1'b0}}, write_issued_bytes};
    assign read_bytes_remaining = read_span_bytes - read_span_issued_bytes;
    assign write_bytes_remaining_to_issue = write_total_bytes - write_issued_bytes;
    assign read_boundary_bytes = 32'd4096 - {20'd0, next_read_address[11:0]};
    assign write_boundary_bytes = 32'd4096 - {20'd0, next_write_address[11:0]};
    assign next_read_useful_bytes = read_bytes_remaining < MAX_BURST_BYTES ?
        (read_bytes_remaining < read_boundary_bytes ? read_bytes_remaining : read_boundary_bytes) :
        (MAX_BURST_BYTES < read_boundary_bytes ? MAX_BURST_BYTES : read_boundary_bytes);
    assign next_write_useful_bytes = write_bytes_remaining_to_issue < MAX_BURST_BYTES ?
        (write_bytes_remaining_to_issue < write_boundary_bytes ?
            write_bytes_remaining_to_issue : write_boundary_bytes) :
        (MAX_BURST_BYTES < write_boundary_bytes ? MAX_BURST_BYTES : write_boundary_bytes);
    assign next_read_actual_bytes = (next_read_useful_bytes + BEAT_BYTES - 1) &
        ~(BEAT_BYTES - 1);
    assign next_write_actual_bytes = (next_write_useful_bytes + BEAT_BYTES - 1) &
        ~(BEAT_BYTES - 1);
    assign next_read_beats = next_read_actual_bytes[BEAT_SHIFT +: 9];
    assign next_write_beats = next_write_actual_bytes[BEAT_SHIFT +: 9];

    assign m_axi_arid = AXI_ID;
    assign m_axi_araddr = next_read_address;
    assign m_axi_arlen = next_read_beats[7:0] - 1'b1;
    assign m_axi_arsize = BEAT_SHIFT[2:0];
    assign m_axi_arburst = 2'b01;
    assign m_axi_arvalid = read_active && !read_stopping &&
        (!read_error_sticky || read_stop_pending) &&
        read_issued_bytes < read_total_bytes &&
        read_outstanding_wide < READ_CREDIT_LIMIT;
    assign read_issue = m_axi_arvalid && m_axi_arready;

    assign read_expected_last = read_response_beat + 1'b1 ==
        read_burst_beats[read_fifo_read];
    // A late RLAST is drained until the slave actually terminates that burst.
    // An early RLAST terminates the malformed burst and releases its metadata.
    assign read_metadata_end = m_axi_rlast;
    assign read_protocol_good = m_axi_rid == AXI_ID && m_axi_rresp == 2'b00 &&
        m_axi_rlast == read_expected_last && read_outstanding_wide != 0;
    assign read_final_beat = read_expected_last &&
        read_delivered_bytes +
            {{(32-$clog2(BEAT_BYTES+1)){1'b0}}, read_accepted_bytes} ==
        read_total_bytes;
    assign m_axi_rready = read_active && (read_stopping || read_error_sticky ||
        !read_protocol_good || read_data_ready);
    assign read_data_valid = read_active && !read_stopping && !read_error_sticky &&
        m_axi_rvalid && read_protocol_good;
    assign read_data = m_axi_rdata;
    assign read_data_last = read_final_beat;
    assign read_data_tag = saved_read_tag;
    assign read_data_span = read_burst_span[read_fifo_read];
    assign read_accept = m_axi_rvalid && m_axi_rready;

    assign read_first_span_end = {1'b0, read_request_address} +
        {{(ADDR_WIDTH-31){1'b0}}, read_request_bytes};
    assign read_second_span_end = {1'b0, read_second_span_address} +
        {{(ADDR_WIDTH-31){1'b0}}, read_request_bytes};
    assign read_pair_extent = {1'b0, read_second_span_address} -
        {1'b0, read_request_address} +
        {{(ADDR_WIDTH-31){1'b0}}, read_request_bytes};
    assign read_pair_stride_wide =
        {{(ADDR_WIDTH-31){1'b0}}, read_pair_stride};
    assign read_address_guard = ({1'b0, {ADDR_WIDTH{1'b1}}} -
        ((ADDR_WIDTH+1)'(1) << 43));
    assign read_request_total_bytes = !read_second_span_valid ?
        read_request_bytes :
        (read_request_bytes == 32'd128 ?
            (32'(read_pair_count) << 8) :
            (32'(read_pair_count) << 9));
    assign read_request_invalid = read_request_bytes == 0 ||
        read_pair_count == 0 ||
        read_request_address[BEAT_SHIFT-1:0] != 0 ||
        read_first_span_end[ADDR_WIDTH] ||
        (!read_second_span_valid &&
         (read_pair_count != 11'd1 || read_pair_stride != 0)) ||
        (read_second_span_valid &&
         ((read_request_bytes != 32'd128 &&
           read_request_bytes != 32'd256) ||
          read_request_address[4:0] != 0 ||
          read_second_span_address[4:0] != 0 ||
          read_second_span_end[ADDR_WIDTH] ||
          read_second_span_address < read_first_span_end[ADDR_WIDTH-1:0] ||
          (read_pair_count == 11'd1 && read_pair_stride != 0) ||
          (read_pair_count > 11'd1 &&
           (read_pair_stride == 0 || read_pair_stride[4:0] != 0 ||
            read_pair_stride_wide < read_pair_extent ||
            {1'b0, read_request_address} > read_address_guard ||
            {1'b0, read_second_span_address} > read_address_guard))));

    always_comb begin
        read_byte_enable = '0;
        for (integer read_byte_index = 0; read_byte_index < BEAT_BYTES;
             read_byte_index = read_byte_index + 1)
            if (read_delivered_bytes + read_byte_index < read_total_bytes)
                read_byte_enable[read_byte_index] = 1'b1;
    end
    assign read_accepted_bytes = $countones(read_byte_enable);

    assign m_axi_awid = AXI_ID;
    assign m_axi_awaddr = next_write_address;
    assign m_axi_awlen = next_write_beats[7:0] - 1'b1;
    assign m_axi_awsize = BEAT_SHIFT[2:0];
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = write_active && !write_stopping &&
        (!write_error_sticky || write_stop_pending) &&
        !write_burst_active && write_issued_bytes < write_total_bytes &&
        write_outstanding_wide < WRITE_CREDIT_LIMIT;
    assign write_aw_accept = m_axi_awvalid && m_axi_awready;
    // The source holds a presented beat until write_data_ready. An earlier
    // burst's error (or abort) may drain only after that beat is accepted.
    assign write_drain_active = (write_stopping || write_error_sticky) &&
        !write_source_held;
    assign write_data_ready = write_active && write_burst_active &&
        !write_drain_active && m_axi_wready;
    assign m_axi_wdata = write_drain_active ? '0 : write_data;
    assign m_axi_wstrb = write_drain_active ? '0 :
        write_byte_enable & write_required_enable;
    assign m_axi_wlast = write_burst_active && write_beats_remaining == 1;
    assign m_axi_wvalid = write_burst_active &&
        (write_drain_active || write_data_valid);
    assign write_accept = m_axi_wvalid && m_axi_wready;
    assign write_accepted_bytes = $countones(m_axi_wstrb);
    assign m_axi_bready = write_front_has_response;
    assign write_response = m_axi_bvalid && m_axi_bready;
    assign write_bytes_before_beat = write_issued_bytes +
        ({23'd0, write_burst_total_beats} - {23'd0, write_beats_remaining}) * BEAT_BYTES;
    assign write_bytes_remaining = write_total_bytes - write_bytes_before_beat;
    assign write_beat_useful_bytes = write_bytes_remaining < BEAT_BYTES ?
        write_bytes_remaining : BEAT_BYTES;
    assign write_beat_error = write_data_last !=
        (write_delivered_bytes + write_beat_useful_bytes == write_total_bytes) ||
        (ALLOW_SPARSE_WRITE_STROBE == 0 &&
         (write_byte_enable & write_required_enable) != write_required_enable) ||
        (ALLOW_SPARSE_WRITE_STROBE != 0 &&
         (((write_byte_enable & write_required_enable) == 0) ||
          ((write_byte_enable & ~write_required_enable) != 0)));

    always_comb begin
        write_required_enable = '0;
        for (integer write_byte_index = 0; write_byte_index < BEAT_BYTES;
             write_byte_index = write_byte_index + 1)
            if (write_byte_index < write_beat_useful_bytes)
                write_required_enable[write_byte_index] = 1'b1;
    end

`ifdef SYNTHESIS
    always_comb begin
        read_credit_stall_cycles = '0;
        write_credit_stall_cycles = '0;
        read_burst_count = '0;
        write_burst_count = '0;
        actual_read_bytes = '0;
        actual_write_bytes = '0;
        useful_read_bytes = '0;
        useful_write_bytes = '0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            read_active <= 1'b0;
            read_stopping <= 1'b0;
            read_stop_pending <= 1'b0;
            read_error_sticky <= 1'b0;
            read_first_base_address <= '0;
            read_second_base_address <= '0;
            read_total_bytes <= '0;
            read_span_bytes <= '0;
            saved_read_pair_stride <= '0;
            read_pairs_remaining <= '0;
            read_has_second_span <= 1'b0;
            read_issue_span <= 1'b0;
            read_issue_base_address <= '0;
            read_span_issued_bytes <= '0;
            read_issued_bytes <= '0;
            read_delivered_bytes <= '0;
            read_issued_bursts <= '0;
            read_completed_bursts <= '0;
            saved_read_tag <= '0;
            read_fifo_write <= '0;
            read_fifo_read <= '0;
            read_response_beat <= '0;
            read_request_done <= 1'b0;
            read_request_error <= 1'b0;

            write_active <= 1'b0;
            write_stopping <= 1'b0;
            write_stop_pending <= 1'b0;
            write_error_sticky <= 1'b0;
            write_source_held <= 1'b0;
            write_base_address <= '0;
            write_total_bytes <= '0;
            write_issued_bytes <= '0;
            write_delivered_bytes <= '0;
            write_issued_bursts <= '0;
            write_completed_bursts <= '0;
            write_burst_active <= 1'b0;
            write_burst_total_beats <= '0;
            write_beats_remaining <= '0;
            write_burst_useful_bytes <= '0;
            write_completion_write <= '0;
            write_completion_read <= '0;
            write_active_entry <= '0;
            write_completion_count <= '0;
            for (integer completion = 0;
                 completion < MAX_WRITE_OUTSTANDING; completion = completion + 1) begin
                write_completion_issued[completion] <= '0;
                write_completion_completed[completion] <= '0;
                write_completion_payload_done[completion] <= 1'b0;
                write_completion_error[completion] <= 1'b0;
            end
            write_request_done <= 1'b0;
            write_request_error <= 1'b0;

            read_outstanding_high_water <= '0;
            write_outstanding_high_water <= '0;
`ifndef SYNTHESIS
            read_credit_stall_cycles <= '0;
            write_credit_stall_cycles <= '0;
            read_burst_count <= '0;
            write_burst_count <= '0;
            actual_read_bytes <= '0;
            actual_write_bytes <= '0;
            useful_read_bytes <= '0;
            useful_write_bytes <= '0;
`endif
        end else begin
            write_source_held <= m_axi_wvalid && !m_axi_wready &&
                !write_drain_active;
            read_request_done <= 1'b0;
            write_request_done <= 1'b0;
            write_request_error <= 1'b0;

            if (read_request_valid && read_request_ready) begin
                read_request_error <= 1'b0;
                if (read_request_invalid) begin
                    read_request_done <= 1'b1;
                    read_request_error <= 1'b1;
                end else begin
                    read_active <= 1'b1;
                    read_stopping <= 1'b0;
                    read_stop_pending <= 1'b0;
                    read_error_sticky <= 1'b0;
                    read_first_base_address <= read_request_address;
                    read_second_base_address <= read_second_span_address;
                    read_span_bytes <= read_request_bytes;
                    saved_read_pair_stride <= read_pair_stride;
                    read_pairs_remaining <= read_pair_count;
                    read_has_second_span <= read_second_span_valid;
                    read_issue_span <= 1'b0;
                    read_issue_base_address <= read_request_address;
                    read_span_issued_bytes <= '0;
                    read_total_bytes <= read_request_total_bytes;
                    read_issued_bytes <= '0;
                    read_delivered_bytes <= '0;
                    read_issued_bursts <= '0;
                    read_completed_bursts <= '0;
                    saved_read_tag <= read_request_tag;
                    read_fifo_write <= '0;
                    read_fifo_read <= '0;
                    read_response_beat <= '0;
                end
            end

            if (write_request_fire) begin
                write_completion_issued[write_completion_write] <= '0;
                write_completion_completed[write_completion_write] <= '0;
                write_completion_payload_done[write_completion_write] <=
                    write_request_invalid;
                write_completion_error[write_completion_write] <=
                    write_request_invalid;
                write_active_entry <= write_completion_write;
                write_completion_write <= write_completion_write + 1'b1;
                if (!write_request_invalid) begin
                    write_active <= 1'b1;
                    write_stopping <= 1'b0;
                    write_stop_pending <= 1'b0;
                    write_error_sticky <= 1'b0;
                    write_base_address <= write_request_address;
                    write_total_bytes <= write_request_bytes;
                    write_issued_bytes <= '0;
                    write_delivered_bytes <= '0;
                    write_burst_active <= 1'b0;
                end
            end

            if (write_completion_pop) begin
                write_request_done <= 1'b1;
                write_request_error <=
                    abort_request || write_completion_error[write_completion_read];
                write_completion_payload_done[write_completion_read] <= 1'b0;
                write_completion_error[write_completion_read] <= 1'b0;
                write_completion_issued[write_completion_read] <= '0;
                write_completion_completed[write_completion_read] <= '0;
                write_completion_read <= write_completion_read + 1'b1;
            end

            case ({write_request_fire, write_completion_pop})
                2'b10: write_completion_count <= write_completion_count + 1'b1;
                2'b01: write_completion_count <= write_completion_count - 1'b1;
                default: write_completion_count <= write_completion_count;
            endcase

            if (abort_request) begin
                for (integer completion = 0;
                     completion < MAX_WRITE_OUTSTANDING; completion = completion + 1)
                    write_completion_error[completion] <= 1'b1;
                if (read_active) begin
                    if (m_axi_arvalid && !m_axi_arready)
                        read_stop_pending <= 1'b1;
                    else
                        read_stopping <= 1'b1;
                end
                if (write_active) begin
                    if (m_axi_awvalid && !m_axi_awready)
                        write_stop_pending <= 1'b1;
                    else
                        write_stopping <= 1'b1;
                end
            end

            if (read_stop_pending && read_issue) begin
                read_stop_pending <= 1'b0;
                read_stopping <= 1'b1;
            end
            if (write_stop_pending && write_aw_accept) begin
                write_stop_pending <= 1'b0;
                write_stopping <= 1'b1;
            end

`ifndef SYNTHESIS
            if (read_active && !read_stopping && !read_error_sticky &&
                read_issued_bytes < read_total_bytes &&
                read_outstanding_wide >= READ_CREDIT_LIMIT)
                read_credit_stall_cycles <= read_credit_stall_cycles + 1'b1;
            if (write_active && !write_stopping && !write_error_sticky &&
                !write_burst_active && write_issued_bytes < write_total_bytes &&
                write_outstanding_wide >= WRITE_CREDIT_LIMIT)
                write_credit_stall_cycles <= write_credit_stall_cycles + 1'b1;
`endif

            if (read_issue) begin
                read_burst_beats[read_fifo_write] <= next_read_beats;
                read_burst_span[read_fifo_write] <= read_issue_span;
                read_fifo_write <= next_read_fifo_pointer(read_fifo_write);
                read_issued_bytes <= read_issued_bytes + next_read_useful_bytes;
                if (read_has_second_span && !read_issue_span &&
                    read_span_issued_bytes + next_read_useful_bytes ==
                        read_span_bytes) begin
                    read_issue_span <= 1'b1;
                    read_issue_base_address <= read_second_base_address;
                    read_span_issued_bytes <= '0;
                end else if (read_has_second_span && read_issue_span &&
                    read_span_issued_bytes + next_read_useful_bytes ==
                        read_span_bytes && read_pairs_remaining > 11'd1) begin
                    read_first_base_address <= read_first_base_address +
                        ADDR_WIDTH'(saved_read_pair_stride);
                    read_second_base_address <= read_second_base_address +
                        ADDR_WIDTH'(saved_read_pair_stride);
                    read_issue_span <= 1'b0;
                    read_issue_base_address <= read_first_base_address +
                        ADDR_WIDTH'(saved_read_pair_stride);
                    read_span_issued_bytes <= '0;
                    read_pairs_remaining <= read_pairs_remaining - 11'd1;
                end else begin
                    read_span_issued_bytes <= read_span_issued_bytes +
                        next_read_useful_bytes;
                end
                read_issued_bursts <= read_issued_bursts + 1'b1;
`ifndef SYNTHESIS
                read_burst_count <= read_burst_count + 1'b1;
                actual_read_bytes <= actual_read_bytes + {32'd0, next_read_actual_bytes};
                useful_read_bytes <= useful_read_bytes + {32'd0, next_read_useful_bytes};
`endif
                if (!(read_accept && read_metadata_end) &&
                    read_outstanding_wide + 1'b1 >
                    {11'd0, read_outstanding_high_water})
                    read_outstanding_high_water <=
                        read_outstanding_wide[4:0] + 1'b1;
            end
            if (read_accept) begin
                if (!read_protocol_good) begin
                    read_error_sticky <= 1'b1;
                    // An AR already presented to the slave cannot be withdrawn.
                    // Accept it, record its metadata, then stop issuing bursts.
                    if (m_axi_arvalid && !m_axi_arready)
                        read_stop_pending <= 1'b1;
                    else
                        read_stopping <= 1'b1;
                end
                if (!read_stopping && !read_error_sticky && read_protocol_good)
                    read_delivered_bytes <= read_delivered_bytes +
                        {{(32-$clog2(BEAT_BYTES+1)){1'b0}}, read_accepted_bytes};
                if (read_metadata_end) begin
                    read_fifo_read <= next_read_fifo_pointer(read_fifo_read);
                    read_response_beat <= '0;
                    read_completed_bursts <= read_completed_bursts + 1'b1;
                end else begin
                    read_response_beat <= read_response_beat + 1'b1;
                end
            end

            if (read_active &&
                ((read_stopping || read_error_sticky) ?
                    (read_outstanding_wide == 0 && !m_axi_arvalid) :
                    (read_issued_bytes == read_total_bytes &&
                     read_delivered_bytes == read_total_bytes &&
                     read_outstanding_wide == 0))) begin
                read_active <= 1'b0;
                read_stop_pending <= 1'b0;
                read_request_done <= 1'b1;
                read_request_error <= read_stopping || read_error_sticky;
            end

            if (write_aw_accept) begin
                write_burst_active <= 1'b1;
                write_burst_total_beats <= next_write_beats;
                write_beats_remaining <= next_write_beats;
                write_burst_useful_bytes <= next_write_useful_bytes;
                write_issued_bursts <= write_issued_bursts + 1'b1;
                write_completion_issued[write_active_entry] <=
                    write_completion_issued[write_active_entry] + 1'b1;
`ifndef SYNTHESIS
                write_burst_count <= write_burst_count + 1'b1;
`endif
                if (!write_response && write_outstanding_wide + 1'b1 >
                    {11'd0, write_outstanding_high_water})
                    write_outstanding_high_water <=
                        write_outstanding_wide[4:0] + 1'b1;
            end
            if (write_accept) begin
                write_delivered_bytes <= write_delivered_bytes + write_beat_useful_bytes;
`ifndef SYNTHESIS
                actual_write_bytes <= actual_write_bytes + BEAT_BYTES_64;
                useful_write_bytes <= useful_write_bytes +
                    {{(64-$clog2(BEAT_BYTES+1)){1'b0}}, write_accepted_bytes};
`endif
                if (write_beat_error) begin
                    write_error_sticky <= 1'b1;
                    write_completion_error[write_active_entry] <= 1'b1;
                end
                write_beats_remaining <= write_beats_remaining - 1'b1;
                if (m_axi_wlast) begin
                    write_burst_active <= 1'b0;
                    write_issued_bytes <= write_issued_bytes + write_burst_useful_bytes;
                    if (write_stopping || write_error_sticky || write_beat_error ||
                        write_issued_bytes + write_burst_useful_bytes ==
                            write_total_bytes) begin
                        write_active <= 1'b0;
                        write_stop_pending <= 1'b0;
                        write_completion_payload_done[write_active_entry] <= 1'b1;
                        if (write_stopping || write_error_sticky || write_beat_error)
                            write_completion_error[write_active_entry] <= 1'b1;
                    end
                end
            end
            if (write_response) begin
                write_completed_bursts <= write_completed_bursts + 1'b1;
                write_completion_completed[write_completion_read] <=
                    write_completion_completed[write_completion_read] + 1'b1;
                if (m_axi_bid != AXI_ID || m_axi_bresp != 2'b00 ||
                    write_outstanding_wide == 0) begin
                    write_completion_error[write_completion_read] <= 1'b1;
                    if (write_active &&
                        write_active_entry == write_completion_read) begin
                        write_error_sticky <= 1'b1;
                        // AW is irrevocable once valid. Accept a stalled next
                        // burst before stopping this logical write.
                        if (m_axi_awvalid && !m_axi_awready)
                            write_stop_pending <= 1'b1;
                        else
                            write_stopping <= 1'b1;
                    end
                end
            end

            // An already presented AW still owns a burst, including its
            // handshake cycle before write_burst_active is registered.
            if (write_active && !write_burst_active && !m_axi_awvalid &&
                (write_stopping || write_error_sticky)) begin
                write_active <= 1'b0;
                write_stop_pending <= 1'b0;
                write_completion_payload_done[write_active_entry] <= 1'b1;
                write_completion_error[write_active_entry] <= 1'b1;
            end
        end
    end

    initial begin
        if (ADDR_WIDTH < 44 || DATA_WIDTH < 32 || DATA_WIDTH % 8 != 0 ||
            (BEAT_BYTES & (BEAT_BYTES - 1)) != 0 || ID_WIDTH < 1 || TAG_WIDTH < 1 ||
            MAX_READ_OUTSTANDING < 1 || MAX_READ_OUTSTANDING > 31 ||
            MAX_WRITE_OUTSTANDING != 8 ||
            MAX_BURST_BYTES < BEAT_BYTES ||
            MAX_BURST_BYTES % BEAT_BYTES != 0 || MAX_BURST_BYTES > 4096 ||
            MAX_BURST_BYTES / BEAT_BYTES > 256 ||
            (ALLOW_SPARSE_WRITE_STROBE != 0 && ALLOW_SPARSE_WRITE_STROBE != 1))
            $error("axi_master parameter relation is invalid");
    end

`ifndef SYNTHESIS
    logic stalled_ar, stalled_aw, stalled_w, stalled_read_data;
    logic stalled_read_request;
    logic [ADDR_WIDTH+8-1:0] held_ar, held_aw;
    logic [DATA_WIDTH+DATA_WIDTH/8:0] held_w;
    logic [DATA_WIDTH+DATA_WIDTH/8+TAG_WIDTH+1:0] held_read_data;
    logic [2*ADDR_WIDTH+TAG_WIDTH+75:0] held_read_request;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_ar <= 1'b0;
            stalled_aw <= 1'b0;
            stalled_w <= 1'b0;
            stalled_read_data <= 1'b0;
            stalled_read_request <= 1'b0;
        end else begin
            if (stalled_read_request)
                assert (read_request_valid &&
                    {read_request_address, read_request_bytes,
                     read_request_tag, read_second_span_valid,
                     read_second_span_address, read_pair_stride,
                     read_pair_count} == held_read_request)
                    else $error("axi_master changed a stalled read request");
            if (stalled_ar)
                assert (m_axi_arvalid && {m_axi_araddr, m_axi_arlen} == held_ar)
                    else $error("axi_master changed stalled AR");
            if (stalled_aw)
                assert (m_axi_awvalid && {m_axi_awaddr, m_axi_awlen} == held_aw)
                    else $error("axi_master changed stalled AW");
            if (stalled_w)
                assert (m_axi_wvalid && {m_axi_wdata, m_axi_wstrb, m_axi_wlast} == held_w)
                    else $error("axi_master changed stalled W");
            if (stalled_read_data && !read_stopping &&
                !read_error_sticky && !abort_request)
                assert (read_data_valid &&
                    {read_data, read_byte_enable, read_data_last,
                     read_data_tag, read_data_span} == held_read_data)
                    else $error(
                        "axi_master changed stalled read data valid=%0d data=%0d be=%0d last=%0d tag=%0d span=%0d active=%0d stopping=%0d sticky=%0d rvalid=%0d protocol=%0d rready=%0d outstanding=%0d expected_last=%0d rlast=%0d rsp_beat=%0d burst_beats=%0d",
                        read_data_valid,
                        read_data == held_read_data[
                            DATA_WIDTH+DATA_WIDTH/8+TAG_WIDTH+1:DATA_WIDTH/8+TAG_WIDTH+2],
                        read_byte_enable == held_read_data[
                            DATA_WIDTH/8+TAG_WIDTH+1:TAG_WIDTH+2],
                        read_data_last == held_read_data[TAG_WIDTH+1],
                        read_data_tag == held_read_data[TAG_WIDTH:1],
                        read_data_span == held_read_data[0],
                        read_active, read_stopping, read_error_sticky,
                        m_axi_rvalid, read_protocol_good, m_axi_rready,
                        read_outstanding_wide, read_expected_last, m_axi_rlast,
                        read_response_beat,
                        read_burst_beats[read_fifo_read]);
            if (read_issue && read_issue_span)
                assert (read_has_second_span)
                    else $error("axi_master issued an unconfigured second span");
            if (read_issue)
                assert ({1'b0, m_axi_araddr[11:0]} +
                        (13'(m_axi_arlen) + 13'd1) * BEAT_BYTES <= 13'd4096)
                    else $error("axi_master accepted a read burst crossing 4 KiB");
            if (read_data_valid && read_data_last && read_has_second_span)
                assert (read_data_span && read_pairs_remaining == 11'd1)
                    else $error("axi_master ended a repeated pair before final Up span");
            if (read_active && read_has_second_span && read_pairs_remaining > 11'd1)
                assert (read_first_base_address +
                            ADDR_WIDTH'(saved_read_pair_stride) >=
                            read_first_base_address &&
                        read_second_base_address +
                            ADDR_WIDTH'(saved_read_pair_stride) >=
                            read_second_base_address)
                    else $error("axi_master repeated pair address overflow");
            assert (read_outstanding_wide <= READ_CREDIT_LIMIT)
                else $error("axi_master exceeded read credits");
            assert (write_outstanding_wide <= WRITE_CREDIT_LIMIT)
                else $error("axi_master exceeded write credits");
            stalled_ar <= m_axi_arvalid && !m_axi_arready;
            stalled_aw <= m_axi_awvalid && !m_axi_awready;
            stalled_w <= m_axi_wvalid && !m_axi_wready;
            stalled_read_data <= read_data_valid && !read_data_ready;
            stalled_read_request <= read_request_valid && !read_request_ready;
            held_ar <= {m_axi_araddr, m_axi_arlen};
            held_aw <= {m_axi_awaddr, m_axi_awlen};
            held_w <= {m_axi_wdata, m_axi_wstrb, m_axi_wlast};
            held_read_data <= {read_data, read_byte_enable, read_data_last,
                read_data_tag, read_data_span};
            held_read_request <= {read_request_address, read_request_bytes,
                read_request_tag, read_second_span_valid,
                read_second_span_address, read_pair_stride,
                read_pair_count};
        end
    end
`endif
endmodule

`default_nettype wire
