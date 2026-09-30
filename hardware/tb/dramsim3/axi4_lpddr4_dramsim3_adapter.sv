`timescale 1ns/1ps
`default_nettype none

module axi4_lpddr4_dramsim3_adapter #(
    parameter int unsigned ADDR_WIDTH = 64,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ID_WIDTH = 4,
    parameter int unsigned CORE_FREQUENCY_MHZ = 500,
    parameter int unsigned SLOT_COUNT = 12
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         inject_read_response_error,
    input  logic                         inject_write_response_error,
    input  logic [ID_WIDTH-1:0]          s_axi_awid,
    input  logic [ADDR_WIDTH-1:0]        s_axi_awaddr,
    input  logic [7:0]                   s_axi_awlen,
    input  logic [2:0]                   s_axi_awsize,
    input  logic [1:0]                   s_axi_awburst,
    input  logic                         s_axi_awvalid,
    output logic                         s_axi_awready,
    input  logic [DATA_WIDTH-1:0]        s_axi_wdata,
    input  logic [DATA_WIDTH/8-1:0]      s_axi_wstrb,
    input  logic                         s_axi_wlast,
    input  logic                         s_axi_wvalid,
    output logic                         s_axi_wready,
    output logic [ID_WIDTH-1:0]          s_axi_bid,
    output logic [1:0]                   s_axi_bresp,
    output logic                         s_axi_bvalid,
    input  logic                         s_axi_bready,
    input  logic [ID_WIDTH-1:0]          s_axi_arid,
    input  logic [ADDR_WIDTH-1:0]        s_axi_araddr,
    input  logic [7:0]                   s_axi_arlen,
    input  logic [2:0]                   s_axi_arsize,
    input  logic [1:0]                   s_axi_arburst,
    input  logic                         s_axi_arvalid,
    output logic                         s_axi_arready,
    output logic [ID_WIDTH-1:0]          s_axi_rid,
    output logic [DATA_WIDTH-1:0]        s_axi_rdata,
    output logic [1:0]                   s_axi_rresp,
    output logic                         s_axi_rlast,
    output logic                         s_axi_rvalid,
    input  logic                         s_axi_rready,
    output logic [4:0]                   max_read_queued,
    output logic [3:0]                   debug_read_slots_used,
    output logic [3:0]                   debug_read_bursts_not_submitted,
    output logic                         debug_read_backend_can_accept,
    output logic                         initial_image_loaded,
    output logic [63:0]                  initial_image_bytes,
    output logic                         backend_idle,
    output logic                         protocol_error
);
    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam logic [2:0] AXI_SIZE = 3'($clog2(BEAT_BYTES));
    localparam int unsigned SLOT_PTR_WIDTH = SLOT_COUNT <= 2 ?
        1 : $clog2(SLOT_COUNT);
    localparam int unsigned SLOT_COUNT_WIDTH = $clog2(SLOT_COUNT + 1);

    import "DPI-C" function int supra_dramsim3_create(
        input string config_path, input string region_manifest_path,
        input string image_path, input string output_directory,
        input int core_frequency_mhz);
    import "DPI-C" function void supra_dramsim3_destroy();
    import "DPI-C" function int supra_dramsim3_reset();
    import "DPI-C" function int supra_dramsim3_can_accept(
        input longint unsigned address, input int write);
    import "DPI-C" function int supra_dramsim3_submit_read(
        input longint unsigned address, input longint unsigned token);
    import "DPI-C" function int supra_dramsim3_submit_write(
        input longint unsigned address, input longint unsigned token,
        input bit [255:0] data_words, input int unsigned byte_enable);
    import "DPI-C" function void supra_dramsim3_tick_core();
    import "DPI-C" function int supra_dramsim3_poll_read(
        output longint unsigned token, output bit [255:0] data_words);
    import "DPI-C" function int supra_dramsim3_poll_write(
        output longint unsigned token);
    import "DPI-C" function int supra_dramsim3_idle();

    logic [ADDR_WIDTH-1:0] read_addr [0:SLOT_COUNT-1];
    logic [3:0] read_beats [0:SLOT_COUNT-1];
    logic [3:0] read_submitted [0:SLOT_COUNT-1];
    logic [7:0] read_received [0:SLOT_COUNT-1];
    logic [DATA_WIDTH-1:0] read_data [0:SLOT_COUNT-1][0:7];
    logic [ID_WIDTH-1:0] read_id [0:SLOT_COUNT-1];
    logic read_error [0:SLOT_COUNT-1];
    logic [SLOT_PTR_WIDTH-1:0] read_head, read_tail, read_submit_slot;
    logic [SLOT_COUNT_WIDTH-1:0] read_count, read_unsubmitted;
    logic [2:0] read_return_beat;

    logic [ADDR_WIDTH-1:0] write_addr [0:SLOT_COUNT-1];
    logic [3:0] write_beats [0:SLOT_COUNT-1];
    logic [3:0] write_submitted [0:SLOT_COUNT-1];
    logic [7:0] write_completed [0:SLOT_COUNT-1];
    logic [ID_WIDTH-1:0] write_id [0:SLOT_COUNT-1];
    logic write_error [0:SLOT_COUNT-1];
    logic write_injected_error [0:SLOT_COUNT-1];
    logic [SLOT_PTR_WIDTH-1:0] write_head, write_tail, write_feed_slot;
    logic [SLOT_COUNT_WIDTH-1:0] write_count, write_unfed;

    logic backend_ready;
    logic read_backend_can_accept;
    logic write_backend_can_accept;
    logic backend_idle_sample;
    logic reset_seen;
    logic sticky_error;
    string config_path;
    string region_manifest_path;
    string image_path;
    string output_directory;
    integer image_file;
    integer image_seek;
    longint image_size;
    longint unsigned read_completion_token;
    longint unsigned write_completion_token;
    bit [255:0] read_completion_data;

    function automatic longint unsigned make_token(
        input logic [SLOT_PTR_WIDTH-1:0] slot, input logic [2:0] beat
    );
        make_token = (longint'(slot) << 8) | longint'(beat);
    endfunction

    function automatic logic [SLOT_PTR_WIDTH-1:0] next_slot(
        input logic [SLOT_PTR_WIDTH-1:0] slot
    );
        if (slot == SLOT_PTR_WIDTH'(SLOT_COUNT - 1))
            next_slot = '0;
        else
            next_slot = slot + 1'b1;
    endfunction

    function automatic logic burst_valid(
        input logic [ADDR_WIDTH-1:0] address,
        input logic [7:0] length,
        input logic [2:0] size,
        input logic [1:0] burst
    );
        logic [ADDR_WIDTH:0] end_address;
        begin
            end_address = {1'b0, address} +
                ((ADDR_WIDTH+1)'({1'b0, length} + 9'd1) << size);
            burst_valid = DATA_WIDTH == 256 && length < 8 &&
                size == AXI_SIZE && burst == 2'b01 &&
                address[$clog2(BEAT_BYTES)-1:0] == '0 &&
                ({1'b0, address} >> 12) ==
                    ((end_address - 1'b1) >> 12);
        end
    endfunction

    logic [ADDR_WIDTH-1:0] next_read_submit_addr;
    logic next_read_backend_ready;
    logic [ADDR_WIDTH-1:0] next_write_addr;
    logic next_write_backend_ready;
    logic [7:0] write_expected_mask;
    logic read_accept, read_return;
    logic write_accept, write_return, write_feed_last;

    always_comb begin : read_submit_select
        next_read_submit_addr = '0;
        next_read_backend_ready = 1'b0;
        if (read_unsubmitted != 0) begin
            next_read_submit_addr = read_addr[read_submit_slot] +
                read_submitted[read_submit_slot] * BEAT_BYTES;
            next_read_backend_ready = !read_error[read_submit_slot] &&
                read_backend_can_accept;
        end
    end

    always_comb begin : write_submit_select
        next_write_addr = '0;
        next_write_backend_ready = 1'b0;
        if (write_unfed != 0) begin
            next_write_addr = write_addr[write_feed_slot] +
                write_submitted[write_feed_slot] * BEAT_BYTES;
            next_write_backend_ready = !write_error[write_feed_slot] &&
                write_backend_can_accept;
        end
    end

    always_comb begin : read_response_select
        s_axi_rid = '0;
        s_axi_rdata = '0;
        s_axi_rresp = 2'b00;
        s_axi_rlast = 1'b0;
        s_axi_rvalid = 1'b0;
        if (read_count != 0) begin
            s_axi_rid = read_id[read_head];
            s_axi_rdata = read_data[read_head][read_return_beat];
            s_axi_rresp = read_error[read_head] ||
                (inject_read_response_error &&
                 {1'b0, read_return_beat} + 4'd2 == read_beats[read_head]) ?
                2'b10 : 2'b00;
            s_axi_rlast = read_return_beat + 1'b1 == read_beats[read_head];
            s_axi_rvalid = read_received[read_head][read_return_beat];
        end
    end

    always_comb begin : write_response_select
        write_expected_mask = '0;
        s_axi_bid = '0;
        s_axi_bresp = 2'b00;
        s_axi_bvalid = 1'b0;
        if (write_count != 0) begin
            write_expected_mask = 8'hff >> (8 - write_beats[write_head]);
            s_axi_bid = write_id[write_head];
            s_axi_bresp = write_error[write_head] ||
                write_injected_error[write_head] ? 2'b10 : 2'b00;
            s_axi_bvalid =
                write_submitted[write_head] == write_beats[write_head] &&
                (write_completed[write_head] & write_expected_mask) ==
                    write_expected_mask;
        end
    end

    assign s_axi_arready = read_count < SLOT_COUNT_WIDTH'(SLOT_COUNT);
    assign s_axi_awready = write_count < SLOT_COUNT_WIDTH'(SLOT_COUNT);
    assign s_axi_wready = write_unfed != 0 &&
        (write_error[write_feed_slot] || next_write_backend_ready);
    assign backend_idle = backend_idle_sample &&
        read_count == 0 && write_count == 0;
    assign protocol_error = sticky_error;
    assign debug_read_slots_used = read_count;
    assign debug_read_bursts_not_submitted = read_unsubmitted;
    assign debug_read_backend_can_accept = read_backend_can_accept;
    assign read_accept = s_axi_arvalid &&
        read_count < SLOT_COUNT_WIDTH'(SLOT_COUNT);
    assign read_return = s_axi_rvalid && s_axi_rready && s_axi_rlast;
    assign write_accept = s_axi_awvalid &&
        write_count < SLOT_COUNT_WIDTH'(SLOT_COUNT);
    assign write_return = s_axi_bvalid && s_axi_bready;
    assign write_feed_last = write_unfed != 0 && s_axi_wvalid && s_axi_wready &&
        write_submitted[write_feed_slot] + 1'b1 ==
        write_beats[write_feed_slot];

    initial begin
        backend_ready = 1'b0;
        initial_image_loaded = 1'b0;
        initial_image_bytes = 64'd0;
        if (!$value$plusargs("DRAMSIM3_CONFIG=%s", config_path) ||
            !$value$plusargs("DRAMSIM3_REGION_MANIFEST=%s", region_manifest_path) ||
            !$value$plusargs("DDR_INITIAL_IMAGE=%s", image_path) ||
            !$value$plusargs("DRAMSIM3_OUTPUT_DIR=%s", output_directory))
            $fatal(1, "DRAMSim3 adapter requires config, region, image and output plusargs");
        image_file = $fopen(image_path, "rb");
        if (image_file == 0) $fatal(1, "DRAMSim3 image open failed path=%s", image_path);
        image_seek = $fseek(image_file, 0, 2);
        image_size = $ftell(image_file);
        $fclose(image_file);
        if (image_seek != 0 || image_size <= 0)
            $fatal(1, "DRAMSim3 image size check failed path=%s", image_path);
`ifdef SUPRA_VCS_DPI
        #1ps;
`endif
        if (supra_dramsim3_create(config_path, region_manifest_path, image_path,
                                 output_directory, CORE_FREQUENCY_MHZ) == 0)
            $fatal(1, "DRAMSim3 backend initialization failed");
`ifdef SUPRA_VCS_DPI
        initial_image_bytes = image_size;
        #1ps;
        initial_image_loaded = 1'b1;
        #1ps;
        backend_ready = 1'b1;
        #1ps;
`else
        initial_image_bytes = image_size;
        initial_image_loaded = 1'b1;
        backend_ready = 1'b1;
`endif
    end

    final begin
        supra_dramsim3_destroy();
    end

    always @(posedge clk) begin : adapter_state
        integer slot;
        integer beat;
        integer poll_result;
        integer read_submit_last_now;
        if (rst) begin
            if (reset_seen == 1'b0 &&
                (read_count != 0 || write_count != 0 ||
                 (backend_ready && supra_dramsim3_idle() == 0)))
                $fatal(1, "DRAMSim3 reset asserted with accepted transaction pending");
            if (reset_seen == 1'b0 && backend_ready &&
                supra_dramsim3_reset() == 0)
                $fatal(1, "DRAMSim3 backend reset failed");
            reset_seen <= 1'b1;
            read_head <= '0;
            read_tail <= '0;
            read_submit_slot <= '0;
            read_count <= '0;
            read_unsubmitted <= '0;
            read_return_beat <= '0;
            write_head <= '0;
            write_tail <= '0;
            write_feed_slot <= '0;
            write_count <= '0;
            write_unfed <= '0;
            max_read_queued <= '0;
            sticky_error <= 1'b0;
            read_backend_can_accept <= 1'b0;
            write_backend_can_accept <= 1'b0;
            backend_idle_sample <= 1'b1;
            for (slot = 0; slot < SLOT_COUNT; slot = slot + 1) begin
                read_received[slot] <= '0;
                read_submitted[slot] <= '0;
                read_error[slot] <= 1'b0;
                write_completed[slot] <= '0;
                write_submitted[slot] <= '0;
                write_error[slot] <= 1'b0;
                write_injected_error[slot] <= 1'b0;
            end
        end else begin
            read_submit_last_now = 0;
            reset_seen <= 1'b0;
            supra_dramsim3_tick_core();
            backend_idle_sample <= supra_dramsim3_idle() != 0;
            if (read_unsubmitted == 0 || read_error[read_submit_slot])
                read_backend_can_accept <= 1'b0;
            else if (!read_backend_can_accept)
                read_backend_can_accept <=
                    supra_dramsim3_can_accept(next_read_submit_addr, 0) != 0;
            if (write_unfed == 0 || write_error[write_feed_slot])
                write_backend_can_accept <= 1'b0;
            else if (!write_backend_can_accept)
                write_backend_can_accept <=
                    supra_dramsim3_can_accept(next_write_addr, 1) != 0;

            poll_result = supra_dramsim3_poll_read(
                read_completion_token, read_completion_data);
            if (poll_result != 0) begin
                slot = int'(read_completion_token[8 +: SLOT_PTR_WIDTH]);
                beat = int'(read_completion_token[2:0]);
                if (slot >= SLOT_COUNT || beat >= 8 ||
                    read_received[slot][beat]) begin
                    sticky_error <= 1'b1;
                    $error("DRAMSim3 invalid or duplicate read completion token=0x%0h",
                           read_completion_token);
                end else begin
                    read_data[slot][beat] <= read_completion_data;
                    read_received[slot][beat] <= 1'b1;
                end
            end
            poll_result = supra_dramsim3_poll_write(write_completion_token);
            if (poll_result != 0) begin
                slot = int'(write_completion_token[8 +: SLOT_PTR_WIDTH]);
                beat = int'(write_completion_token[2:0]);
                if (slot >= SLOT_COUNT || beat >= 8 ||
                    write_completed[slot][beat]) begin
                    sticky_error <= 1'b1;
                    $error("DRAMSim3 invalid or duplicate write completion token=0x%0h",
                           write_completion_token);
                end else begin
                    write_completed[slot][beat] <= 1'b1;
                end
            end

            if (read_accept) begin
                read_addr[read_tail] <= s_axi_araddr;
                read_beats[read_tail] <= {1'b0, s_axi_arlen[2:0]} + 1'b1;
                read_submitted[read_tail] <= '0;
                read_received[read_tail] <= '0;
                read_id[read_tail] <= s_axi_arid;
                read_error[read_tail] <= !burst_valid(
                    s_axi_araddr, s_axi_arlen, s_axi_arsize, s_axi_arburst);
                if (!burst_valid(s_axi_araddr, s_axi_arlen,
                                 s_axi_arsize, s_axi_arburst)) begin
                    read_received[read_tail] <=
                        8'hff >> (7 - s_axi_arlen[2:0]);
                    for (beat = 0; beat < 8; beat = beat + 1)
                        read_data[read_tail][beat] <= '0;
                    sticky_error <= 1'b1;
                end
                read_tail <= next_slot(read_tail);
                if (read_count + 1'b1 > max_read_queued)
                    max_read_queued <= read_count + 1'b1;
            end

            if (read_unsubmitted != 0) begin
                if (read_error[read_submit_slot]) begin
                    read_submitted[read_submit_slot] <=
                        read_beats[read_submit_slot];
                    read_submit_slot <= next_slot(read_submit_slot);
                    read_submit_last_now = 1;
                end else if (next_read_backend_ready) begin
                    if (supra_dramsim3_submit_read(
                            next_read_submit_addr,
                            make_token(read_submit_slot,
                                       read_submitted[read_submit_slot][2:0])) == 0) begin
                        read_backend_can_accept <= 1'b0;
                    end else begin
                        if (read_submitted[read_submit_slot] + 1'b1 ==
                            read_beats[read_submit_slot]) begin
                            read_submit_slot <= next_slot(read_submit_slot);
                            read_backend_can_accept <= 1'b0;
                            read_submit_last_now = 1;
                        end else begin
                            read_backend_can_accept <= supra_dramsim3_can_accept(
                                next_read_submit_addr + BEAT_BYTES, 0) != 0;
                        end
                        read_submitted[read_submit_slot] <=
                            read_submitted[read_submit_slot] + 1'b1;
                    end
                end
            end

            if (s_axi_rvalid && s_axi_rready) begin
                if (s_axi_rlast) begin
                    read_received[read_head] <= '0;
                    read_head <= next_slot(read_head);
                    read_return_beat <= '0;
                end else begin
                    read_return_beat <= read_return_beat + 1'b1;
                end
            end

            if (write_accept) begin
                write_addr[write_tail] <= s_axi_awaddr;
                write_beats[write_tail] <= {1'b0, s_axi_awlen[2:0]} + 1'b1;
                write_submitted[write_tail] <= '0;
                write_completed[write_tail] <= '0;
                write_id[write_tail] <= s_axi_awid;
                write_error[write_tail] <= !burst_valid(
                    s_axi_awaddr, s_axi_awlen, s_axi_awsize, s_axi_awburst);
                write_injected_error[write_tail] <= 1'b0;
                if (!burst_valid(s_axi_awaddr, s_axi_awlen,
                                 s_axi_awsize, s_axi_awburst))
                    sticky_error <= 1'b1;
                write_tail <= next_slot(write_tail);
            end

            if (s_axi_wvalid && s_axi_wready) begin
                if (s_axi_wlast !=
                    (write_submitted[write_feed_slot] + 1'b1 ==
                     write_beats[write_feed_slot])) begin
                    write_error[write_feed_slot] <= 1'b1;
                    sticky_error <= 1'b1;
                end
                if (s_axi_wlast && inject_write_response_error)
                    write_injected_error[write_feed_slot] <= 1'b1;
                if (write_error[write_feed_slot]) begin
                    write_completed[write_feed_slot]
                        [write_submitted[write_feed_slot][2:0]] <= 1'b1;
                end else if (supra_dramsim3_submit_write(
                        next_write_addr,
                        make_token(write_feed_slot,
                                   write_submitted[write_feed_slot][2:0]),
                        s_axi_wdata, s_axi_wstrb) == 0) begin
                    $fatal(1, "DRAMSim3 write submit failed after can-accept");
                end
                if (write_submitted[write_feed_slot] + 1'b1 ==
                    write_beats[write_feed_slot]) begin
                    write_feed_slot <= next_slot(write_feed_slot);
                    write_backend_can_accept <= 1'b0;
                end else begin
                    write_backend_can_accept <= supra_dramsim3_can_accept(
                        next_write_addr + BEAT_BYTES, 1) != 0;
                end
                write_submitted[write_feed_slot] <=
                    write_submitted[write_feed_slot] + 1'b1;
            end

            if (s_axi_bvalid && s_axi_bready) begin
                write_completed[write_head] <= '0;
                write_injected_error[write_head] <= 1'b0;
                write_head <= next_slot(write_head);
            end

            case ({read_accept, read_return})
                2'b10: read_count <= read_count + 1'b1;
                2'b01: read_count <= read_count - 1'b1;
                default: read_count <= read_count;
            endcase
            case ({read_accept, read_submit_last_now != 0})
                2'b10: read_unsubmitted <= read_unsubmitted + 1'b1;
                2'b01: read_unsubmitted <= read_unsubmitted - 1'b1;
                default: read_unsubmitted <= read_unsubmitted;
            endcase
            case ({write_accept, write_return})
                2'b10: write_count <= write_count + 1'b1;
                2'b01: write_count <= write_count - 1'b1;
                default: write_count <= write_count;
            endcase
            case ({write_accept, write_feed_last})
                2'b10: write_unfed <= write_unfed + 1'b1;
                2'b01: write_unfed <= write_unfed - 1'b1;
                default: write_unfed <= write_unfed;
            endcase
        end
    end

    initial begin
        if (SLOT_COUNT < 1 || SLOT_COUNT > 15)
            $fatal(1, "DRAMSim3 adapter requires 1..15 AXI slots");
        for (integer init_slot = 0; init_slot < SLOT_COUNT;
             init_slot = init_slot + 1) begin
            read_addr[init_slot] = '0;
            read_beats[init_slot] = '0;
            read_submitted[init_slot] = '0;
            read_received[init_slot] = '0;
            read_id[init_slot] = '0;
            read_error[init_slot] = 1'b0;
            write_addr[init_slot] = '0;
            write_beats[init_slot] = '0;
            write_submitted[init_slot] = '0;
            write_completed[init_slot] = '0;
            write_id[init_slot] = '0;
            write_error[init_slot] = 1'b0;
            write_injected_error[init_slot] = 1'b0;
            for (integer init_beat = 0; init_beat < 8;
                 init_beat = init_beat + 1)
                read_data[init_slot][init_beat] = '0;
        end
        reset_seen = 1'b0;
        read_head = '0;
        read_tail = '0;
        read_submit_slot = '0;
        read_count = '0;
        read_unsubmitted = '0;
        read_return_beat = '0;
        write_head = '0;
        write_tail = '0;
        write_feed_slot = '0;
        write_count = '0;
        write_unfed = '0;
        max_read_queued = '0;
        sticky_error = 1'b0;
        read_backend_can_accept = 1'b0;
        write_backend_can_accept = 1'b0;
        backend_idle_sample = 1'b1;
    end

endmodule

`default_nettype wire
