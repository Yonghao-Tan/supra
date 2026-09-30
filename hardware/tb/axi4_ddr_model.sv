`timescale 1ns/1ps
`default_nettype none

module axi4_ddr_model #(
    parameter integer ADDR_WIDTH = 64,
    parameter integer DATA_WIDTH = 128,
    parameter integer ID_WIDTH = 4,
    parameter integer MEMORY_BYTES = 1048576,
    parameter integer BACKING_BYTES = MEMORY_BYTES,
    parameter [ADDR_WIDTH-1:0] BASE_ADDRESS = 64'h0000_0000_1000_0000
) (
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         random_stall,
    input  wire                         performance_config,
    input  wire                         inject_read_id_error,
    input  wire                         inject_read_last_error,
    input  wire                         inject_read_response_error,
    input  wire                         inject_write_id_error,
    input  wire                         inject_write_response_error,
    input  wire [ID_WIDTH-1:0]          s_axi_awid,
    input  wire [ADDR_WIDTH-1:0]        s_axi_awaddr,
    input  wire [7:0]                   s_axi_awlen,
    input  wire [2:0]                   s_axi_awsize,
    input  wire [1:0]                   s_axi_awburst,
    input  wire                         s_axi_awvalid,
    output wire                         s_axi_awready,
    input  wire [DATA_WIDTH-1:0]        s_axi_wdata,
    input  wire [DATA_WIDTH/8-1:0]      s_axi_wstrb,
    input  wire                         s_axi_wlast,
    input  wire                         s_axi_wvalid,
    output wire                         s_axi_wready,
    output reg  [ID_WIDTH-1:0]          s_axi_bid,
    output reg  [1:0]                   s_axi_bresp,
    output reg                          s_axi_bvalid,
    input  wire                         s_axi_bready,
    input  wire [ID_WIDTH-1:0]          s_axi_arid,
    input  wire [ADDR_WIDTH-1:0]        s_axi_araddr,
    input  wire [7:0]                   s_axi_arlen,
    input  wire [2:0]                   s_axi_arsize,
    input  wire [1:0]                   s_axi_arburst,
    input  wire                         s_axi_arvalid,
    output wire                         s_axi_arready,
    output reg  [ID_WIDTH-1:0]          s_axi_rid,
    output reg  [DATA_WIDTH-1:0]        s_axi_rdata,
    output reg  [1:0]                   s_axi_rresp,
    output reg                          s_axi_rlast,
    output reg                          s_axi_rvalid,
    input  wire                         s_axi_rready,
    output reg  [4:0]                   max_read_queued,
    input  wire                         image_check_request,
    output reg                          initial_image_loaded,
    output reg  [63:0]                  initial_image_bytes,
    output reg                          image_check_done,
    output reg                          image_check_pass,
    output reg  [31:0]                  image_check_mismatch_count,
    output reg  [63:0]                  expected_image_bytes
);
    localparam integer BEAT_BYTES = DATA_WIDTH / 8;
    localparam logic [2:0] AXI_SIZE = 3'($clog2(BEAT_BYTES));
`ifdef VCS_SPLIT_DDR_STORAGE
    localparam integer STORAGE_BANK_BYTES = (BACKING_BYTES + 3) / 4;
`else
    reg [7:0] memory [0:BACKING_BYTES-1];
`endif
    reg [ADDR_WIDTH-1:0] read_address [0:7];
    reg [8:0] read_beats [0:7];
    reg [ID_WIDTH-1:0] read_id [0:7];
    reg [5:0] read_delay [0:7];
    reg [3:0] read_head;
    reg [3:0] read_tail;
    reg [4:0] read_count;
    reg write_active;
    reg [ADDR_WIDTH-1:0] write_address;
    reg [8:0] write_beats;
    reg [ID_WIDTH-1:0] write_active_id;
    reg write_response_pending;
    reg [5:0] write_response_delay;
    reg [15:0] lfsr;
    reg [63:0] cycle_count;
    reg [2:0] performance_phase;
    integer byte_index;
    integer memory_index;
    integer queue_index;
    integer image_file;
    integer image_read_bytes;
    integer image_extra_bytes;
    reg [7:0] image_extra_byte;
    string initial_image_path;
    string expected_image_path;
    wire performance_beat_enable = performance_phase != 3'd4;
    wire performance_next_beat_enable = performance_phase != 3'd3;
    wire allow_aw = performance_config || !random_stall || lfsr[0];
    wire allow_w_slot = performance_config ? performance_beat_enable :
        (!random_stall || lfsr[1]);
    wire allow_ar = performance_config || !random_stall || lfsr[2];
    wire write_data_pending = write_active && s_axi_wvalid;
    wire [ADDR_WIDTH:0] aw_end = {1'b0, s_axi_awaddr} + ((s_axi_awlen + 1) << s_axi_awsize);
    wire [ADDR_WIDTH:0] ar_end = {1'b0, s_axi_araddr} + ((s_axi_arlen + 1) << s_axi_arsize);
    wire [ADDR_WIDTH:0] aw_last_address = aw_end - 1'b1;
    wire read_enqueue = s_axi_arvalid && s_axi_arready;
    wire read_dequeue = s_axi_rvalid && s_axi_rready && read_beats[read_head] == 1;

`ifdef VCS_SPLIT_DDR_STORAGE
    wire storage_write_valid = s_axi_wvalid && s_axi_wready && s_axi_bresp == 2'b00;
    wire [31:0] storage_write_index = write_address - BASE_ADDRESS;
    wire [3:0] storage_image_loaded;
    wire [63:0] storage_image_bytes [0:3];

    axi4_ddr_storage_bank #(
        .DATA_WIDTH(DATA_WIDTH), .BANK_BYTES(STORAGE_BANK_BYTES),
        .BANK_FILE_OFFSET(0)
    ) storage_bank_0 (
        .clk, .write_valid(storage_write_valid), .write_index(storage_write_index),
        .write_data(s_axi_wdata), .write_strobe(s_axi_wstrb),
        .image_loaded(storage_image_loaded[0]), .image_bytes(storage_image_bytes[0])
    );
    axi4_ddr_storage_bank #(
        .DATA_WIDTH(DATA_WIDTH), .BANK_BYTES(STORAGE_BANK_BYTES),
        .BANK_FILE_OFFSET(STORAGE_BANK_BYTES)
    ) storage_bank_1 (
        .clk, .write_valid(storage_write_valid), .write_index(storage_write_index),
        .write_data(s_axi_wdata), .write_strobe(s_axi_wstrb),
        .image_loaded(storage_image_loaded[1]), .image_bytes(storage_image_bytes[1])
    );
    axi4_ddr_storage_bank #(
        .DATA_WIDTH(DATA_WIDTH), .BANK_BYTES(STORAGE_BANK_BYTES),
        .BANK_FILE_OFFSET(2 * STORAGE_BANK_BYTES)
    ) storage_bank_2 (
        .clk, .write_valid(storage_write_valid), .write_index(storage_write_index),
        .write_data(s_axi_wdata), .write_strobe(s_axi_wstrb),
        .image_loaded(storage_image_loaded[2]), .image_bytes(storage_image_bytes[2])
    );
    axi4_ddr_storage_bank #(
        .DATA_WIDTH(DATA_WIDTH), .BANK_BYTES(STORAGE_BANK_BYTES),
        .BANK_FILE_OFFSET(3 * STORAGE_BANK_BYTES)
    ) storage_bank_3 (
        .clk, .write_valid(storage_write_valid), .write_index(storage_write_index),
        .write_data(s_axi_wdata), .write_strobe(s_axi_wstrb),
        .image_loaded(storage_image_loaded[3]), .image_bytes(storage_image_bytes[3])
    );

    function automatic [7:0] stored_byte(input integer index);
        integer bank_index;
        integer local_index;
        begin
            bank_index = index / STORAGE_BANK_BYTES;
            local_index = index % STORAGE_BANK_BYTES;
            case (bank_index)
                0: stored_byte = storage_bank_0.peek_byte(local_index);
                1: stored_byte = storage_bank_1.peek_byte(local_index);
                2: stored_byte = storage_bank_2.peek_byte(local_index);
                3: stored_byte = storage_bank_3.peek_byte(local_index);
                default: stored_byte = 8'd0;
            endcase
        end
    endfunction

    function automatic [DATA_WIDTH-1:0] stored_beat(input integer index);
        integer stored_byte_index;
        begin
            stored_beat = '0;
            for (stored_byte_index = 0; stored_byte_index < BEAT_BYTES;
                 stored_byte_index = stored_byte_index + 1)
                stored_beat[stored_byte_index*8 +: 8] =
                    stored_byte(index + stored_byte_index);
        end
    endfunction
`else
    function automatic [7:0] stored_byte(input integer index);
        stored_byte = memory[index];
    endfunction
`endif

    assign s_axi_awready = !write_active && !write_response_pending &&
        !s_axi_bvalid && allow_aw;
    // A held R response already owns its payload register, not the memory
    // service slot. Blocking W until RREADY deadlocks streaming read/modify/
    // write clients when their read sink waits for write acceptance.
    // Keep at most one accepted data beat per cycle in this shared-bus model.
    assign s_axi_wready = write_active && allow_w_slot &&
        !(s_axi_rvalid && s_axi_rready);
    assign s_axi_arready = read_count < 8 && allow_ar;

    initial begin
        if (BACKING_BYTES <= 0 || BACKING_BYTES > MEMORY_BYTES)
            $fatal(1, "DDR backing bytes must be within logical memory capacity");
`ifdef VCS_SPLIT_DDR_STORAGE
`else
        initial_image_loaded = 1'b0;
        initial_image_bytes = 64'd0;
        if ($value$plusargs("DDR_INITIAL_IMAGE=%s", initial_image_path)) begin
            image_file = $fopen(initial_image_path, "rb");
            if (image_file == 0)
                $fatal(1, "DDR initial image open failed path=%s", initial_image_path);
            image_read_bytes = $fread(memory, image_file, 0, BACKING_BYTES);
            image_extra_bytes = $fread(image_extra_byte, image_file);
            if (image_read_bytes <= 0 || image_extra_bytes != 0) begin
                $fclose(image_file);
                $fatal(1,
                    "DDR initial image size invalid path=%s bytes=%0d capacity=%0d overflow=%0d",
                    initial_image_path, image_read_bytes, BACKING_BYTES,
                    image_extra_bytes);
            end
            $fclose(image_file);
            initial_image_bytes = {{32{1'b0}}, image_read_bytes[31:0]};
            initial_image_loaded = 1'b1;
            $display("DDR initial image loaded path=%s bytes=%0d capacity=%0d",
                     initial_image_path, image_read_bytes, BACKING_BYTES);
        end
`endif
        if (!$value$plusargs("DDR_EXPECTED_IMAGE=%s", expected_image_path))
            expected_image_path = "";
    end

`ifdef VCS_SPLIT_DDR_STORAGE
    always_comb begin
        initial_image_loaded = &storage_image_loaded;
        initial_image_bytes = storage_image_bytes[0] + storage_image_bytes[1] +
            storage_image_bytes[2] + storage_image_bytes[3];
    end
`endif

    task automatic check_expected_image;
        integer expected_file;
        integer expected_read_bytes;
        integer expected_index;
        integer printed_mismatches;
        reg [7:0] expected_byte;
        begin
            image_check_done = 1'b0;
            image_check_pass = 1'b0;
            image_check_mismatch_count = 32'd0;
            expected_image_bytes = 64'd0;
            expected_index = 0;
            printed_mismatches = 0;
            if (expected_image_path == "") begin
                image_check_mismatch_count = 32'd1;
                image_check_done = 1'b1;
                $display("DDR expected image check failed: +DDR_EXPECTED_IMAGE is missing");
            end else begin
                expected_file = $fopen(expected_image_path, "rb");
                if (expected_file == 0) begin
                    image_check_mismatch_count = 32'd1;
                    image_check_done = 1'b1;
                    $display("DDR expected image open failed path=%s",
                             expected_image_path);
                end else begin
                    expected_read_bytes = 1;
                    while (expected_read_bytes != 0 && expected_index < BACKING_BYTES) begin
                        expected_read_bytes = $fread(expected_byte, expected_file);
                        if (expected_read_bytes != 0) begin
                            if (stored_byte(expected_index) !== expected_byte) begin
                                image_check_mismatch_count =
                                    image_check_mismatch_count + 1'b1;
                                if (printed_mismatches < 8) begin
                                    $display(
                                        "DDR expected mismatch offset=%0d actual=%02x expected=%02x",
                                        expected_index, stored_byte(expected_index),
                                        expected_byte);
                                    printed_mismatches = printed_mismatches + 1;
                                end
                            end
                            expected_index = expected_index + 1;
                        end
                    end
                    if (expected_index == BACKING_BYTES) begin
                        expected_read_bytes = $fread(expected_byte, expected_file);
                        if (expected_read_bytes != 0)
                            image_check_mismatch_count =
                                image_check_mismatch_count + 1'b1;
                    end
                    $fclose(expected_file);
                    expected_image_bytes = {{32{1'b0}}, expected_index[31:0]};
                    if (initial_image_loaded &&
                        expected_image_bytes != initial_image_bytes)
                        image_check_mismatch_count =
                            image_check_mismatch_count + 1'b1;
                    image_check_pass = image_check_mismatch_count == 0;
                    image_check_done = 1'b1;
                    $display(
                        "DDR expected image checked path=%s bytes=%0d mismatches=%0d pass=%0d",
                        expected_image_path, expected_image_bytes,
                        image_check_mismatch_count, image_check_pass);
                end
            end
        end
    endtask

    // File comparison is a zero-time testbench operation requested after completion.
    always @(posedge clk) begin
        if (rst) begin
            image_check_done = 1'b0;
            image_check_pass = 1'b0;
            image_check_mismatch_count = 32'd0;
            expected_image_bytes = 64'd0;
        end else begin
            image_check_done = 1'b0;
            if (image_check_request === 1'b1)
                check_expected_image;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            read_head <= 0;
            read_tail <= 0;
            read_count <= 0;
            write_active <= 1'b0;
            write_response_pending <= 1'b0;
            write_response_delay <= 4'd0;
            s_axi_bvalid <= 1'b0;
            s_axi_rvalid <= 1'b0;
            s_axi_rlast <= 1'b0;
            lfsr <= 16'h1ace;
            cycle_count <= 0;
            performance_phase <= 0;
            max_read_queued <= 0;
        end else begin
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
            cycle_count <= cycle_count + 1'b1;
            performance_phase <= performance_phase == 3'd4 ?
                3'd0 : performance_phase + 1'b1;
            if (s_axi_awvalid && s_axi_awready) begin
                write_active <= 1'b1;
                write_address <= s_axi_awaddr;
                write_beats <= s_axi_awlen + 1'b1;
                write_active_id <= s_axi_awid;
                s_axi_bresp <= (s_axi_awburst != 2'b01 ||
                    s_axi_awsize != AXI_SIZE ||
                    s_axi_awaddr < BASE_ADDRESS || aw_end > BASE_ADDRESS + MEMORY_BYTES ||
                    aw_end > BASE_ADDRESS + BACKING_BYTES ||
                    s_axi_awaddr[ADDR_WIDTH-1:12] != aw_last_address[ADDR_WIDTH-1:12]) ? 2'b10 : 2'b00;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                memory_index = write_address - BASE_ADDRESS;
                if (s_axi_bresp == 2'b00) begin
`ifndef VCS_SPLIT_DDR_STORAGE
                    for (byte_index = 0; byte_index < BEAT_BYTES; byte_index = byte_index + 1)
                        if (s_axi_wstrb[byte_index]) memory[memory_index + byte_index] <= s_axi_wdata[byte_index*8 +: 8];
`endif
                end
                write_address <= write_address + BEAT_BYTES;
                write_beats <= write_beats - 1'b1;
                if (s_axi_wlast) begin
                    if (write_beats != 1) s_axi_bresp <= 2'b10;
                    if (inject_write_response_error) s_axi_bresp <= 2'b10;
                    write_active <= 1'b0;
                    write_response_pending <= 1'b1;
                    write_response_delay <= random_stall ?
                        {3'b000, lfsr[7:5]} : 6'd0;
                    s_axi_bid <= inject_write_id_error ? write_active_id ^ {{(ID_WIDTH-1){1'b0}}, 1'b1} : write_active_id;
                end
            end
            if (write_response_pending && write_response_delay != 0)
                write_response_delay <= write_response_delay - 1'b1;
            if (write_response_pending && write_response_delay == 0 && !s_axi_bvalid) begin
                write_response_pending <= 1'b0;
                s_axi_bvalid <= 1'b1;
            end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;

            // Each accepted AR matures independently. Responses remain ordered
            // because only read_head may drive R, but queued request latency is
            // allowed to overlap as it would in an outstanding-capable slave.
            for (queue_index = 0; queue_index < 8; queue_index = queue_index + 1)
                if (read_delay[queue_index] != 0)
                    read_delay[queue_index] <= read_delay[queue_index] - 1'b1;
            if (read_enqueue) begin
                read_address[read_tail] <= s_axi_araddr;
                read_beats[read_tail] <= s_axi_arlen + 1'b1;
                read_id[read_tail] <= s_axi_arid;
                read_delay[read_tail] <= random_stall ?
                    {4'b0000, lfsr[4:3]} + 6'd3 : 6'd3;
                read_tail <= (read_tail + 1'b1) & 4'h7;
                if (read_count + 1'b1 > max_read_queued) max_read_queued <= read_count + 1'b1;
            end
            if (!s_axi_rvalid && read_count != 0 && read_delay[read_head] == 0 &&
                !write_data_pending &&
                (!performance_config || performance_next_beat_enable)) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rid <= inject_read_id_error ? read_id[read_head] ^ {{(ID_WIDTH-1){1'b0}}, 1'b1} : read_id[read_head];
                s_axi_rlast <= (read_beats[read_head] == 1) ^ inject_read_last_error;
                if (read_address[read_head] < BASE_ADDRESS ||
                    read_address[read_head] + BEAT_BYTES > BASE_ADDRESS + MEMORY_BYTES) begin
                    s_axi_rresp <= 2'b11;
                    s_axi_rdata <= 0;
                end else begin
                    s_axi_rresp <= 2'b00;
                    if (inject_read_response_error && read_beats[read_head] == 2) s_axi_rresp <= 2'b10;
                    memory_index = read_address[read_head] - BASE_ADDRESS;
                    if (memory_index + BEAT_BYTES > BACKING_BYTES)
                        s_axi_rdata <= 0;
                    else begin
`ifdef VCS_SPLIT_DDR_STORAGE
                        s_axi_rdata <= stored_beat(memory_index);
`else
                        for (byte_index = 0; byte_index < BEAT_BYTES; byte_index = byte_index + 1)
                            s_axi_rdata[byte_index*8 +: 8] <= memory[memory_index + byte_index];
`endif
                    end
                end
            end
            if (s_axi_rvalid && s_axi_rready) begin
                if (read_beats[read_head] == 1) begin
                    s_axi_rvalid <= 1'b0;
                    read_head <= (read_head + 1'b1) & 4'h7;
                end else begin
                    read_address[read_head] <= read_address[read_head] + BEAT_BYTES;
                    read_beats[read_head] <= read_beats[read_head] - 1'b1;
                    // Keep RVALID asserted only when the next cycle is a
                    // configured service slot.
                    if (!write_data_pending &&
                        (!performance_config || performance_next_beat_enable)) begin
                        s_axi_rvalid <= 1'b1;
                        s_axi_rid <= inject_read_id_error ? read_id[read_head] ^
                            {{(ID_WIDTH-1){1'b0}}, 1'b1} : read_id[read_head];
                        s_axi_rlast <= (read_beats[read_head] == 2) ^ inject_read_last_error;
                        if (read_address[read_head] + 2 * BEAT_BYTES > BASE_ADDRESS + MEMORY_BYTES) begin
                            s_axi_rresp <= 2'b11;
                            s_axi_rdata <= 0;
                        end else begin
                            s_axi_rresp <= inject_read_response_error && read_beats[read_head] == 3 ?
                                2'b10 : 2'b00;
                            memory_index = read_address[read_head] + BEAT_BYTES - BASE_ADDRESS;
                            if (memory_index + BEAT_BYTES > BACKING_BYTES)
                                s_axi_rdata <= 0;
                            else begin
`ifdef VCS_SPLIT_DDR_STORAGE
                                s_axi_rdata <= stored_beat(memory_index);
`else
                                for (byte_index = 0; byte_index < BEAT_BYTES; byte_index = byte_index + 1)
                                    s_axi_rdata[byte_index*8 +: 8] <= memory[memory_index + byte_index];
`endif
                            end
                        end
                    end else begin
                        s_axi_rvalid <= 1'b0;
                    end
                end
            end
            case ({read_enqueue, read_dequeue})
                2'b10: read_count <= read_count + 1'b1;
                2'b01: read_count <= read_count - 1'b1;
                default: read_count <= read_count;
            endcase
        end
    end
endmodule

`ifdef VCS_SPLIT_DDR_STORAGE
module axi4_ddr_storage_bank #(
    parameter integer DATA_WIDTH = 128,
    parameter integer BANK_BYTES = 1024,
    parameter integer BANK_FILE_OFFSET = 0
) (
    input  logic                  clk,
    input  logic                  write_valid,
    input  logic [31:0]           write_index,
    input  logic [DATA_WIDTH-1:0] write_data,
    input  logic [DATA_WIDTH/8-1:0] write_strobe,
    output logic                  image_loaded,
    output logic [63:0]           image_bytes
);
    localparam integer BEAT_BYTES = DATA_WIDTH / 8;
    logic [7:0] memory [0:BANK_BYTES-1];
    string image_path;
    integer image_file;
    integer seek_status;
    integer read_bytes;
    integer write_byte_index;
    integer local_write_index;

    initial begin
        image_loaded = 1'b0;
        image_bytes = 64'd0;
        if ($value$plusargs("DDR_INITIAL_IMAGE=%s", image_path)) begin
            image_file = $fopen(image_path, "rb");
            if (image_file == 0)
                $fatal(1, "DDR storage bank image open failed path=%s", image_path);
            seek_status = $fseek(image_file, BANK_FILE_OFFSET, 0);
            if (seek_status != 0)
                $fatal(1, "DDR storage bank image seek failed offset=%0d",
                       BANK_FILE_OFFSET);
            read_bytes = $fread(memory, image_file, 0, BANK_BYTES);
            $fclose(image_file);
            image_bytes = read_bytes;
            image_loaded = 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (write_valid && write_index >= BANK_FILE_OFFSET &&
            write_index + BEAT_BYTES <= BANK_FILE_OFFSET + BANK_BYTES) begin
            local_write_index = write_index - BANK_FILE_OFFSET;
            for (write_byte_index = 0; write_byte_index < BEAT_BYTES;
                 write_byte_index = write_byte_index + 1)
                if (write_strobe[write_byte_index])
                    memory[local_write_index + write_byte_index] <=
                        write_data[write_byte_index*8 +: 8];
        end
    end

    function automatic [7:0] peek_byte(input integer local_index);
        if (local_index >= 0 && local_index < BANK_BYTES)
            peek_byte = memory[local_index];
        else
            peek_byte = 8'd0;
    endfunction
endmodule
`endif

`default_nettype wire
