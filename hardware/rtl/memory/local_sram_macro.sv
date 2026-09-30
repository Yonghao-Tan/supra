`default_nettype none

// One physical true-dual-port SRAM. Array outputs are registered once here so
// no bank selection or compute logic remains on the macro output timing arc.
module local_sram_macro #(
    parameter integer DEPTH = 1024,
    parameter bit ASSERT_ON_CONFLICT = 1'b0
) (
    input  logic         clk,
    input  logic         rst,
    input  logic         a_req_valid,
    output logic         a_req_ready,
    input  logic         a_write,
    input  logic [11:0]  a_address,
    input  logic [127:0] a_write_data,
    input  logic [127:0] a_write_mask_n,
    output logic         a_read_valid,
    output logic [127:0] a_read_data,
    input  logic         b_req_valid,
    output logic         b_req_ready,
    input  logic         b_write,
    input  logic [11:0]  b_address,
    input  logic [127:0] b_write_data,
    input  logic [127:0] b_write_mask_n,
    output logic         b_read_valid,
    output logic [127:0] b_read_data,
    output logic         same_address_conflict
);
    logic [127:0] macro_qa;
    logic [127:0] macro_qb;
    logic [1:0] a_read_valid_pipe;
    logic [1:0] b_read_valid_pipe;

    assign a_req_ready = !rst;
    assign b_req_ready = !rst;
`ifdef SYNTHESIS
    assign same_address_conflict = 1'b0;
`else
    assign same_address_conflict = a_req_valid && b_req_valid &&
        a_address == b_address && (a_write || b_write);
`endif

    local_sram_array #(.DEPTH(DEPTH)) memory (
        .clk(clk), .a_enable(a_req_valid && a_req_ready),
        .a_write(a_write), .a_address(a_address), .a_write_data(a_write_data),
        .a_write_mask_n(a_write_mask_n), .a_read_data(macro_qa),
        .b_enable(b_req_valid && b_req_ready), .b_write(b_write),
        .b_address(b_address), .b_write_data(b_write_data),
        .b_write_mask_n(b_write_mask_n), .b_read_data(macro_qb));
    always_ff @(posedge clk) begin
        if (rst) begin
            a_read_valid_pipe <= '0;
            b_read_valid_pipe <= '0;
        end else begin
            a_read_valid_pipe <= {a_read_valid_pipe[0],
                a_req_valid && a_req_ready && !a_write};
            b_read_valid_pipe <= {b_read_valid_pipe[0],
                b_req_valid && b_req_ready && !b_write};
            a_read_data <= macro_qa;
            b_read_data <= macro_qb;
        end
    end

    assign a_read_valid = a_read_valid_pipe[1];
    assign b_read_valid = b_read_valid_pipe[1];

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (DEPTH == 2048) begin
                assert (!a_req_valid || a_address[11] == 1'b0)
                    else $error("local_sram_macro A address %0d exceeds depth 2048 write=%0b",
                        a_address, a_write);
                assert (!b_req_valid || b_address[11] == 1'b0)
                    else $error("local_sram_macro B address %0d exceeds depth 2048 write=%0b",
                        b_address, b_write);
            end
            if (DEPTH == 1024) begin
                assert (!a_req_valid || a_address[11:10] == 2'b00)
                    else $error("local_sram_macro A address %0d exceeds depth 1024 write=%0b",
                        a_address, a_write);
                assert (!b_req_valid || b_address[11:10] == 2'b00)
                    else $error("local_sram_macro B address %0d exceeds depth 1024 write=%0b",
                        b_address, b_write);
            end
            if (ASSERT_ON_CONFLICT)
                assert (!same_address_conflict)
                    else $error("local_sram_macro received an illegal same-address access");
        end
    end
`endif
endmodule

`default_nettype wire
