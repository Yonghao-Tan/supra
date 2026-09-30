`default_nettype none

module ready_valid_fifo #(
    parameter integer DATA_WIDTH = 64,
    parameter integer DEPTH = 64,
    parameter integer PTR_WIDTH = (DEPTH <= 2) ? 1 : $clog2(DEPTH),
    parameter bit ALLOW_FULL_POP_PUSH = 1'b1
) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic                  input_valid,
    output logic                  input_ready,
    input  logic [DATA_WIDTH-1:0] input_data,
    output logic                  output_valid,
    input  logic                  output_ready,
    output logic [DATA_WIDTH-1:0] output_data,
    output logic [PTR_WIDTH:0]    occupancy
);
    logic [DATA_WIDTH-1:0] storage [0:DEPTH-1];
    localparam logic [PTR_WIDTH-1:0] LAST_POINTER = PTR_WIDTH'(DEPTH - 1);
    localparam logic [PTR_WIDTH:0] DEPTH_COUNT = (PTR_WIDTH + 1)'(DEPTH);
    logic [PTR_WIDTH-1:0] write_pointer;
    logic [PTR_WIDTH-1:0] read_pointer;
    logic push_accepted;
    logic pop_accepted;

    assign output_valid = occupancy != 0;
    assign input_ready = occupancy < DEPTH_COUNT ||
        (ALLOW_FULL_POP_PUSH && output_valid && output_ready);
    assign output_data = storage[read_pointer];
    assign push_accepted = input_valid && input_ready;
    assign pop_accepted = output_valid && output_ready;

    always_ff @(posedge clk) begin
        if (rst) begin
            write_pointer <= {PTR_WIDTH{1'b0}};
            read_pointer <= {PTR_WIDTH{1'b0}};
            occupancy <= {(PTR_WIDTH+1){1'b0}};
        end else begin
            if (push_accepted) begin
                storage[write_pointer] <= input_data;
                write_pointer <= write_pointer == LAST_POINTER ?
                    {PTR_WIDTH{1'b0}} : write_pointer + 1'b1;
            end
            if (pop_accepted)
                read_pointer <= read_pointer == LAST_POINTER ?
                    {PTR_WIDTH{1'b0}} : read_pointer + 1'b1;
            case ({push_accepted, pop_accepted})
                2'b10: occupancy <= occupancy + 1'b1;
                2'b01: occupancy <= occupancy - 1'b1;
                default: occupancy <= occupancy;
            endcase
        end
    end

    initial begin
        if (DATA_WIDTH < 1 || DEPTH < 2 ||
            PTR_WIDTH != (DEPTH <= 2 ? 1 : $clog2(DEPTH)))
            $error("ready_valid_fifo parameter relation is invalid");
    end

`ifndef SYNTHESIS
    logic stalled_output;
    logic [DATA_WIDTH-1:0] stalled_output_data;
    always_ff @(posedge clk) begin
        if (rst) begin
            stalled_output <= 1'b0;
            stalled_output_data <= {DATA_WIDTH{1'b0}};
        end else begin
            assert (occupancy <= DEPTH_COUNT)
                else $error("ready_valid_fifo occupancy exceeds DEPTH");
            if (stalled_output)
                // Stability includes inactive payload bits; their X values may persist.
                assert (output_valid === 1'b1 && output_data === stalled_output_data)
                    else $error("ready_valid_fifo changed a stalled output");
            stalled_output <= output_valid && !output_ready;
            stalled_output_data <= output_data;
        end
    end
`endif
endmodule

`default_nettype wire
