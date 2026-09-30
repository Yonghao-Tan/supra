`default_nettype none

package bf16_silu_pwl_pkg;
    // Deployment coefficients are fixed at build time and selected by input interval.
    function automatic logic [31:0] segment_coefficients(input logic [3:0] segment);
`ifdef SUPRA_SILU_SEGMENT_0
        case (segment)
            4'd0: segment_coefficients = `SUPRA_SILU_SEGMENT_0;
            4'd1: segment_coefficients = `SUPRA_SILU_SEGMENT_1;
            4'd2: segment_coefficients = `SUPRA_SILU_SEGMENT_2;
            4'd3: segment_coefficients = `SUPRA_SILU_SEGMENT_3;
            4'd4: segment_coefficients = `SUPRA_SILU_SEGMENT_4;
            4'd5: segment_coefficients = `SUPRA_SILU_SEGMENT_5;
            4'd6: segment_coefficients = `SUPRA_SILU_SEGMENT_6;
            4'd7: segment_coefficients = `SUPRA_SILU_SEGMENT_7;
            4'd8: segment_coefficients = `SUPRA_SILU_SEGMENT_8;
            4'd9: segment_coefficients = `SUPRA_SILU_SEGMENT_9;
            4'd10: segment_coefficients = `SUPRA_SILU_SEGMENT_10;
            4'd11: segment_coefficients = `SUPRA_SILU_SEGMENT_11;
            4'd12: segment_coefficients = `SUPRA_SILU_SEGMENT_12;
            4'd13: segment_coefficients = `SUPRA_SILU_SEGMENT_13;
            4'd14: segment_coefficients = `SUPRA_SILU_SEGMENT_14;
            default: segment_coefficients = `SUPRA_SILU_SEGMENT_15;
        endcase
`elsif SUPRA_SILU_MONOTONE
        // Shared deployed calibration: silu_table_monotone.json.
        case (segment)
            4'd0: segment_coefficients = 32'hbd5bbbda;
            4'd1: segment_coefficients = 32'hbe40bcf7;
            4'd2: segment_coefficients = 32'hbeb5bd92;
            4'd3: segment_coefficients = 32'hbedfbdc9;
            4'd4: segment_coefficients = 32'hbec5bd94;
            4'd5: segment_coefficients = 32'hbe883c09;
            4'd6: segment_coefficients = 32'hbdf43e22;
            4'd7: segment_coefficients = 32'h00003ecc;
            4'd8: segment_coefficients = 32'h00003f17;
            4'd9: segment_coefficients = 32'hbde63f56;
            4'd10: segment_coefficients = 32'hbe843f7d;
            4'd11: segment_coefficients = 32'hbec33f89;
            4'd12: segment_coefficients = 32'hbeda3f8c;
            4'd13: segment_coefficients = 32'hbeb33f89;
            4'd14: segment_coefficients = 32'hbe293f83;
            default: segment_coefficients = 32'h00003f80;
        endcase
`else
        case (segment)
            4'd0: segment_coefficients = 32'hbd58bbd6;
            4'd1: segment_coefficients = 32'hbe3bbcef;
            4'd2: segment_coefficients = 32'hbeb5bd92;
            4'd3: segment_coefficients = 32'hbedfbdc9;
            4'd4: segment_coefficients = 32'hbec5bd94;
            4'd5: segment_coefficients = 32'hbe883c09;
            4'd6: segment_coefficients = 32'hbdf43e22;
            4'd7: segment_coefficients = 32'h00003ecc;
            4'd8: segment_coefficients = 32'h00003f17;
            4'd9: segment_coefficients = 32'hbde63f56;
            4'd10: segment_coefficients = 32'hbe843f7d;
            4'd11: segment_coefficients = 32'hbec33f89;
            4'd12: segment_coefficients = 32'hbeda3f8c;
            4'd13: segment_coefficients = 32'hbeb33f89;
            4'd14: segment_coefficients = 32'hbe293f83;
            default: segment_coefficients = 32'ha6fc3f80;
        endcase
`endif
    endfunction
    function automatic logic is_nan(input logic [15:0] value);
        is_nan = value[14:7] == 8'hff && value[6:0] != 0;
    endfunction

    function automatic logic [31:0] coefficients(input logic [15:0] value);
        logic [14:0] magnitude;
        magnitude = value[14:0];
        if (is_nan(value))
            coefficients = segment_coefficients(4'd0);
        else if (magnitude == 15'd0)
            coefficients = segment_coefficients(4'd8);
        else if (value[15]) begin
            if (magnitude >= 15'h4100)
                coefficients = {16'h0000, 16'h0000};
            else if (magnitude > 15'h40c0)
                coefficients = segment_coefficients(4'd0);
            else if (magnitude > 15'h4080)
                coefficients = segment_coefficients(4'd1);
            else if (magnitude > 15'h4040)
                coefficients = segment_coefficients(4'd2);
            else if (magnitude > 15'h4000)
                coefficients = segment_coefficients(4'd3);
            else if (magnitude > 15'h3fc0)
                coefficients = segment_coefficients(4'd4);
            else if (magnitude > 15'h3f80)
                coefficients = segment_coefficients(4'd5);
            else if (magnitude > 15'h3f00)
                coefficients = segment_coefficients(4'd6);
            else
                coefficients = segment_coefficients(4'd7);
        end else begin
            if (magnitude >= 15'h4100)
                coefficients = {16'h0000, 16'h3f80};
            else if (magnitude >= 15'h40c0)
                coefficients = segment_coefficients(4'd15);
            else if (magnitude >= 15'h4080)
                coefficients = segment_coefficients(4'd14);
            else if (magnitude >= 15'h4040)
                coefficients = segment_coefficients(4'd13);
            else if (magnitude >= 15'h4000)
                coefficients = segment_coefficients(4'd12);
            else if (magnitude >= 15'h3fc0)
                coefficients = segment_coefficients(4'd11);
            else if (magnitude >= 15'h3f80)
                coefficients = segment_coefficients(4'd10);
            else if (magnitude >= 15'h3f00)
                coefficients = segment_coefficients(4'd9);
            else
                coefficients = segment_coefficients(4'd8);
        end
    endfunction

    function automatic logic [15:0] input_value(input logic [15:0] value);
        if (!is_nan(value) && value[15] && value[14:0] >= 15'h4100)
            input_value = 16'h0000;
        else
            input_value = value;
    endfunction
endpackage

`default_nettype wire
