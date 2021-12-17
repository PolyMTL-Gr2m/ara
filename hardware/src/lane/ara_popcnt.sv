// Copyright 2018 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

////////////////////////////////////////////////////////////////////////////////
// Engineer:       Andreas Traber - atraber@student.ethz.ch                   //
//                                                                            //
// Additional contributions by:                                               //
//                 Davide Schiavone - pschiavo@iis.ee.ethz.ch                 //
//                 MohammadHossein AskariHemmat-m.h.askari.hemmat@gmail.com   //
//                                                                            //
// Design Name:    ara_popcnt                                                 //
// Project Name:   Ara                                                        //
// Language:       SystemVerilog                                              //
//                                                                            //
// Description:    Count the number of '1's in a word                         //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

module ara_popcnt import ara_pkg::*; (
    input  logic [63:0] in_i,
    output logic [0:0][63:0] result_o_w64,
    output logic [1:0][31:0] result_o_w32,
    output logic [3:0][15:0] result_o_w16,
    output logic [7:0][ 7:0] result_o_w8
  );
  logic [31:0][1:0] cnt_l1;
  logic [15:0][2:0] cnt_l2;
  logic [ 7:0][3:0] cnt_l3;
  logic [ 3:0][4:0] cnt_l4;
  logic [ 1:0][5:0] cnt_l5;
  logic [ 0:0][6:0] cnt_l6;

  genvar l, m, n, p, q;
  generate
    for (l = 0; l < 32; l++) begin : gen_cnt_l1
      assign cnt_l1[l] = {1'b0, in_i[2*l]} + {1'b0, in_i[2*l+1]} ;
    end
  endgenerate

  generate
    for (m = 0; m < 16; m++) begin : gen_cnt_l2
      assign cnt_l2[m] = {1'b0, cnt_l1[2*m]} + {1'b0, cnt_l1[2*m+1]};
    end
  endgenerate

  generate
    for (n = 0; n < 8; n++) begin : gen_cnt_l3
      assign cnt_l3[n] = {1'b0, cnt_l2[2*n]} + {1'b0, cnt_l2[2*n+1]};
      assign result_o_w8[n] = {4'b0, cnt_l3[n]};
    end
  endgenerate

  generate
    for (p = 0; p < 4; p++) begin : gen_cnt_l4
      assign cnt_l4[p] = {1'b0, cnt_l3[2*p]} + {1'b0, cnt_l3[2*p+1]};
      assign result_o_w16[p] = {11'b0, cnt_l4[p]};
    end
  endgenerate

  generate
    for (q = 0; q < 2; q++) begin : gen_cnt_l5
      assign cnt_l5[q] = {1'b0, cnt_l4[2*q]} + {1'b0, cnt_l4[2*q+1]};
      assign result_o_w32[q] = {26'b0, cnt_l5[q]};
    end
  endgenerate
  assign cnt_l6 = {1'b0, cnt_l5[0]} + {1'b0, cnt_l5[1]};
  assign result_o_w64[0] = {57'b0, cnt_l6};

endmodule
