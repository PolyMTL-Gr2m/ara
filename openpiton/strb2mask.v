`include "define.tmp.h"

module strb2mask (
    input  wire    clk,
    input  wire    rst_n,
    input  wire [63:0]   i_addr, 
    input  wire enable,
    input  wire [7:0]    new_strb,
    output wire done,
    output wire [2:0]    size,
    output wire [63:0]    o_addr    
);

    localparam BASE_1B = 8'b1000_0000;
    localparam BASE_2B = 8'b1100_0000;
    localparam BASE_4B = 8'b1111_0000;
    localparam BASE_8B = 8'b1111_1111;

    reg [7:0] q;
    wire [7:0] remain;
  
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        q <= 0;
      end 
      else if (enable && done) begin 
        q <= new_strb;
      end
      else if (enable && !done) begin
        q <= remain;
      end 
      else begin 
        q <= q;
      end 
    end
    
    always@(*) begin
      if (q & (BASE_8B >> 0) == (BASE_8B >> 0)) begin
        remain = q ^ (BASE_8B >> 0);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_8B;
        o_addr = i_addr;
      end
      else if (q & (BASE_4B >> 4) == (BASE_4B >> 4)) begin
        remain = q ^ (BASE_4B >> 4);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_4B;
        o_addr = i_addr;
      end
      else if (q & (BASE_2B >> 6) == (BASE_2B >> 6)) begin
        remain = q ^ (BASE_2B >> 6);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_2B;
        o_addr = i_addr;
      end
      else if (q & (BASE_1B >> 7) == (BASE_1B >> 7)) begin
        remain = q ^ (BASE_1B >> 7);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_1B;
        o_addr = i_addr + 64'd0;
      end
      else if (q & (BASE_1B >> 6) == (BASE_1B >> 6)) begin 
        remain = q ^ (BASE_1B >> 6);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_1B;
        o_addr = i_addr + 64'd1;
      end
      else if (q & (BASE_2B >> 4) == (BASE_2B >> 4)) begin
        remain = q ^ (BASE_2B >> 4);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_2B;
        o_addr = i_addr + 64'd2;
      end
      else if (q & BASE_1B >> 5 == (BASE_1B >> 5)) begin
        remain = q ^ (BASE_1B >> 5);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_1B;
        o_addr = i_addr + 64'd2;
      end
      else if (q & (BASE_1B >> 4) == (BASE_1B >> 4)) begin
        remain = q ^ (BASE_1B >> 4);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_1B;
        o_addr = i_addr + 64'd3;
      end
      else if (q & (BASE_4B >> 0) == (BASE_4B >> 0)) begin
        remain = q ^ (BASE_4B >> 0);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_4B;
        o_addr = i_addr + 64'd4;
      end 
      else if (q & (BASE_2B >> 2) == (BASE_2B >> 2)) begin
        remain = q ^ (BASE_2B >> 2);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_2B;
        o_addr = i_addr + 64'd4;
      end
      else if (q & (BASE_1B >> 3) == (BASE_1B >> 3)) begin
        remain = q ^ (BASE_1B >> 3);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_1B;
        o_addr = i_addr + 64'd4;
      end
      else if (q & (BASE_1B >> 2) == (BASE_1B >> 2)) begin
        remain = q ^ (BASE_1B >> 2);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_1B;
        o_addr = i_addr + 64'd5;
      end
      else if (q & (BASE_2B >> 0) == (BASE_2B >> 0)) begin
        remain = q ^ (BASE_2B >> 0);
        done = (remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_2B;
        o_addr = i_addr + 64'd6;
      end
      else if (q & (BASE_1B >> 1) == (BASE_1B >> 1)) begin
        remain = q ^ (BASE_1B >> 1);
        done = ( remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_1B;
        o_addr = i_addr + 64'd6;
      end
      else if (q & (BASE_1B >> 0) == (BASE_1B >> 0)) begin
        remain = q ^ (BASE_1B >> 0);
        done = ( remain == 8'b0) ? 1 :0;
        size = `MSG_DATA_SIZE_1B;
        o_addr = i_addr + 64'd7;
      end
      else begin 
        done = 1;
        size = `MSG_DATA_SIZE_0B;
        o_addr = i_addr;
      end
    end
    
    
  endmodule