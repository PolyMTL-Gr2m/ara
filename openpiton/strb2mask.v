`include "define.tmp.h"

module strb2mask (
    input wire clk,
    input wire rst,
    input wire [7:0] m_axi_wstrb,
    output reg [2:0] pmesh_data_size,
    output reg [5:0] pmesh_addr,
    input wire s_channel_valid,
    output wire s_channel_ready,
    input wire d_channel_ready,
    output wire d_channel_valid
);


localparam BASE_1B = 8'b1000_0000;
localparam BASE_2B = 8'b1100_0000;
localparam BASE_4B = 8'b1111_0000;
localparam BASE_8B = 8'b1111_1111;

reg [7:0] source_q;
reg [7:0] source_d;
reg [7:0] input_data;
reg [7:0] target [14:0];
reg [14:0] all_match; 
reg [14:0] part_match;
reg [7:0] output_mask;
reg [7:0] pmesh_mask;
reg [7:0] reverse_source;
reg [7:0] reverse_target [14:0];

wire need_split;
wire split;
wire tx;
wire rx;
reg tx_delay_stage_1;
reg last_split;

integer i;
integer j, k;

// FSM for destination valid control 
localparam NOT_VALID = 0;
localparam VALID = 1;
reg valid_state;
reg valid_state_next;

//FSM for source ready control
reg [1:0] ready_state, next_ready_state;
localparam NOT_READY = 0;
localparam READY = 1;
localparam WAIT = 2;

assign tx = s_channel_ready & s_channel_valid;
assign rx = d_channel_ready & d_channel_valid;

assign split = (((last_split) & (!need_split)) | (need_split));
assign need_split = (|part_match) & ~(|all_match);

always@ (*) begin
    source_d = source_q;
end

always@ (*) begin // for split output 
    reverse_source[0] = source_q[7];
    reverse_source[1] = source_q[6];
    reverse_source[2] = source_q[5];
    reverse_source[3] = source_q[4];
    reverse_source[4] = source_q[3];
    reverse_source[5] = source_q[2];
    reverse_source[6] = source_q[1];
    reverse_source[7] = source_q[0];
    for (i = 0; i < 15; i = i + 1) begin
        reverse_target[i][0] = target[i][7];
        reverse_target[i][1] = target[i][6];
        reverse_target[i][2] = target[i][5];
        reverse_target[i][3] = target[i][4];
        reverse_target[i][4] = target[i][3];
        reverse_target[i][5] = target[i][2];
        reverse_target[i][6] = target[i][1];
        reverse_target[i][7] = target[i][0];
    end
end

always@ (*) begin  // target generate
    target[0] = BASE_8B >> 0; 
    target[1] = BASE_4B >> 4; 
    target[2] = BASE_2B >> 6; 
    target[3] = BASE_1B >> 7; 
    target[4] = BASE_1B >> 6;
    target[5] = BASE_2B >> 4;
    target[6] = BASE_1B >> 5;
    target[7] = BASE_1B >> 4;
    target[8] = BASE_4B >> 0;
    target[9] = BASE_2B >> 2;
    target[10] = BASE_1B >> 3;
    target[11] = BASE_1B >> 2;
    target[12] = BASE_2B >> 0;
    target[13] = BASE_1B >> 1;
    target[14] = BASE_1B >> 0;
    for (j = 0; j < 15; j = j + 1) begin
        all_match[j] = (source_q == target[j]);
    end
end

always@ (*) begin 
    for (k = 0; k < 15; k = k + 1) begin
        part_match[k] = (source_q > target[k]) & (reverse_source > reverse_target[k]);
    end
end

always@ (*) begin
    if ((|all_match)) begin
        input_data = m_axi_wstrb;
    end
    else if ((|part_match))begin
        casex (part_match)
            15'b????_????_????_??1: begin input_data = source_q - target[0]; end
            15'b????_????_????_?1?: begin input_data = source_q - target[1]; end
            15'b????_????_????_1??: begin input_data = source_q - target[2]; end
            15'b????_????_???1_???: begin input_data = source_q - target[3]; end
            15'b????_????_??1?_???: begin input_data = source_q - target[4]; end
            15'b????_????_?1??_???: begin input_data = source_q - target[5]; end
            15'b????_????_1???_???: begin input_data = source_q - target[6]; end
            15'b????_???1_????_???: begin input_data = source_q - target[7]; end
            15'b????_??1?_????_???: begin input_data = source_q - target[8]; end
            15'b????_?1??_????_???: begin input_data = source_q - target[9]; end
            15'b????_1???_????_???: begin input_data = source_q - target[10]; end
            15'b???1_????_????_???: begin input_data = source_q - target[11]; end
            15'b??1?_????_????_???: begin input_data = source_q - target[12]; end
            15'b?1??_????_????_???: begin input_data = source_q - target[13]; end
            15'b1???_????_????_???: begin input_data = source_q - target[14]; end
            default:  begin input_data = source_q; end
        endcase
    end
    else begin
        input_data = source_q;
    end
end

always@ (*) begin
    if ((|all_match)) begin
         casex (all_match)
            15'b????_????_????_??1: output_mask = target[0];
            15'b????_????_????_?1?: output_mask = target[1];
            15'b????_????_????_1??: output_mask = target[2];
            15'b????_????_???1_???: output_mask = target[3];
            15'b????_????_??1?_???: output_mask = target[4];
            15'b????_????_?1??_???: output_mask = target[5];
            15'b????_????_1???_???: output_mask = target[6];
            15'b????_???1_????_???: output_mask = target[7];
            15'b????_??1?_????_???: output_mask = target[8];
            15'b????_?1??_????_???: output_mask = target[9];
            15'b????_1???_????_???: output_mask = target[10];
            15'b???1_????_????_???: output_mask = target[11];
            15'b??1?_????_????_???: output_mask = target[12];
            15'b?1??_????_????_???: output_mask = target[13];
            15'b1???_????_????_???: output_mask = target[14];
            default:  output_mask = BASE_8B;
        endcase
    end
    else if ((|part_match)) begin
        casex (part_match)
            15'b????_????_????_??1: begin output_mask = target[0];end
            15'b????_????_????_?1?: begin output_mask = target[1];end
            15'b????_????_????_1??: begin output_mask = target[2];end
            15'b????_????_???1_???: begin output_mask = target[3];end
            15'b????_????_??1?_???: begin output_mask = target[4];end
            15'b????_????_?1??_???: begin output_mask = target[5];end
            15'b????_????_1???_???: begin output_mask = target[6];end
            15'b????_???1_????_???: begin output_mask = target[7];end
            15'b????_??1?_????_???: begin output_mask = target[8];end
            15'b????_?1??_????_???: begin output_mask = target[9];end
            15'b????_1???_????_???: begin output_mask = target[10];end
            15'b???1_????_????_???: begin output_mask = target[11];end
            15'b??1?_????_????_???: begin output_mask = target[12];end
            15'b?1??_????_????_???: begin output_mask = target[13];end
            15'b1???_????_????_???: begin output_mask = target[14];end
            default:  begin output_mask = BASE_8B; end
        endcase
    end
    else output_mask = pmesh_mask;
end


always@ (*) begin   
    casex (pmesh_mask)
        target[0]: begin pmesh_data_size = `MSG_DATA_SIZE_8B;  pmesh_addr = 0; end
        target[1]: begin pmesh_data_size = `MSG_DATA_SIZE_4B;  pmesh_addr = 0; end
        target[2]: begin pmesh_data_size = `MSG_DATA_SIZE_2B;  pmesh_addr = 0; end
        target[3]: begin pmesh_data_size = `MSG_DATA_SIZE_1B;  pmesh_addr = 0; end
        target[4]: begin pmesh_data_size = `MSG_DATA_SIZE_1B;  pmesh_addr = 1; end
        target[5]: begin pmesh_data_size = `MSG_DATA_SIZE_2B;  pmesh_addr = 2; end
        target[6]: begin pmesh_data_size = `MSG_DATA_SIZE_1B;  pmesh_addr = 2; end
        target[7]: begin pmesh_data_size = `MSG_DATA_SIZE_1B;  pmesh_addr = 3; end
        target[8]: begin pmesh_data_size = `MSG_DATA_SIZE_4B;  pmesh_addr = 4; end
        target[9]: begin pmesh_data_size = `MSG_DATA_SIZE_2B;  pmesh_addr = 4; end
        target[10]: begin pmesh_data_size = `MSG_DATA_SIZE_1B;  pmesh_addr = 4; end
        target[11]: begin pmesh_data_size = `MSG_DATA_SIZE_1B;  pmesh_addr = 5; end
        target[12]: begin pmesh_data_size = `MSG_DATA_SIZE_2B;  pmesh_addr = 6; end
        target[13]: begin pmesh_data_size = `MSG_DATA_SIZE_1B;  pmesh_addr = 6; end
        target[14]: begin pmesh_data_size = `MSG_DATA_SIZE_1B;  pmesh_addr = 7; end
        default: begin  pmesh_data_size = `MSG_DATA_SIZE_8B;  pmesh_addr = 0; end // is this the right default?
    endcase 
end


// FSN for output valid
always@(posedge clk) begin
    if (rst) valid_state <= NOT_VALID;
    else valid_state <= valid_state_next;
end

always@ (*) begin
  if (valid_state == NOT_VALID) begin
      if (tx_delay_stage_1) valid_state_next = VALID;
      else valid_state_next = valid_state;
  end
  else if (valid_state == VALID) begin
      if (tx_delay_stage_1| split) valid_state_next = VALID;
      else if (d_channel_ready) valid_state_next = NOT_VALID;
      else valid_state_next = valid_state;
  end
end

assign d_channel_valid = (valid_state == VALID);


always@(posedge clk) begin
    if (rst)begin
        tx_delay_stage_1 <= 0;
        last_split <= 0;

    end
    else begin
        tx_delay_stage_1 <= tx;
        last_split <= need_split;
    end
end

always@ (posedge clk) begin
    if (rst) ready_state <= NOT_READY;
    else ready_state <= next_ready_state;
end

always@(*) begin
    if (ready_state == NOT_READY) begin
        if (d_channel_ready && ~need_split) next_ready_state = READY;
        else next_ready_state = NOT_READY;
    end
    else if (ready_state == READY) begin
        if (tx) next_ready_state = WAIT;
        else next_ready_state = READY;
    end
    else if (ready_state == WAIT) begin
        if (rx) next_ready_state = NOT_READY;
        else next_ready_state = WAIT;
    end
    else next_ready_state = ready_state;
end

assign s_channel_ready = (ready_state == READY) ? 1'b1 : 1'b0;

always@ (posedge clk) begin 
    if (rst) begin
        pmesh_mask <= BASE_8B;
    end
    else begin
        pmesh_mask <= output_mask;
    end 
end
always@ (posedge clk) begin
    if (rst) source_q <= BASE_8B;
    else if (tx | (need_split & rx)) source_q <= input_data;
    else source_q <= source_d;
end

endmodule 

