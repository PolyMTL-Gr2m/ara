// Copyright (c) 2019 multiple authors
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the authors nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE AUTHORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
module axi_noc_bridge #(
    parameter AXI_DATA_WIDTH = 64, 
    parameter AXI_ADDR_WIDTH = 64
)
(

    input logic                                             clk, 
    input logic                                             rst_n,

    output logic                                            noc_valid_out,
    output logic [`NOC_DATA_WIDTH-1:0]                      noc_data_out,
    input  logic                                            noc_ready_in,
   
    input  logic                                            noc_valid_in,
    input  logic [`NOC_DATA_WIDTH-1:0]                      noc_data_in,
    output logic                                            noc_ready_out,

    input  logic [`MSG_SRC_CHIPID_WIDTH-1:0]                src_chipid,
    input  logic [`MSG_SRC_X_WIDTH-1:0]                     src_xpos,
    input  logic [`MSG_SRC_Y_WIDTH-1:0]                     src_ypos,
    input  logic [`MSG_SRC_FBITS_WIDTH-1:0]                 src_fbits,

    input  logic [`MSG_DST_CHIPID_WIDTH-1:0]                dest_chipid,
    input  logic [`MSG_DST_X_WIDTH-1:0]                     dest_xpos,
    input  logic [`MSG_DST_Y_WIDTH-1:0]                     dest_ypos,
    input  logic [`MSG_DST_FBITS_WIDTH-1:0]                 dest_fbits,

  `ifndef ARA_REQ2MEM
    input  logic [`HOME_ID_WIDTH-1:0]                       system_tile_count,
    input  logic [`HOME_ALLOC_METHOD_WIDTH-1:0]             home_alloc_method,
  `endif 

    // write address channel
    input logic [AXI_ADDR_WIDTH - 1:0]                      m_axi_awaddr, 
    input logic [7:0]                                       m_axi_awlen, 
    input logic [2:0]                                       m_axi_awsize, 
    input logic [1:0]                                       m_axi_awburst, 
    input logic [3:0]                                       m_axi_awcache,
    input logic [4:0]                                       m_axi_awid,

    // handshake logic 
    input logic                                             m_axi_awvalid, 
    output logic                                            m_axi_awready,

    //write data channel
    input logic [AXI_DATA_WIDTH - 1:0]                      m_axi_wdata,
    input logic [(AXI_DATA_WIDTH/8) -1 :0]                  m_axi_wstrb,
    input logic                                             m_axi_wlast, 
    // handshake logic 
    input logic                                             m_axi_wvalid, 
    output logic                                            m_axi_wready,

    // write response channel
    output logic [1:0]                                      m_axi_bresp,
    output logic [4:0]                                      m_axi_bid,
    input logic                                             m_axi_bready, 
    output logic                                            m_axi_bvalid,

    // read address channel 
    input logic [AXI_ADDR_WIDTH - 1:0]                      m_axi_araddr, 
    input logic [7:0]                                       m_axi_arlen, 
    input logic [2:0]                                       m_axi_arsize, 
    input logic [1:0]                                       m_axi_arburst, 
    input logic [3:0]                                       m_axi_arcache,
    input logic [4:0]                                       m_axi_arid,
    // handshake logic 
    input logic                                             m_axi_arvalid, 
    output logic                                            m_axi_arready, 

    // read data channel 
    output logic [AXI_DATA_WIDTH -1:0]                      m_axi_rdata, 
    output logic [1:0]                                      m_axi_rresp, 
    output logic                                            m_axi_rlast, 
    output logic                                            m_axi_ruser, 
    output logic [4:0]                                      m_axi_rid, 

    // handshake logic 
    output logic                                            m_axi_rvalid, 
    input logic                                             m_axi_rready
 );


localparam MSG_TYPE_INVAL          = 2'd0; // Invalid Message
localparam MSG_TYPE_LOAD           = 2'd1; // Load Request
localparam MSG_TYPE_STORE          = 2'd2; // Store Request

localparam MIN_NOC_DATA_WIDTH      = 64; // 8 Bytes
localparam NOC_HDR_LEN             = 3;

localparam NOC_PAYLOAD_LEN = (AXI_DATA_WIDTH < MIN_NOC_DATA_WIDTH) ?
                        3'b1 : AXI_DATA_WIDTH / MIN_NOC_DATA_WIDTH;
localparam NULL_PAYLOAD_LEN = 2; // we write 16 bytes null data for L2 version load request

localparam AX_SIZE_1B = 0;
localparam AX_SIZE_2B = 1;
localparam AX_SIZE_4B = 2;
localparam AX_SIZE_8B = 3;
localparam AX_SIZE_16B = 4;


typedef enum logic [1:0] {
    MSG_STATE_IDLE = 2'd0,

  `ifdef ARA_REQ2MEM
    MSG_STATE_WAIT_STRB = 2'd1,
    MSG_STATE_HEADER = 2'd2,
    MSG_STATE_NOC_DATA = 2'd3
  `else 
    MSG_STATE_DEST_CAL = 2'd1,
    MSG_STATE_HEADER = 2'd2,
    MSG_STATE_NOC_DATA = 2'd3
  `endif
} flit_state;        // state for flit output

flit_state flit_state_f, flit_state_next;

`ifdef ARA_REQ2MEM
    typedef enum logic [1:0] {
        IDLE = 2'd0,
        VALID = 2'd1,
        WAIT_ = 2'd2
    } wstrb_fifo_state;  // state for strobe conversion 

    wstrb_fifo_state wstrb_fifo_state_f, wstrb_fifo_state_next;
`endif 

    /* flit fields */
logic [`NOC_DATA_WIDTH-1:0]               msg_address;
logic [`MSG_LENGTH_WIDTH-1:0]             msg_length;
logic [`MSG_TYPE_WIDTH-1:0]               msg_type;
logic [`MSG_MSHRID_WIDTH-1:0]             msg_mshrid;
logic [`MSG_DATA_SIZE_WIDTH-1:0]          msg_data_size;
logic [`MSG_OPTIONS_1]                    msg_options_1;
logic [`MSG_OPTIONS_2_]                   msg_options_2;
logic [`MSG_OPTIONS_3_]                   msg_options_3;

logic                                     axi2noc_msg_type_store;
logic                                     axi2noc_msg_type_load;
logic                                     flit_ready;
logic [`NOC_DATA_WIDTH-1:0]               flit;
logic [`NOC_DATA_WIDTH-1:0]               noc_data;

logic                                     type_fifo_wval;
logic                                     type_fifo_full;
logic [1:0]                               type_fifo_wdata;
logic                                     type_fifo_empty;
logic [1:0]                               type_fifo_out;
logic                                     type_fifo_ren;

logic                                     awaddr_fifo_wval;
logic                                     awaddr_fifo_full;
logic [AXI_ADDR_WIDTH-1:0]                awaddr_fifo_wdata;
logic                                     awaddr_fifo_empty;
logic [AXI_ADDR_WIDTH-1:0]                awaddr_fifo_out;
logic                                     awaddr_fifo_ren;
logic [AXI_ADDR_WIDTH-1:0]                awaddr_buffer_q, awaddr_buffer_d;

logic                                     wdata_fifo_wval;
logic                                     wdata_fifo_full;
logic [AXI_DATA_WIDTH-1:0]                wdata_fifo_wdata;
logic                                     wdata_fifo_empty;
logic [AXI_DATA_WIDTH-1:0]                wdata_fifo_out_buffer; // for endian conversion 
logic [AXI_DATA_WIDTH-1:0]                wdata_fifo_out;
logic                                     wdata_fifo_ren;


logic                                     araddr_fifo_wval;
logic                                     araddr_fifo_full;
logic [AXI_ADDR_WIDTH-1:0]                araddr_fifo_wdata;
logic                                     araddr_fifo_empty;
logic [AXI_ADDR_WIDTH-1:0]                araddr_fifo_out;
logic                                     araddr_fifo_ren;
logic [AXI_ADDR_WIDTH-1:0]                araddr_buffer_q, araddr_buffer_d;

logic [AXI_DATA_WIDTH/8 - 1: 0]           wstrb_fifo_out;
logic [AXI_DATA_WIDTH/8 - 1: 0]           wstrb_fifo_wdata; 
logic                                     wstrb_fifo_full;
logic                                     wstrb_fifo_empty;
logic                                     wstrb_fifo_ren;
logic                                     wstrb_fifo_wval;
logic [AXI_DATA_WIDTH/8 - 1: 0]           wstrb_fifo_mux_out;

logic [7:0]                               awlen_fifo_out;
logic [7:0]                               awlen_fifo_wdata;
logic                                     awlen_fifo_empty;
logic                                     awlen_fifo_full;
logic                                     awlen_fifo_wval;
logic                                     awlen_fifo_ren;

logic                                     [8:0] awlen_buffer_q;
logic                                     [8:0] awlen_buffer_d; 
logic                                     need_split_w_transaction;
logic                                     last_write_transfer;
logic                                     waddr_aligned_with_16B;
logic                                     write_word_select;

logic [2:0]                               awsize_fifo_out;
logic [2:0]                               awsize_fifo_wdata;
logic                                     awsize_fifo_empty;
logic                                     awsize_fifo_full;
logic                                     awsize_fifo_wval;
logic                                     awsize_fifo_ren;

logic                                     [3:0] awsize_buffer_q;
logic                                     [3:0] awsize_buffer_d; 

logic [7:0]                               arlen_fifo_out;
logic [7:0]                               arlen_fifo_wdata;
logic                                     arlen_fifo_empty;
logic                                     arlen_fifo_full;
logic                                     arlen_fifo_wval;
logic                                     arlen_fifo_ren;

logic                                     [8:0] arlen_buffer_q;
logic                                     [8:0] arlen_buffer_d; 
logic                                     need_split_r_transaction;
logic                                     last_read_transfer;
logic                                     raddr_aligned_with_16B;
logic                                     read_word_select;
logic                                     read_size; 

logic [2:0]                               arsize_fifo_out;
logic [2:0]                               arsize_fifo_wdata;
logic                                     arsize_fifo_empty;
logic                                     arsize_fifo_full;
logic                                     arsize_fifo_wval;
logic                                     arsize_fifo_ren;

logic                                     [3:0] arsize_buffer_q;
logic                                     [3:0] arsize_buffer_d; 

logic                                     fifo_has_packet;
logic                                     noc_store_done;
logic                                     noc_load_done;
logic [MIN_NOC_DATA_WIDTH-1:0]            out_data [0:NOC_PAYLOAD_LEN-1];
logic                                     noc_last_header;
logic                                     noc_last_data;
logic [2:0]                               noc_cnt;
logic                                     fifo_rst;

// deal with simultaneous read write operation
logic                                     simu_wr_detected; 
logic                                     read_buffer_full; 

logic outstanding_load_req_d;
logic outstanding_load_req_q;
logic outstanding_load_req;

logic outstanding_store_req_d;
logic outstanding_store_req_q;
logic outstanding_store_req;

`ifdef ARA_REQ2MEM
    logic [`MSG_DATA_SIZE_WIDTH - 1:0]        pmesh_data_size;
    logic [$clog2(AXI_ADDR_WIDTH) - 1:0]      pmesh_addr;
    logic [`MSG_DATA_SIZE_WIDTH:0]            buf_pmesh_data_size;
    logic [$clog2(AXI_ADDR_WIDTH) - 1:0]      buf_pmesh_addr;
    logic                                     wstrb_fifoside_valid;
    logic                                     wstrb_fifoside_ready;
    logic                                     wstrb_outputside_valid;
    logic                                     wstrb_outputside_ready;
`else 
    logic                                     cal_dest_stage0;
    logic                                     cal_dest_stage1; 
    logic                                     cal_dest_stage2 ;
    logic [`HOME_ID_WIDTH-1:0]                lhid_s0;
    logic [`HOME_ID_WIDTH-1:0]                lhid_s1;
    logic [`HOME_ID_WIDTH-1:0]                home_addr_bits_s0;
    logic [`PHY_ADDR_WIDTH-1:0]               axi2noc_req_address_s0;
    logic                                     special_l2_addr_s0;
    logic [`NOC_X_WIDTH-1:0]                  lhid_s1_x;
    logic [`NOC_Y_WIDTH-1:0]                  lhid_s1_y;
`endif

    logic previous_trans_complete;
    logic L2_request_ack;

/******** Where the magic happens ********/
    noc_response_axi #(
        `ifdef ARA_REQ2MEM
          .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
          .AXI_RESP_WIDTH(2)
        `else 
          .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
          .AXI_RESP_WIDTH(2),
          .MSG_TYPE_INVAL(MSG_TYPE_INVAL), 
          .MSG_TYPE_STORE(MSG_TYPE_STORE), 
          .MSG_TYPE_LOAD(MSG_TYPE_LOAD)
        `endif
      ) noc_response_axi(
          .clk(clk),
          .rst_n(rst_n),
          .noc_valid_in(noc_valid_in),
          .noc_data_in(noc_data_in),
          .noc_ready_out(noc_ready_out),
        `ifndef ARA_REQ2MEM 
          .transaction_type_wr_data({last_write_transfer, last_read_transfer, read_size, read_word_select, type_fifo_out}), 
          .transaction_type_wr(noc_load_done || noc_store_done),
        `endif
          .m_axi_rdata(m_axi_rdata),
          .m_axi_rresp(m_axi_rresp),
          .m_axi_rlast(m_axi_rlast),
          .m_axi_rvalid(m_axi_rvalid),
          .m_axi_rready(m_axi_rready),
          .m_axi_bresp(m_axi_bresp),
          .m_axi_bvalid(m_axi_bvalid),
          .m_axi_bready(m_axi_bready), 
          .previous_trans_complete (previous_trans_complete),
          .L2_request_ack (L2_request_ack) 
      );

assign m_axi_ruser = 0;
assign m_axi_rid = 5'b0;
assign m_axi_bid = 5'b0;

assign fifo_rst = !rst_n;
/**************************************************************************/
/*control signal of store buffer, which is for simultaneous read and write*/
/**************************************************************************/  
assign simu_wr_detected = awaddr_fifo_wval && araddr_fifo_wval && !read_buffer_full;

always@(posedge clk or negedge rst_n) begin 
    if (!rst_n) begin 
        read_buffer_full <= 0;
    end 
    else if (simu_wr_detected) begin
        read_buffer_full <= 1;
    end 
    else if (type_fifo_wval) begin 
        read_buffer_full <= 0;
    end 
    else begin 
        read_buffer_full <= read_buffer_full;
    end
end 
/****************************************************************************/

assign write_channel_ready = !awaddr_fifo_full && !wdata_fifo_full && !wstrb_fifo_full;
// assign m_axi_awready = !awaddr_fifo_full && !type_fifo_full && !read_buffer_full && previous_trans_complete;
// assign m_axi_wready = !wdata_fifo_full && !wstrb_fifo_full && !type_fifo_full && previous_trans_complete;
// assign m_axi_arready = !araddr_fifo_full && !type_fifo_full && !read_buffer_full && previous_trans_complete;
assign m_axi_awready = !awaddr_fifo_full && !type_fifo_full && !read_buffer_full;
assign m_axi_wready = !wdata_fifo_full && !wstrb_fifo_full && !type_fifo_full;
assign m_axi_arready = !araddr_fifo_full && !type_fifo_full && !read_buffer_full;
assign axi2noc_msg_type_store = (m_axi_awvalid && m_axi_awready) && (!read_buffer_full); //give priority to load request if read buffer has something
assign axi2noc_msg_type_load = (m_axi_arvalid && m_axi_arready) || read_buffer_full;

/* fifo for storing packet type */
sync_fifo #(
    .DSIZE(2),
    .ASIZE(5),
    .MEMSIZE(16) // should be 2 ^ (ASIZE-1)
) type_fifo (
    .rdata(type_fifo_out),
    .empty(type_fifo_empty),
    .clk(clk),
    .ren(type_fifo_ren),
    .wdata(type_fifo_wdata),
    .full(type_fifo_full),
    .wval(type_fifo_wval),
    .reset(fifo_rst)
);

assign type_fifo_wval = (axi2noc_msg_type_store || axi2noc_msg_type_load) && !type_fifo_full;
assign type_fifo_ren = ((outstanding_load_req && ~outstanding_load_req_d) || (outstanding_store_req && ~outstanding_store_req_d)) && !type_fifo_empty && (last_read_transfer || last_write_transfer);
assign type_fifo_wdata = (axi2noc_msg_type_store) ? MSG_TYPE_STORE :
                            (axi2noc_msg_type_load) ? MSG_TYPE_LOAD : MSG_TYPE_INVAL;

sync_fifo #(
    .DSIZE(AXI_ADDR_WIDTH),
    .ASIZE(5),
    .MEMSIZE(16) // should be 2 ^ (ASIZE-1)
) awaddr_fifo (
    .rdata(awaddr_fifo_out),
    .empty(awaddr_fifo_empty),
    .clk(clk),
    .ren(awaddr_fifo_ren),
    .wdata(awaddr_fifo_wdata),
    .full(awaddr_fifo_full),
    .wval(awaddr_fifo_wval),
    .reset(fifo_rst)
);

assign awaddr_fifo_wval = m_axi_awvalid && m_axi_awready; 
assign awaddr_fifo_wdata = m_axi_awaddr;
assign waddr_aligned_with_16B = (type_fifo_out == MSG_TYPE_STORE) ? (~awaddr_buffer_q[3] && ~awaddr_buffer_q[2] && ~awaddr_buffer_q[1] && ~awaddr_buffer_q[0]) &&  (awsize_buffer_q == AX_SIZE_8B): 0;

if (AXI_DATA_WIDTH == MIN_NOC_DATA_WIDTH) begin 
    assign awaddr_fifo_ren = ((outstanding_store_req && ~outstanding_store_req_d) && !awaddr_fifo_empty && ((awlen_buffer_q <= 2 && waddr_aligned_with_16B) || (awlen_buffer_q == 1)));
    assign write_word_select = (awaddr_buffer_q[3]) ? 1 : 0; 

    always_comb begin 
        if (flit_state_f == MSG_STATE_IDLE && !need_split_w_transaction && type_fifo_out == MSG_TYPE_STORE) awaddr_buffer_d = awaddr_fifo_out;
        else if ((outstanding_store_req && ~outstanding_store_req_d) && need_split_w_transaction) awaddr_buffer_d =  (awlen_buffer_q >= 2 && waddr_aligned_with_16B ) ? (awaddr_buffer_q + (1 << (awsize_buffer_q + 1))) : (awaddr_buffer_q + (1 << (awsize_buffer_q)));
        else awaddr_buffer_d = awaddr_buffer_q; 
    end 
end
else begin 
    assign awaddr_fifo_ren = ((outstanding_store_req && ~outstanding_store_req_d) && !awaddr_fifo_empty && (awlen_buffer_q == 1));
    assign write_word_select = 0; // no meaning for 128 bits

    always_comb begin 
        if (flit_state_f == MSG_STATE_IDLE && !need_split_w_transaction && type_fifo_out == MSG_TYPE_STORE) awaddr_buffer_d = awaddr_fifo_out;
        else if ((outstanding_store_req && ~outstanding_store_req_d) && need_split_w_transaction) awaddr_buffer_d =  awaddr_buffer_q + (1 << (awsize_buffer_q));
        else awaddr_buffer_d = awaddr_buffer_q; 
    end 

end 

always_ff@(posedge clk or negedge rst_n) begin 
    if (!rst_n) 
        awaddr_buffer_q <= 0;
    else 
        awaddr_buffer_q <= awaddr_buffer_d;
end 

/* fifo for wdata */
sync_fifo #(
        .DSIZE(AXI_DATA_WIDTH),
        .ASIZE(5),
        .MEMSIZE(16) // should be 2 ^ (ASIZE-1)    
) waddr_fifo (
        .rdata(wdata_fifo_out_buffer),
        .empty(wdata_fifo_empty),
        .clk(clk),
        .ren(wdata_fifo_ren),
        .wdata(wdata_fifo_wdata),
        .full(wdata_fifo_full),
        .wval(wdata_fifo_wval),
        .reset(fifo_rst)
);

assign wdata_fifo_out = {<<8{wdata_fifo_out_buffer}};
// assign wdata_fifo_out = {wdata_fifo_out_buffer[7:0], wdata_fifo_out_buffer[15:8], wdata_fifo_out_buffer[23:16], wdata_fifo_out_buffer[31:24], 
//                             wdata_fifo_out_buffer[39:32], wdata_fifo_out_buffer[47:40], wdata_fifo_out_buffer[55:48], wdata_fifo_out_buffer[63:56]};
assign wdata_fifo_wval = m_axi_wvalid && m_axi_wready;
assign wdata_fifo_wdata = m_axi_wdata;

if (AXI_DATA_WIDTH == MIN_NOC_DATA_WIDTH)
    always_comb begin 
        if (flit_state_f == MSG_STATE_NOC_DATA && noc_ready_in && !wdata_fifo_empty) begin 
            if (waddr_aligned_with_16B && (awlen_buffer_q >= 2)) begin
                wdata_fifo_ren = (noc_cnt == 0) || noc_store_done;
            end 
            else begin
                wdata_fifo_ren = noc_store_done;
            end 
        end 
        else wdata_fifo_ren = 0;
    end 
else begin 
    always_comb begin 
        wdata_fifo_ren = noc_store_done; 
    end 
end

/* fifo for wlen */
sync_fifo #(
        .DSIZE(8),
        .ASIZE(5),
        .MEMSIZE(16) // should be 2 ^ (ASIZE-1)
) wlen_fifo (
        .rdata(awlen_fifo_out),
        .empty(awlen_fifo_empty),
        .clk(clk),
        .ren(awlen_fifo_ren),
        .wdata(awlen_fifo_wdata),
        .full(awlen_fifo_full),
        .wval(awlen_fifo_wval),
        .reset(fifo_rst)
);

assign awlen_fifo_wval = m_axi_awvalid && m_axi_awready;
assign awlen_fifo_wdata = m_axi_awlen;
assign need_split_w_transaction = (awlen_buffer_q > 0);
assign awlen_fifo_ren = ((outstanding_store_req && ~outstanding_store_req_d) && !awlen_fifo_empty && last_write_transfer);

if (AXI_DATA_WIDTH == MIN_NOC_DATA_WIDTH) begin 
    assign last_write_transfer = (awlen_buffer_q == 1) || (awlen_buffer_q == 2 && waddr_aligned_with_16B);
    always_comb begin 
        if ((flit_state_f == MSG_STATE_IDLE) && (awlen_buffer_q == 0) && fifo_has_packet && (type_fifo_out == MSG_TYPE_STORE)) awlen_buffer_d = awlen_fifo_out + 1; // no beat left, next transaction
        else if (outstanding_store_req && ~outstanding_store_req_d) awlen_buffer_d = (awlen_buffer_q >= 2 && waddr_aligned_with_16B) ? (awlen_buffer_q - 2) : (awlen_buffer_q - 1);
        else awlen_buffer_d = awlen_buffer_q;
    end 
end 

else begin  
    assign last_write_transfer = (awlen_buffer_q == 1);
    always_comb begin 
        if ((flit_state_f == MSG_STATE_IDLE) && (awlen_buffer_q == 0) && fifo_has_packet && (type_fifo_out == MSG_TYPE_STORE)) awlen_buffer_d = awlen_fifo_out + 1; // no beat left, next transaction
        else if (outstanding_store_req && ~outstanding_store_req_d) awlen_buffer_d = (awlen_buffer_q - 1);
        else awlen_buffer_d = awlen_buffer_q;
    end 
end 

always_ff @(posedge clk or negedge rst_n) begin 
    if (!rst_n) awlen_buffer_q <= 0;
    else awlen_buffer_q <= awlen_buffer_d;
end 

/**fifo for wsize**/
sync_fifo #(
        .DSIZE(8),
        .ASIZE(5),
        .MEMSIZE(16) // should be 2 ^ (ASIZE-1)
) awsize_fifo (
        .rdata(awsize_fifo_out),
        .empty(awsize_fifo_empty),
        .clk(clk),
        .ren(awsize_fifo_ren),
        .wdata(awsize_fifo_wdata),
        .full(awsize_fifo_full),
        .wval(awsize_fifo_wval),
        .reset(fifo_rst)
);

assign awsize_fifo_wval = m_axi_awvalid && m_axi_awready;
assign awsize_fifo_wdata = m_axi_awsize;
assign awsize_fifo_ren = awlen_fifo_ren;

always_comb begin 
    if ((flit_state_f == MSG_STATE_IDLE) && (awlen_buffer_q == 0) && fifo_has_packet && (type_fifo_out == MSG_TYPE_STORE)) awsize_buffer_d = awsize_fifo_out; // no beat left, next transaction
    else awsize_buffer_d = awsize_buffer_q;
end 

always_ff @(posedge clk or negedge rst_n) begin 
    if (!rst_n) awsize_buffer_q <= 0;
    else awsize_buffer_q <= awsize_buffer_d;
end 

//assign wstrb_fifo_mux_out = (wstrb_fifo_empty) ? 8'b0000_0000 : {wstrb_fifo_out[0], wstrb_fifo_out[1], wstrb_fifo_out[2], wstrb_fifo_out[3], wstrb_fifo_out[4], wstrb_fifo_out[5], wstrb_fifo_out[6], wstrb_fifo_out[7]};
assign wstrb_fifo_mux_out = (wstrb_fifo_empty) ? 0 : {<<1{wstrb_fifo_out}};

sync_fifo #(
        .DSIZE(AXI_DATA_WIDTH/8),
        .ASIZE(5),
        .MEMSIZE(16) // should be 2 ^ (ASIZE-1)
) wstrb_fifo (
        .rdata(wstrb_fifo_out),
        .empty(wstrb_fifo_empty),
        .clk(clk),
        .ren(wstrb_fifo_ren),
        .wdata(wstrb_fifo_wdata),
        .full(wstrb_fifo_full),
        .wval(wstrb_fifo_wval),
        .reset(fifo_rst)
);

if (AXI_DATA_WIDTH == MIN_NOC_DATA_WIDTH) begin
    always_comb begin 
            if ((flit_state_f == MSG_STATE_HEADER && noc_ready_in && (!wstrb_fifo_empty))) begin
                if (waddr_aligned_with_16B && (awlen_buffer_q >= 2)) begin 
                    wstrb_fifo_ren = (noc_cnt == 2);
                end 
                else wstrb_fifo_ren = 0;
            end 
            else wstrb_fifo_ren = (outstanding_store_req && ~outstanding_store_req_d);
    end 
end 

else begin 
    always_comb begin 
        wstrb_fifo_ren = (outstanding_store_req && ~outstanding_store_req_d);
    end 
end 

assign wstrb_fifo_wval = m_axi_wvalid && m_axi_wready;
assign wstrb_fifo_wdata = m_axi_wstrb;

/** fifo for read len***/
sync_fifo #(
    .DSIZE(8),
    .ASIZE(5),
    .MEMSIZE(16) // should be 2 ^ (ASIZE-1)
) arlen_fifo (
    .rdata(arlen_fifo_out),
    .empty(arlen_fifo_empty),
    .clk(clk),
    .ren(arlen_fifo_ren),
    .wdata(arlen_fifo_wdata),
    .full(arlen_fifo_full),
    .wval(arlen_fifo_wval),
    .reset(fifo_rst)
);

assign arlen_fifo_wval = m_axi_arvalid && m_axi_arready;
assign arlen_fifo_wdata = m_axi_arlen;
assign need_split_r_transaction = (arlen_buffer_q > 0);
assign arlen_fifo_ren = ((outstanding_load_req && ~outstanding_load_req_d) && !arlen_fifo_empty && last_read_transfer); 

if (AXI_DATA_WIDTH == MIN_NOC_DATA_WIDTH) begin
    assign read_word_select = (araddr_buffer_q[3] == 1) ? 1 : 0; 
    assign last_read_transfer = ((arlen_buffer_q == 2) && raddr_aligned_with_16B) || (arlen_buffer_q == 1);
    assign read_size = (arlen_buffer_q >= 2) && raddr_aligned_with_16B; // 1 -> 16B, 0 -> 8B

    always_comb begin 
        if ((flit_state_f == MSG_STATE_IDLE) && (arlen_buffer_q == 0) && fifo_has_packet && (type_fifo_out == MSG_TYPE_LOAD)) arlen_buffer_d = arlen_fifo_out + 1; // no beat left, next transaction
        else if (outstanding_load_req && ~outstanding_load_req_d)  arlen_buffer_d = (arlen_buffer_q == 1) ? arlen_buffer_q - 1 :
                                                    (arlen_buffer_q >= 2 && !raddr_aligned_with_16B) ? arlen_buffer_q - 1:
                                                    arlen_buffer_q -2;   
        else arlen_buffer_d = arlen_buffer_q;
    end 
end
else begin
    assign read_word_select = 0; // no need for selection in 128 bits version 
    assign last_read_transfer = (arlen_buffer_q == 1);
    assign read_size = 1; // 1 -> 16B, 0 -> 8B

    always_comb begin 
        if ((flit_state_f == MSG_STATE_IDLE) && (arlen_buffer_q == 0) && fifo_has_packet && (type_fifo_out == MSG_TYPE_LOAD)) arlen_buffer_d = arlen_fifo_out + 1; // no beat left, next transaction
        else if (outstanding_load_req && ~outstanding_load_req_d)  arlen_buffer_d = arlen_buffer_q - 1;  
        else arlen_buffer_d = arlen_buffer_q;
    end 
end 

always_ff @(posedge clk or negedge rst_n) begin 
    if (!rst_n) arlen_buffer_q <= 0;
    else arlen_buffer_q <= arlen_buffer_d;
end

/** fifo for read len***/
sync_fifo #(
    .DSIZE(8),
    .ASIZE(5),
    .MEMSIZE(16) // should be 2 ^ (ASIZE-1)
) arsize_fifo (
    .rdata(arsize_fifo_out),
    .empty(arsize_fifo_empty),
    .clk(clk),
    .ren(arsize_fifo_ren),
    .wdata(arsize_fifo_wdata),
    .full(arsize_fifo_full),
    .wval(arsize_fifo_wval),
    .reset(fifo_rst)
);

assign arsize_fifo_wval = m_axi_arvalid && m_axi_arready;
assign arsize_fifo_wdata = m_axi_arsize;
assign arsize_fifo_ren = arlen_fifo_ren;


always_comb begin 
    if ((flit_state_f == MSG_STATE_IDLE) && (arlen_buffer_q == 0) && fifo_has_packet && (type_fifo_out == MSG_TYPE_LOAD)) arsize_buffer_d = arsize_fifo_out; // no beat left, next transaction  
    else arsize_buffer_d = arsize_buffer_q;
end 

always_ff @(posedge clk or negedge rst_n) begin 
    if (!rst_n) arsize_buffer_q <= 0;
    else arsize_buffer_q <= arsize_buffer_d;
end

/* fifo for read addr */
sync_fifo #(
        .DSIZE(AXI_ADDR_WIDTH),
        .ASIZE(5),
        .MEMSIZE(16) // should be 2 ^ (ASIZE-1)    
) raddr_fifo (
        .rdata(araddr_fifo_out),
        .empty(araddr_fifo_empty),
        .clk(clk),
        .ren(araddr_fifo_ren),
        .wdata(araddr_fifo_wdata),
        .full(araddr_fifo_full),
        .wval(araddr_fifo_wval),
        .reset(fifo_rst)
);
assign raddr_aligned_with_16B = (type_fifo_out == MSG_TYPE_LOAD) ? ((~araddr_buffer_q[3]) && (~araddr_buffer_q[2]) && (~araddr_buffer_q[1]) && (~araddr_buffer_q[0])) && (arsize_buffer_q == AX_SIZE_8B): 0;  
assign araddr_fifo_wval = m_axi_arvalid && m_axi_arready;
assign araddr_fifo_wdata = m_axi_araddr;
assign araddr_fifo_ren = ((outstanding_load_req && ~outstanding_load_req_d)  && !araddr_fifo_empty && last_read_transfer);

if (AXI_DATA_WIDTH == MIN_NOC_DATA_WIDTH) begin 
    always_comb begin 
        if (flit_state_f == MSG_STATE_IDLE && !need_split_r_transaction && type_fifo_out == MSG_TYPE_LOAD) araddr_buffer_d = araddr_fifo_out;
        else if ((outstanding_load_req && ~outstanding_load_req_d)  && need_split_r_transaction ) araddr_buffer_d = (raddr_aligned_with_16B) ? araddr_buffer_q + (1 << (arsize_buffer_q + 1)) : araddr_buffer_q + (1 << (arsize_buffer_q)) ; 
        else araddr_buffer_d = araddr_buffer_q; 
    end 
end 

else begin 
    always_comb begin 
        if (flit_state_f == MSG_STATE_IDLE && !need_split_r_transaction && type_fifo_out == MSG_TYPE_LOAD) araddr_buffer_d = araddr_fifo_out;
        else if ((outstanding_load_req && ~outstanding_load_req_d)  && need_split_r_transaction ) araddr_buffer_d = araddr_buffer_q + (1 << (arsize_buffer_q)) ; 
        else araddr_buffer_d = araddr_buffer_q; 
    end 
end 

always_ff@(posedge clk or negedge rst_n) begin 
    if (!rst_n) 
        araddr_buffer_q <= 0;
    else 
        araddr_buffer_q <= araddr_buffer_d;
end 

/*write channel ready: address, data, strobe, read channel ready: address */
assign fifo_has_packet = (type_fifo_out == MSG_TYPE_STORE) ? (!awaddr_fifo_empty && !wdata_fifo_empty && !wstrb_fifo_empty && !awlen_fifo_empty) :
                            (type_fifo_out == MSG_TYPE_LOAD) ? !araddr_fifo_empty && !arlen_fifo_empty : 1'b0;

assign noc_store_done = noc_last_data && type_fifo_out == MSG_TYPE_STORE;
assign noc_load_done = noc_last_header && type_fifo_out == MSG_TYPE_LOAD;

always_ff@(posedge clk) begin 
    if (!rst_n) begin
        outstanding_load_req_q <= 0;
        outstanding_store_req_q <= 0;
    end else begin
        outstanding_load_req_q <= outstanding_load_req_d;
        outstanding_store_req_q <= outstanding_store_req_d;
    end
end 

always_comb begin
    if (noc_load_done) begin
        outstanding_load_req_d = 1;
    end else if (L2_request_ack && outstanding_load_req_q) begin
        outstanding_load_req_d = 0;
    end else begin
        outstanding_load_req_d = outstanding_load_req_q;
    end
end

always_comb begin
    if (noc_store_done) begin
        outstanding_store_req_d = 1;
    end else if (L2_request_ack && outstanding_store_req_q) begin
        outstanding_store_req_d = 0;
    end else begin
        outstanding_store_req_d = outstanding_store_req_q;
    end
end

assign outstanding_load_req = outstanding_load_req_d || outstanding_load_req_q;
assign outstanding_store_req = outstanding_store_req_d || outstanding_store_req_q;

/* set defaults for the flit */
always_comb
begin
    msg_type = `MSG_TYPE_RESERVED;
    msg_data_size = `MSG_DATA_SIZE_8B;
    msg_length = `MSG_LENGTH_WIDTH'b0;
    msg_address = {{`MSG_ADDR_WIDTH-`PHY_ADDR_WIDTH{1'b0}}, {`PHY_ADDR_WIDTH{1'b0}}}; 
    unique case (type_fifo_out)
        MSG_TYPE_STORE: begin
        `ifdef ARA_REQ2MEM
            msg_type = `MSG_TYPE_NC_STORE_REQ; // axi peripheral is writing to the memory?
            msg_data_size = buf_pmesh_data_size; // fix it for now
            msg_address = {{`MSG_ADDR_WIDTH-`PHY_ADDR_WIDTH{1'b0}}, awaddr_fifo_out[`PHY_ADDR_WIDTH-1:3], buf_pmesh_addr[2:0]};
            msg_length = `MSG_LENGTH_WIDTH'd2 + NOC_PAYLOAD_LEN; // 2 extra headers + 1 data
        `else 
            msg_type = `MSG_TYPE_SWAPWB_REQ;
            msg_data_size = `MSG_DATA_SIZE_16B;
            msg_address = {{`MSG_ADDR_WIDTH-`PHY_ADDR_WIDTH{1'b0}}, awaddr_buffer_q[`PHY_ADDR_WIDTH-1:0]};
            msg_length = `MSG_LENGTH_WIDTH'd2 + 2;
        `endif
            
        end

        MSG_TYPE_LOAD: begin
        `ifdef ARA_REQ2MEM // to memeory 
            msg_type = `MSG_TYPE_NC_LOAD_REQ; 
            msg_data_size = `MSG_DATA_SIZE_8B; 
            msg_length = `MSG_LENGTH_WIDTH'd2; 
            msg_address = {{`MSG_ADDR_WIDTH-`PHY_ADDR_WIDTH{1'b0}}, araddr_fifo_out[`PHY_ADDR_WIDTH-1:0]};
        `else // we can use load noshare to load the data from L2 right now
            msg_type = `MSG_TYPE_LOAD_NOSHARE_REQ; // for L2 LOADNOSHARE test 
            msg_data_size = `MSG_DATA_SIZE_16B;
            msg_length = `MSG_LENGTH_WIDTH'd2; 
            msg_address = {{`MSG_ADDR_WIDTH-`PHY_ADDR_WIDTH{1'b0}}, araddr_buffer_q[`PHY_ADDR_WIDTH-1:0]};
        `endif
        end
        MSG_TYPE_INVAL: begin 
            msg_type = `MSG_TYPE_RESERVED;
            msg_data_size = `MSG_DATA_SIZE_8B;
            msg_length = `MSG_LENGTH_WIDTH'b0;
            msg_address = {{`MSG_ADDR_WIDTH-`PHY_ADDR_WIDTH{1'b0}}, {`PHY_ADDR_WIDTH{1'b0}}}; 
        end
        default: begin
            msg_type = `MSG_TYPE_RESERVED;
            msg_data_size = `MSG_DATA_SIZE_8B;
            msg_length = `MSG_LENGTH_WIDTH'b0;
            msg_address = {{`MSG_ADDR_WIDTH-`PHY_ADDR_WIDTH{1'b0}}, {`PHY_ADDR_WIDTH{1'b0}}}; 
        end
    endcase
end

always_ff@(posedge clk or negedge rst_n)
begin
    if (!rst_n) begin
        noc_cnt <= 3'b0;
    end
    else begin
        noc_cnt <= (noc_last_header | noc_last_data)  ? 3'b0 :
                    (noc_cnt == 3'b0 && fifo_has_packet && flit_state_f == MSG_STATE_HEADER) ? noc_cnt + 1: // since ready signal of NoC spilitter depends on producer's valid sinal, we need to let flit header0 ready first
                    (fifo_has_packet && noc_ready_in) ? noc_cnt + 1 : noc_cnt;
    end
end



always_comb
begin
    if (noc_ready_in) begin
        noc_last_header = (flit_state_f == MSG_STATE_HEADER &&
                                    noc_cnt == NOC_HDR_LEN) ? 1'b1 : 1'b0;
        noc_last_data = (flit_state_f == MSG_STATE_NOC_DATA &&
                                    noc_cnt == (msg_length - NOC_HDR_LEN)) ? 1'b1 : 1'b0;
    end
    else begin 
        noc_last_header = 1'b0;
        noc_last_data = 1'b0;
    end 
end


always_ff@(posedge clk or negedge rst_n) 
begin 
    if (!rst_n) begin
        flit_state_f <= MSG_STATE_IDLE;
    end
    else begin
        flit_state_f <= flit_state_next;
    end
end 


always_comb //@(posedge clk or negedge rst_n)
begin
    flit_state_next = flit_state_f;
    unique case (flit_state_f)
        MSG_STATE_IDLE: begin
          if (~(outstanding_load_req || outstanding_store_req)) begin  
          `ifdef ARA_REQ2MEM
            if ((fifo_has_packet && type_fifo_out == MSG_TYPE_STORE))
                flit_state_next = MSG_STATE_WAIT_STRB;
            else if ((fifo_has_packet && type_fifo_out == MSG_TYPE_LOAD))
                flit_state_next = MSG_STATE_HEADER;
          `else 
            if ((awlen_buffer_q > 0 || arlen_buffer_q > 0) && ((type_fifo_out == MSG_TYPE_STORE) || (type_fifo_out == MSG_TYPE_LOAD)))
                flit_state_next = MSG_STATE_DEST_CAL;
          `endif 
          end else begin
            flit_state_next = flit_state_f;
          end
        end
      `ifdef ARA_REQ2MEM
        MSG_STATE_WAIT_STRB:begin
            if (wstrb_outputside_ready & wstrb_outputside_valid)
                flit_state_next = MSG_STATE_HEADER;
            else flit_state_next = flit_state_f;
        end
      `else 
        MSG_STATE_DEST_CAL: begin
            flit_state_next = MSG_STATE_HEADER;
        end
      `endif
      
        MSG_STATE_HEADER: begin
            if (noc_last_header && type_fifo_out == MSG_TYPE_STORE)
                flit_state_next = MSG_STATE_NOC_DATA;
            else if (noc_last_header && type_fifo_out == MSG_TYPE_LOAD)
              `ifdef ARA_REQ2MEM
                flit_state_next = MSG_STATE_IDLE;
              `else
                flit_state_next = MSG_STATE_IDLE;
              `endif 
            else flit_state_next = flit_state_f;
        end

        MSG_STATE_NOC_DATA: begin
            if (noc_store_done)
                flit_state_next = MSG_STATE_IDLE;
            else if (noc_load_done)
                flit_state_next = MSG_STATE_IDLE;
            else 
                flit_state_next = flit_state_f;
        end
        default: flit_state_next = MSG_STATE_IDLE;
    endcase
end

/*****destination x and y index calculate*************/

`ifndef ARA_REQ2MEM
    l15_home_encoder    l15_home_encoder(
    .home_in        (home_addr_bits_s0),
    .num_homes      (system_tile_count),
    .lhid_out       (lhid_s0)
    );

    flat_id_to_xy lhid_to_xy (
        .flat_id(lhid_s1[`HOME_ID_WIDTH-1:0]),
        .x_coord(lhid_s1_x),
        .y_coord(lhid_s1_y)
    );

    always_ff@(posedge clk or negedge rst_n) begin 
        if (!rst_n) lhid_s1 <= `HOME_ID_WIDTH'b0;
        else if (cal_dest_stage1) lhid_s1 <= lhid_s0;
        else lhid_s1 <= lhid_s1;
    end 

    always_comb 
    begin 
        cal_dest_stage0 = (flit_state_f == MSG_STATE_IDLE);
        cal_dest_stage1 = (flit_state_f == MSG_STATE_DEST_CAL);
        cal_dest_stage2 = (flit_state_f == MSG_STATE_HEADER);
    end 

    always_comb
    begin
        //special l2 addresses start with 0xA
        special_l2_addr_s0 = (axi2noc_req_address_s0[39:36] == 4'b1010);
    end

    always_comb begin
        unique case (type_fifo_out) 
            MSG_TYPE_STORE: axi2noc_req_address_s0 = awaddr_buffer_q[`PHY_ADDR_WIDTH-1:0];
            MSG_TYPE_LOAD: axi2noc_req_address_s0 = araddr_buffer_q[`PHY_ADDR_WIDTH-1:0];
            MSG_TYPE_INVAL:axi2noc_req_address_s0 = `PHY_ADDR_WIDTH'b0;
            default: axi2noc_req_address_s0 = `PHY_ADDR_WIDTH'b0;
        endcase
    end 

    always_comb
    begin
        if (special_l2_addr_s0)
        begin
            home_addr_bits_s0 = axi2noc_req_address_s0[`HOME_ID_ADDR_POS_HIGH];
        end
        else
        begin
            unique case (home_alloc_method) 
            `HOME_ALLOC_LOW_ORDER_BITS:
            begin
                home_addr_bits_s0 = axi2noc_req_address_s0[`HOME_ID_ADDR_POS_LOW];
            end
            `HOME_ALLOC_MIDDLE_ORDER_BITS:
            begin
                home_addr_bits_s0 = axi2noc_req_address_s0[`HOME_ID_ADDR_POS_MIDDLE];
            end
            `HOME_ALLOC_HIGH_ORDER_BITS:
            begin
                home_addr_bits_s0 = axi2noc_req_address_s0[`HOME_ID_ADDR_POS_HIGH];
            end
            `HOME_ALLOC_MIXED_ORDER_BITS:
            begin
                home_addr_bits_s0 = (axi2noc_req_address_s0[`HOME_ID_ADDR_POS_LOW] ^ axi2noc_req_address_s0[`HOME_ID_ADDR_POS_MIDDLE]);
            end
            default: home_addr_bits_s0 = `MSG_LHID_WIDTH'b0;
            endcase
        end
    end
`endif 

always_comb
begin
    msg_mshrid = {`MSG_MSHRID_WIDTH{1'b0}};
    msg_options_1 = {`MSG_OPTIONS_1_WIDTH{1'b0}};
    msg_options_2 = 16'b0;
    msg_options_3 = 30'b0;

    flit[`MSG_OPTIONS_1] = msg_options_1;

    flit[`MSG_AMO_MASK0_] = {`MSG_AMO_MASK0_WIDTH{1'b0}};
    flit[`MSG_DATA_SIZE_] = msg_data_size;
    flit[`MSG_OPTIONS_2_] = msg_options_2;

    flit[`MSG_AMO_MASK1_] = {`MSG_AMO_MASK1_WIDTH{1'b0}};
    flit[`MSG_OPTIONS_3_] = msg_options_2;

    flit[`NOC_DATA_WIDTH -1:0] = {`NOC_DATA_WIDTH{1'b0}};
    flit_ready = 1'b0;
    unique case (flit_state_f)
        MSG_STATE_HEADER: begin
            unique case (noc_cnt)
                3'b001: begin
                    flit[`MSG_DST_CHIPID] = dest_chipid;
                    `ifdef ARA_REQ2MEM
                    flit[`MSG_DST_X] = dest_xpos;
                    flit[`MSG_DST_Y] = dest_ypos;
                    `else 
                    flit[`MSG_DST_X] = lhid_s1_x;
                    flit[`MSG_DST_Y] = lhid_s1_y;
                    `endif
                    flit[`MSG_DST_FBITS] = dest_fbits; // to memory or L2 cache
                    flit[`MSG_LENGTH] = msg_length;
                    flit[`MSG_TYPE] = msg_type;
                    flit[`MSG_MSHRID] = msg_mshrid;
                    flit[`MSG_OPTIONS_1] = msg_options_1;
                    flit_ready = 1'b1;
                end

                3'b010: begin
                `ifdef ARA_REQ2MEM
                    flit[`MSG_ADDR_] = msg_address;
                    flit[`MSG_OPTIONS_2_] = msg_options_2;
                    flit[`MSG_DATA_SIZE_] = msg_data_size;
                `else 
                    flit[`MSG_ADDR_] = msg_address;
                    flit[`MSG_DATA_SIZE_] = msg_data_size;
                    if (AXI_DATA_WIDTH == MIN_NOC_DATA_WIDTH) begin
                        flit[`MSG_AMO_MASK0_] = (type_fifo_out == MSG_TYPE_STORE && waddr_aligned_with_16B &&  awlen_buffer_q >= 2) ? wstrb_fifo_mux_out : 
                        (write_word_select == 0) ? wstrb_fifo_mux_out : 8'b0;
                    end 
                    else begin 
                        flit[`MSG_AMO_MASK0_] = (type_fifo_out == MSG_TYPE_STORE) ? wstrb_fifo_mux_out[15:8] : 0;
                    end 
                `endif 
                    flit_ready = 1'b1;                  
                end

                3'b011: begin
                `ifdef ARA_REQ2MEM
                    flit[`MSG_SRC_CHIPID_] = src_chipid;
                    flit[`MSG_SRC_X_] = src_xpos;
                    flit[`MSG_SRC_Y_] = src_ypos;
                    flit[`MSG_SRC_FBITS_] = src_fbits;
                    flit[`MSG_OPTIONS_3_] = msg_options_3;
                `else 
                    flit[`MSG_SRC_CHIPID_] = src_chipid;
                    flit[`MSG_SRC_X_] = src_xpos;
                    flit[`MSG_SRC_Y_] = src_ypos;
                    flit[`MSG_SRC_FBITS_] = src_fbits;
                    if (AXI_DATA_WIDTH == MIN_NOC_DATA_WIDTH) begin 
                        flit[`MSG_AMO_MASK1_] =  (type_fifo_out == MSG_TYPE_STORE && waddr_aligned_with_16B && awlen_buffer_q >= 2) ? wstrb_fifo_mux_out : 
                        (write_word_select == 1) ? wstrb_fifo_mux_out : 8'b0;
                    end
                    else begin 
                        flit[`MSG_AMO_MASK1_] = (type_fifo_out == MSG_TYPE_STORE) ? wstrb_fifo_mux_out[7:0] : 0;
                    end 

                `endif 
                    flit_ready = 1'b1;
                end
                default: begin
                    flit_ready = 1'b0;
                end
            endcase
        end

        MSG_STATE_NOC_DATA: begin
            if (AXI_DATA_WIDTH == MIN_NOC_DATA_WIDTH) begin
                flit[`NOC_DATA_WIDTH-1:0] = (type_fifo_out == MSG_TYPE_STORE) ? wdata_fifo_out : {`NOC_DATA_WIDTH{1'b0}}; //wdata_fifo_out;
            end 
            else begin
                flit[`NOC_DATA_WIDTH-1:0] = (type_fifo_out == MSG_TYPE_STORE && noc_cnt == 0) ? wdata_fifo_out[127:64] :
                                                (type_fifo_out == MSG_TYPE_STORE && noc_cnt == 1) ? wdata_fifo_out[63:0] :
                                                {`NOC_DATA_WIDTH{1'b0}}; //wdata_fifo_out;
            end 
            flit_ready = 1'b1;
        end

        default: begin
            flit[`NOC_DATA_WIDTH-1:0] = {`NOC_DATA_WIDTH{1'b0}};
            flit_ready = 1'b0;
        end
    endcase
end

assign noc_valid_out = flit_ready;
assign noc_data_out = flit;

endmodule
