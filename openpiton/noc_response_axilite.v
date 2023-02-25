/* In this version, let's assume that AXI_LITE_DATA_WIDTH >= `NOC_DATA_WDITH
In the subsequent version, we'll add support for axi data widths smaller than
noc data width*/

`include "define.tmp.h"

module noc_response_axilite #(
    parameter SLAVE_RESP_BYTEWIDTH = 4,
    parameter SWAP_ENDIANESS       = 0,
    // shift unaligned read data
    parameter ALIGN_RDATA          = 1,
    parameter AXI_LITE_DATA_WIDTH  = 512,
    parameter AXI_LITE_RESP_WIDTH  = 2
) (
    // Clock + Reset
    input  wire                                   clk,
    input  wire                                   rst,

    input  wire                                   noc_valid_in,
    input  wire [`NOC_DATA_WIDTH-1:0]             noc_data_in,
    output wire                                   noc_ready_out,

    // Memory Splitter <- AXI SPI
    output  reg                                   noc_valid_out,
    output  reg  [`NOC_DATA_WIDTH-1:0]            noc_data_out,
    input  wire                                   noc_ready_in,

    // AXI Read Data Channel Signals
    output  reg [AXI_LITE_DATA_WIDTH-1:0]         m_axi_rdata,
    output  reg [AXI_LITE_RESP_WIDTH-1:0]         m_axi_rresp,
    output  reg                                   m_axi_rvalid,
    input  wire                                   m_axi_rready,

    // AXI Write Response Channel Signals
    output  reg [AXI_LITE_RESP_WIDTH-1:0]         m_axi_bresp,
    output  reg                                   m_axi_bvalid,
    input wire                                    m_axi_bready,

    // this does not belong to axi lite and is non-standard
    output  reg  [`C_M_AXI_LITE_SIZE_WIDTH-1:0]   w_reqbuf_size,
    output  reg  [`C_M_AXI_LITE_SIZE_WIDTH-1:0]   r_reqbuf_size
);

//==============================================================================
// Local Parameters
//==============================================================================

// States for Incoming Piton Messages
localparam MSG_STATE_INVAL      = 3'd0; // Invalid Message
localparam MSG_STATE_HEADER_0   = 3'd1; // Header 0
localparam MSG_STATE_HEADER_1   = 3'd2; // Header 1
localparam MSG_STATE_HEADER_2   = 3'd3; // Header 2
localparam MSG_STATE_DATA       = 3'd4; // Data Lines

// Types for Incoming Piton Messages
localparam MSG_TYPE_INVAL       = 2'd0; // Invalid Message
localparam MSG_TYPE_LOAD        = 2'd1; // Load Request
localparam MSG_TYPE_STORE       = 2'd2; // Store Request

// States for Buffer Status
localparam BUF_STATUS_INCOMP    = 2'd0; // Buffer not yet filled by one complete request/response
localparam BUF_STATUS_COMP      = 2'd1; // Buffer contains a complete but unsent request
localparam BUF_STATUS_WAITRESP  = 2'd2; // Request sent via AXI, waiting on response
localparam BUF_STATUS_RESPSEND  = 2'd3; // Response waiting to forward back to memory splitter

// ACK's
localparam LOAD_ACK = 1'd0;
localparam STORE_ACK = 1'd1;

//==============================================================================
// Local Variables
//==============================================================================

// Meta-registers for tracking incoming Piton packets (used in parsing)
 reg  [2:0]                          splitter_io_msg_state_f;
 reg  [1:0]                          splitter_io_msg_type_f;
 reg  [`MSG_LENGTH_WIDTH-1:0]        splitter_io_msg_counter_f;

// Buffer registers for load requests
 reg  [`NOC_DATA_WIDTH-1:0]          r_req_buf_header0_f;
 reg  [`NOC_DATA_WIDTH-1:0]          r_req_buf_header1_f;
 reg  [`NOC_DATA_WIDTH-1:0]          r_req_buf_header2_f;
 reg  [1:0]                          r_req_buf_status_f;

// Buffer registers for store requests
 reg  [`NOC_DATA_WIDTH-1:0]          w_req_buf_header0_f;
 reg  [`NOC_DATA_WIDTH-1:0]          w_req_buf_header1_f;
 reg  [`NOC_DATA_WIDTH-1:0]          w_req_buf_header2_f;
 reg  [`NOC_DATA_WIDTH-1:0]          w_req_buf_data0_f;
 wire [1:0]                          w_req_buf_status;
 reg  [1:0]                          w_addr_req_buf_status_f;
 reg  [1:0]                          w_data_req_buf_status_f;

// Buffer registers for load responses
 reg  [`NOC_DATA_WIDTH-1:0]          r_resp_buf_header0_f;
 reg  [AXI_LITE_DATA_WIDTH-1:0] r_resp_buf_data0_f;
 reg  [AXI_LITE_RESP_WIDTH-1:0] r_resp_buf_rresp_f;
 reg  [1:0]                          r_resp_buf_status_f;

// Buffer registers for store responses
 reg  [`NOC_DATA_WIDTH-1:0]          w_resp_buf_header0_f;
 reg  [AXI_LITE_RESP_WIDTH-1:0] w_resp_buf_bresp_f;
 wire [1:0]                          w_resp_buf_status;
 reg  [1:0]                          w_addr_resp_buf_status_f;
 reg  [1:0]                          w_data_resp_buf_status_f;

// Helper Signals for saving requests
 wire                         splitter_io_go;
 wire                         splitter_io_load_go;
 wire                         splitter_io_store_go;

 wire                         splitter_io_msg_is_load;
 wire                         splitter_io_msg_is_store;
 wire                         splitter_io_msg_is_load_next;
 wire                         splitter_io_msg_is_store_next;

 wire [2:0]                   splitter_io_msg_state_next;
 wire [2:0]                   splitter_io_msg_type_mux_out;
 wire [2:0]                   splitter_io_msg_type_next;
 wire [`MSG_LENGTH_WIDTH-1:0] splitter_io_msg_counter_next;

// Helper Signals for sending responses
 wire                         m_axi_ar_go;
 wire                         m_axi_w_go;
 wire                         m_axi_aw_go;

 wire                         m_axi_b_go;
 wire                         m_axi_r_go;
 reg  [`NOC_DATA_WIDTH-1:0]   a_axi_rdata_shifted;
 wire [`NOC_DATA_WIDTH-1:0]   a_axi_rdata_masked;

 wire [`NOC_DATA_WIDTH-1:0]   r_resp_buf_header0_next;
 wire [`NOC_DATA_WIDTH-1:0]   w_resp_buf_header0_next;

 reg  [`MSG_LENGTH_WIDTH-1:0] io_splitter_ack_load_counter_f;
 reg                          io_splitter_arb_f;
 reg                          io_splitter_ack_mux_sel;

 wire                         r_resp_buf_val;
 wire                         w_resp_buf_val;
 wire [`NOC_DATA_WIDTH-1:0]   io_splitter_ack_store;
 wire [`NOC_DATA_WIDTH-1:0]   io_splitter_ack_load;
 wire                         io_splitter_ack_load_go;
 wire                         io_splitter_ack_store_go;


// Types for Incoming Piton Messages
localparam MSG_TYPE_INVAL_ACK       = 2'd0; // Invalid Message
localparam MSG_TYPE_LOAD_ACK        = 2'd1; // Load Request Ack
localparam MSG_TYPE_STORE_ACK       = 2'd2; // Store Request Ack

reg  [2:0]                   noc_io_msg_state_f;
reg  [1:0]                   noc_io_msg_type_f;
reg  [`MSG_LENGTH_WIDTH-1:0] noc_io_msg_counter_f;
reg  [1:0]                   noc_msg_type_mux_out;
reg  [1:0]                   noc_msg_type_mux_out_next;

wire                         noc_io_go;
//wire [1:0]                   noc_msg_type_mux_out;
wire [2:0]                   noc_io_msg_state_next;
wire [2:0]                   noc_io_msg_type_mux_out;
wire [2:0]                   noc_io_msg_type_next;
wire [`MSG_LENGTH_WIDTH-1:0] noc_io_msg_counter_next;

reg                          msg_data_done;
reg  [2:0]                   msg_state_f;
reg  [2:0]                   msg_state_next;
reg  [`MSG_LENGTH_WIDTH-1:0] msg_payload_len;
reg  [`MSG_LENGTH_WIDTH-1:0] msg_counter_next;
reg  [`MSG_LENGTH_WIDTH-1:0] msg_counter_f;


// Should we read data from noc_data_in?
assign noc_io_go = noc_valid_in && noc_ready_out;

always @(posedge clk)
begin
    if (rst) begin
        msg_state_f <= MSG_STATE_HEADER_0;
        msg_counter_f <= `MSG_LENGTH_WIDTH'b0;
    end
    else begin
        msg_state_f <= msg_state_next;
        msg_counter_f <= msg_counter_next;
    end
end

always @(*)
begin
    msg_state_next = msg_state_f;
    msg_counter_next = msg_counter_f;
    msg_data_done = 1'b0;
    m_axi_bresp = {AXI_LITE_RESP_WIDTH{1'b0}};
    m_axi_bvalid = 1'b0;
    case (msg_state_f)
        MSG_STATE_HEADER_0: begin
            if (noc_io_go && (noc_data_in[`MSG_TYPE] == `MSG_TYPE_NC_LOAD_MEM_ACK)) begin
                
                if (noc_data_in[`MSG_LENGTH] == `MSG_LENGTH_WIDTH'd0) 
                begin
                    msg_state_next = MSG_STATE_HEADER_0;
                end
                else
                begin
                    msg_state_next = MSG_STATE_DATA;
                end

                msg_counter_next = `MSG_LENGTH_WIDTH'd0;
                msg_payload_len = noc_data_in[`MSG_LENGTH];
            end
            else if (noc_io_go && (noc_data_in[`MSG_TYPE] == `MSG_TYPE_NC_STORE_MEM_ACK)) begin
                
                if (noc_data_in[`MSG_LENGTH] == `MSG_LENGTH_WIDTH'd0) 
                begin
                    msg_state_next = MSG_STATE_HEADER_0;
                    m_axi_bresp = {AXI_LITE_RESP_WIDTH{1'b0}};
                    m_axi_bvalid = 1'b1;
                end
                else
                begin
                    msg_state_next = MSG_STATE_DATA;
                end

                msg_counter_next = `MSG_LENGTH_WIDTH'd0;
                msg_payload_len = noc_data_in[`MSG_LENGTH];
            end
        end
        MSG_STATE_DATA: begin
            if (msg_counter_f >= msg_payload_len) begin
                msg_data_done = 1'b1;
                msg_state_next = MSG_STATE_HEADER_0;
                msg_payload_len = `MSG_LENGTH_WIDTH'd0;
                msg_counter_next = `MSG_LENGTH_WIDTH'd0;
            end
            else begin
                msg_counter_next = (noc_io_go) ? msg_counter_f + 1'b1 : msg_counter_f;
            end
        end
    endcase
end

//assign msg_type = (noc_io_go && msg_state_f == MSG_STATE_HEADER_0) ?
//                        noc_data_in[`MSG_TYPE] :

//--------------------------------------------------------------------------
// Forward data to AXI Read Channel
//--------------------------------------------------------------------------

wire                                ren;
wire                                full;
wire                                empty;
wire [AXI_LITE_DATA_WIDTH-1:0]      rdata;

reg                                 wval;
reg [AXI_LITE_DATA_WIDTH-1:0]       wdata;           

/* fifo for read data */
sync_fifo #(
	.DSIZE(AXI_LITE_DATA_WIDTH),
	.ASIZE(5),
	.MEMSIZE(16) // should be 2 ^ (ASIZE-1)    
) raddr_fifo (
	.rdata(rdata),
	.empty(empty),
	.clk(clk),
	.ren(ren),
	.wdata(wdata),
	.full(full),
	.wval(wval),
	.reset(rst)
);

assign noc_ready_out = !full;
assign ren = (!empty && m_axi_rready);

generate
    if (AXI_LITE_DATA_WIDTH == `NOC_DATA_WIDTH) begin
        always @(posedge clk)
        begin
            if (rst) begin
                wval <= 
                wdata <= {AXI_LITE_DATA_WIDTH{1'b0}};
            end
            else begin
                wval <= (msg_state_f == MSG_STATE_DATA && noc_io_go && !full);
                wdata <= noc_data_in;
            end
        end        
    end
    else if (AXI_LITE_DATA_WIDTH >= `NOC_DATA_WIDTH) begin
        reg [AXI_LITE_DATA_WIDTH-1:0]                      data;
        always @(posedge clk)
        begin
            if (rst) begin
                wval <= 1'b0;
                wdata <= {AXI_LITE_DATA_WIDTH{1'b0}};
            end
            else begin
                wval <= 1'b0;

                if (msg_state_f == MSG_STATE_DATA && noc_io_go && !full)
                begin
                    data[msg_counter_f*`NOC_DATA_WIDTH +: `NOC_DATA_WIDTH] = noc_data_in;
                end

                if (msg_data_done) begin
                    wval <= 1'b1;
                    wdata <= data;
                end
            end
        end
    end
endgenerate

assign m_axi_rvalid = ren;
assign m_axi_rdata = rdata;
assign m_axi_rresp = {AXI_LITE_RESP_WIDTH{1'b0}};

endmodule
