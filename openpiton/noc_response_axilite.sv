`include "define.tmp.h"

module noc_response_axilite #(
    parameter AXI_LITE_DATA_WIDTH  = 512,
  `ifndef ARA_REQ2MEM
    parameter MSG_TYPE_INVAL       = 2'd0, // Invalid Message
    parameter MSG_TYPE_LOAD        = 2'd1,// Load Request
    parameter MSG_TYPE_STORE       = 2'd2, // Store Request
  `endif
    parameter AXI_LITE_RESP_WIDTH  = 2
) (
    // Clock + Reset
    input  logic                                   clk,
    input  logic                                   rst_n,

    //NoC channel output 
    input  logic                                   noc_valid_in,
    input  logic [`NOC_DATA_WIDTH-1:0]             noc_data_in,
    output logic                                   noc_ready_out,

    //Signal to indicate current transaction type 
  `ifndef ARA_REQ2MEM
    input  logic [2:0]                             transaction_type_wr_data, 
    input  logic                                   transaction_type_wr,
  `endif

    // AXI Read Data Channel Signals
    output logic [AXI_LITE_DATA_WIDTH-1:0]         m_axi_rdata,
    output logic [AXI_LITE_RESP_WIDTH-1:0]         m_axi_rresp,
    output logic                                   m_axi_rvalid,
    input  logic                                   m_axi_rready,

    // AXI Write Response Channel Signals
    output logic [AXI_LITE_RESP_WIDTH-1:0]         m_axi_bresp,
    output logic                                   m_axi_bvalid,
    input  logic                                   m_axi_bready
);

`ifdef ARA_REQ2MEM
    typedef enum logic {
        MSG_STATE_HEADER_0   = 1'b0, // Header 0
        MSG_STATE_DATA       = 1'b1 // Data Lines
    } msg_state; 
`else 
    typedef enum logic [2:0] {
        MSG_STATE_HEADER_0   = 2'b00, // Header 0
        MSG_STATE_DATA       = 2'b01, // Data Lines
        MSG_STATE_STORE      = 2'b10  // for the fifo read when store happens
    } msg_state;
`endif

msg_state msg_state_f, msg_state_next;

logic                          noc_io_go;
logic                          msg_data_done;
logic  [`MSG_LENGTH_WIDTH-1:0] msg_payload_len_f;
logic  [`MSG_LENGTH_WIDTH-1:0] msg_payload_len_next;
logic  [`MSG_LENGTH_WIDTH-1:0] msg_counter_next;
logic  [`MSG_LENGTH_WIDTH-1:0] msg_counter_f;

`ifndef ARA_REQ2MEM
    logic store_ack; 
    logic load_ack;
    logic [2:0] transaction_type_rd_data; 
    logic transaction_type_rd; 
    logic transaction_fifo_empty;
    logic transaction_fifo_full; 

    always_comb begin 
        store_ack = (noc_data_in[`MSG_TYPE] == `MSG_TYPE_NODATA_ACK);
        load_ack = (noc_data_in[`MSG_TYPE] == `MSG_TYPE_DATA_ACK);
    end 

    /* fifo for storing transaction type */
    sync_fifo #(
        .DSIZE(3),
        .ASIZE(5),
        .MEMSIZE(16) // should be 2 ^ (ASIZE-1)
    ) type_fifo (
        .rdata(transaction_type_rd_data),
        .empty(transaction_fifo_empty),
        .clk(clk),
        .ren(transaction_type_rd),
        .wdata(transaction_type_wr_data),
        .full(transaction_fifo_full),
        .wval(transaction_type_wr),
        .reset(!rst_n)
    );
`endif

assign noc_io_go = noc_valid_in && noc_ready_out;

always_ff @(posedge clk or negedge rst_n)
begin
    if (!rst_n) begin
        msg_state_f <= MSG_STATE_HEADER_0;
        msg_counter_f <= `MSG_LENGTH_WIDTH'b0;
        msg_payload_len_f <= `MSG_LENGTH_WIDTH'b0;
    end
    else begin
        msg_state_f <= msg_state_next;
        msg_counter_f <= msg_counter_next;
        msg_payload_len_f <= msg_payload_len_next;
    end
end

always_comb
begin
    msg_state_next = msg_state_f;
    msg_counter_next = msg_counter_f;
    msg_data_done = 1'b0;
    m_axi_bresp = {AXI_LITE_RESP_WIDTH{1'b0}};
    m_axi_bvalid = 1'b0;
    msg_payload_len_next = msg_payload_len_f;

  `ifndef ARA_REQ2MEM    
    transaction_type_rd = 0;
  `endif

    unique case (msg_state_f)
        MSG_STATE_HEADER_0: begin
        `ifdef ARA_REQ2MEM
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
                msg_payload_len_next = noc_data_in[`MSG_LENGTH];
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
                msg_payload_len_next = noc_data_in[`MSG_LENGTH];
            end
        `elsif LOAD_NOSHARE_TEST
            if (noc_io_go && (load_ack)) begin
                if (transaction_type_rd_data[2:1] == MSG_TYPE_STORE && (~transaction_fifo_empty)) 
                begin
                    msg_state_next = MSG_STATE_STORE;
                    m_axi_bresp = {AXI_LITE_RESP_WIDTH{1'b0}};
                    m_axi_bvalid = 1'b1;
                end
                else if (transaction_type_rd_data[2:1] == MSG_TYPE_LOAD && (~transaction_fifo_empty))
                begin
                    msg_state_next = MSG_STATE_DATA;
                end
                else msg_state_next = MSG_STATE_HEADER_0;
                msg_counter_next = `MSG_LENGTH_WIDTH'd0;
                msg_payload_len_next = noc_data_in[`MSG_LENGTH];
            end
        `else 
            if (noc_io_go && (load_ack)) begin
                
                if (transaction_type_rd_data[2:1] == MSG_TYPE_STORE && (~transaction_fifo_empty)) 
                begin
                    msg_state_next = MSG_STATE_STORE;
                    m_axi_bresp = {AXI_LITE_RESP_WIDTH{1'b0}};
                    m_axi_bvalid = 1'b1;
                end
                else if (transaction_type_rd_data[2:1] == MSG_TYPE_LOAD && (~transaction_fifo_empty))
                begin
                    msg_state_next = MSG_STATE_DATA;
                end
                else msg_state_next = MSG_STATE_HEADER_0;
                msg_counter_next = `MSG_LENGTH_WIDTH'd0;
                msg_payload_len_next = noc_data_in[`MSG_LENGTH];
            end
        `endif 
        end
        MSG_STATE_DATA: begin
            if (msg_counter_f >= msg_payload_len_f) begin
                msg_data_done = 1'b1;
                msg_state_next = MSG_STATE_HEADER_0;
                msg_payload_len_next = `MSG_LENGTH_WIDTH'd0;
                msg_counter_next = `MSG_LENGTH_WIDTH'd0;
              `ifndef ARA_REQ2MEM
                transaction_type_rd = 1;
              `endif
            end
            else begin
                msg_counter_next = (noc_io_go) ? msg_counter_f + 1'b1 : msg_counter_f;
            end
        end
      `ifndef ARA_REQ2MEM
        MSG_STATE_STORE:begin 
            msg_state_next = MSG_STATE_HEADER_0;
            m_axi_bresp = {AXI_LITE_RESP_WIDTH{1'b0}};
            m_axi_bvalid = 1'b0;
            transaction_type_rd = 1;
            msg_counter_next = `MSG_LENGTH_WIDTH'd0;
            msg_payload_len_next = `MSG_LENGTH_WIDTH'd0;
        end 
      `endif
        default: begin 
            msg_state_next = msg_state_f;
            msg_counter_next = msg_counter_f;
            msg_data_done = 1'b0;
            m_axi_bresp = {AXI_LITE_RESP_WIDTH{1'b0}};
            m_axi_bvalid = 1'b0;
            msg_payload_len_next = msg_payload_len_f;
          `ifndef ARA_REQ2MEM    
            transaction_type_rd = 0;
          `endif
        end 
    endcase
end

//--------------------------------------------------------------------------
// Forward data to AXI Read Channel
//--------------------------------------------------------------------------

logic                                ren;
logic                                full;
logic                                empty;
logic [AXI_LITE_DATA_WIDTH-1:0]      rdata;

logic                                 wval;
logic [AXI_LITE_DATA_WIDTH-1:0]       wdata;  
logic [AXI_LITE_DATA_WIDTH-1:0]       data;         

/* fifo for read data */
sync_fifo #(
	.DSIZE(AXI_LITE_DATA_WIDTH),
	.ASIZE(5),
	.MEMSIZE(16)  
) raddr_fifo (
	.rdata(rdata),
	.empty(empty),
	.clk(clk),
	.ren(ren),
	.wdata(wdata),
	.full(full),
	.wval(wval),
	.reset(!rst_n)
);

// Noc interface signal 
assign noc_ready_out = !full;

// fifo control signal
assign ren = (m_axi_rvalid && m_axi_rready);

// AXI Read channel signal 
assign m_axi_rvalid = !empty;
assign m_axi_rdata = m_axi_rvalid ? rdata : {AXI_LITE_DATA_WIDTH{1'b0}};
assign m_axi_rresp = {AXI_LITE_RESP_WIDTH{1'b0}};

generate
    if (AXI_LITE_DATA_WIDTH == `NOC_DATA_WIDTH) begin
        always_ff@(posedge clk or negedge rst_n)
        begin
            if (!rst_n) begin
                wval <= 0;
                wdata <= {AXI_LITE_DATA_WIDTH{1'b0}};
            end
            else begin
              `ifdef ARA_REQ2MEM
                wval <= (msg_state_f == MSG_STATE_DATA && noc_io_go && !full);
              `else 
                wval <= (msg_state_f == MSG_STATE_DATA && noc_io_go && !full && msg_counter_f[0] == transaction_type_rd_data[0]);
              `endif 
                wdata <= {<<8{noc_data_in}}; // for endian conversion
            end
        end        
    end
    else if (AXI_LITE_DATA_WIDTH >= `NOC_DATA_WIDTH) begin
        always_ff@(posedge clk or negedge rst_n)
        begin
            if (!rst_n) begin
                wval <= 1'b0;
                wdata <= {AXI_LITE_DATA_WIDTH{1'b0}};
            end
            else begin
                if (msg_data_done) begin
                    wval <= 1'b1;
                    wdata <= data;
                end
                else begin 
                    wval <= 1'b0;
                    wdata <= {AXI_LITE_DATA_WIDTH{1'b0}};
                end
            end
        end

        always_comb begin 
            data[msg_counter_f*`NOC_DATA_WIDTH +: `NOC_DATA_WIDTH] = (msg_state_f == MSG_STATE_DATA && noc_io_go && !full) ? noc_data_in : 0;
        end 
    end
endgenerate

endmodule 