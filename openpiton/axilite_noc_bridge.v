`include "define.tmp.h"

module axilite_noc_bridge #(
    parameter AXI_LITE_DATA_WIDTH = 64
) (
    input  wire                                   clk,
    input  wire                                   rst,

    input wire                                    noc2_valid_in,
    input wire [`NOC_DATA_WIDTH-1:0]              noc2_data_in,
    output                                        noc2_ready_out,

    output  					                  noc2_valid_out,
    output [`NOC_DATA_WIDTH-1:0] 		          noc2_data_out,
    input wire 					                  noc2_ready_in,
   
    input wire 					                  noc3_valid_in,
    input wire [`NOC_DATA_WIDTH-1:0] 		      noc3_data_in,
    output       				                  noc3_ready_out,

    output   					                  noc3_valid_out,
    output [`NOC_DATA_WIDTH-1:0]     		      noc3_data_out,
    input wire      			                  noc3_ready_in,

    input wire [`MSG_SRC_CHIPID_WIDTH-1:0]        src_chipid,
    input wire [`MSG_SRC_X_WIDTH-1:0]             src_xpos,
    input wire [`MSG_SRC_Y_WIDTH-1:0]             src_ypos,
    input wire [`MSG_SRC_FBITS_WIDTH-1:0]         src_fbits,

    input wire [`MSG_DST_CHIPID_WIDTH-1:0]        dest_chipid,
    input wire [`MSG_DST_X_WIDTH-1:0]             dest_xpos,
    input wire [`MSG_DST_Y_WIDTH-1:0]             dest_ypos,
    input wire [`MSG_DST_FBITS_WIDTH-1:0]         dest_fbits,

    // AXI Write Address Channel Signals
    input  wire  [`C_M_AXI_LITE_ADDR_WIDTH-1:0]   m_axi_awaddr,
    input  wire                                   m_axi_awvalid,
    output wire                                   m_axi_awready,

    // AXI Write Data Channel Signals
    input  wire  [AXI_LITE_DATA_WIDTH-1:0]        m_axi_wdata,
    input  wire  [AXI_LITE_DATA_WIDTH/8-1:0]      m_axi_wstrb,
    input  wire                                   m_axi_wvalid,
    output wire                                   m_axi_wready,

    // AXI Read Address Channel Signals
    input  wire  [`C_M_AXI_LITE_ADDR_WIDTH-1:0]   m_axi_araddr,
    input  wire                                   m_axi_arvalid,
    output wire                                   m_axi_arready,

    // AXI Read Data Channel Signals
    output  reg [AXI_LITE_DATA_WIDTH-1:0]         m_axi_rdata,
    output  reg [`C_M_AXI_LITE_RESP_WIDTH-1:0]    m_axi_rresp,
    output  reg                                   m_axi_rvalid,
    input  wire                                   m_axi_rready,

    // AXI Write Response Channel Signals
    output  reg [`C_M_AXI_LITE_RESP_WIDTH-1:0]    m_axi_bresp,
    output  reg                                   m_axi_bvalid,
    input  wire                                   m_axi_bready
);

// States for Incoming Piton Messages
`define MSG_STATE_INVALID    3'd0 // Invalid Message
`define MSG_STATE_HEADER_0   3'd1 // Header 0
`define MSG_STATE_HEADER_1   3'd2 // Header 1
`define MSG_STATE_HEADER_2   3'd3 // Header 2
`define MSG_STATE_DATA       3'd4 // Data Lines

`define MSG_TYPE_INVAL       2'd0 // Invalid Message
`define MSG_TYPE_LOAD        2'd1 // Load Request
`define MSG_TYPE_STORE       2'd2 // Store Request

`define MIN_NOC_DATA_WIDTH      64 // 8 Bytes
`define NOC_HDR_LEN             3
`define MSG_STATE_IDLE          3'd0
`define MSG_STATE_WAIT_STRB     3'd1 // new state for converting strobe 
`define MSG_STATE_HEADER        3'd2
`define MSG_STATE_NOC_DATA      3'd3

localparam NOC_PAYLOAD_LEN = (AXI_LITE_DATA_WIDTH < `MIN_NOC_DATA_WIDTH) ?
                        3'b1 : AXI_LITE_DATA_WIDTH/`MIN_NOC_DATA_WIDTH;

/* flit fields */
reg [`NOC_DATA_WIDTH-1:0]               msg_address;
reg [`MSG_LENGTH_WIDTH-1:0]             msg_length;
reg [`MSG_TYPE_WIDTH-1:0]               msg_type;
reg [`MSG_MSHRID_WIDTH-1:0]             msg_mshrid;
reg [`MSG_OPTIONS_1]                    msg_options_1;
reg [`MSG_OPTIONS_2_]                   msg_options_2;
reg [`MSG_OPTIONS_3_]                   msg_options_3;
reg [`MSG_OPTIONS_4]                    msg_options_4;

reg [`MSG_LENGTH_WIDTH-1:0]             axi2noc_msg_counter;
wire                                    axi2noc_msg_type_store;
wire                                    axi2noc_msg_type_load;
wire [1:0]                              axi2noc_msg_type;
wire                                    axi_valid_ready;
reg [2:0]                               flit_state;
reg [2:0]                               flit_state_next;
reg                                     flit_ready;
reg [`NOC_DATA_WIDTH-1:0]               flit;
reg [`NOC_DATA_WIDTH-1:0]               noc_data;

wire                                    type_fifo_wval;
wire                                    type_fifo_full;
wire [1:0]                              type_fifo_wdata;
wire                                    type_fifo_empty;
wire [1:0]                              type_fifo_out;
reg                                     type_fifo_ren;

wire                                    awaddr_fifo_wval;
wire                                    awaddr_fifo_full;
wire [`C_M_AXI_LITE_ADDR_WIDTH-1:0]     awaddr_fifo_wdata;
wire                                    awaddr_fifo_empty;
wire [`C_M_AXI_LITE_ADDR_WIDTH-1:0]     awaddr_fifo_out;
reg                                     awaddr_fifo_ren;

wire                                    wdata_fifo_wval;
wire                                    wdata_fifo_full;
wire [`C_M_AXI_LITE_ADDR_WIDTH-1:0]     wdata_fifo_wdata;
wire                                    wdata_fifo_empty;
wire [`C_M_AXI_LITE_ADDR_WIDTH-1:0]     wdata_fifo_out;
reg                                     wdata_fifo_ren;

wire                                    araddr_fifo_wval;
wire                                    araddr_fifo_full;
wire [`C_M_AXI_LITE_ADDR_WIDTH-1:0]     araddr_fifo_wdata;
wire                                    araddr_fifo_empty;
wire [`C_M_AXI_LITE_ADDR_WIDTH-1:0]     araddr_fifo_out;
reg                                     araddr_fifo_ren;

wire [AXI_LITE_DATA_WIDTH/8 - 1: 0]     wstrb_fifo_out;
wire [AXI_LITE_DATA_WIDTH/8 - 1: 0]     wstrb_fifo_wdata; 
wire wstrb_fifo_full;
wire wstrb_fifo_empty;
wire wstrb_fifo_ren;
wire wstrb_fifo_wval;

wire                                    fifo_has_packet;
wire                                    noc_store_done;
wire                                    noc_load_done;
wire [`MIN_NOC_DATA_WIDTH-1:0]          out_data[0:NOC_PAYLOAD_LEN-1];
reg                                     noc_last_header;
reg                                     noc_last_data;
reg [2:0]                               noc_cnt;


/* Dump store addr and data to file. */
integer file;
initial begin
    file = $fopen("axilite_noc.log", "w");
end

always @(posedge clk)
begin
    if (awaddr_fifo_ren)
    begin
        $fwrite(file, "awaddr-fifo %064x\n", awaddr_fifo_out);
        $fflush(file);
    end
    if (wdata_fifo_ren)
    begin
        $fwrite(file, "wdata-fifo %064x\n", wdata_fifo_out);
        $fflush(file);
    end
    if (araddr_fifo_ren)
    begin
        $fwrite(file, "araddr-fifo %064x\n", araddr_fifo_out);
        $fflush(file);
    end
    /*if (noc2_valid_out && noc2_ready_in) begin
        $fwrite(file, "bridge-write-data %064x\n", noc2_data_out);
        $fflush(file);
    end */
end

/* Dump store addr and data to file. */
integer file3;
initial begin
    file3 = $fopen("axilite_noc2_store.log", "w");
end
always @(posedge clk)
begin
    if ((type_fifo_out == `MSG_TYPE_STORE) && noc2_valid_out && noc2_ready_in)
    begin
        $fwrite(file3, "noc2_data_out %064x\n", noc2_data_out);
        $fflush(file3);
    end
end

/* Dump store addr and data to file. */
integer file2;
initial begin
    file2 = $fopen("axilite_noc3_load.log", "w");
end
always @(posedge clk)
begin
    if ((type_fifo_out == `MSG_TYPE_LOAD) && noc3_valid_in && noc3_ready_out)
    begin
        $fwrite(file2, "noc3_data_in %064x\n", noc3_data_in);
        $fflush(file2);
    end
end

/* Dump axi read data to file. */
integer file4;
initial begin
    file4 = $fopen("axilite_read_data.log", "w");
end
always @(posedge clk)
begin
    if (m_axi_rvalid && m_axi_rready)
    begin
        $fwrite(file4, "rdata %x\n", m_axi_rdata);
        $fflush(file4);
    end
end

/******** Where the magic happens ********/
noc_response_axilite #(
    .AXI_LITE_DATA_WIDTH(AXI_LITE_DATA_WIDTH)
) noc_response_axilite(
    .clk(clk),
    .rst(rst),
    .noc_valid_in(noc3_valid_in),
    .noc_data_in(noc3_data_in),
    .noc_ready_out(noc3_ready_out),
    .noc_valid_out(),
    .noc_data_out(),
    .noc_ready_in(1'b1),
    .m_axi_rdata(m_axi_rdata),
    .m_axi_rresp(m_axi_rresp),
    .m_axi_rvalid(m_axi_rvalid),
    .m_axi_rready(m_axi_rready),
    .m_axi_bresp(m_axi_bresp),
    .m_axi_bvalid(m_axi_bvalid),
    .m_axi_bready(m_axi_bready),
    .w_reqbuf_size(),
    .r_reqbuf_size()
);


assign noc3_ready_out = 1'b1;

//assign write_channel_ready = !awaddr_fifo_full && !wdata_fifo_full;
assign write_channel_ready = !awaddr_fifo_full && !wdata_fifo_full && !wstrb_fifo_full;
assign m_axi_awready = write_channel_ready && !type_fifo_full;
assign m_axi_wready = write_channel_ready && !type_fifo_full;
assign m_axi_arready = !araddr_fifo_full && !type_fifo_full;

assign axi2noc_msg_type_store = m_axi_awvalid && m_axi_wvalid;
assign axi2noc_msg_type_load = m_axi_arvalid;
assign axi2noc_msg_type = (axi2noc_msg_type_store) ? `MSG_TYPE_STORE :
                            (axi2noc_msg_type_load) ? `MSG_TYPE_LOAD :
                                                        `MSG_TYPE_INVAL;

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
	.reset(rst)
);

assign type_fifo_wval = (axi2noc_msg_type_store ||axi2noc_msg_type_load) && !type_fifo_full;
assign type_fifo_wdata = (axi2noc_msg_type_store) ? `MSG_TYPE_STORE :
                            (axi2noc_msg_type_load) ? `MSG_TYPE_LOAD : `MSG_TYPE_INVAL;
//assign type_fifo_ren = (flit_state == `MSG_STATE_INVALID) && !type_fifo_empty;
// benefits of using XOR?
//assign type_fifo_ren = (noc_store_done ^ noc_load_done) && !type_fifo_empty;
assign type_fifo_ren = (noc_store_done ^ noc_load_done) && !type_fifo_empty;


/* fifo for storing addresses */
sync_fifo #(
	.DSIZE(`C_M_AXI_LITE_ADDR_WIDTH),
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
	.reset(rst)
);

assign awaddr_fifo_wval = m_axi_awvalid && m_axi_awready; //write_channel_ready;// && noc2_ready_in;
assign awaddr_fifo_wdata = m_axi_awaddr;
//assign awaddr_fifo_ren = (noc_store_done && !awaddr_fifo_empty); // noc_last_data only occurs for stores
assign awaddr_fifo_ren = (noc_store_done && !awaddr_fifo_empty);


/* fifo for wdata */
sync_fifo #(
	.DSIZE(`NOC_DATA_WIDTH),
	.ASIZE(5),
	.MEMSIZE(16) // should be 2 ^ (ASIZE-1)    
) waddr_fifo (
	.rdata(wdata_fifo_out),
	.empty(wdata_fifo_empty),
	.clk(clk),
	.ren(wdata_fifo_ren),
	.wdata(wdata_fifo_wdata),
	.full(wdata_fifo_full),
	.wval(wdata_fifo_wval),
	.reset(rst)
);

assign wdata_fifo_wval = m_axi_wvalid && m_axi_wready; // write_channel_ready;// && noc2_ready_in;
assign wdata_fifo_wdata = m_axi_wdata;
//assign wdata_fifo_ren = (noc_store_done && !wdata_fifo_empty); // noc_last_data only occurs for stores
assign wdata_fifo_ren = (noc_store_done && !wdata_fifo_empty);



/****************/ 
// declaration for strobe to mask conversion 
// control intreface signal
wire [2:0] pmesh_data_size;
wire [5:0] pmesh_addr;

reg[2:0] buf_pmesh_data_size;
reg [5:0] buf_pmesh_addr;

// input side 
wire wstrb_fifoside_valid;
wire wstrb_fifoside_ready;

// output side 
wire wstrb_outputside_valid;
wire wstrb_outputside_ready;

wire [AXI_LITE_DATA_WIDTH/8 - 1: 0] fifo_out_mux;
strb2mask strb2mask_ins (
    .clk (clk),
    .rst (rst),
    .m_axi_wstrb (fifo_out_mux),
    .pmesh_data_size (pmesh_data_size),
    .pmesh_addr (pmesh_addr), 
    .s_channel_valid (wstrb_fifoside_valid),
    .s_channel_ready(wstrb_fifoside_ready),
    .d_channel_ready(wstrb_outputside_ready), 
    .d_channel_valid(wstrb_outputside_valid)
);
assign fifo_out_mux = (wstrb_fifo_empty) ? 8'b1111_1111 : wstrb_fifo_out;

sync_fifo #(
	.DSIZE(AXI_LITE_DATA_WIDTH/8),
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
	.reset(rst)
);
assign wstrb_fifo_ren = (noc_store_done && !wstrb_fifo_empty);
assign wstrb_fifo_wval = m_axi_wvalid && m_axi_wready;
assign wstrb_fifo_wdata = m_axi_wstrb;
assign wstrb_outputside_ready = (flit_state == 3'd1) && (type_fifo_out == `MSG_TYPE_STORE);



reg [1:0] fifo_valid_state;
reg [1:0] fifo_valid_next_state; 
localparam IDLE = 0;
localparam VALID = 1;
localparam WAIT = 2;

//valid state machine 
always@ (posedge clk) begin
    if (rst) begin
        fifo_valid_state <= IDLE;
    end
    else begin
        fifo_valid_state <= fifo_valid_next_state;
    end
end
//state transfer logic 
always@ (*) begin
    if (fifo_valid_state == IDLE) begin
        if (fifo_has_packet && type_fifo_out == `MSG_TYPE_STORE) begin
            fifo_valid_next_state = VALID;
        end
        else fifo_valid_next_state = fifo_valid_state;
    end
    else if (fifo_valid_state == VALID) begin
        if (wstrb_fifoside_ready) fifo_valid_next_state = WAIT;
        else fifo_valid_next_state = fifo_valid_state;
    end
    else if (fifo_valid_state == WAIT) begin
        if (noc_store_done) fifo_valid_next_state = IDLE;
        else fifo_valid_next_state = fifo_valid_state;
    end
    else fifo_valid_next_state = fifo_valid_state;
end

assign wstrb_fifoside_valid = (fifo_valid_state == VALID);





/* fifo for read addr */
sync_fifo #(
	.DSIZE(`C_M_AXI_LITE_ADDR_WIDTH),
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
	.reset(rst)
);

assign araddr_fifo_wval = m_axi_arvalid && m_axi_arready;
assign araddr_fifo_wdata = m_axi_araddr;
assign araddr_fifo_ren = (noc_load_done && !araddr_fifo_empty);


/* start the state machine when fifo is not empty and noc is ready 
* We need to toggle between address and data fifos.
*/




/*assign fifo_has_packet = (type_fifo_out == `MSG_TYPE_STORE) ? (!awaddr_fifo_empty && !wdata_fifo_empty) :
                        (type_fifo_out == `MSG_TYPE_LOAD) ? !araddr_fifo_empty : 1'b0;*/
assign fifo_has_packet = (type_fifo_out == `MSG_TYPE_STORE) ? (!awaddr_fifo_empty && !wdata_fifo_empty && !wstrb_fifo_empty) :
                           (type_fifo_out == `MSG_TYPE_LOAD) ? !araddr_fifo_empty : 1'b0;

assign noc_store_done = noc_last_data && type_fifo_out == `MSG_TYPE_STORE && ~wstrb_outputside_valid;
//assign noc_store_done = noc_last_data && type_fifo_out == `MSG_TYPE_STORE
assign noc_load_done = noc_last_header && type_fifo_out == `MSG_TYPE_LOAD;



generate begin
    genvar k;
    if (AXI_LITE_DATA_WIDTH < `MIN_NOC_DATA_WIDTH) begin
        for (k=0; k<`MIN_NOC_DATA_WIDTH/AXI_LITE_DATA_WIDTH; k = k + 1)
        begin: DATA_GEN
            assign out_data[(k+1)*AXI_LITE_DATA_WIDTH-1 : k*AXI_LITE_DATA_WIDTH] = wdata_fifo_out;
        end
    end
    else begin
        for (k=0; k<NOC_PAYLOAD_LEN; k = k + 1)
        begin: DATA_GEN
            assign out_data[k] = wdata_fifo_out[(k+1)*`MIN_NOC_DATA_WIDTH-1 : k*`MIN_NOC_DATA_WIDTH];
        end
    end
end
endgenerate

reg [2:0]           msg_data_size;

/* set defaults for the flit */
always @(*)
begin
    case (type_fifo_out)
        `MSG_TYPE_STORE: begin
            msg_type = `MSG_TYPE_NC_STORE_REQ; // axilite peripheral is writing to the memory?
            msg_length = 2'd2 + NOC_PAYLOAD_LEN; // 2 extra headers + 1 data
            msg_data_size = buf_pmesh_data_size; // fix it for now
            //msg_data_size = 3'b001;
            msg_address = {{`MSG_ADDR_WIDTH-`PHY_ADDR_WIDTH{1'b0}}, awaddr_fifo_out[`PHY_ADDR_WIDTH-1:3], buf_pmesh_addr[2:0]};
        end

        `MSG_TYPE_LOAD: begin
            msg_type = `MSG_TYPE_NC_LOAD_REQ; // axilite peripheral is reading from the memory?
            msg_length = 2'd2; // only 2 extra headers
            msg_data_size = `MSG_DATA_SIZE_8B; // fix it for now. 
            //msg_data_size = 3'b001;
            msg_address = {{`MSG_ADDR_WIDTH-`PHY_ADDR_WIDTH{1'b0}}, araddr_fifo_out[`PHY_ADDR_WIDTH-1:0]};
        end
        
        default: begin
            msg_length = 2'b0;
            msg_data_size = `MSG_DATA_SIZE_8B;
            //msg_data_size = 3'b001;
        end
    endcase
end

always @(posedge clk)
begin
    if (rst) begin
        noc_cnt <= 3'b0;
    end
    else begin
        noc_cnt <= (noc_last_header | noc_last_data)  ? 3'b0 :
                    (fifo_has_packet && noc2_ready_in) ? noc_cnt + 1 : noc_cnt;
    end
end

always@(posedge clk) begin
    if (rst) begin
        buf_pmesh_addr <= 0;
        buf_pmesh_data_size <= 0;
    end
    else if ((flit_state == `MSG_STATE_WAIT_STRB) & (wstrb_outputside_ready & wstrb_outputside_valid)) begin
        buf_pmesh_addr <= pmesh_addr;
        buf_pmesh_data_size <= pmesh_data_size;
    end
end

always @(*)
begin
    noc_last_header = 1'b0;
    noc_last_data = 1'b0;
    if (noc2_ready_in) begin
        noc_last_header = (flit_state == `MSG_STATE_HEADER &&
                                    noc_cnt == `NOC_HDR_LEN) ? 1'b1 : 1'b0;
        noc_last_data = (flit_state == `MSG_STATE_NOC_DATA &&
                                    noc_cnt == NOC_PAYLOAD_LEN-1) ? 1'b1 : 1'b0;
    end
end



always @(posedge clk)
begin
    if (rst) begin
        flit_state <= `MSG_STATE_IDLE;
    end
    else begin
        case (flit_state)
            `MSG_STATE_IDLE: begin
                if ((fifo_has_packet && type_fifo_out == `MSG_TYPE_STORE) && noc2_ready_in)
                    //flit_state <= `MSG_STATE_HEADER;
                    flit_state <= `MSG_STATE_WAIT_STRB;
                else if ((fifo_has_packet && type_fifo_out == `MSG_TYPE_LOAD) && noc2_ready_in)
                    flit_state <= `MSG_STATE_HEADER;
            end
            `MSG_STATE_WAIT_STRB:begin
                if (wstrb_outputside_ready & wstrb_outputside_valid)
                    flit_state <= `MSG_STATE_HEADER;
            end
            `MSG_STATE_HEADER: begin
                if (noc_last_header && type_fifo_out == `MSG_TYPE_STORE)
                    flit_state <= `MSG_STATE_NOC_DATA;
                else if (noc_load_done)
                    flit_state <= `MSG_STATE_IDLE;
            end

            `MSG_STATE_NOC_DATA: begin
                if (noc_store_done)
                    flit_state <= `MSG_STATE_IDLE;
                else 
                    flit_state <= `MSG_STATE_WAIT_STRB;
            end
        endcase
    end
end

always @(*)
begin
    msg_mshrid = {`MSG_MSHRID_WIDTH{1'b0}};
    msg_options_1 = {`MSG_OPTIONS_1_WIDTH{1'b0}};
    msg_options_2 = 16'b0;
    msg_options_3 = 30'b0;
    case (flit_state)
        `MSG_STATE_HEADER: begin
            case (noc_cnt)
                3'b001: begin
                    flit[`MSG_DST_CHIPID] = dest_chipid;
                    flit[`MSG_DST_X] = dest_xpos;
                    flit[`MSG_DST_Y] = dest_ypos;
                    flit[`MSG_DST_FBITS] = dest_fbits; // towards memory?
                    flit[`MSG_LENGTH] = msg_length;
                    flit[`MSG_TYPE] = msg_type;
                    flit[`MSG_MSHRID] = msg_mshrid;
                    flit[`MSG_OPTIONS_1] = msg_options_1;
                    flit_ready = 1'b1;
                end

                3'b010: begin
                    flit[`MSG_ADDR_] = msg_address;
                    flit[`MSG_OPTIONS_2_] = msg_options_2;
                    flit[`MSG_DATA_SIZE_] = msg_data_size;
                    flit_ready = 1'b1;                  
                end

                3'b011: begin
                    flit[`MSG_SRC_CHIPID_] = src_chipid;
                    flit[`MSG_SRC_X_] = src_xpos;
                    flit[`MSG_SRC_Y_] = src_ypos;
                    flit[`MSG_SRC_FBITS_] = src_fbits;
                    flit[`MSG_OPTIONS_3_] = msg_options_3;
                    flit_ready = 1'b1;
                end
            endcase
        end

        `MSG_STATE_NOC_DATA: begin
            flit[`NOC_DATA_WIDTH-1:0] = out_data[noc_cnt]; //wdata_fifo_out;
            flit_ready = 1'b1;
        end

        default: begin
            flit[`NOC_DATA_WIDTH-1:0] = {`NOC_DATA_WIDTH{1'b0}};
            flit_ready = 1'b0;
        end
    endcase
end

assign noc2_valid_out = flit_ready;
assign noc2_data_out = flit;

endmodule