// Copyright 2022 Polytechnique Montreal
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//
// Author: MohammadHossein AskariHemmat
// Date: 19.12.2022
// Description: Ara Top-level wrapper to connect to OpenPiton NoC.
module ara_verilog_wrap
  import axi_pkg::*; 
  import ara_pkg::*;
  import ariane_pkg::*;
  #(
    // RVV Parameters
    parameter int unsigned               NrLanes               = 4,// Number of parallel vector lanes.
    // Support for floating-point data types
    parameter fpu_support_e              FPUSupport            = FPUSupportNone,
    // External support for vfrec7, vfrsqrt7
    parameter fpext_support_e            FPExtSupport          = FPUSupportNone,
    // Support for fixed-point data types
    parameter fixpt_support_e            FixPtSupport          = FPUSupportNone,
    // AXI Interface
    parameter int unsigned               AxiDataWidth          = 32*NrLanes,
    parameter int unsigned               AxiAddrWidth          = 64,
    parameter int unsigned               AxiUserWidth          = 1,
    parameter int unsigned               AxiIdWidth            = 5, // only one slave and one master
    // Ariane Cache
    parameter int unsigned               RASDepth              = 2,
    parameter int unsigned               BTBEntries            = 32,
    parameter int unsigned               BHTEntries            = 128,
    // debug module base address
    parameter logic [63:0]               DmBaseAddress         = 64'h0,
    // swap endianess in l15 adapter
    parameter bit                        SwapEndianess         = 1,
    // PMA configuration
    // idempotent region
    parameter int unsigned               NrNonIdempotentRules  =  1,
    parameter logic [NrMaxRules*64-1:0]  NonIdempotentAddrBase = 64'h00C0000000,
    parameter logic [NrMaxRules*64-1:0]  NonIdempotentLength   = 64'hFFFFFFFFFF,
    // executable regions
    parameter int unsigned               NrExecuteRegionRules  =  0,
    parameter logic [NrMaxRules*64-1:0]  ExecuteRegionAddrBase = '0,
    parameter logic [NrMaxRules*64-1:0]  ExecuteRegionLength   = '0,
    // cacheable regions
    parameter int unsigned               NrCachedRegionRules   =  0,
    parameter logic [NrMaxRules*64-1:0]  CachedRegionAddrBase  = '0,
    parameter logic [NrMaxRules*64-1:0]  CachedRegionLength    = '0,
    // PMP
    parameter int unsigned               NrPMPEntries          =  8
  ) (
    input                       clk_i,
    input                       reset_l,      // this is an openpiton-specific name, do not change (hier. paths in TB use this)
    output                      spc_grst_l,   // this is an openpiton-specific name, do not change (hier. paths in TB use this)
    // Core ID, Cluster ID and boot address are considered more or less static
    input  [riscv::VLEN-1:0]    boot_addr_i,  // reset boot address
    input  [riscv::XLEN-1:0]    hart_id_i,    // hart id in a multicore environment (reflected in a CSR)
    input wire [`NOC_CHIPID_WIDTH-1:0]  default_chipid,
    input wire [`NOC_X_WIDTH-1:0]       default_coreid_x,
    input wire [`NOC_Y_WIDTH-1:0]       default_coreid_y,
    
    // Interrupt inputs
    input  [1:0]                irq_i,        // level sensitive IR lines, mip & sip (async)
    input                       ipi_i,        // inter-processor interrupts (async)
    // Timer facilities
    input                       time_irq_i,   // timer interrupt in (async)
    input                       debug_req_i,  // debug request (async)

    // NOC Signals

`ifdef ARA_REQ2MEM // NoC channel direct memory request to main memory
    output [`NOC_DATA_WIDTH-1:0] noc2_valid_out,
    output [`NOC_DATA_WIDTH-1:0] noc2_data_out ,
    input  [`NOC_DATA_WIDTH-1:0] noc2_ready_in ,

    input  [`NOC_DATA_WIDTH-1:0] noc3_valid_in ,
    input  [`NOC_DATA_WIDTH-1:0] noc3_data_in  ,
    output [`NOC_DATA_WIDTH-1:0] noc3_ready_out,
`else // NoC channel for memory request to L2 Cache 
    output                       noc1_valid_out,
    output [`NOC_DATA_WIDTH-1:0] noc1_data_out ,
    input                        noc1_ready_in , 

    input                        noc2_valid_in ,
    input  [`NOC_DATA_WIDTH-1:0] noc2_data_in  ,
    output                       noc2_ready_out,
    input wire [`HOME_ID_WIDTH-1:0] config_system_tile_count_5_0,
    input wire [`HOME_ALLOC_METHOD_WIDTH-1:0] config_home_alloc_method,
`endif 



  `ifdef PITON_ARIANE
    // L15 (memory side)
    // output [$size(wt_cache_pkg::l15_req_t)-1:0]  l15_req_o,
    // input  [$size(wt_cache_pkg::l15_rtrn_t)-1:0] l15_rtrn_i
    output wt_cache_pkg::l15_req_t    l15_req_o,
    input  wt_cache_pkg::l15_rtrn_t   l15_rtrn_i
  `else
    // AXI (memory side)
    output [$size(ariane_axi::req_t)-1:0]             axi_req_o,
    input  [$size(ariane_axi::resp_t)-1:0]            axi_resp_i
  `endif
   );
  

   // assign bitvector to packed struct and vice versa
  `ifdef PITON_ARIANE
    // L15 (memory side)
    wt_cache_pkg::l15_req_t  l15_req;
    wt_cache_pkg::l15_rtrn_t l15_rtrn;
  
    assign l15_req_o = l15_req;
    assign l15_rtrn  = l15_rtrn_i;
  `else
    ariane_axi::req_t             axi_req;
    ariane_axi::resp_t            axi_resp;
  
    assign axi_req_o = axi_req;
    assign axi_resp  = axi_resp_i;
  `endif

  //`ifndef ARIANE_ACCELERATOR_PORT
  //  `define ARIANE_ACCELERATOR_PORT
  //`endif

  `include "axi/assign.svh"
  `include "axi/typedef.svh"
  

  localparam int unsigned AxiStrbWidth = AxiDataWidth / 32'd8;
  typedef logic [AxiIdWidth-1:0] axi_id_t;
  typedef logic [AxiAddrWidth-1:0] axi_addr_t;
  typedef logic [AxiUserWidth-1:0] axi_user_t;
  typedef logic [AxiDataWidth-1:0] axi_data_t;
  typedef logic [AxiStrbWidth-1:0] axi_strb_t;
  ///////////
  //  AXI  //
  ///////////
  // Ariane's AXI port data width
  localparam AxiNarrowDataWidth = 64;
  localparam AxiNarrowStrbWidth = AxiNarrowDataWidth / 8;
  // Ara's AXI port data width
  localparam AxiWideDataWidth   = AxiDataWidth;
  localparam AXiWideStrbWidth   = AxiWideDataWidth / 8;

  //localparam AxiSocIdWidth  = AxiIdWidth - $clog2(NrAXIMasters);
  //localparam AxiCoreIdWidth = AxiSocIdWidth - 1;
  localparam AxiSocIdWidth  = AxiIdWidth;
  localparam AxiCoreIdWidth = AxiIdWidth;

  // Internal types
  typedef logic [AxiNarrowDataWidth-1:0] axi_narrow_data_t;
  typedef logic [AxiNarrowStrbWidth-1:0] axi_narrow_strb_t;
  typedef logic [AxiSocIdWidth-1:0] axi_soc_id_t;
  typedef logic [AxiCoreIdWidth-1:0] axi_core_id_t;

  ///////////
  //  AXI  //
  ///////////
  // AXI Typedefs
  `AXI_TYPEDEF_ALL(system, axi_addr_t, axi_id_t, axi_data_t, axi_strb_t, axi_user_t)
  `AXI_TYPEDEF_ALL(ara_axi, axi_addr_t, axi_core_id_t, axi_data_t, axi_strb_t, axi_user_t)
  // `AXI_TYPEDEF_ALL(ariane_axi, axi_addr_t, axi_core_id_t, axi_narrow_data_t, axi_narrow_strb_t, axi_user_t)

  `AXI_TYPEDEF_ALL(soc_narrow, axi_addr_t, axi_soc_id_t, axi_narrow_data_t, axi_narrow_strb_t,
  axi_user_t)
  `AXI_TYPEDEF_ALL(soc_wide, axi_addr_t, axi_soc_id_t, axi_data_t, axi_strb_t, axi_user_t)
  `AXI_LITE_TYPEDEF_ALL(soc_narrow_lite, axi_addr_t, axi_narrow_data_t, axi_narrow_strb_t)
  

  system_req_t system_axi_req;
  system_resp_t system_axi_resp;
  ara_axi_req_t  ara_axi_req;
  ara_axi_resp_t ara_axi_resp;

   
  
    //////////////////////
    //  Ara and Ariane  //
    //////////////////////
  
    import acc_pkg::accelerator_req_t;
    import acc_pkg::accelerator_resp_t;
  
    // Accelerator ports
    accelerator_req_t                     acc_req;
    accelerator_resp_t                    acc_resp;
  
  
    /////////////////////////////
    // Core wakeup mechanism
    /////////////////////////////
  
    // // this is a workaround since interrupts are not fully supported yet.
    // // the logic below catches the initial wake up interrupt that enables the cores.
    // logic wake_up_d, wake_up_q;
    // logic rst_n;
  
    // assign wake_up_d = wake_up_q || ((l15_rtrn.l15_returntype == wt_cache_pkg::L15_INT_RET) && l15_rtrn.l15_val);
  
    // always_ff @(posedge clk_i or negedge reset_l) begin : p_regs
    //   if(~reset_l) begin
    //     wake_up_q <= 0;
    //   end else begin
    //     wake_up_q <= wake_up_d;
    //   end
    // end
  
    // // reset gate this
    // assign rst_n = wake_up_q & reset_l;
  
    // this is a workaround,
    // we basically wait for 32k cycles such that the SRAMs in openpiton can initialize
    // 128KB..8K cycles
    // 256KB..16K cycles
    // etc, so this should be enough for 512k per tile
  
    logic [15:0] wake_up_cnt_d, wake_up_cnt_q;
    logic rst_n;
  
    assign wake_up_cnt_d = (wake_up_cnt_q[$high(wake_up_cnt_q)]) ? wake_up_cnt_q : wake_up_cnt_q + 1;
  
    always_ff @(posedge clk_i or negedge reset_l) begin : p_regs
      if(~reset_l) begin
        wake_up_cnt_q <= 0;
      end else begin
        wake_up_cnt_q <= wake_up_cnt_d;
      end
    end
  
    // reset gate this
    assign rst_n = wake_up_cnt_q[$high(wake_up_cnt_q)] & reset_l;
  
  
    /////////////////////////////
    // synchronizers
    /////////////////////////////
    
    logic rst_ni;
    logic [1:0] irq;
    logic ipi, time_irq, debug_req;
  
    // reset synchronization
    synchronizer i_sync (
      .clk         ( clk_i      ),
      .presyncdata ( rst_n      ),
      .syncdata    ( spc_grst_l )
    );

    assign rst_ni = spc_grst_l;
  
    // interrupts
    for (genvar k=0; k<$size(irq_i); k++) begin
      synchronizer i_irq_sync (
        .clk         ( clk_i      ),
        .presyncdata ( irq_i[k]   ),
        .syncdata    ( irq[k]     )
      );
    end
  
    synchronizer i_ipi_sync (
      .clk         ( clk_i      ),
      .presyncdata ( ipi_i      ),
      .syncdata    ( ipi        )
    );
  
    synchronizer i_timer_sync (
      .clk         ( clk_i      ),
      .presyncdata ( time_irq_i ),
      .syncdata    ( time_irq   )
    );
  
    synchronizer i_debug_sync (
      .clk         ( clk_i       ),
      .presyncdata ( debug_req_i ),
      .syncdata    ( debug_req   )
    );
  
    /////////////////////////////
    // ariane instance
    /////////////////////////////
    localparam Axi64BitCompliant = 1'b0; // for compiling correct 
    localparam ariane_pkg::ariane_cfg_t ArianeOpenPitonCfg = '{
      RASDepth:              RASDepth,
      BTBEntries:            BTBEntries,
      BHTEntries:            BHTEntries,
      // idempotent region
      NrNonIdempotentRules:  NrNonIdempotentRules,
      NonIdempotentAddrBase: NonIdempotentAddrBase,
      NonIdempotentLength:   NonIdempotentLength,
      NrExecuteRegionRules:  NrExecuteRegionRules,
      ExecuteRegionAddrBase: ExecuteRegionAddrBase,
      ExecuteRegionLength:   ExecuteRegionLength,
      // cached region
      NrCachedRegionRules:   NrCachedRegionRules,
      CachedRegionAddrBase:  CachedRegionAddrBase,
      CachedRegionLength:    CachedRegionLength,
      // cache config
      AxiCompliant:          Axi64BitCompliant, // Not sure it's a typo (Original: Axi64BitCompliant:1'b0 )
      SwapEndianess:         SwapEndianess,
      // debug
      DmBaseAddress:         DmBaseAddress,
      NrPMPEntries:          NrPMPEntries
    };

  localparam ariane_pkg::cva6_cfg_t CVA6Cfg = '{
    NrCommitPorts: 2                 ,
    IsRVFI       : 0                 ,
    AxiAddrWidth : AxiAddrWidth      ,
    AxiDataWidth : AxiNarrowDataWidth,
    AxiIdWidth   : AxiIdWidth        ,
    AxiUserWidth : AxiUserWidth       
  };

  ariane #(
    .CVA6Cfg          (CVA6Cfg                    ),
    .cvxif_req_t      (acc_pkg::accelerator_req_t ),
    .cvxif_resp_t     (acc_pkg::accelerator_resp_t),
    .ArianeCfg        (ArianeOpenPitonCfg         ),
    .noc_req_t        (wt_cache_pkg::l15_req_t    ),
    .noc_resp_t       (wt_cache_pkg::l15_rtrn_t   )
  ) ariane (
    .clk_i            (clk_i                 ),
    .rst_ni           (spc_grst_l            ),
    .boot_addr_i,
    .hart_id_i  ,
    .irq_i            (irq                   ),
    .ipi_i            (ipi                   ),
    .time_irq_i       (time_irq              ),
    .debug_req_i      (debug_req             ),
    .cvxif_req_o      (acc_req               ),
    .cvxif_resp_i     (acc_resp              ),
    .noc_req_o        (l15_req               ),
    .noc_resp_i       (l15_rtrn              )
  );

  logic  scan_enable_i;
  assign scan_enable_i = 0;

  ara #(
    .NrLanes     (NrLanes          ),
    .FPUSupport  (FPUSupport       ),
    .FPExtSupport(FPExtSupport     ),
    .FixPtSupport(FixPtSupport     ),
    .AxiDataWidth(AxiWideDataWidth ),
    .AxiAddrWidth(AxiAddrWidth     ),
    .axi_ar_t    (ara_axi_ar_chan_t),
    .axi_r_t     (ara_axi_r_chan_t ),
    .axi_aw_t    (ara_axi_aw_chan_t),
    .axi_w_t     (ara_axi_w_chan_t ),
    .axi_b_t     (ara_axi_b_chan_t ),
    .axi_req_t   (ara_axi_req_t    ),
    .axi_resp_t  (ara_axi_resp_t   )
  ) i_ara (
    .clk_i           (clk_i         ),
    .rst_ni          (rst_ni        ),
    .scan_enable_i   (scan_enable_i ),
    .scan_data_i     (1'b0          ),
    .scan_data_o     (/* Unused */  ),
    .acc_req_i       (acc_req       ),
    .acc_resp_o      (acc_resp      ),
    .axi_req_o       (ara_axi_req   ),
    .axi_resp_i      (ara_axi_resp  )
  );
  
  // Buses
  soc_narrow_req_t  periph_narrow_axi_req;
  soc_narrow_resp_t periph_narrow_axi_resp;
  soc_narrow_req_t  periph_cut_narrow_axi_req;
  soc_narrow_resp_t periph_cut_narrow_axi_resp;

  /////////////////////////
  //  NOC Bridge         //
  /////////////////////////

  soc_narrow_lite_req_t  axi_lite_noc_req;
  soc_narrow_lite_resp_t axi_lite_noc_resp;

`ifdef AXI_LITE_BRIDGE 
  axi_dw_converter #(
    .AxiSlvPortDataWidth(AxiWideDataWidth    ),
    .AxiMstPortDataWidth(AxiNarrowDataWidth  ),
    .AxiAddrWidth       (AxiAddrWidth        ),
    .AxiIdWidth         (AxiCoreIdWidth       ),
    .AxiMaxReads        (2                  ),
    //.ar_chan_t          (soc_wide_ar_chan_t  ),
    .ar_chan_t          (ara_axi_ar_chan_t  ),
    .mst_r_chan_t       (soc_narrow_r_chan_t ),
    //.slv_r_chan_t       (soc_wide_r_chan_t   ),
    .slv_r_chan_t       (ara_axi_r_chan_t   ),
    .aw_chan_t          (soc_narrow_aw_chan_t),
    .b_chan_t           (soc_narrow_b_chan_t ),
    .mst_w_chan_t       (soc_narrow_w_chan_t ),
    //.slv_w_chan_t       (soc_wide_w_chan_t   ),
    .slv_w_chan_t       (ara_axi_w_chan_t   ),
    .axi_mst_req_t      (soc_narrow_req_t    ),
    .axi_mst_resp_t     (soc_narrow_resp_t   ),
    //.axi_slv_req_t      (soc_wide_req_t      ),
    //.axi_slv_resp_t     (soc_wide_resp_t     )
    .axi_slv_req_t      (ara_axi_req_t      ),
    .axi_slv_resp_t     (ara_axi_resp_t     )
  ) i_axi_slave_ctrl_dwc_noc (
    .clk_i     (clk_i                       ),
    .rst_ni    (rst_ni                      ),
    //.slv_req_i (periph_wide_axi_req[NOC]    ),
    //.slv_resp_o(periph_wide_axi_resp[NOC]   ),
    .slv_req_i (ara_axi_req                 ),
    .slv_resp_o(ara_axi_resp                ),
    .mst_req_o (periph_narrow_axi_req  ),
    .mst_resp_i(periph_narrow_axi_resp )
  );
  axi_to_axi_lite #(
    .AxiAddrWidth   (AxiAddrWidth          ),
    .AxiDataWidth   (AxiNarrowDataWidth    ),
    .AxiIdWidth     (AxiSocIdWidth         ),
    .AxiUserWidth   (AxiUserWidth          ),
    .AxiMaxReadTxns (1                     ),
    .AxiMaxWriteTxns(1                     ),
    .FallThrough    (1'b0                  ),
    .full_req_t     (soc_narrow_req_t      ),
    .full_resp_t    (soc_narrow_resp_t     ),
    .lite_req_t     (soc_narrow_lite_req_t ),
    .lite_resp_t    (soc_narrow_lite_resp_t)
  ) i_axi_to_axi_lite_noc (
    .clk_i     (clk_i                        ),
    .rst_ni    (rst_ni                       ),
    .test_i    (1'b0                         ),
    .slv_req_i (periph_narrow_axi_req   ),
    .slv_resp_o(periph_narrow_axi_resp  ),
    .mst_req_o (axi_lite_noc_req             ),
    .mst_resp_i(axi_lite_noc_resp            )
  );

axilite_noc_bridge #(
  .AXI_LITE_DATA_WIDTH (AxiNarrowDataWidth), 
  .AXI_LITE_ADDR_WIDTH (AxiAddrWidth), 
  .AXI_LITE_RESP_WIDTH (2)
) axilite_noc_bridge (
    .clk           (clk_i),
    .rst_n         (rst_ni),
  `ifdef ARA_REQ2MEM // direct memory request to main memory 
    .noc_valid_out (noc2_valid_out          ),
    .noc_data_out  (noc2_data_out           ),
    .noc_ready_in  (noc2_ready_in           ),
    .noc_valid_in  (noc3_valid_in),
    .noc_data_in   (noc3_data_in),
    .noc_ready_out (noc3_ready_out),
  `else 
    .noc_valid_out (noc1_valid_out          ),
    .noc_data_out  (noc1_data_out           ),
    .noc_ready_in  (noc1_ready_in           ),
    .noc_valid_in  (noc2_valid_in),
    .noc_data_in   (noc2_data_in),
    .noc_ready_out (noc2_ready_out),
  `endif 
    .src_chipid     (default_chipid), // 14 // 0
    .src_xpos       (default_coreid_x), // 8  // 1
    .src_ypos       (default_coreid_y), // 8  // 1
    .src_fbits      (`NOC_FBITS_ARA), // 4
  `ifdef ARA_REQ2MEM // direct memory request to main memory 
    .dest_chipid    (14'b1000_0000_0000_00), // 14 //  all zero 
    .dest_xpos      (8'd0), // 8
    .dest_ypos      (8'd0), // 8
    .dest_fbits     (`NOC_FBITS_MEM), //4
  `else // memory request to L2 cache 
    .dest_chipid    (14'b0000_0000_0000_00), // 14
    .dest_xpos      (8'd0), // 8
    .dest_ypos      (8'd0), // 8
    .dest_fbits     (`NOC_FBITS_L2), //4
    .system_tile_count (config_system_tile_count_5_0),
    .home_alloc_method (config_home_alloc_method),
  `endif 
    .m_axi_awaddr   (axi_lite_noc_req.aw.addr  ),
    .m_axi_awvalid  (axi_lite_noc_req.aw_valid ),
    .m_axi_awready  (axi_lite_noc_resp.aw_ready),
    .m_axi_wdata    (axi_lite_noc_req.w.data   ),
    .m_axi_wstrb    (axi_lite_noc_req.w.strb   ),
    .m_axi_wvalid   (axi_lite_noc_req.w_valid  ),
    .m_axi_wready   (axi_lite_noc_resp.w_ready ),
    .m_axi_araddr   (axi_lite_noc_req.ar.addr  ),
    .m_axi_arvalid  (axi_lite_noc_req.ar_valid ),
    .m_axi_arready  (axi_lite_noc_resp.ar_ready),
    .m_axi_rdata    (axi_lite_noc_resp.r.data  ),
    .m_axi_rresp    (axi_lite_noc_resp.r.resp  ),
    .m_axi_rvalid   (axi_lite_noc_resp.r_valid ),
    .m_axi_rready   (axi_lite_noc_req.r_ready  ),
    .m_axi_bresp    (axi_lite_noc_resp.b.resp  ),
    .m_axi_bvalid   (axi_lite_noc_resp.b_valid ),
    .m_axi_bready   (axi_lite_noc_req.b_ready  )
);
`elsif AXI128_BRIDGE
  axi_noc_bridge #(
    .AXI_DATA_WIDTH (AxiDataWidth  ),
    .AXI_ADDR_WIDTH (AxiAddrWidth  )
  )i_axi_noc_bridge
  (
    .clk     (clk_i                        ),
    .rst_n    (rst_ni                       ),
  `ifdef ARA_REQ2MEM // direct memory request to main memory 
    .noc_valid_out (noc2_valid_out          ),
    .noc_data_out  (noc2_data_out           ),
    .noc_ready_in  (noc2_ready_in           ),
    .noc_valid_in  (noc3_valid_in),
    .noc_data_in   (noc3_data_in),
    .noc_ready_out (noc3_ready_out),
  `else 
    .noc_valid_out (noc1_valid_out          ),
    .noc_data_out  (noc1_data_out           ),
    .noc_ready_in  (noc1_ready_in           ),
    .noc_valid_in  (noc2_valid_in),
    .noc_data_in   (noc2_data_in),
    .noc_ready_out (noc2_ready_out),
  `endif 
    .src_chipid     (default_chipid), // 14 // 0
    .src_xpos       (default_coreid_x), // 8  // 1
    .src_ypos       (default_coreid_y), // 8  // 1
    .src_fbits      (`NOC_FBITS_ARA), // 4
  `ifdef ARA_REQ2MEM // direct memory request to main memory 
    .dest_chipid    (14'b1000_0000_0000_00), // 14 //  all zero 
    .dest_xpos      (8'd0), // 8
    .dest_ypos      (8'd0), // 8
    .dest_fbits     (`NOC_FBITS_MEM), //4
  `else // memory request to L2 cache 
    .dest_chipid    (14'b0000_0000_0000_00), // 14
    .dest_xpos      (8'd0), // 8
    .dest_ypos      (8'd0), // 8
    .dest_fbits     (`NOC_FBITS_L2), //4
    .system_tile_count (config_system_tile_count_5_0),
    .home_alloc_method (config_home_alloc_method),
  `endif 
    // write address channel
    .m_axi_awaddr   (ara_axi_req.aw.addr ),
    .m_axi_awlen    (ara_axi_req.aw.len  ),
    .m_axi_awsize   (ara_axi_req.aw.size ),
    .m_axi_awburst  (ara_axi_req.aw.burst),
    .m_axi_awcache (ara_axi_req.aw.cache),
    .m_axi_awid     (ara_axi_req.aw.id),
    // handshake logic
    .m_axi_awvalid  (ara_axi_req.aw_valid ),
    .m_axi_awready  (ara_axi_resp.aw_ready),

    //write data channel
    .m_axi_wdata    (ara_axi_req.w.data   ),
    .m_axi_wstrb    (ara_axi_req.w.strb   ),
    .m_axi_wlast    (ara_axi_req.w.last   ),
    // handshake logic
    .m_axi_wvalid   (ara_axi_req.w_valid  ),
    .m_axi_wready   (ara_axi_resp.w_ready ),

    // write response channel
    .m_axi_bresp    (ara_axi_resp.b.resp  ),
    .m_axi_bid      (ara_axi_resp.b.id ),
    .m_axi_bvalid   (ara_axi_resp.b_valid  ),
    .m_axi_bready   (ara_axi_req.b_ready ),

    // read address channel 
    .m_axi_araddr   (ara_axi_req.ar.addr  ),
    .m_axi_arlen    (ara_axi_req.ar.len   ), 
    .m_axi_arsize   (ara_axi_req.ar.size  ), 
    .m_axi_arburst  (ara_axi_req.ar.burst ), 
    .m_axi_arcache (ara_axi_req.ar.cache ), 
    .m_axi_arid     (ara_axi_req.ar.id),
    // handshake logic 
    .m_axi_arvalid  (ara_axi_req.ar_valid ),
    .m_axi_arready  (ara_axi_resp.ar_ready),

    // read data channel
    .m_axi_rdata    (ara_axi_resp.r.data  ),
    .m_axi_rresp    (ara_axi_resp.r.resp  ),
    .m_axi_rlast    (ara_axi_resp.r.last  ),
    .m_axi_ruser    (ara_axi_resp.r.user),
    .m_axi_rid      (ara_axi_resp.r.id),
    // handshake logic
    .m_axi_rvalid   (ara_axi_resp.r_valid ),
    .m_axi_rready   (ara_axi_req.r_ready  )
  );
`elsif ARA_REQ2MEM
  axi_dw_converter #(
    .AxiSlvPortDataWidth(AxiWideDataWidth    ),
    .AxiMstPortDataWidth(AxiNarrowDataWidth  ),
    .AxiAddrWidth       (AxiAddrWidth        ),
    .AxiIdWidth         (AxiCoreIdWidth       ),
    .AxiMaxReads        (1                  ),
    .ar_chan_t          (ara_axi_ar_chan_t  ),
    .mst_r_chan_t       (soc_narrow_r_chan_t ),
    .slv_r_chan_t       (ara_axi_r_chan_t   ),
    .aw_chan_t          (soc_narrow_aw_chan_t),
    .b_chan_t           (soc_narrow_b_chan_t ),
    .mst_w_chan_t       (soc_narrow_w_chan_t ),
    .slv_w_chan_t       (ara_axi_w_chan_t   ),
    .axi_mst_req_t      (soc_narrow_req_t    ),
    .axi_mst_resp_t     (soc_narrow_resp_t   ),
    .axi_slv_req_t      (ara_axi_req_t      ),
    .axi_slv_resp_t     (ara_axi_resp_t     )
  ) i_axi_slave_ctrl_dwc_noc (
    .clk_i     (clk_i                       ),
    .rst_ni    (rst_ni                      ),
    .slv_req_i (ara_axi_req                 ),
    .slv_resp_o(ara_axi_resp                ),
    .mst_req_o (periph_narrow_axi_req  ),
    .mst_resp_i(periph_narrow_axi_resp )
  );

  axi_noc_bridge_mem #(
    .AxiDataWidth (AxiNarrowDataWidth  ),
    .AxiAddrWidth (AxiAddrWidth        )
  )i_axi_noc_bridge
  (
    .clk     (clk_i                        ),
    .rst_n    (rst_ni                       ),
    .noc_valid_out (noc2_valid_out          ),
    .noc_data_out  (noc2_data_out           ),
    .noc_ready_in  (noc2_ready_in           ),
    .noc_valid_in  (noc3_valid_in),
    .noc_data_in   (noc3_data_in),
    .noc_ready_out (noc3_ready_out),
    .src_chipid     (default_chipid), // 14 // 0
    .src_xpos       (default_coreid_x), // 8  // 1
    .src_ypos       (default_coreid_y), // 8  // 1
    .src_fbits      (`NOC_FBITS_ARA), // 4
    .dest_chipid    (14'b1000_0000_0000_00), // 14 //  all zero 
    .dest_xpos      (8'd0), // 8
    .dest_ypos      (8'd0), // 8
    .dest_fbits     (`NOC_FBITS_MEM), //4
    // write address channel
    .m_axi_awaddr   (periph_narrow_axi_req.aw.addr ),
    .m_axi_awlen    (periph_narrow_axi_req.aw.len  ),
    .m_axi_awsize   (periph_narrow_axi_req.aw.size ),
    .m_axi_awburst  (periph_narrow_axi_req.aw.burst),
    .m_axi_awcache (periph_narrow_axi_req.aw.cache),
    .m_axi_awid     (periph_narrow_axi_req.aw.id),
    // handshake logic
    .m_axi_awvalid  (periph_narrow_axi_req.aw_valid ),
    .m_axi_awready  (periph_narrow_axi_resp.aw_ready),

    //write data channel
    .m_axi_wdata    (periph_narrow_axi_req.w.data   ),
    .m_axi_wstrb    (periph_narrow_axi_req.w.strb   ),
    .m_axi_wlast    (periph_narrow_axi_req.w.last   ),
    // handshake logic
    .m_axi_wvalid   (periph_narrow_axi_req.w_valid  ),
    .m_axi_wready   (periph_narrow_axi_resp.w_ready ),

    // write response channel
    .m_axi_bresp    (periph_narrow_axi_resp.b.resp  ),
    .m_axi_bid      (periph_narrow_axi_resp.b.id ),
    .m_axi_bvalid   (periph_narrow_axi_resp.b_valid  ),
    .m_axi_bready   (periph_narrow_axi_req.b_ready ),

    // read address channel 
    .m_axi_araddr   (periph_narrow_axi_req.ar.addr  ),
    .m_axi_arlen    (periph_narrow_axi_req.ar.len   ), 
    .m_axi_arsize   (periph_narrow_axi_req.ar.size  ), 
    .m_axi_arburst  (periph_narrow_axi_req.ar.burst ), 
    .m_axi_arcache (periph_narrow_axi_req.ar.cache ), 
    .m_axi_arid     (periph_narrow_axi_req.ar.id),
    // handshake logic 
    .m_axi_arvalid  (periph_narrow_axi_req.ar_valid ),
    .m_axi_arready  (periph_narrow_axi_resp.ar_ready),

    // read data channel
    .m_axi_rdata    (periph_narrow_axi_resp.r.data  ),
    .m_axi_rresp    (periph_narrow_axi_resp.r.resp  ),
    .m_axi_rlast    (periph_narrow_axi_resp.r.last  ),
    .m_axi_ruser    (periph_narrow_axi_resp.r.user),
    .m_axi_rid      (periph_narrow_axi_resp.r.id),
    // handshake logic
    .m_axi_rvalid   (periph_narrow_axi_resp.r_valid ),
    .m_axi_rready   (periph_narrow_axi_req.r_ready  )
  );

`else 
  axi_dw_converter #(
    .AxiSlvPortDataWidth(AxiWideDataWidth    ),
    .AxiMstPortDataWidth(AxiNarrowDataWidth  ),
    .AxiAddrWidth       (AxiAddrWidth        ),
    .AxiIdWidth         (AxiCoreIdWidth       ),
    .AxiMaxReads        (1                  ),
    //.ar_chan_t          (soc_wide_ar_chan_t  ),
    .ar_chan_t          (ara_axi_ar_chan_t  ),
    .mst_r_chan_t       (soc_narrow_r_chan_t ),
    //.slv_r_chan_t       (soc_wide_r_chan_t   ),
    .slv_r_chan_t       (ara_axi_r_chan_t   ),
    .aw_chan_t          (soc_narrow_aw_chan_t),
    .b_chan_t           (soc_narrow_b_chan_t ),
    .mst_w_chan_t       (soc_narrow_w_chan_t ),
    //.slv_w_chan_t       (soc_wide_w_chan_t   ),
    .slv_w_chan_t       (ara_axi_w_chan_t   ),
    .axi_mst_req_t      (soc_narrow_req_t    ),
    .axi_mst_resp_t     (soc_narrow_resp_t   ),
    //.axi_slv_req_t      (soc_wide_req_t      ),
    //.axi_slv_resp_t     (soc_wide_resp_t     )
    .axi_slv_req_t      (ara_axi_req_t      ),
    .axi_slv_resp_t     (ara_axi_resp_t     )
  ) i_axi_slave_ctrl_dwc_noc (
    .clk_i     (clk_i                       ),
    .rst_ni    (rst_ni                      ),
    //.slv_req_i (periph_wide_axi_req[NOC]    ),
    //.slv_resp_o(periph_wide_axi_resp[NOC]   ),
    .slv_req_i (ara_axi_req                 ),
    .slv_resp_o(ara_axi_resp                ),
    .mst_req_o (periph_narrow_axi_req  ),
    .mst_resp_i(periph_narrow_axi_resp )
  );
  axi_noc_bridge #(
    .AXI_DATA_WIDTH (AxiNarrowDataWidth  ),
    .AXI_ADDR_WIDTH (AxiAddrWidth  )
  )i_axi_noc_bridge
  (
    .clk     (clk_i                        ),
    .rst_n    (rst_ni                       ),
  `ifdef ARA_REQ2MEM // direct memory request to main memory 
    .noc_valid_out (noc2_valid_out          ),
    .noc_data_out  (noc2_data_out           ),
    .noc_ready_in  (noc2_ready_in           ),
    .noc_valid_in  (noc3_valid_in),
    .noc_data_in   (noc3_data_in),
    .noc_ready_out (noc3_ready_out),
  `else 
    .noc_valid_out (noc1_valid_out          ),
    .noc_data_out  (noc1_data_out           ),
    .noc_ready_in  (noc1_ready_in           ),
    .noc_valid_in  (noc2_valid_in),
    .noc_data_in   (noc2_data_in),
    .noc_ready_out (noc2_ready_out),
  `endif 
    .src_chipid     (default_chipid), // 14 // 0
    .src_xpos       (default_coreid_x), // 8  // 1
    .src_ypos       (default_coreid_y), // 8  // 1
    .src_fbits      (`NOC_FBITS_ARA), // 4
  `ifdef ARA_REQ2MEM // direct memory request to main memory 
    .dest_chipid    (14'b1000_0000_0000_00), // 14 //  all zero 
    .dest_xpos      (8'd0), // 8
    .dest_ypos      (8'd0), // 8
    .dest_fbits     (`NOC_FBITS_MEM), //4
  `else // memory request to L2 cache 
    .dest_chipid    (14'b0000_0000_0000_00), // 14
    .dest_xpos      (8'd0), // 8
    .dest_ypos      (8'd0), // 8
    .dest_fbits     (`NOC_FBITS_L2), //4
    .system_tile_count (config_system_tile_count_5_0),
    .home_alloc_method (config_home_alloc_method),
  `endif 
    // write address channel
    .m_axi_awaddr   (periph_narrow_axi_req.aw.addr ),
    .m_axi_awlen    (periph_narrow_axi_req.aw.len  ),
    .m_axi_awsize   (periph_narrow_axi_req.aw.size ),
    .m_axi_awburst  (periph_narrow_axi_req.aw.burst),
    .m_axi_awcache (periph_narrow_axi_req.aw.cache),
    .m_axi_awid     (periph_narrow_axi_req.aw.id),
    // handshake logic
    .m_axi_awvalid  (periph_narrow_axi_req.aw_valid ),
    .m_axi_awready  (periph_narrow_axi_resp.aw_ready),

    //write data channel
    .m_axi_wdata    (periph_narrow_axi_req.w.data   ),
    .m_axi_wstrb    (periph_narrow_axi_req.w.strb   ),
    .m_axi_wlast    (periph_narrow_axi_req.w.last   ),
    // handshake logic
    .m_axi_wvalid   (periph_narrow_axi_req.w_valid  ),
    .m_axi_wready   (periph_narrow_axi_resp.w_ready ),

    // write response channel
    .m_axi_bresp    (periph_narrow_axi_resp.b.resp  ),
    .m_axi_bid      (periph_narrow_axi_resp.b.id ),
    .m_axi_bvalid   (periph_narrow_axi_resp.b_valid  ),
    .m_axi_bready   (periph_narrow_axi_req.b_ready ),

    // read address channel 
    .m_axi_araddr   (periph_narrow_axi_req.ar.addr  ),
    .m_axi_arlen    (periph_narrow_axi_req.ar.len   ), 
    .m_axi_arsize   (periph_narrow_axi_req.ar.size  ), 
    .m_axi_arburst  (periph_narrow_axi_req.ar.burst ), 
    .m_axi_arcache (periph_narrow_axi_req.ar.cache ), 
    .m_axi_arid     (periph_narrow_axi_req.ar.id),
    // handshake logic 
    .m_axi_arvalid  (periph_narrow_axi_req.ar_valid ),
    .m_axi_arready  (periph_narrow_axi_resp.ar_ready),

    // read data channel
    .m_axi_rdata    (periph_narrow_axi_resp.r.data  ),
    .m_axi_rresp    (periph_narrow_axi_resp.r.resp  ),
    .m_axi_rlast    (periph_narrow_axi_resp.r.last  ),
    .m_axi_ruser    (periph_narrow_axi_resp.r.user),
    .m_axi_rid      (periph_narrow_axi_resp.r.id),
    // handshake logic
    .m_axi_rvalid   (periph_narrow_axi_resp.r_valid ),
    .m_axi_rready   (periph_narrow_axi_req.r_ready  )
  );
`endif

endmodule : ara_verilog_wrap

