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
    parameter fpu_support_e              FPUSupport            = FPUSupportHalfSingleDouble,
    // AXI Interface
    parameter int unsigned               AxiDataWidth          = 32*NrLanes,
    parameter int unsigned               AxiAddrWidth          = 64,
    parameter int unsigned               AxiUserWidth          = 1,
    parameter int unsigned               AxiIdWidth            = 5,
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
    // Interrupt inputs
    input  [1:0]                irq_i,        // level sensitive IR lines, mip & sip (async)
    input                       ipi_i,        // inter-processor interrupts (async)
    // Timer facilities
    input                       time_irq_i,   // timer interrupt in (async)
    input                       debug_req_i,  // debug request (async)

    // NOC Signals
    input  [`NOC_DATA_WIDTH-1:0] noc2_valid_in ,
    input  [`NOC_DATA_WIDTH-1:0] noc2_data_in  ,
    output [`NOC_DATA_WIDTH-1:0] noc2_ready_out,
    output [`NOC_DATA_WIDTH-1:0] noc2_valid_out,
    output [`NOC_DATA_WIDTH-1:0] noc2_data_out ,
    input  [`NOC_DATA_WIDTH-1:0] noc2_ready_in ,

  `ifdef PITON_ARIANE
    // L15 (memory side)
    output [$size(wt_cache_pkg::l15_req_t)-1:0]  l15_req_o,
    input  [$size(wt_cache_pkg::l15_rtrn_t)-1:0] l15_rtrn_i
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

  localparam AxiSocIdWidth  = AxiIdWidth - $clog2(NrAXIMasters);
  localparam AxiCoreIdWidth = AxiSocIdWidth - 1;

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
  //`AXI_TYPEDEF_ALL(ariane_axi, axi_addr_t, axi_core_id_t, axi_narrow_data_t, axi_narrow_strb_t,
  //  axi_user_t)
  `AXI_TYPEDEF_ALL(soc_narrow, axi_addr_t, axi_soc_id_t, axi_narrow_data_t, axi_narrow_strb_t,
  axi_user_t)
  `AXI_TYPEDEF_ALL(soc_wide, axi_addr_t, axi_soc_id_t, axi_data_t, axi_strb_t, axi_user_t)
  `AXI_LITE_TYPEDEF_ALL(soc_narrow_lite, axi_addr_t, axi_narrow_data_t, axi_narrow_strb_t)
  
  //ara_axi_req_t  ara_axi_req;
  //ara_axi_resp_t ara_axi_resp;



  
   
  
    //////////////////////
    //  Ara and Ariane  //
    //////////////////////
  
    import ariane_pkg::accelerator_req_t;
    import ariane_pkg::accelerator_resp_t;
  
    // Accelerator ports
    accelerator_req_t                     acc_req;
    logic                                 acc_req_valid;
    logic                                 acc_req_ready;
    accelerator_resp_t                    acc_resp;
    logic                                 acc_resp_valid;
    logic                                 acc_resp_ready;
    logic                                 acc_cons_en; // unused
    logic              [AxiAddrWidth-1:0] inval_addr;
    logic                                 inval_valid;
    logic                                 inval_ready;
  
  
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
  
    logic [1:0] irq;
    logic ipi, time_irq, debug_req;
  
    // reset synchronization
    synchronizer i_sync (
      .clk         ( clk_i      ),
      .presyncdata ( rst_n      ),
      .syncdata    ( spc_grst_l )
    );
  
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
      AxiCompliant:     Axi64BitCompliant, // Not sure it's a typo (Original: Axi64BitCompliant:1'b0 )
      SwapEndianess:         SwapEndianess,
      // debug
      DmBaseAddress:         DmBaseAddress,
      NrPMPEntries:          NrPMPEntries
    };

    
    ariane #(
      .ArianeCfg ( ArianeOpenPitonCfg )
    ) ariane (
      .clk_i            ( clk_i                ),
      .rst_ni           ( spc_grst_l           ),
      .boot_addr_i                             ,// constant
      .hart_id_i                               ,// constant
      .irq_i            ( irq                  ),
      .ipi_i            ( ipi                  ),
      .time_irq_i       ( time_irq             ),
      .debug_req_i      ( debug_req            ),
      // Accelerator ports, connection to Ara
      .acc_req_o        (acc_req               ),
      .acc_req_valid_o  (acc_req_valid         ),
      .acc_req_ready_i  (acc_req_ready         ),
      .acc_resp_i       (acc_resp              ),
      .acc_resp_valid_i (acc_resp_valid        ),
      .acc_resp_ready_o (acc_resp_ready        ),
      // .acc_cons_en_o    (acc_cons_en           ),
      // .inval_addr_i     (inval_addr            ),
      // .inval_valid_i    (inval_valid           ),
      // .inval_ready_o    (inval_ready           ),
    `ifdef PITON_ARIANE
      .l15_req_o   ( l15_req   ),
      .l15_rtrn_i  ( l15_rtrn  )
  `else
      .axi_req_o   ( axi_req   ),
      .axi_resp_i  ( axi_resp  )
  `endif
    );

  logic  scan_enable_i;
  assign scan_enable_i = 0;

  ara #(
    .NrLanes     (NrLanes         ),
    .FPUSupport  (FPUSupport      ),
    .AxiDataWidth(AxiWideDataWidth),
    .AxiAddrWidth(AxiAddrWidth    ),
    .axi_ar_t    (ara_axi_ar_chan_t    ),
    .axi_r_t     (ara_axi_r_chan_t     ),
    .axi_aw_t    (ara_axi_aw_chan_t    ),
    .axi_w_t     (ara_axi_w_chan_t     ),
    .axi_b_t     (ara_axi_b_chan_t     ),
    .axi_req_t   (ara_axi_req_t   ),
    .axi_resp_t  (ara_axi_resp_t  )
  ) i_ara (
    .clk_i           (clk_i         ),
    .rst_ni          (rst_ni        ),
    .scan_enable_i   (scan_enable_i ),
    .scan_data_i     (1'b0          ),
    .scan_data_o     (/* Unused */  ),
    .acc_req_i       (acc_req       ),
    .acc_req_valid_i (acc_req_valid ),
    .acc_req_ready_o (acc_req_ready ),
    .acc_resp_o      (acc_resp      ),
    .acc_resp_valid_o(acc_resp_valid),
    .acc_resp_ready_i(acc_resp_ready),
    .axi_req_o       (ara_axi_req   ),
    .axi_resp_i      (ara_axi_resp  )
  );

  //////////////////////
  //  Memory Regions  //
  //////////////////////

  localparam NrAXIMasters = 1; // Actually masters, but slaves on the crossbar

  typedef enum int unsigned {
    NOC = 0,
    CTRL  = 1
  } axi_slaves_e;
  localparam NrAXISlaves = CTRL + 1;

  // Memory Map
  // 1GByte of DDR (split between two chips on Genesys2)
  localparam logic [63:0] DRAMLength = 64'h40000000;
  localparam logic [63:0] CTRLLength = 64'h1000;
  localparam logic [63:0] NOCL2Length = 64'h1000;

  typedef enum logic [63:0] {
    DRAMBase = 64'h8000_0000,
    CTRLBase = 64'hD000_0000, 
    NOCL2Base = 64'hD00_1000
  } soc_bus_start_e;


  // Buses
  system_req_t  system_axi_req;
  system_resp_t system_axi_resp;

  soc_wide_req_t    [NrAXISlaves-1:0] periph_wide_axi_req;
  soc_wide_resp_t   [NrAXISlaves-1:0] periph_wide_axi_resp;
  soc_narrow_req_t  [NrAXISlaves-1:0] periph_narrow_axi_req;
  soc_narrow_resp_t [NrAXISlaves-1:0] periph_narrow_axi_resp;

  ////////////////
  //  Crossbar  //
  ////////////////

  localparam axi_pkg::xbar_cfg_t XBarCfg = '{
    NoSlvPorts        : NrAXIMasters,
    NoMstPorts        : NrAXISlaves,
    MaxMstTrans       : 4,
    MaxSlvTrans       : 4,
    FallThrough       : 1'b0,
    LatencyMode       : axi_pkg::CUT_MST_PORTS,
    AxiIdWidthSlvPorts: AxiSocIdWidth,
    AxiIdUsedSlvPorts : AxiSocIdWidth,
    UniqueIds         : 1'b0,
    AxiAddrWidth      : AxiAddrWidth,
    AxiDataWidth      : AxiWideDataWidth,
    NoAddrRules       : NrAXISlaves
  };

  axi_pkg::xbar_rule_64_t [NrAXISlaves-1:0] routing_rules;
  assign routing_rules = '{
    '{idx: CTRL, start_addr: CTRLBase, end_addr: CTRLBase + CTRLLength},
    '{idx: NOC, start_addr: NOCL2Base, end_addr: NOCL2Base + NOCL2Length}
  };

  axi_xbar #(
    .Cfg          (XBarCfg                ),
    .slv_aw_chan_t(system_aw_chan_t       ),
    .mst_aw_chan_t(soc_wide_aw_chan_t     ),
    .w_chan_t     (system_w_chan_t        ),
    .slv_b_chan_t (system_b_chan_t        ),
    .mst_b_chan_t (soc_wide_b_chan_t      ),
    .slv_ar_chan_t(system_ar_chan_t       ),
    .mst_ar_chan_t(soc_wide_ar_chan_t     ),
    .slv_r_chan_t (system_r_chan_t        ),
    .mst_r_chan_t (soc_wide_r_chan_t      ),
    .slv_req_t    (system_req_t           ),
    .slv_resp_t   (system_resp_t          ),
    .mst_req_t    (soc_wide_req_t         ),
    .mst_resp_t   (soc_wide_resp_t        ),
    .rule_t       (axi_pkg::xbar_rule_64_t)
  ) i_soc_xbar (
    .clk_i                (clk_i               ),
    .rst_ni               (rst_ni              ),
    .test_i               (1'b0                ),
    .slv_ports_req_i      (system_axi_req      ),
    .slv_ports_resp_o     (system_axi_resp     ),
    .mst_ports_req_o      (periph_wide_axi_req ),
    .mst_ports_resp_i     (periph_wide_axi_resp),
    .addr_map_i           (routing_rules       ),
    .en_default_mst_port_i('0                  ),
    .default_mst_port_i   ('0                  )
  );

  /////////////////////////
  //  Control registers  //
  /////////////////////////

  soc_narrow_lite_req_t  axi_lite_ctrl_registers_req;
  soc_narrow_lite_resp_t axi_lite_ctrl_registers_resp;
  logic [63:0] event_trigger;

  axi_dw_converter #(
    .AxiSlvPortDataWidth(AxiWideDataWidth    ),
    .AxiMstPortDataWidth(AxiNarrowDataWidth  ),
    .AxiAddrWidth       (AxiAddrWidth        ),
    .AxiIdWidth         (AxiSocIdWidth       ),
    .AxiMaxReads        (2                   ),
    .ar_chan_t          (soc_wide_ar_chan_t  ),
    .mst_r_chan_t       (soc_narrow_r_chan_t ),
    .slv_r_chan_t       (soc_wide_r_chan_t   ),
    .aw_chan_t          (soc_narrow_aw_chan_t),
    .b_chan_t           (soc_narrow_b_chan_t ),
    .mst_w_chan_t       (soc_narrow_w_chan_t ),
    .slv_w_chan_t       (soc_wide_w_chan_t   ),
    .axi_mst_req_t      (soc_narrow_req_t    ),
    .axi_mst_resp_t     (soc_narrow_resp_t   ),
    .axi_slv_req_t      (soc_wide_req_t      ),
    .axi_slv_resp_t     (soc_wide_resp_t     )
  ) i_axi_slave_ctrl_dwc_csr (
    .clk_i     (clk_i                       ),
    .rst_ni    (rst_ni                      ),
    .slv_req_i (periph_wide_axi_req[CTRL]   ),
    .slv_resp_o(periph_wide_axi_resp[CTRL]  ),
    .mst_req_o (periph_narrow_axi_req[CTRL] ),
    .mst_resp_i(periph_narrow_axi_resp[CTRL])
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
  ) i_axi_to_axi_lite_csr (
    .clk_i     (clk_i                        ),
    .rst_ni    (rst_ni                       ),
    .test_i    (1'b0                         ),
    .slv_req_i (periph_narrow_axi_req[CTRL]  ),
    .slv_resp_o(periph_narrow_axi_resp[CTRL] ),
    .mst_req_o (axi_lite_ctrl_registers_req  ),
    .mst_resp_i(axi_lite_ctrl_registers_resp )
  );

  ctrl_registers #(
    .DRAMBaseAddr   (DRAMBase              ),
    .DRAMLength     (DRAMLength            ),
    .DataWidth      (AxiNarrowDataWidth    ),
    .AddrWidth      (AxiAddrWidth          ),
    .axi_lite_req_t (soc_narrow_lite_req_t ),
    .axi_lite_resp_t(soc_narrow_lite_resp_t)
  ) i_ctrl_registers (
    .clk_i                (clk_i                       ),
    .rst_ni               (rst_ni                      ),
    .axi_lite_slave_req_i (axi_lite_ctrl_registers_req ),
    .axi_lite_slave_resp_o(axi_lite_ctrl_registers_resp),
    .hw_cnt_en_o          (hw_cnt_en_o                 ),
    .dram_base_addr_o     (/* Unused */                ),
    .dram_end_addr_o      (/* Unused */                ),
    .exit_o               (exit_o                      ),
    .event_trigger_o      (event_trigger)
  );


  /////////////////////////
  //  NOC Bridge         //
  /////////////////////////

  soc_narrow_lite_req_t  axi_lite_noc_req;
  soc_narrow_lite_resp_t axi_lite_noc_resp;

  axi_dw_converter #(
    .AxiSlvPortDataWidth(AxiWideDataWidth    ),
    .AxiMstPortDataWidth(AxiNarrowDataWidth  ),
    .AxiAddrWidth       (AxiAddrWidth        ),
    .AxiIdWidth         (AxiSocIdWidth       ),
    .AxiMaxReads        (2                   ),
    .ar_chan_t          (soc_wide_ar_chan_t  ),
    .mst_r_chan_t       (soc_narrow_r_chan_t ),
    .slv_r_chan_t       (soc_wide_r_chan_t   ),
    .aw_chan_t          (soc_narrow_aw_chan_t),
    .b_chan_t           (soc_narrow_b_chan_t ),
    .mst_w_chan_t       (soc_narrow_w_chan_t ),
    .slv_w_chan_t       (soc_wide_w_chan_t   ),
    .axi_mst_req_t      (soc_narrow_req_t    ),
    .axi_mst_resp_t     (soc_narrow_resp_t   ),
    .axi_slv_req_t      (soc_wide_req_t      ),
    .axi_slv_resp_t     (soc_wide_resp_t     )
  ) i_axi_slave_ctrl_dwc_noc (
    .clk_i     (clk_i                       ),
    .rst_ni    (rst_ni                      ),
    .slv_req_i (periph_wide_axi_req[NOC]    ),
    .slv_resp_o(periph_wide_axi_resp[NOC]   ),
    .mst_req_o (periph_narrow_axi_req[NOC]  ),
    .mst_resp_i(periph_narrow_axi_resp[NOC] )
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
    .slv_req_i (periph_narrow_axi_req[NOC]   ),
    .slv_resp_o(periph_narrow_axi_resp[NOC]  ),
    .mst_req_o (axi_lite_noc_req             ),
    .mst_resp_i(axi_lite_noc_resp            )
  );

/*axilite_noc_bridge #(
  .AXI_LITE_DATA_WIDTH (     64     )
) axi_noc_bridge (
    .clk            (clk_i),
    .rst            (rst_ni),
    .noc2_valid_in  (noc2_valid_in           ),
    .noc2_data_in   (noc2_data_in            ),
    .noc2_ready_out (noc2_ready_out          ),
    .noc2_valid_out (noc2_valid_out          ),
    .noc2_data_out  (noc2_data_out           ),
    .noc2_ready_in  (noc2_ready_in           ),
    .noc3_valid_in  (),
    .noc3_data_in   (),
    .noc3_ready_out (),
    .noc3_valid_out (),
    .noc3_data_out  (),
    .noc3_ready_in  (),
    .src_chipid     (),
    .src_xpos       (),
    .src_ypos       (),
    .src_fbits      (),
    .dest_chipid    (),
    .dest_xpos      (),
    .dest_ypos      (),
    .dest_fbits     (),
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
);*/

endmodule : ara_verilog_wrap
