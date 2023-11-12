module ariane import ariane_pkg::*; #(
  parameter ariane_pkg::cva6_cfg_t CVA6Cfg = cva6_cfg_empty,
  parameter type rvfi_instr_t = logic,
  parameter type cvxif_req_t = acc_pkg::accelerator_req_t,
  parameter type cvxif_resp_t = acc_pkg::accelerator_resp_t,
  //
  parameter ariane_pkg::ariane_cfg_t ArianeCfg     = ariane_pkg::ArianeDefaultConfig,
  parameter int unsigned AxiAddrWidth = ariane_axi::AddrWidth,
  parameter int unsigned AxiDataWidth = ariane_axi::DataWidth,
  parameter int unsigned AxiIdWidth   = ariane_axi::IdWidth,
  parameter type axi_ar_chan_t = ariane_axi::ar_chan_t,
  parameter type axi_aw_chan_t = ariane_axi::aw_chan_t,
  parameter type axi_w_chan_t  = ariane_axi::w_chan_t,
  parameter type noc_req_t = ariane_axi::req_t,
  parameter type noc_resp_t = ariane_axi::resp_t
) (
  input  logic                         clk_i,
  input  logic                         rst_ni,
  // Core ID, Cluster ID and boot address are considered more or less static
  input  logic [riscv::VLEN-1:0]       boot_addr_i,  // reset boot address
  input  logic [riscv::XLEN-1:0]       hart_id_i,    // hart id in a multicore environment (reflected in a CSR)

  // Interrupt inputs
  input  logic [1:0]                   irq_i,        // level sensitive IR lines, mip & sip (async)
  input  logic                         ipi_i,        // inter-processor interrupts (async)
  // Timer facilities
  input  logic                         time_irq_i,   // timer interrupt in (async)
  input  logic                         debug_req_i,  // debug request (async)
  // RISC-V formal interface port (`rvfi`):
  // Can be left open when formal tracing is not needed.
  output rvfi_instr_t [CVA6Cfg.NrCommitPorts-1:0] rvfi_o,
  // Accel interface
  output cvxif_req_t        cvxif_req_o,
  input  cvxif_resp_t       cvxif_resp_i,

  // memory side
  output noc_req_t                     noc_req_o,
  input  noc_resp_t                    noc_resp_i
);

  // cva6 #(
  //   .CVA6Cfg ( CVA6Cfg ),
  //   .rvfi_instr_t ( rvfi_instr_t ),
  //   .cvxif_req_t  (accelerator_req_t ),
  //   .cvxif_resp_t (accelerator_resp_t),
  //   .ArianeCfg     (ArianeCfg    ),
  //   .axi_ar_chan_t (axi_ar_chan_t),
  //   .axi_aw_chan_t (axi_aw_chan_t),
  //   .axi_w_chan_t  (axi_w_chan_t ),
  //   .noc_req_t     (noc_req_t    ),
  //   .noc_resp_t    (noc_resp_t   )
  // ) i_cva6 (
  //   .clk_i                ( clk_i                     ),
  //   .rst_ni               ( rst_ni                    ),
  //   .boot_addr_i          ( boot_addr_i               ),
  //   .hart_id_i            ( hart_id_i                 ),
  //   .irq_i                ( irq_i                     ),
  //   .ipi_i                ( ipi_i                     ),
  //   .time_irq_i           ( time_irq_i                ),
  //   .debug_req_i          ( debug_req_i               ),
  //   .rvfi_o               ( rvfi_o                    ),
  //   .cvxif_req_o          ( cvxif_req_o               ),
  //   .cvxif_resp_i         ( cvxif_resp_i              ),
  //   .noc_req_o            ( noc_req_o                 ),
  //   .noc_resp_i           ( noc_resp_i                )
  // );


  cva6 #(
    .CVA6Cfg              (CVA6Cfg       ),
    .rvfi_instr_t         (rvfi_instr_t  ),
    .cvxif_req_t          (cvxif_req_t   ),
    .cvxif_resp_t         (cvxif_resp_t  ),
    .ArianeCfg            (ArianeCfg     ),
    .axi_ar_chan_t        (axi_ar_chan_t ),
    .axi_aw_chan_t        (axi_aw_chan_t ),
    .axi_w_chan_t         (axi_w_chan_t  ),
    .noc_req_t            (noc_req_t     ),
    .noc_resp_t           (noc_resp_t    )
  ) i_cva6 (
    .clk_i                ( clk_i                     ),
    .rst_ni               ( rst_ni                    ),
    .boot_addr_i          ( boot_addr_i               ),
    .hart_id_i            ( hart_id_i                 ),
    .irq_i                ( irq_i                     ),
    .ipi_i                ( ipi_i                     ),
    .time_irq_i           ( time_irq_i                ),
    .debug_req_i          ( debug_req_i               ),
    .rvfi_o               ( rvfi_o                    ),
    .cvxif_req_o          ( cvxif_req_o               ),
    .cvxif_resp_i         ( cvxif_resp_i              ),
    .noc_req_o            ( noc_req_o                 ),
    .noc_resp_i           ( noc_resp_i                )
  );

  if (ariane_pkg::CVXIF_PRESENT) begin : gen_example_coprocessor
    cvxif_example_coprocessor i_cvxif_coprocessor (
      .clk_i                ( clk_i                          ),
      .rst_ni               ( rst_ni                         ),
      .cvxif_req_i          ( cvxif_req                      ),
      .cvxif_resp_o         ( cvxif_resp                     )
    );
  end

endmodule // ariane