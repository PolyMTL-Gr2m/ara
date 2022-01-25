// Copyright 2022 ETH Zurich, University of Bologna and Polytechnique Montreal.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: MohammadHossein AskariHemmat <m.h.askari.hemmat@gmail.com>

#ifndef BITSERIAL_MATMUL_H
#define BITSERIAL_MATMUL_H

#include <stdint.h>

void bitserial_matmul_init();
void bitserial_matmul_64(uint64_t* c, uint64_t* a, uint64_t* b, int aprec, int bprec);

#endif
