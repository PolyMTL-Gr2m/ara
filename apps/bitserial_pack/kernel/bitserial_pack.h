// Copyright 2022 ETH Zurich, University of Bologna and Polytechnique Montreal.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: MohammadHossein AskariHemmat <m.h.askari.hemmat@gmail.com>

#ifndef BITSERIAL_PACK_H
#define BITSERIAL_PACK_H

#include <stdint.h>

void bitserial_pack_init();
// 2 bit precision packing
void bitserial_pack_64(uint64_t* mat_i, uint64_t* out, uint64_t len_mat_i);

#endif


