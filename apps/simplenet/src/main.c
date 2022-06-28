// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matteo Perotti

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iconv2d_tensor8.h"

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

// Define Matrix dimensions:
// o = i ° f, with i=[MxN], f=[FxF], o=[MxN]
// The filter is a square matrix, and F is odd

// Matrices defined in data.S
extern int8_t input_layer[] __attribute__((    aligned(4 * NR_LANES)));
extern int8_t convolutional_0_wbuf[] __attribute__((    aligned(4 * NR_LANES)));
extern int8_t convolutional_0_obuf[] __attribute__((    aligned(4 * NR_LANES)));
extern int8_t convolutional_1_wbuf[] __attribute__((    aligned(4 * NR_LANES)));
extern int8_t convolutional_1_obuf[] __attribute__((    aligned(4 * NR_LANES)));

// 1 x W x C x R   *   K x W x F x F   =    K x C x R
//iconv2d_tensor8(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {
int main() {
    printf("\n");
    printf("============\n");
    printf("=  CONV2D  =\n");
    printf("============\n");
    printf("\n");
    printf("\n");
    int64_t runtime;

    // Call the main kernel, and measure cycles
    start_timer();
    iconv2d_tensor8(convolutional_0_obuf, input_layer, convolutional_0_wbuf, 32, 32, 3, 3, 128);
    stop_timer();
    runtime = get_timer();
    // Performance metrics
    printf("The execution took %d cycles for layer 1.\n", runtime);

    start_timer();
    iconv2d_tensor8(convolutional_1_obuf, convolutional_0_obuf, convolutional_1_wbuf, 32, 32, 128, 3, 128);
    stop_timer();
    runtime = get_timer();
    // Performance metrics
    printf("The execution took %d cycles for layer 2.\n", runtime);

    return 0;
}
