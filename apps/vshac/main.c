// Author : Théo Dupuis
// Mail   : theo.dupuis@polymtl.ca
// Groupe de Recherche en Microelectronique et Microsystemes (GR2M)
// Electrical Engineering Department, Polytechnique Montreal

#define VLEN_TEST 8

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif


// Vector to shift
int8_t  i8[VLEN_TEST]  = {0, 3, 7, 15, 31, 63, 127, 255};

// Vector shift amount
int8_t  s8[VLEN_TEST]  = {4, 3, 3, 2, 2, 1, 1, 0};

// Vector output
int8_t  o8[VLEN_TEST];
int8_t  golden_o8[VLEN_TEST];

// Pointer used for input, shift amount and output
// required to load values from memory using inline assembly

int8_t *  ptr8  = i8;


int main() {
printf("==============\n");
printf("= VSHAC TEST =\n");
printf("==============\n");
printf("\n");

printf("TEST 8b \n");

// setup full register size on 8b, but we only use VLEN_TEST elements
asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(VLEN_TEST));

printf("Vector-Vector shift and accumulate with vshac...\n");

// load 8b input into v2
asm volatile("vle8.v v2, (%0)" : "+&r" (ptr8));
// load 8b shift into v1
ptr8 = s8;
asm volatile("vle8.v v1, (%0)" : "+&r" (ptr8));
// load the value '2' in each element of v0 to check for accumulation
asm volatile("vmv.v.i v0, 2");



// Execute the intruction (logical left shift and accumulate)
// vshac.vv v0, v2, v1
// => v0[i] <- v0[i] + (v2[i] << v1[i])
asm volatile (".byte 0x57, 0xa0, 0x20, 0xba");

// store the result
ptr8 = o8;
asm volatile("vse8.v v0, (%0)" : "+&r"(ptr8));


printf("Verifying result...\n");

int error = 0;
for(int n = 0; n < VLEN_TEST; n++){
  golden_o8[n] = 2 + (i8[n] << s8[n]);
  if(golden_o8[n] != o8[n])
    error = 1;
}

if (error == 0)
  printf("--* passed *--\n\n");
else
  printf("--* failed *--\n\n");




printf("Vector-Scalar shift and accumulate with vshac...\n");

// load 8b input into v2
ptr8 = i8;
asm volatile("vle8.v v2, (%0)" : "+&r" (ptr8));
// load 8b shift into v1
// ::"t0" is used in the clobber list to avoid the compiler from using it
ptr8 = s8;
asm volatile("lb t0, (%0)" : "+&r" (ptr8) ::"t0");
// load the value '2' in each element of v0 to check for accumulation
asm volatile("vmv.v.i v0, 2");



// Execute the intruction (logical left shift and accumulate)
// vshac.vx v0, v2, t0
// (can also be expressed as : vshac.vx v0, v2, x5)
// => v0[i] <- v0[i] + (v2[i] << t0)
asm volatile (".byte 0x57, 0xe0, 0x22, 0xba");


// store the result
ptr8 = o8;
asm volatile("vse8.v v0, (%0)" : "+&r"(ptr8));


printf("Verifying result...\n");

error = 0;
for(int n = 0; n < VLEN_TEST; n++){
  golden_o8[n] = 2 + (i8[n] << s8[0]);
  if(golden_o8[n] != o8[n])
    error = 1;
}

if (error == 0)
  printf("--* passed *--\n\n");
else
  printf("--* failed *--\n\n");
  

}
