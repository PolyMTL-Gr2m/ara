// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>

#ifndef __VECTOR_MACROS_H__
#define __VECTOR_MACROS_H__

#include <stdint.h>
#include "dataset.h"
#include "rvv_debug_macros.h"
#include "encoding.h"
#include "float_conversion.h"

#ifdef __SPIKE__
#include <stdio.h>

// We need to activate the FP and V extensions manually
#define enable_vec() do { asm volatile ("csrs mstatus, %[bits];" :: [bits] "r" (MSTATUS_VS & (MSTATUS_VS >> 1))); } while (0);
#define enable_fp()  do { asm volatile ("csrs mstatus, %[bits];" :: [bits] "r" (MSTATUS_FS & (MSTATUS_FS >> 1))); } while (0);
#else
#include <stdio.h>

// #ifndef __MEM_FLUSH__
//   #define __MEM_FLUSH__
// #endif
// The FP and V extensions are activated in the crt0 script
#define enable_vec()
#define enable_fp()
#endif

/**************
 *  Counters  *
 **************/

// Counter for how many tests have failed
int num_failed;
// Pointer to the current test case
int test_case;

/************
 *  Macros  *
 ************/

#define read_vtype(buf) do { asm volatile ("csrr %[BUF], vtype" : [BUF] "=r" (buf)); } while (0);
#define read_vl(buf)    do { asm volatile ("csrr %[BUF], vl" : [BUF] "=r" (buf)); } while (0);
#define read_vxsat(buf) do { asm volatile ("csrr %[BUF], vxsat" : [BUF] "=r" (buf)); } while (0);
#define reset_vxsat     do { asm volatile ("csrw vxsat, %0" :: "rK"(0)); } while (0);
#define set_vxrm(val) do { asm volatile ("csrw vxrm, %0" :: "rK"(val)); } while (0);

#define vtype(golden_vtype, vlmul, vsew, vta, vma) (golden_vtype = vlmul << 0 | vsew << 3 | vta << 6 | vma << 7)

// Checking vtype and vl value which is set by configuration setting instructions
// by comparing with golden_vtype value used as a reference. Also check for
// illegal values of vlmul and vsew which violtate: ELEN >= SEW/LMUL
#define check_vtype_vl(casenum, vtype, golden_vtype, avl, vl, vsew, vlmul)                                                     \
  if((vlmul==5 && (vsew == 1 || vsew == 2 || vsew ==3)) || (vlmul==6 && (vsew == 2 || vsew ==3)) || (vlmul==7 && vsew==3)){   \
  if((vtype != 0x8000000000000000) || (vl != 0)){    \
  return;                                                  \
  }}                                                        \
  else if (vtype != golden_vtype || avl != vl) {                                                                        \
    num_failed++;                                                                                                  \
    return;                                                                                                        \
  }                                                                                                                \

#define check_vxsat(casenum, vxsat, golden_vxsat)                                                                  \
  if (vxsat != golden_vxsat) {                                                                        \
    num_failed++;                                                                                                  \
    return;                                                                                                        \
  }                                                                                                                \

// In order to avoid that scalar loads run ahead of vector stores,
// we use an instruction to ensure that all vector stores have been
// committed before continuing with scalar memory operations.
#define MEMORY_BARRIER  //asm volatile ("fence"); 
#define MEMBARRIER


#ifdef __MEM_FLUSH__
  #define OPENPITON_FLUSH(datatype,vreg,size) \
    int temp = size; \
    uint64_t addr = ((uint64_t)V ##vreg & 0x03ffffffc0) | 0xac00000000 | ((uint64_t)V ##vreg >> 34); \
    volatile uint64_t my_var = *((volatile uint64_t *)addr); \
    addr = addr + 64; \
    my_var = *((volatile uint64_t *)addr); \
    addr = addr + 64; \
    my_var = *((volatile uint64_t *)addr); \
    addr = addr + 64; \
    my_var = *((volatile uint64_t *)addr);
#else 
  #define OPENPITON_FLUSH(datatype,vreg,size)
#endif

// Zero-initialized variables can be problematic on bare-metal.
// Therefore, initialize them during runtime.
#define INIT_CHECK()  \
  num_failed = 0;     \
  test_case  = 0;     \

// Check at the final of the execution whether all the tests passed or not.
// Returns the number of failed tests.
#define EXIT_CHECK()                                                \
  do {                                                              \
    MEMORY_BARRIER;                                                 \
    if (num_failed > 0) {                                           \
      return num_failed;                                            \
    }                                                               \
    else {                                                          \
      return 0;                                                     \
    }                                                               \
  } while(0);                                                       \

// Check the result against a scalar golden value
#define XCMP(casenum,act,exp)                                           \
  if (act != exp) {                                                     \
    num_failed++;                                                       \
    return;                                                             \
  }                                                                     \

// Check the result against a floating-point scalar golden value
#define FCMP(casenum,act,exp)                                           \
  if(act != exp) {                                                      \
    num_failed++;                                                       \
    return;                                                             \
  }                                                                     \

// uint64_t addr = ((uint64_t)vact & 0x03ffffffc0) | 0xac00000000 | ((uint64_t)vact >> 34); 
// volatile uint8_t my_var = *((volatile uint8_t *)addr);                              
// Check the results against a vector of golden values
#define VCMP(T,str,casenum,vexp,act...)                                               \
  T vact[] = {act};                                                                   \
  MEMORY_BARRIER;                                                                     \
  for (unsigned int i = 0; i < sizeof(vact)/sizeof(T); i++) {                         \
    if (vexp[i] != vact[i]) {                                                         \
      num_failed++;                                                                   \
      return;                                                                         \
    }                                                                                 \
  }                                                                                   \

//Macro used to compare large number of elements
#define LVCMP(T,str,casenum,elements, vexp ,vact)                                     \
  MEMORY_BARRIER;                                                                     \
  for (unsigned int i = 0; i < elements; i++) {                                       \
    if (vexp[i] != vact[i]) {                                                         \
      num_failed++;                                                                   \
      return;                                                                         \
    }                                                                                 \
  }                                                                                   \


// Check the results against an in-memory vector of golden values
#define VMCMP(T,str,casenum,vexp,vgold,size)                                          \
  MEMORY_BARRIER;                                                                     \
  for (unsigned int i = 0; i < size; i++) {                                           \
    if (vexp[i] != vgold[i]) {                                                        \
      num_failed++;                                                                   \
      return;                                                                         \
    }                                                                                 \
  }                                                                                   \

// Macros to set vector length, type and multiplier
// Don't use this to set VL == 0 since the compiler puts rs1 == x0
#define VSET(VLEN,VTYPE,LMUL)                                                          \
  do {                                                                                 \
  asm volatile ("vsetvli t0, %[A]," #VTYPE "," #LMUL ", ta, ma \n" :: [A] "r" (VLEN)); \
  } while(0)

// Macros to set vector length equal to zero
#define VSET_ZERO(VTYPE,LMUL)                                                              \
  do {                                                                                     \
    int vset_zero_buf;                                                                     \
    asm volatile("li %0, 0" : "=r" (vset_zero_buf));                                       \
    asm volatile("vsetvli x0, %0," #VTYPE "," #LMUL ", ta, ma \n" :: "r" (vset_zero_buf)); \
  } while(0)

#define VSETMAX(VTYPE,LMUL)                                                            \
  do {                                                                                 \
  int64_t scalar = -1;                                                                 \
  asm volatile ("vsetvli t1, %[A]," #VTYPE "," #LMUL", ta, ma \n":: [A] "r" (scalar)); \
  } while(0)

// Macro to load a vector register with data from the stack
    // volatile uint64_t *addr = (volatile uint64_t *)0xc00000000; 
    // volatile char a = *((volatile char *)addr); 
    // volatile char b = a + 1; 
    //int vector_size = sizeof(vec);                                          
    //OPENPITON_FLUSH(datatype,vreg,vector_size);                                         
#define VLOAD(datatype,loadtype,vreg,vec...)                                \
  do {                                                                      \
    volatile datatype V ##vreg[] = {vec};                                   \
    int vector_size = sizeof(vec);                                          \
    OPENPITON_FLUSH(datatype,vreg,vector_size);                             \
    MEMORY_BARRIER;                                                         \
    asm volatile ("vl"#loadtype".v "#vreg", (%0)  \n":: [V] "r"(V ##vreg)); \
  } while(0)

// Macro to store a vector register into the pointer vec

#define VSTORE(T, storetype, vreg, vec)                                   \
  do {                                                                    \
    T* vec ##_t = (T*) vec;                                               \
    asm volatile ("vs"#storetype".v "#vreg", (%0)\n" : "+r" (vec ##_t));  \
    MEMORY_BARRIER;                                                       \
  } while(0)

// Macro to reset the whole register back to zero
#define VCLEAR(register)                                                                          \
  do {                                                                                            \
    MEMORY_BARRIER;                                                                               \
    uint64_t vtype; uint64_t vl; uint64_t vlmax;                                                  \
    asm volatile("csrr %[vtype], vtype" : [vtype] "=r" (vtype));                                  \
    asm volatile("csrr %[vl], vl" : [vl] "=r" (vl));                                              \
    asm volatile("vsetvl %[vlmax], zero, %[vtype]" : [vlmax] "=r" (vlmax) : [vtype] "r" (vtype)); \
    asm volatile("vmv.v.i "#register", 0");                                                       \
    asm volatile("vsetvl zero, %[vl], %[vtype]" :: [vl] "r" (vl), [vtype] "r" (vtype));           \
  } while(0)

#define CLEAR(register) VCLEAR(register)

// Macro to initialize a vector with progressive values from a counter
#define INIT_MEM_CNT(vec_name, size) \
  counter = 0;                          \
  for (int i = 0 ; i < size; i++) {     \
    vec_name[i] = counter;              \
    counter++;                          \
  }                                     \

// Macro to initialize a vector with zeroes
// The vector is initialized on the stack, use this function with caution
// Easy to go in the UART address space
#define INIT_MEM_ZEROES(vec_name, size) \
  for (int i = 0 ; i < size; i++) {        \
    vec_name[i] = 0;                       \
  }                                        \

/***************************
 *  Type-dependant macros  *
 ***************************/

// Vector comparison
#define VCMP_U64(casenum,vect,act...) {VSTORE_U64(vect); VCMP(uint64_t,%x,casenum,Ru64,act)}
#define VCMP_U32(casenum,vect,act...) {VSTORE_U32(vect); VCMP(uint32_t,%x,casenum,Ru32,act)}
#define VCMP_U16(casenum,vect,act...) {VSTORE_U16(vect); VCMP(uint16_t,%x,casenum,Ru16,act)}
#define VCMP_U8(casenum,vect,act...)  {VSTORE_U8(vect); VCMP(uint8_t, %x,casenum, Ru8,act)}

#define VEC_CMP_64(casenum,vect,act...) {VSTORE_U64(vect); VCMP(uint64_t,%x,casenum,Ru64,act)}
#define VEC_CMP_32(casenum,vect,act...) {VSTORE_U32(vect); VCMP(uint32_t,%x,casenum,Ru32,act)}
#define VEC_CMP_16(casenum,vect,act...) {VSTORE_U16(vect); VCMP(uint16_t,%x,casenum,Ru16,act)}
#define VEC_CMP_8(casenum,vect,act...)  {VSTORE_U8(vect); VCMP(uint8_t, %x,casenum, Ru8,act)}

#define VEC_CMP_U64(casenum,vect,act...) {VSTORE_U64(vect); VCMP(uint64_t,%x,casenum,Ru64,act)}
#define VEC_CMP_U32(casenum,vect,act...) {VSTORE_U32(vect); VCMP(uint32_t,%x,casenum,Ru32,act)}
#define VEC_CMP_U16(casenum,vect,act...) {VSTORE_U16(vect); VCMP(uint16_t,%x,casenum,Ru16,act)}
#define VEC_CMP_U8(casenum,vect,act...)  {VSTORE_U8(vect); VCMP(uint8_t, %x,casenum, Ru8,act)}

#define LVCMP_U8(casenum,vect,act)   {uint64_t vl; read_vl(vl); VSTORE_L8(vect);     \
                                       LVCMP(uint8_t, %x,casenum,vl, Lu8, act)}
#define LVCMP_U16(casenum,vect,act)  {uint64_t vl; read_vl(vl); VSTORE_L16(vect);    \
                                       LVCMP(uint16_t, %x,casenum,vl,Lu16,act)}
#define LVCMP_U32(casenum,vect,act)  {uint64_t vl; read_vl(vl); VSTORE_L32(vect);    \
                                       LVCMP(uint32_t, %x,casenum,vl,Lu32,act)}
#define LVCMP_U64(casenum,vect,act)  {uint64_t vl; read_vl(vl); VSTORE_L64(vect);    \
                                       LVCMP(uint64_t, %x,casenum,vl,Lu64,act)}

#define VVCMP_U64(casenum,ptr64,act...) {VCMP(uint64_t,%x,casenum,ptr64,act)}
#define VVCMP_U32(casenum,ptr32,act...) {VCMP(uint32_t,%x,casenum,ptr32,act)}
#define VVCMP_U16(casenum,ptr16,act...) {VCMP(uint16_t,%x,casenum,ptr16,act)}
#define VVCMP_U8(casenum,ptr8,act...)   {VCMP(uint8_t, %x,casenum,ptr8, act)}

#define LVVCMP_U64(casenum,ptr64,act) {uint64_t vl; read_vl(vl); LVCMP(uint64_t,%x,casenum,vl, ptr64,act)}
#define LVVCMP_U32(casenum,ptr32,act) {uint64_t vl; read_vl(vl); LVCMP(uint32_t,%x,casenum,vl, ptr32,act)}
#define LVVCMP_U16(casenum,ptr16,act) {uint64_t vl; read_vl(vl); LVCMP(uint16_t,%x,casenum,vl, ptr16,act)}
#define LVVCMP_U8(casenum,ptr8,act)   {uint64_t vl; read_vl(vl); LVCMP(uint8_t, %x,casenum,vl, ptr8, act)}

#define VCMP_I64(casenum,vect,act...) {VSTORE_I64(vect); VCMP(int64_t,%ld, casenum,Ri64,act)}
#define VCMP_I32(casenum,vect,act...) {VSTORE_I32(vect); VCMP(int32_t,%d,  casenum,Ri32,act)}
#define VCMP_I16(casenum,vect,act...) {VSTORE_I16(vect); VCMP(int16_t,%hd, casenum,Ri16,act)}
#define VCMP_I8(casenum,vect,act...)  {VSTORE_I8(vect);  VCMP(int8_t, %hhd,casenum,Ri8, act)}

#define VCMP_F64(casenum,vect,act...) {VSTORE_F64(vect); VCMP(double,%lf,casenum,Rf64,act)}
#define VCMP_F32(casenum,vect,act...) {VSTORE_F32(vect); VCMP(float, %f ,casenum,Rf32,act)}

// Vector load
#define VLOAD_64(vreg,vec...) VLOAD(uint64_t,e64,vreg,vec)
#define VLOAD_32(vreg,vec...) VLOAD(uint32_t,e32,vreg,vec)
#define VLOAD_16(vreg,vec...) VLOAD(uint16_t,e16,vreg,vec)
#define VLOAD_8(vreg,vec...)  VLOAD(uint8_t, e8, vreg,vec)

#define VLOAD_U64(vreg,vec...) VLOAD(uint64_t,e64,vreg,vec)
#define VLOAD_U32(vreg,vec...) VLOAD(uint32_t,e32,vreg,vec)
#define VLOAD_U16(vreg,vec...) VLOAD(uint16_t,e16,vreg,vec)
#define VLOAD_U8(vreg,vec...)  VLOAD(uint8_t, e8, vreg,vec)

// Vector store
#define VSTORE_U64(vreg) VSTORE(uint64_t,e64,vreg,Ru64)
#define VSTORE_U32(vreg) VSTORE(uint32_t,e32,vreg,Ru32)
#define VSTORE_U16(vreg) VSTORE(uint16_t,e16,vreg,Ru16)
#define VSTORE_U8(vreg)  VSTORE(uint8_t ,e8 ,vreg,Ru8 )

#define VSTORE_L64(vreg) VSTORE(uint64_t,e64,vreg,Lu64)
#define VSTORE_L32(vreg) VSTORE(uint32_t,e32,vreg,Lu32)
#define VSTORE_L16(vreg) VSTORE(uint16_t,e16,vreg,Lu16)
#define VSTORE_L8(vreg)  VSTORE(uint8_t ,e8 ,vreg,Lu8 )

#define VSTORE_I64(vreg) VSTORE(int64_t,e64,vreg,Ri64)
#define VSTORE_I32(vreg) VSTORE(int32_t,e32,vreg,Ri32)
#define VSTORE_I16(vreg) VSTORE(int16_t,e16,vreg,Ri16)
#define VSTORE_I8(vreg)  VSTORE(int8_t ,e8 ,vreg,Ri8 )

#define VSTORE_F64(vreg) VSTORE(double,e64,vreg,Rf64)
#define VSTORE_F32(vreg) VSTORE(float, e32,vreg,Rf32)

#endif // __VECTOR_MACROS_H__
