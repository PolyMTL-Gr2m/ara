// Author : Theo Dupuis
// GR2M - 2022
// Polytechnique Montreal

#include "ibsconv2d_tensor32.h"
#include <stdio.h>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                           Description : Functions for cross-correlation between                             //		
//                                                                                                             //
//                      1 x Cin x Hin x Win  * Cout x Cin x F x F   =    Cout x Hout x Wout                    //			
//                          input              kernels                       output                            //	
//																																					//
//																																					//				
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// *o    : tensor convolution output pointer 
// *i    : input tensor pointer
// *f    : kernel/filter tensor pointer
// Hin   : number of input cows
// Win   : number of input column
// Cin   : number of input channels 
// F     : size of the kernel/filter 
// Cout  : number of kernel/filter corresponding to the number of output channels

/////////////////////////////////////////////////////////////////
//
//                      1 Bits precision packing
//
/////////////////////////////////////////////////////////////////

/*void bitpack32_vec_1_to_32_2H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in){

   asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(len));
   
    int8_t *i_1 = tensor;
    int8_t *i_2 = tensor + C_in * W_in;
    
   for(int loop = 0; loop < 4; loop ++){
		
	asm volatile("vlse64.v v8, (%0), %1; addi %0, %0, 8" : "+&r" (i_2) : "r"(C_in));
	asm volatile("vlse64.v v12, (%0), %1; addi %0, %0, 8" : "+&r" (i_1) : "r"(C_in));
	
	
	////////////////////////////////////////
	asm volatile("vsll.vi v8, v8, 1");
	asm volatile("vor.vv v16, v12, v8");

	///////////////////////////////////// TO REPLACE WITH VSHACC
	
	// v0 = vpback(v16)
	asm volatile(".byte 0x57, 0x00, 0x08, 0x0E");
	
	}

	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(len));
		
 	asm volatile("vnsrl.wi v2, v0, 0");
 	asm volatile("vnsrl.wx v4, v0, %0" :: "r"(32));
}*/

void bitpack32_vec_1_to_32_2H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in){

   asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len));
   
    uint64_t const next_line = C_in * W_in;
   
    int8_t *i_1 = tensor;
    int8_t *i_2 = tensor + next_line;
    
   for(int loop = 0; loop < 4; loop ++){
		
		asm volatile("vlse64.v v8, (%0), %1; addi %0, %0, 8" : "+&r" (i_2) : "r"(C_in));
		asm volatile("vlse64.v v10, (%0), %1; addi %0, %0, 8" : "+&r" (i_1) : "r"(C_in));
		
		
		////////////////////////////////////////
		#ifndef PERF_VHSACC
		asm volatile("vsll.vi v8, v8, 1");
		#endif
		
		asm volatile("vor.vv v8, v8, v10");
		///////////////////////////////////// TO REPLACE WITH VSHACC
		
		// v0 = vpback(v8)
		asm volatile(".byte 0x57, 0x00, 0x04, 0x0E");
		
	
	}

	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(len));
		
 	asm volatile("vnsrl.wi v3, v0, 0");
 	asm volatile("vnsrl.wx v4, v0, %0" :: "r"(32));
 	
}

void bitpack32_vec_1_to_32_4H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in){

   asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len));
   
    uint64_t const next_line = C_in * W_in;
   
    int8_t *i_1 = tensor;
    int8_t *i_2 = tensor + next_line;
    int8_t *i_3 = tensor + 2 * next_line;
    int8_t *i_4 = tensor + 3 * next_line;
    
   for(int loop = 0; loop < 4; loop ++){
		
		asm volatile("vlse64.v v8, (%0), %1; addi %0, %0, 8" : "+&r" (i_2) : "r"(C_in));
		asm volatile("vlse64.v v10, (%0), %1; addi %0, %0, 8" : "+&r" (i_1) : "r"(C_in));
		
		asm volatile("vlse64.v v12, (%0), %1; addi %0, %0, 8" : "+&r" (i_4) : "r"(C_in));
		asm volatile("vlse64.v v14, (%0), %1; addi %0, %0, 8" : "+&r" (i_3) : "r"(C_in));
		
		
		////////////////////////////////////////
		#ifndef PERF_VHSACC
		asm volatile("vsll.vi v8, v8, 1");
		asm volatile("vsll.vi v12, v12, 1");
		#endif
		
		asm volatile("vor.vv v8, v8, v10");
		asm volatile("vor.vv v12, v12, v14");
		///////////////////////////////////// TO REPLACE WITH VSHACC
		
		// v0 = vpback(v8)
		asm volatile(".byte 0x57, 0x00, 0x04, 0x0E");
		
		// v2 = vpback(v12)
		asm volatile(".byte 0x57, 0x01, 0x26, 0x0E");
	
	}

	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(len));
		
 	asm volatile("vnsrl.wi v3, v0, 0");
 	asm volatile("vnsrl.wx v4, v0, %0" :: "r"(32));
 	
 	asm volatile("vnsrl.wi v5, v2, 0");
 	asm volatile("vnsrl.wx v6, v2, %0" :: "r"(32));
 	
}

void bitpack32_vec_1_to_32_6H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in){

   asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len));
   
    uint64_t const next_line = C_in * W_in;
   
    int8_t *i_1 = tensor;
    int8_t *i_2 = tensor + next_line;
    int8_t *i_3 = tensor + 2 * next_line;
    int8_t *i_4 = tensor + 3 * next_line;
    int8_t *i_5 = tensor + 4 * next_line;
    int8_t *i_6 = tensor + 5 * next_line;
    
   for(int loop = 0; loop < 4; loop ++){
		
		asm volatile("vlse64.v v8, (%0), %1; addi %0, %0, 8" : "+&r" (i_2) : "r"(C_in));
		asm volatile("vlse64.v v10, (%0), %1; addi %0, %0, 8" : "+&r" (i_1) : "r"(C_in));
		
		asm volatile("vlse64.v v12, (%0), %1; addi %0, %0, 8" : "+&r" (i_4) : "r"(C_in));
		asm volatile("vlse64.v v14, (%0), %1; addi %0, %0, 8" : "+&r" (i_3) : "r"(C_in));
		
		asm volatile("vlse64.v v16, (%0), %1; addi %0, %0, 8" : "+&r" (i_6) : "r"(C_in));
		asm volatile("vlse64.v v18, (%0), %1; addi %0, %0, 8" : "+&r" (i_5) : "r"(C_in));
		
		
		////////////////////////////////////////
		#ifndef PERF_VHSACC
		asm volatile("vsll.vi v8, v8, 1");
		asm volatile("vsll.vi v12, v12, 1");
		asm volatile("vsll.vi v16, v16, 1");
		#endif
		
		asm volatile("vor.vv v8, v8, v10");
		asm volatile("vor.vv v12, v12, v14");
		asm volatile("vor.vv v16, v16, v18");
		///////////////////////////////////// TO REPLACE WITH VSHACC
		
		// v0 = vpback(v8)
		asm volatile(".byte 0x57, 0x00, 0x04, 0x0E");
		
		// v2 = vpback(v12)
		asm volatile(".byte 0x57, 0x01, 0x26, 0x0E"); //f
		
		// v4 = vpback(v16)
		asm volatile(".byte 0x57, 0x02, 0x48, 0x0E");
	
	}

	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(len));
	
 	asm volatile("vnsrl.wi v7, v4, 0");
 	asm volatile("vnsrl.wx v8, v4, %0" :: "r"(32));
		
 	asm volatile("vnsrl.wi v3, v0, 0");
 	asm volatile("vnsrl.wx v4, v0, %0" :: "r"(32));
 	
 	asm volatile("vnsrl.wi v5, v2, 0");
 	asm volatile("vnsrl.wx v6, v2, %0" :: "r"(32));
 	
}

void bitpack_filter32_vec_1_to_32(int8_t * tensor, int32_t* packed_data, uint64_t len, uint64_t C_in){

	 uint64_t len_ = len + (len & 1); //to process data in case of odd or even
	 	
    asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len_ >> 1));
    
    asm volatile("vmv.v.i v20, 0"); 
    
    asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len_ >> 1));
    
    //asm volatile("vmv.v.i v0, 0");
    
    int8_t *i_1 = tensor; 
    int8_t *i_2 = (tensor + C_in); 
    
    int32_t *o = packed_data;

    
    // we use a 64b pointer *i, but careful because *tensor is a bit pointer
    
   
    uint64_t stride_i = (C_in << 1);
    
    for(int c = 0 ; c < (C_in >> 5); c ++){
 			  
			for (int8_t loop = 0 ; loop < 4 ; loop ++){
				
				asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len_ >> 1));
			
				asm volatile("vlse64.v v18, (%0), %1" : "+&r" (i_1) : "r"(stride_i));
				
				i_1 += 8;
				
				asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len >> 1));
				
				asm volatile("vlse64.v v20, (%0), %1" : "+&r" (i_2) : "r"(stride_i));
				
				i_2 += 8; 
				
				asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len_ >> 1)); 
				
				/////////////////////////////////////
				#ifndef PERF_VHSACC
			   asm volatile("vsll.vi v20, v20, 1");
			   #endif
			   
			   asm volatile("vor.vv v16, v20, v18");
			   ///////////////////////////////////// TO REPLACE WITH VSHACC
	 			
	 			// v0 = vpback(v16)
	 			asm volatile(".byte 0x57, 0x00, 0x08, 0x0E");
		 	
 			}	
 			
 			asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len_ >> 1));

 			asm volatile("vse64.v v0, (%0)" : "+&r"(o));
		 	o += len_;
	 }
}

/////////////////////////////////////////////////////////////////
//
//                      2 Bits precision packing
//
/////////////////////////////////////////////////////////////////




void bitpack32_vec_2_to_32_2H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in){

   asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len));
   
    uint64_t const next_line = C_in * W_in;
   
    int8_t *i_1 = tensor;
    int8_t *i_2 = tensor + next_line;
    
   for(int loop = 0; loop < 4; loop ++){
		
		asm volatile("vlse64.v v0, (%0), %1; addi %0, %0, 8" : "+&r" (i_1) : "r"(C_in));
		asm volatile("vlse64.v v2, (%0), %1; addi %0, %0, 8" : "+&r" (i_2) : "r"(C_in));
		
		// v8  = vpback(v0)
		asm volatile(".byte 0x57, 0x04, 0x80, 0x0E");
		// v10 = vpback(v2)
		asm volatile(".byte 0x57, 0x05, 0xA1, 0x0E");
		
	
	}

	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(len));
		
 	asm volatile("vnsrl.wi v0, v8, 0");
 	asm volatile("vnsrl.wi v1, v10, 0");
 	
 	asm volatile("vnsrl.wx v2, v8, %0" :: "r"(32));
 	asm volatile("vnsrl.wx v3, v10, %0" :: "r"(32));
 	
}

void bitpack32_vec_2_to_32_4H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in){

   asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len));
   
    uint64_t const next_line = C_in * W_in;
   
    int8_t *i_1 = tensor;
    int8_t *i_2 = tensor + next_line;
    int8_t *i_3 = tensor + 2 * next_line;
    int8_t *i_4 = tensor + 3 * next_line;
    
   for(int loop = 0; loop < 4; loop ++){
		
		asm volatile("vlse64.v v0, (%0), %1; addi %0, %0, 8" : "+&r" (i_1) : "r"(C_in));
		asm volatile("vlse64.v v2, (%0), %1; addi %0, %0, 8" : "+&r" (i_2) : "r"(C_in));
		asm volatile("vlse64.v v4, (%0), %1; addi %0, %0, 8" : "+&r" (i_3) : "r"(C_in));
		asm volatile("vlse64.v v6, (%0), %1; addi %0, %0, 8" : "+&r" (i_4) : "r"(C_in));
		
		// v8  = vpback(v0)
		asm volatile(".byte 0x57, 0x04, 0x80, 0x0E");
		// v10 = vpback(v2)
		asm volatile(".byte 0x57, 0x05, 0xA1, 0x0E");
		// v12 = vpback(v4)
		asm volatile(".byte 0x57, 0x06, 0xC2, 0x0E");
		// v14 = vpback(v6)
		asm volatile(".byte 0x57, 0x07, 0xE3, 0x0E");
		
	
	}

	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(len));
		
	asm volatile("vnsrl.wi v0, v8, 0");	
	asm volatile("vnsrl.wi v1, v10, 0");
	asm volatile("vnsrl.wi v2, v12, 0");	
	asm volatile("vnsrl.wi v3, v14, 0");
		
	asm volatile("vnsrl.wx v4, v8, %0" :: "r"(32));
 	asm volatile("vnsrl.wx v5, v10, %0" :: "r"(32));
 	asm volatile("vnsrl.wx v6, v12, %0" :: "r"(32));
	asm volatile("vnsrl.wx v7, v14, %0" :: "r"(32));
	
}

void bitpack32_vec_2_to_32_6H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in){

   asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(len));
   
    uint64_t const next_line = C_in * W_in;
   
    int8_t *i_1 = tensor;
    int8_t *i_2 = tensor + next_line;
    int8_t *i_3 = tensor + 2 * next_line;
    int8_t *i_4 = tensor + 3 * next_line;
    int8_t *i_5 = tensor + 4 * next_line;
    int8_t *i_6 = tensor + 5 * next_line;
    
   for(int loop = 0; loop < 4; loop ++){
		
		asm volatile("vlse64.v v0, (%0), %1; addi %0, %0, 8" : "+&r" (i_1) : "r"(C_in));
		asm volatile("vlse64.v v2, (%0), %1; addi %0, %0, 8" : "+&r" (i_2) : "r"(C_in));
		asm volatile("vlse64.v v4, (%0), %1; addi %0, %0, 8" : "+&r" (i_3) : "r"(C_in));
		
		// v8  = vpback(v0)
		asm volatile(".byte 0x57, 0x04, 0x80, 0x0E");
		// v10 = vpback(v2)
		asm volatile(".byte 0x57, 0x05, 0xA1, 0x0E");
		// v12 = vpback(v4)
		asm volatile(".byte 0x57, 0x06, 0xC2, 0x0E");
		
		
		
		asm volatile("vlse64.v v0, (%0), %1; addi %0, %0, 8" : "+&r" (i_4) : "r"(C_in));
		asm volatile("vlse64.v v2, (%0), %1; addi %0, %0, 8" : "+&r" (i_5) : "r"(C_in));
		asm volatile("vlse64.v v4, (%0), %1; addi %0, %0, 8" : "+&r" (i_6) : "r"(C_in));
		
		// v14 = vpback(v0)
		asm volatile(".byte 0x57, 0x07, 0xE0, 0x0E");
		// v16 = vpback(v2)
		asm volatile(".byte 0x57, 0x08, 0x01, 0x0F");
		// v18 = vpback(v4)
		asm volatile(".byte 0x57, 0x09, 0x22, 0x0F");
		
	
	}

	asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(len));
		
	asm volatile("vnsrl.wi v0, v8,  0");	
	asm volatile("vnsrl.wi v1, v10, 0");
	asm volatile("vnsrl.wi v2, v12, 0");	
	asm volatile("vnsrl.wi v3, v14, 0");
	asm volatile("vnsrl.wi v4, v16, 0");
	asm volatile("vnsrl.wi v5, v18, 0");
		
	asm volatile("vnsrl.wx v6,  v8,  %0" :: "r"(32));
 	asm volatile("vnsrl.wx v7,  v10, %0" :: "r"(32));
 	asm volatile("vnsrl.wx v8,  v12, %0" :: "r"(32));
	asm volatile("vnsrl.wx v9,  v14, %0" :: "r"(32));
	asm volatile("vnsrl.wx v10, v16, %0" :: "r"(32));
	asm volatile("vnsrl.wx v11, v18, %0" :: "r"(32));
 	
}

void bitpack_filter32_vec_2_to_32(int8_t * tensor, int32_t* packed_data, uint64_t len, uint64_t C_in){
 
    // tensor       : input tensor
    // packed_data  : output packed data
    // len			  : number of elements in the filter
    // C_in			  : 
        
    int64_t * filter_ptr = tensor;
    int32_t * o = packed_data;

    uint64_t stride_i = C_in;

    asm volatile("vsetvli zero, %0, e64, m1, ta, ma" ::"r"(len));
    
    for(int c = 0 ; c < (C_in >> 5); c ++){
    
    //asm volatile("vmv.v.i v0, 0");
    
    // load an len * 8 * 8b inputs in 64b registers  and extract the 2 LSB to pack them
    // e.g. load 0x0203020103020103 in a 64b register means you load an 8b vector of v8b = {2, 3, 2, 1, 3, 2, 1, 3}
    // The input layout has to be NHWC to be able to vectorize the packing 
    
    // The way it works for a 3x3x64 input is:
    // 1. consider the 3x3 first plane as a vector, we load only the first 8 channels in an element (1)
    // 2. with a stride on the load instruction, we load the 8 channels of the second value (2)
    //
    //      ■■■■■■■■■   channel 64
    //    ...............
    //   ■■■■■■■■■   channel 1
    // ■■■■■■■■■   channel 0
    // 1 2
    //
    // 3. The vbpack instructions in the loop allows us to pack the data with an accumulator
    // 4. The 32 MSB are the MSB packed bits
    // 5. The 32 LSB are the LSB packed bits
    			  
			for (int8_t loop = 0 ; loop < 4 ; loop ++){

				asm volatile("vlse64.v v8, (%0), %1" : "+&r" (filter_ptr) : "r"(C_in));
	 			
			   filter_ptr += 1;
	 			
	 			//	funct6	vm	vs2	vs1		funct3	vd			OPIVV	
	 			//	0000 11|1  |0 0000|0100	0|000		0000 0|101 0111	
	 			//		0		E		0		4		0			0		57
	 			//
	 			
	 			// v0 = vbpack(v0, v8) // careful, vd and vs2 must be the same
	 			asm volatile(".byte 0x57, 0x00, 0x04, 0x0E");
		 	
 			}
 			asm volatile("vse64.v v0, (%0)" : "+&r"(o));
		 	o += (len << 1); 
	 }
}

