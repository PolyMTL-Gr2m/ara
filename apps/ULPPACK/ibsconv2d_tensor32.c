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

void ulppack_conv2d(int16_t *o, int8_t *i, int8_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out, int64_t precA, int64_t precW) {
	
  int8_t *i_;
  int16_t *o_;
  int8_t *f_;
  
  //helper variable
  
  for(int64_t c = 0; c < C_out; c++) {
  		// First iteration round, c = 0 for the adress of the first value of the first filter
		o_ = o + c * (H_in - F + 1) * (W_in - F + 1);      // Output is incremented 
		i_ = i;                                            // Since we aren't working on batch, we only consider one input
		f_ = f + c * F * F * C_in;
		#ifdef VMACSR
			if(F == 7)
				if((precA <= 2 && precW < 2) || (precA < 2 && precW <= 2))
					ulppack_conv2d_vec8_7x7(o_, i_, f_, H_in, W_in, C_in, F, C_out);
				else 
					ulppack_conv2d_vec16_7x7(o_, i_, f_, H_in, W_in, C_in, F, C_out);
		
		#else
		if(F == 3)
			ulppack_conv2d32_W2_A2_vec_3x3(o_, i_, f_, H_in, W_in, C_in, F, C_out);
		else if (F == 7)
			if (precA <= 1 && precW <= 1)
				ulppack_conv2d_vec_7x7_A1W1(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			else if (precA <= 2 && precW <= 2)
		 		ulppack_conv2d_vec_7x7_A2W2(o_, i_, f_, H_in, W_in, C_in, F, C_out);
			else if (precA <= 3 && precW <= 3)
		 		ulppack_conv2d_vec_7x7_A3W3(o_, i_, f_, H_in, W_in, C_in, F, C_out);
		#endif
	}
}

void print_tensor_(uint16_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
  printf("0x%8X\n", (uint64_t)tensor);
  for (uint64_t k = 0; k < num_depth; ++k) {
  	for (uint64_t i = 0; i < num_rows; ++i) {
    	for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%lu ", tensor[(i+k*num_rows) * num_columns  + j ]);
    	}
    	printf("\n");
  	 }
  	 printf("\n");
	}
}


///////////////////////////////////////////
//                                       //
//                 7x7                   //
//                                       //
///////////////////////////////////////////

// precision ensuring no overflow on dot-product

// Weights (W)
// precision
//
// +---+---+
// | 7|| X |
// +---+---+
// | 6|| X | 
// +---+---+---+
// | 5|| X | X |
// +---+---+---+---+
// | 4|| X | X | X |
// +---+---+---+---+---+
// | 3|| X | X | X | X |
// +---+---+---+---+---+---+
// | 2|| X | X | X | X | X |
// +---+---+---+---+---+---+---+---+
// | 1|| X | X | X | X | X | X | X |
// +---+===+===+===+===+===+===+===+
// bits| 1 | 2 | 3 | 4 | 5 | 6 | 7 | Activation (A)
//     +---+---+---+---+---+---+---+ precision

void ulppack_conv2d_vec16_7x7(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

uint8_t *f_ = f_ptr; 

uint64_t const ldo = W_in - F + 1;

uint64_t ldi = W_in;

uint64_t vlen;

uint16_t f_packed[F * F * (C_in >> 1)];
uint16_t *f_loop = f_packed;
	
	// PACKING THE FILTER
	
	for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){

	
		uint64_t ldf = F * F;
		
		asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(F * F));

		asm volatile("vle8.v v1, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		asm volatile("vle8.v v2, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		
		asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(F * F));
		
		asm volatile("vzext.vf2 v0, v1");
		asm volatile("vzext.vf2 v1, v2");
		
		asm volatile("vmacc.vx v0, %0, v1" ::"r"(256));
		
		asm volatile("vse16.v v0, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf << 1));
	}	
	
	for (int width = 0 ; width < (W_in - 6) ; width += VLEN_MF2_OUT) // IF CONVOLUTION NEED TO BE TILED (C > VLEN_WIDE)
	{

	
		int8_t *i_ = i_ptr + width; 									// input pointer realtive to the tile (constant throughout the tile)
		int16_t *o_ = o_ptr + width;	
		
		if(width > W_in - VLEN_MF2) 	// if we are at the right border of the input
			vlen = (W_in - width);		 	// we set the vector length to fit the last inputs
		else
			vlen = VLEN_MF2;						// else we go full length
		
		int8_t * i__ = i_;							// inside tile pointer (change at each load)
		
		int64_t vlen_out = vlen - 6;
		
		for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
			
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
			i__ = i_ + 2 * channels * H_in * W_in;
			
			asm volatile("vle8.v v6, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v7, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v8, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v9, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v10,(%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v11,(%0)"                 : "+&r" (i__));
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
			
			asm volatile("vzext.vf2 v0, v6");
			asm volatile("vzext.vf2 v1, v7");
			asm volatile("vzext.vf2 v2, v8");
			asm volatile("vzext.vf2 v3, v9");
			asm volatile("vzext.vf2 v4, v10");
			asm volatile("vzext.vf2 v5, v11");
			
			asm volatile("vsll.vi v0, v0, 8");
			asm volatile("vsll.vi v1, v1, 8");
			asm volatile("vsll.vi v2, v2, 8");
			asm volatile("vsll.vi v3, v3, 8");
			asm volatile("vsll.vi v4, v4, 8");
			asm volatile("vsll.vi v5, v5, 8");	
			
			//asm volatile("vse16.v v5, (%0)" : "+&r"(o_));
			//print_tensor_(o_, 1, W_in, 1);
			
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
			i__ += (H_in - F + 2) * W_in;
			
			
			asm volatile("vle8.v v6,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v7,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v8,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v9,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v10, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v11, (%0)"                 : "+&r" (i__));

			asm volatile("vwadd.wv v0, v0, v6");
			asm volatile("vwadd.wv v1, v1, v7");
			asm volatile("vwadd.wv v2, v2, v8");
			asm volatile("vwadd.wv v3, v3, v9");
			asm volatile("vwadd.wv v4, v4, v10");
			asm volatile("vwadd.wv v5, v5, v11");
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
			
			//asm volatile("vse16.v v5, (%0)" : "+&r"(o_));
			//print_tensor_(o_, 1, W_in, 1);
			
		
			// Each iteration of tis loop processes one column of the filter
			// Each iteration of tis loop processes one column of the filter
			for(int f_w = 0; f_w < F ; f_w++){
			
				f_loop = f_packed + f_w + channels * F * F;
				
				uint8_t ldf = F << 1;

				// the register "t0->t5" in clobber list specify to the compiler
				// to avoid using those register since we need them to compute the conv2d
				// the .byte directive should be used carefully with scalar registers 
				
				asm volatile("lh t0, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t0");
				asm volatile("lh t1, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t1");
				asm volatile("lh t2, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t2");
				asm volatile("lh t3, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t3");
				asm volatile("lh t4, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t4");
				asm volatile("lh t5, (%0)"                 : "+&r"(f_loop) :          :"t5");

				if(f_w > 0 || channels > 0){
					//MACC_SR4
					
					//vmacsr.v.x v25, t0, v0
					asm volatile (".byte 0xd7, 0xec, 0x02, 0xba");
					//vmacsr.v.x v26, t0, v1
					asm volatile (".byte 0x57, 0xed, 0x12, 0xba");
					//vmacsr.v.x v27, t0, v2
					asm volatile (".byte 0xd7, 0xed, 0x22, 0xba");
					//vmacsr.v.x v28, t0, v3
					asm volatile (".byte 0x57, 0xee, 0x32, 0xba");
					//vmacsr.v.x v29, t0, v4
					asm volatile (".byte 0xd7, 0xee, 0x42, 0xba");
					//vmacsr.v.x v30, t0, v5
					asm volatile (".byte 0x57, 0xef, 0x52, 0xba");

				}
				else{
					//vmulsr.v.x v25, t0, v0
					asm volatile (".byte 0xd7, 0xec, 0x02, 0xa2");
					//vmulsr.v.x v26, t0, v1
					asm volatile (".byte 0x57, 0xed, 0x12, 0xa2");
					//vmulsr.v.x v27, t0, v2
					asm volatile (".byte 0xd7, 0xed, 0x22, 0xa2");
					//vmulsr.v.x v28, t0, v3
					asm volatile (".byte 0x57, 0xee, 0x32, 0xa2");
					//vmulsr.v.x v29, t0, v4
					asm volatile (".byte 0xd7, 0xee, 0x42, 0xa2");
					//vmulsr.v.x v30, t0, v5
					asm volatile (".byte 0x57, 0xef, 0x52, 0xa2");

				}
				
				//vmacsr.v.x v25, t1, v1
				asm volatile (".byte 0xd7, 0x6c, 0x13, 0xba");
				//vmacsr.v.x v26, t1, v2
				asm volatile (".byte 0x57, 0x6d, 0x23, 0xba");
				//vmacsr.v.x v27, t1, v3
				asm volatile (".byte 0xd7, 0x6d, 0x33, 0xba");
				//vmacsr.v.x v28, t1, v4
				asm volatile (".byte 0x57, 0x6e, 0x43, 0xba");
				//vmacsr.v.x v29, t1, v5
				asm volatile (".byte 0xd7, 0x6e, 0x53, 0xba");

				//vmacsr.v.x v25, t2, v2
				asm volatile (".byte 0xd7, 0xec, 0x23, 0xba");
				//vmacsr.v.x v26, t2, v3
				asm volatile (".byte 0x57, 0xed, 0x33, 0xba");
				//vmacsr.v.x v27, t2, v4
				asm volatile (".byte 0xd7, 0xed, 0x43, 0xba");
				//vmacsr.v.x v28, t2, v5
				asm volatile (".byte 0x57, 0xee, 0x53, 0xba");

				//vmacsr.v.x v25, t3, v3
				asm volatile (".byte 0xd7, 0x6c, 0x3e, 0xba");
				//vmacsr.v.x v26, t3, v4
				asm volatile (".byte 0x57, 0x6d, 0x4e, 0xba");
				//vmacsr.v.x v27, t3, v5
				asm volatile (".byte 0xd7, 0x6d, 0x5e, 0xba");

				//vmacsr.v.x v25, t4, v4
				asm volatile (".byte 0xd7, 0xec, 0x4e, 0xba");
				//vmacsr.v.x v26, t4, v5
				asm volatile (".byte 0x57, 0xed, 0x5e, 0xba");
				
				//vmacsr.v.x v25, t5, v5
				asm volatile (".byte 0xd7, 0x6c, 0x5f, 0xba");



				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v1, v1, 1");
					asm volatile("vslidedown.vi v2, v2, 1");
					asm volatile("vslidedown.vi v3, v3, 1");
					asm volatile("vslidedown.vi v4, v4, 1");
					asm volatile("vslidedown.vi v5, v5, 1");
				}
				
			}

		}
		
		i_ += (F - 1) * W_in;
		
		for (int height = F - 1 ; height < H_in ; height += 6){
		
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
					
			for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
			
				i__ = i_ + 2 * channels * H_in * W_in;
				
				asm volatile("vle8.v v6, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 1)
				asm volatile("vle8.v v7, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 2)
				asm volatile("vle8.v v8, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 3)
				asm volatile("vle8.v v9, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 4)
				asm volatile("vle8.v v10,(%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 5)
				asm volatile("vle8.v v11,(%0)"                 : "+&r" (i__));
				
				
				asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
				
				asm volatile("vzext.vf2 v0, v6");
				if(height < H_in - 1)
				asm volatile("vzext.vf2 v1, v7");
				if(height < H_in - 2)
				asm volatile("vzext.vf2 v2, v8");
				if(height < H_in - 3)
				asm volatile("vzext.vf2 v3, v9");
				if(height < H_in - 4)
				asm volatile("vzext.vf2 v4, v10");
				if(height < H_in - 5)
				asm volatile("vzext.vf2 v5, v11");
				
				asm volatile("vsll.vi v0, v0, 8");
				if(height < H_in - 1)
				asm volatile("vsll.vi v1, v1, 8");
				if(height < H_in - 2)
				asm volatile("vsll.vi v2, v2, 8");
				if(height < H_in - 3)
				asm volatile("vsll.vi v3, v3, 8");
				if(height < H_in - 4)
				asm volatile("vsll.vi v4, v4, 8");
				if(height < H_in - 5)
				asm volatile("vsll.vi v5, v5, 8");	
				
				//asm volatile("vse16.v v5, (%0)" : "+&r"(o_));
				//print_tensor_(o_, 1, W_in, 1);
				
				asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
				
				i__ = i_ + (2 * channels + 1) * H_in * W_in;
				
				
				asm volatile("vle8.v v6,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 1)
				asm volatile("vle8.v v7,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 2)
				asm volatile("vle8.v v8,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 3)
				asm volatile("vle8.v v9,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 4)
				asm volatile("vle8.v v10, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 5)
				asm volatile("vle8.v v11, (%0)"                 : "+&r" (i__));

				asm volatile("vwadd.wv v0, v0, v6");
				if(height < H_in - 1)
				asm volatile("vwadd.wv v1, v1, v7");
				if(height < H_in - 2)
				asm volatile("vwadd.wv v2, v2, v8");
				if(height < H_in - 3)
				asm volatile("vwadd.wv v3, v3, v9");
				if(height < H_in - 4)
				asm volatile("vwadd.wv v4, v4, v10");
				if(height < H_in - 5)
				asm volatile("vwadd.wv v5, v5, v11");
				
				asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
				
				
				for(int f_w = 0; f_w < F ; f_w++){ // go through every column
					
					f_loop = f_packed + f_w + channels * F * F;
					
					int8_t ldf = F << 1;

					// the register "t0->t5" in clobber list specify to the compiler
					// to avoid using those register since we need them to compute the conv2d
					// the .byte directive should be used carefully with scalar registers 
	
					asm volatile("lh t0, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t0");
					asm volatile("lh t1, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t1");
					asm volatile("lh t2, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t2");
					asm volatile("lh t3, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t3");
					asm volatile("lh t4, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t4");
					asm volatile("lh t5, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t5");
					asm volatile("lh t6, (%0)" 					 : "+&r"(f_loop) :          :"t6");

					
					// Row 0					
					if(f_w > 0 || channels > 0){
						//vmacsr.v.x v25, t0, v0
						asm volatile (".byte 0xd7, 0xec, 0x02, 0xba");
						//vmacsr.v.x v24, t1, v0
						asm volatile (".byte 0x57, 0x6c, 0x03, 0xba");
						//vmacsr.v.x v23, t2, v0
						asm volatile (".byte 0xd7, 0xeb, 0x03, 0xba");
						//vmacsr.v.x v22, t3, v0
						asm volatile (".byte 0x57, 0x6b, 0x0e, 0xba");
						//vmacsr.v.x v21, t4, v0
						asm volatile (".byte 0xd7, 0xea, 0x0e, 0xba");
						//vmacsr.v.x v20, t5, v0
						asm volatile (".byte 0x57, 0x6a, 0x0f, 0xba");
						//vmacsr.v.x v19, t6, v0
						asm volatile (".byte 0xd7, 0xe9, 0x0f, 0xba");

					}
					else{
						//vmulsr.v.x v24, t1, v0
						asm volatile (".byte 0x57, 0x6c, 0x03, 0xa2");
						//vmulsr.v.x v23, t2, v0
						asm volatile (".byte 0xd7, 0xeb, 0x03, 0xa2");
						//vmulsr.v.x v22, t3, v0
						asm volatile (".byte 0x57, 0x6b, 0x0e, 0xa2");
						//vmulsr.v.x v21, t4, v0
						asm volatile (".byte 0xd7, 0xea, 0x0e, 0xa2");
						//vmulsr.v.x v20, t5, v0
						asm volatile (".byte 0x57, 0x6a, 0x0f, 0xa2");
						//vmulsr.v.x v19, t6, v0
						asm volatile (".byte 0xd7, 0xe9, 0x0f, 0xa2");
						
						asm volatile("vadd.vv v24, v24, v30");
						asm volatile("vadd.vv v23, v23, v29");
						asm volatile("vadd.vv v22, v22, v28");
						asm volatile("vadd.vv v21, v21, v27");
						asm volatile("vadd.vv v20, v20, v26");
						asm volatile("vadd.vv v19, v19, v25");
					
						asm volatile (".byte 0xd7, 0xec, 0x02, 0xa2");
						//vmulsr.vx v25, v0, t0
					}
					
					
					// Row 1
					if(height < H_in - 1){
						if(f_w > 0 || channels > 0)
							//vmacsr.v.x v26, t0, v1
							asm volatile (".byte 0x57, 0xed, 0x12, 0xba");
						else
							asm volatile(" .byte 0x57, 0xed, 0x12, 0xA2");
							//vop v26, v5, v1
							

						//vmacsr.v.x v25, t1, v1
						asm volatile (".byte 0xd7, 0x6c, 0x13, 0xba");
						//vmacsr.v.x v24, t2, v1
						asm volatile (".byte 0x57, 0xec, 0x13, 0xba");
						//vmacsr.v.x v23, t3, v1
						asm volatile (".byte 0xd7, 0x6b, 0x1e, 0xba");
						//vmacsr.v.x v22, t4, v1
						asm volatile (".byte 0x57, 0xeb, 0x1e, 0xba");
						//vmacsr.v.x v21, t5, v1
						asm volatile (".byte 0xd7, 0x6a, 0x1f, 0xba");
						//vmacsr.v.x v20, t6, v1
						asm volatile (".byte 0x57, 0xea, 0x1f, 0xba");


					
					
					
						// Row 2
						if(height < H_in - 2){
							if(f_w > 0 || channels > 0)
								asm volatile(" .byte 0xd7, 0xed, 0x22, 0xba");
								//vop v27, v5, v2
							else
								asm volatile(" .byte 0xd7, 0xed, 0x22, 0xA2");
								//vop v27, v5, v2
								
							asm volatile(" .byte 0x57, 0x6d, 0x23, 0xba");
							//vop v26, v6, v2
							asm volatile(" .byte 0xd7, 0xec, 0x23, 0xba");
							//vop v25, v7, v2
							asm volatile(" .byte 0x57, 0x6c, 0x2e, 0xba");
							//vop v24, v28, v2
							asm volatile(" .byte 0xd7, 0xeb, 0x2e, 0xba");
							//vop v23, v29, v2
							asm volatile(" .byte 0x57, 0x6b, 0x2f, 0xba");
							//vop v22, v30, v2
							asm volatile(" .byte 0xd7, 0xea, 0x2f, 0xba");
							//vop v21, v31, v2
						
						
							// Row 3
							if(height < H_in - 3){

								if(f_w > 0 || channels > 0)
									asm volatile(" .byte 0x57, 0xee, 0x32, 0xba");
									//vop v28, v5, v3
								else
									asm volatile(" .byte 0x57, 0xee, 0x32, 0xA2");
									//vop v28, v5, v3

								
								asm volatile(" .byte 0xd7, 0x6d, 0x33, 0xba");
								//vop v27, v6, v3
								asm volatile(" .byte 0x57, 0xed, 0x33, 0xba");
								//vop v26, v7, v3
								asm volatile(" .byte 0xd7, 0x6c, 0x3e, 0xba");
								//vop v25, v28, v3
								asm volatile(" .byte 0x57, 0xec, 0x3e, 0xba");
								//vop v24, v29, v3
								asm volatile(" .byte 0xd7, 0x6b, 0x3f, 0xba");
								//vop v23, v30, v3
								asm volatile(" .byte 0x57, 0xeb, 0x3f, 0xba");
								//vop v22, v31, v3
							
							
								// Row 4
								if(height < H_in - 4){

									if(f_w > 0 || channels > 0)
										asm volatile(" .byte 0xd7, 0xee, 0x42, 0xba");
										//vop v29, v5, v4
									else
										asm volatile(" .byte 0xd7, 0xee, 0x42, 0xA2");
										//vop v29, v5, v4


									
									asm volatile(" .byte 0x57, 0x6e, 0x43, 0xba");
									//vop v28, v6, v4
									asm volatile(" .byte 0xd7, 0xed, 0x43, 0xba");
									//vop v27, v7, v4
									asm volatile(" .byte 0x57, 0x6d, 0x4e, 0xba");
									//vop v26, v28, v4
									asm volatile(" .byte 0xd7, 0xec, 0x4e, 0xba");
									//vop v25, v29, v4
									asm volatile(" .byte 0x57, 0x6c, 0x4f, 0xba");
									//vop v24, v30, v4
									asm volatile(" .byte 0xd7, 0xeb, 0x4f, 0xba");
									//vop v23, v31, v4

								
								
									// Row 5
									if(height < H_in - 5){

										if(f_w > 0 || channels > 0)
											asm volatile(" .byte 0x57, 0xef, 0x52, 0xba");
											//vop v30, v5, v5
										else
											asm volatile(" .byte 0x57, 0xef, 0x52, 0xA2");
											//vop v30, v5, v5
										
										asm volatile(" .byte 0xd7, 0x6e, 0x53, 0xba");
										//vop v29, v6, v5
										asm volatile(" .byte 0x57, 0xee, 0x53, 0xba");
										//vop v28, v7, v5
										asm volatile(" .byte 0xd7, 0x6d, 0x5e, 0xba");
										//vop v27, v28, v5
										asm volatile(" .byte 0x57, 0xed, 0x5e, 0xba");
										//vop v26, v29, v5
										asm volatile(" .byte 0xd7, 0x6c, 0x5f, 0xba");
										//vop v25, v30, v5
										asm volatile(" .byte 0x57, 0xec, 0x5f, 0xba");
										//vop v24, v31, v5
									}
								}
							}
						}
					}
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						if(height < H_in - 1){
							asm volatile("vslidedown.vi v1, v1, 1");
							if(height < H_in - 2){
								asm volatile("vslidedown.vi v2, v2, 1");
								if(height < H_in - 3){
									asm volatile("vslidedown.vi v3, v3, 1");
									if(height < H_in - 4){
										asm volatile("vslidedown.vi v4, v4, 1");
										if(height < H_in - 5){
											asm volatile("vslidedown.vi v5, v5, 1");
										}
									}
								}
							}
						}
					}
				}
			}
			
			i_ += (F - 1) * W_in;
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen_out));
			
			asm volatile("vse16.v v19, (%0)" : "+&r"(o_));
			o_ += ldo;
			if(height < H_in - 1){
				asm volatile("vse16.v v20, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 2){
				asm volatile("vse16.v v21, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 3){
				asm volatile("vse16.v v22, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 4){
				asm volatile("vse16.v v23, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 5){
				asm volatile("vse16.v v24, (%0)" : "+&r"(o_));;
				o_ += ldo;
			}
		}
	}
}

// precision ensuring no overflow on dot-product

// Weights (W)
// precision
//
// +---+---+---+
// | 2|| X |   |
// +---+---+---+
// | 1|| X | X |
// +---+===+===+
// bits| 1 | 2 |  Activation (A)
//     +---+---+ precision

void ulppack_conv2d_vec8_7x7(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

uint8_t *f_ = f_ptr; 

uint64_t const ldo = W_in - F + 1;

uint64_t ldi = W_in;

uint64_t vlen;

uint8_t f_packed[F * F * (C_in >> 1)];
uint8_t *f_loop = f_packed;
	
	// PACKING THE FILTER

	
	for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
		
		uint64_t ldf = F * F;
		
		asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(F * F));

		asm volatile("vle8.v v1, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		asm volatile("vle8.v v0, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		
		asm volatile("vmacc.vx v0, %0, v1" ::"r"(16));
				
		asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf));
			
	}
	
	for (int width = 0 ; width < (W_in - 6) ; width += VLEN_MF2_OUT) // IF CONVOLUTION NEED TO BE TILED (C > VLEN_WIDE)
	{

	
		int8_t *i_ = i_ptr + width; 									// input pointer realtive to the tile (constant throughout the tile)
		int16_t *o_ = o_ptr + width;	
		
		if(width > W_in - VLEN_MF2) 	// if we are at the right border of the input
			vlen = (W_in - width);		 	// we set the vector length to fit the last inputs
		else
			vlen = VLEN_MF2;						// else we go full length
		
		int8_t * i__ = i_;							// inside tile pointer (change at each load)
		
		int64_t vlen_out = vlen - 6;
		
		for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
			
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
			i__ = i_ + 2 * channels * H_in * W_in;
			
			asm volatile("vle8.v v0, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v1, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v2, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v3, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v4, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v5, (%0)"                 : "+&r" (i__));
			
			i__ += (H_in - F + 2) * W_in;
			
			asm volatile("vle8.v v19,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v20,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v21,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v22,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v23, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v24, (%0)"                 : "+&r" (i__));
			
			// PACKING 2 8b data into 1 8b register (with 4b interval)
			
			int8_t mul_shift = 16;
			
			asm volatile("vmacc.vx v0, %0, v19" ::"r"(mul_shift));
			asm volatile("vmacc.vx v1, %0, v20" ::"r"(mul_shift));
			asm volatile("vmacc.vx v2, %0, v21" ::"r"(mul_shift));
			asm volatile("vmacc.vx v3, %0, v22" ::"r"(mul_shift));
			asm volatile("vmacc.vx v4, %0, v23" ::"r"(mul_shift));
			asm volatile("vmacc.vx v5, %0, v24" ::"r"(mul_shift));

			// Each iteration of tis loop processes one column of the filter
			for(int f_w = 0; f_w < F ; f_w++){
			
				f_loop = f_packed + f_w + channels * F * F;
				
				int8_t ldf = F;

				// the register "t0->t5" in clobber list specify to the compiler
				// to avoid using those register since we need them to compute the conv2d
				// the .byte directive should be used carefully with scalar registers 
				
				asm volatile("lb t0, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t0");
				asm volatile("lb t1, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t1");
				asm volatile("lb t2, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t2");
				asm volatile("lb t3, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t3");
				asm volatile("lb t4, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t4");
				asm volatile("lb t5, (%0)"                 : "+&r"(f_loop) :          :"t5");

				if(f_w > 0){
					//MACC_SR4
					
					//vmacsr.v.x v25, t0, v0
					asm volatile (".byte 0xd7, 0xec, 0x02, 0xba");
					//vmacsr.v.x v26, t0, v1
					asm volatile (".byte 0x57, 0xed, 0x12, 0xba");
					//vmacsr.v.x v27, t0, v2
					asm volatile (".byte 0xd7, 0xed, 0x22, 0xba");
					//vmacsr.v.x v28, t0, v3
					asm volatile (".byte 0x57, 0xee, 0x32, 0xba");
					//vmacsr.v.x v29, t0, v4
					asm volatile (".byte 0xd7, 0xee, 0x42, 0xba");
					//vmacsr.v.x v30, t0, v5
					asm volatile (".byte 0x57, 0xef, 0x52, 0xba");

				}
				else{
					//vmulsr.v.x v25, t0, v0
					asm volatile (".byte 0xd7, 0xec, 0x02, 0xa2");
					//vmulsr.v.x v26, t0, v1
					asm volatile (".byte 0x57, 0xed, 0x12, 0xa2");
					//vmulsr.v.x v27, t0, v2
					asm volatile (".byte 0xd7, 0xed, 0x22, 0xa2");
					//vmulsr.v.x v28, t0, v3
					asm volatile (".byte 0x57, 0xee, 0x32, 0xa2");
					//vmulsr.v.x v29, t0, v4
					asm volatile (".byte 0xd7, 0xee, 0x42, 0xa2");
					//vmulsr.v.x v30, t0, v5
					asm volatile (".byte 0x57, 0xef, 0x52, 0xa2");

				}
				
				//vmacsr.v.x v25, t1, v1
				asm volatile (".byte 0xd7, 0x6c, 0x13, 0xba");
				//vmacsr.v.x v26, t1, v2
				asm volatile (".byte 0x57, 0x6d, 0x23, 0xba");
				//vmacsr.v.x v27, t1, v3
				asm volatile (".byte 0xd7, 0x6d, 0x33, 0xba");
				//vmacsr.v.x v28, t1, v4
				asm volatile (".byte 0x57, 0x6e, 0x43, 0xba");
				//vmacsr.v.x v29, t1, v5
				asm volatile (".byte 0xd7, 0x6e, 0x53, 0xba");

				//vmacsr.v.x v25, t2, v2
				asm volatile (".byte 0xd7, 0xec, 0x23, 0xba");
				//vmacsr.v.x v26, t2, v3
				asm volatile (".byte 0x57, 0xed, 0x33, 0xba");
				//vmacsr.v.x v27, t2, v4
				asm volatile (".byte 0xd7, 0xed, 0x43, 0xba");
				//vmacsr.v.x v28, t2, v5
				asm volatile (".byte 0x57, 0xee, 0x53, 0xba");

				//vmacsr.v.x v25, t3, v3
				asm volatile (".byte 0xd7, 0x6c, 0x3e, 0xba");
				//vmacsr.v.x v26, t3, v4
				asm volatile (".byte 0x57, 0x6d, 0x4e, 0xba");
				//vmacsr.v.x v27, t3, v5
				asm volatile (".byte 0xd7, 0x6d, 0x5e, 0xba");

				//vmacsr.v.x v25, t4, v4
				asm volatile (".byte 0xd7, 0xec, 0x4e, 0xba");
				//vmacsr.v.x v26, t4, v5
				asm volatile (".byte 0x57, 0xed, 0x5e, 0xba");
				
				//vmacsr.v.x v25, t5, v5
				asm volatile (".byte 0xd7, 0x6c, 0x5f, 0xba");



				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v1, v1, 1");
					asm volatile("vslidedown.vi v2, v2, 1");
					asm volatile("vslidedown.vi v3, v3, 1");
					asm volatile("vslidedown.vi v4, v4, 1");
					asm volatile("vslidedown.vi v5, v5, 1");
				}
				
			}
			
			// Compute every iteration on intermediate 8b registers
			// And avoid overflow by storing every 2 channel result on 16b registers
						
			
			if(channels > 0){
				asm volatile("vwadd.wv v12,  v12,  v25");
				asm volatile("vwadd.wv v13,  v13,  v26");
				asm volatile("vwadd.wv v14,  v14,  v27");
				asm volatile("vwadd.wv v15,  v15,  v28");
				asm volatile("vwadd.wv v16,  v16,  v29");
				asm volatile("vwadd.wv v17,  v17,  v30");
			}
			else{
			
				asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
				
				asm volatile("vzext.vf2 v12, v25");
				asm volatile("vzext.vf2 v13, v26");
				asm volatile("vzext.vf2 v14, v27");
				asm volatile("vzext.vf2 v15, v28");
				asm volatile("vzext.vf2 v16, v29");
				asm volatile("vzext.vf2 v17, v30");
			}

		}
		
		
		i_ += (F - 1) * W_in;
		
		for (int height = F - 1 ; height < H_in ; height += 6){
		
					
			for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
			
				asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
				i__ = i_ + 2 * channels * H_in * W_in;
				
				
				asm volatile("vle8.v v0, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 1){
					asm volatile("vle8.v v1, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
					if(height < H_in - 2){
						asm volatile("vle8.v v2, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
						if(height < H_in - 3){
							asm volatile("vle8.v v3, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
							if(height < H_in - 4){
								asm volatile("vle8.v v4, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
								if(height < H_in - 5){
									asm volatile("vle8.v v5, (%0)"                 : "+&r" (i__));

								}
							}
						}
					}
				}
				
				i__ = i_ + (2 * channels + 1) * H_in * W_in;
				
				asm volatile("vle8.v v19,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 1){
					asm volatile("vle8.v v20,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
					if(height < H_in - 2){
						asm volatile("vle8.v v21,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
						if(height < H_in - 3){
							asm volatile("vle8.v v22,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
							if(height < H_in - 4){
								asm volatile("vle8.v v23, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
									if(height < H_in - 5){
										asm volatile("vle8.v v24, (%0)"                 : "+&r" (i__));
									}
								}
							}
						}
					}
				// PACKING 2 8b data into 1 8b register (with 4b interval)
				
				int8_t mul_shift = 16;
				
				asm volatile("vmacc.vx v0, %0, v19"  ::"r"(mul_shift));
				if(height < H_in - 1){
					asm volatile("vmacc.vx v1, %0, v20"  ::"r"(mul_shift));
					if(height < H_in - 2){
						asm volatile("vmacc.vx v2, %0, v21"  ::"r"(mul_shift));
						if(height < H_in - 3){
							asm volatile("vmacc.vx v3, %0, v22"  ::"r"(mul_shift));
							if(height < H_in - 4){
								asm volatile("vmacc.vx v4, %0, v23" ::"r"(mul_shift));
								if(height < H_in - 5){
									asm volatile("vmacc.vx v5, %0, v24" ::"r"(mul_shift));
								}
							}
						}
					}
				}
				
				
				for(int f_w = 0; f_w < F ; f_w++){ // go through every column
					
					f_loop = f_packed + f_w + channels * F * F;
					
					int8_t ldf = F;

					// the register "t0->t5" in clobber list specify to the compiler
					// to avoid using those register since we need them to compute the conv2d
					// the .byte directive should be used carefully with scalar registers 
	
					asm volatile("lb t0, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t0");
					asm volatile("lb t1, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t1");
					asm volatile("lb t2, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t2");
					asm volatile("lb t3, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t3");
					asm volatile("lb t4, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t4");
					asm volatile("lb t5, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf) :"t5");
					asm volatile("lb t6, (%0)" 					 : "+&r"(f_loop) :          :"t6");

					
					// Row 0					
					if(f_w > 0){
						//vmacsr.v.x v25, t0, v0
						asm volatile (".byte 0xd7, 0xec, 0x02, 0xba");
						//vmacsr.v.x v24, t1, v0
						asm volatile (".byte 0x57, 0x6c, 0x03, 0xba");
						//vmacsr.v.x v23, t2, v0
						asm volatile (".byte 0xd7, 0xeb, 0x03, 0xba");
						//vmacsr.v.x v22, t3, v0
						asm volatile (".byte 0x57, 0x6b, 0x0e, 0xba");
						//vmacsr.v.x v21, t4, v0
						asm volatile (".byte 0xd7, 0xea, 0x0e, 0xba");
						//vmacsr.v.x v20, t5, v0
						asm volatile (".byte 0x57, 0x6a, 0x0f, 0xba");
						//vmacsr.v.x v19, t6, v0
						asm volatile (".byte 0xd7, 0xe9, 0x0f, 0xba");

					}
					else{
						//vmulsr.vx v25, v0, t0			
						asm volatile (".byte 0xd7, 0xec, 0x02, 0xa2");
						//vmulsr.v.x v24, t1, v0
						asm volatile (".byte 0x57, 0x6c, 0x03, 0xa2");
						//vmulsr.v.x v23, t2, v0
						asm volatile (".byte 0xd7, 0xeb, 0x03, 0xa2");
						//vmulsr.v.x v22, t3, v0
						asm volatile (".byte 0x57, 0x6b, 0x0e, 0xa2");
						//vmulsr.v.x v21, t4, v0
						asm volatile (".byte 0xd7, 0xea, 0x0e, 0xa2");
						//vmulsr.v.x v20, t5, v0
						asm volatile (".byte 0x57, 0x6a, 0x0f, 0xa2");
						//vmulsr.v.x v19, t6, v0
						asm volatile (".byte 0xd7, 0xe9, 0x0f, 0xa2");
						
					}
					
					
					// Row 1
					if(height < H_in - 1){
						if(f_w > 0)
							//vmacsr.v.x v26, t0, v1
							asm volatile (".byte 0x57, 0xed, 0x12, 0xba");
						else
							asm volatile(" .byte 0x57, 0xed, 0x12, 0xA2");
							//vop v26, v5, v1
							

						//vmacsr.v.x v25, t1, v1
						asm volatile (".byte 0xd7, 0x6c, 0x13, 0xba");
						//vmacsr.v.x v24, t2, v1
						asm volatile (".byte 0x57, 0xec, 0x13, 0xba");
						//vmacsr.v.x v23, t3, v1
						asm volatile (".byte 0xd7, 0x6b, 0x1e, 0xba");
						//vmacsr.v.x v22, t4, v1
						asm volatile (".byte 0x57, 0xeb, 0x1e, 0xba");
						//vmacsr.v.x v21, t5, v1
						asm volatile (".byte 0xd7, 0x6a, 0x1f, 0xba");
						//vmacsr.v.x v20, t6, v1
						asm volatile (".byte 0x57, 0xea, 0x1f, 0xba");


					
					
					
						// Row 2
						if(height < H_in - 2){
							if(f_w > 0)
								asm volatile(" .byte 0xd7, 0xed, 0x22, 0xba");
								//vop v27, v5, v2
							else
								asm volatile(" .byte 0xd7, 0xed, 0x22, 0xA2");
								//vop v27, v5, v2
								
							asm volatile(" .byte 0x57, 0x6d, 0x23, 0xba");
							//vop v26, v6, v2
							asm volatile(" .byte 0xd7, 0xec, 0x23, 0xba");
							//vop v25, v7, v2
							asm volatile(" .byte 0x57, 0x6c, 0x2e, 0xba");
							//vop v24, v28, v2
							asm volatile(" .byte 0xd7, 0xeb, 0x2e, 0xba");
							//vop v23, v29, v2
							asm volatile(" .byte 0x57, 0x6b, 0x2f, 0xba");
							//vop v22, v30, v2
							asm volatile(" .byte 0xd7, 0xea, 0x2f, 0xba");
							//vop v21, v31, v2
						
						
							// Row 3
							if(height < H_in - 3){

								if(f_w > 0)
									asm volatile(" .byte 0x57, 0xee, 0x32, 0xba");
									//vop v28, v5, v3
								else
									asm volatile(" .byte 0x57, 0xee, 0x32, 0xA2");
									//vop v28, v5, v3

								
								asm volatile(" .byte 0xd7, 0x6d, 0x33, 0xba");
								//vop v27, v6, v3
								asm volatile(" .byte 0x57, 0xed, 0x33, 0xba");
								//vop v26, v7, v3
								asm volatile(" .byte 0xd7, 0x6c, 0x3e, 0xba");
								//vop v25, v28, v3
								asm volatile(" .byte 0x57, 0xec, 0x3e, 0xba");
								//vop v24, v29, v3
								asm volatile(" .byte 0xd7, 0x6b, 0x3f, 0xba");
								//vop v23, v30, v3
								asm volatile(" .byte 0x57, 0xeb, 0x3f, 0xba");
								//vop v22, v31, v3
							
							
								// Row 4
								if(height < H_in - 4){

									if(f_w > 0)
										asm volatile(" .byte 0xd7, 0xee, 0x42, 0xba");
										//vop v29, v5, v4
									else
										asm volatile(" .byte 0xd7, 0xee, 0x42, 0xA2");
										//vop v29, v5, v4


									
									asm volatile(" .byte 0x57, 0x6e, 0x43, 0xba");
									//vop v28, v6, v4
									asm volatile(" .byte 0xd7, 0xed, 0x43, 0xba");
									//vop v27, v7, v4
									asm volatile(" .byte 0x57, 0x6d, 0x4e, 0xba");
									//vop v26, v28, v4
									asm volatile(" .byte 0xd7, 0xec, 0x4e, 0xba");
									//vop v25, v29, v4
									asm volatile(" .byte 0x57, 0x6c, 0x4f, 0xba");
									//vop v24, v30, v4
									asm volatile(" .byte 0xd7, 0xeb, 0x4f, 0xba");
									//vop v23, v31, v4

								
								
									// Row 5
									if(height < H_in - 5){

										if(f_w > 0)
											asm volatile(" .byte 0x57, 0xef, 0x52, 0xba");
											//vop v30, v5, v5
										else
											asm volatile(" .byte 0x57, 0xef, 0x52, 0xA2");
											//vop v30, v5, v5
										
										asm volatile(" .byte 0xd7, 0x6e, 0x53, 0xba");
										//vop v29, v6, v5
										asm volatile(" .byte 0x57, 0xee, 0x53, 0xba");
										//vop v28, v7, v5
										asm volatile(" .byte 0xd7, 0x6d, 0x5e, 0xba");
										//vop v27, v28, v5
										asm volatile(" .byte 0x57, 0xed, 0x5e, 0xba");
										//vop v26, v29, v5
										asm volatile(" .byte 0xd7, 0x6c, 0x5f, 0xba");
										//vop v25, v30, v5
										asm volatile(" .byte 0x57, 0xec, 0x5f, 0xba");
										//vop v24, v31, v5
									}
								}
							}
						}
					}
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						if(height < H_in - 1){
							asm volatile("vslidedown.vi v1, v1, 1");
							if(height < H_in - 2){
								asm volatile("vslidedown.vi v2, v2, 1");
								if(height < H_in - 3){
									asm volatile("vslidedown.vi v3, v3, 1");
									if(height < H_in - 4){
										asm volatile("vslidedown.vi v4, v4, 1");
										if(height < H_in - 5){
											asm volatile("vslidedown.vi v5, v5, 1");
										}
									}
								}
							}
						}
					}
				}
				
				if(channels > 0){
					asm volatile("vwadd.wv v6,  v6,  v19");
					asm volatile("vwadd.wv v7,  v7,  v20");
					asm volatile("vwadd.wv v8,  v8,  v21");
					asm volatile("vwadd.wv v9,  v9,  v22");
					asm volatile("vwadd.wv v10, v10, v23");
					asm volatile("vwadd.wv v11, v11, v24");
					asm volatile("vwadd.wv v12, v12, v25");
					asm volatile("vwadd.wv v13, v13, v26");
					asm volatile("vwadd.wv v14, v14, v27");
					asm volatile("vwadd.wv v15, v15, v28");
					asm volatile("vwadd.wv v16, v16, v29");
					asm volatile("vwadd.wv v17, v17, v30");
					
				}
				else{
					
					asm volatile("vwadd.wv v6,  v12,  v19");
					asm volatile("vwadd.wv v7,  v13,  v20");
					asm volatile("vwadd.wv v8,  v14,  v21");
					asm volatile("vwadd.wv v9,  v15,  v22");
					asm volatile("vwadd.wv v10, v16, v23");
					asm volatile("vwadd.wv v11, v17, v24");
					
					asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
					
					asm volatile("vzext.vf2 v12,  v25");
					asm volatile("vzext.vf2 v13,  v26");
					asm volatile("vzext.vf2 v14,  v27");
					asm volatile("vzext.vf2 v15,  v28");
					asm volatile("vzext.vf2 v16,  v29");
					asm volatile("vzext.vf2 v17,  v30");
					
				}
			}
			i_ += (F - 1) * W_in;
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen_out));
			
			asm volatile("vse16.v v6, (%0)" : "+&r"(o_));
			o_ += ldo;
			if(height < H_in - 1){
				asm volatile("vse16.v v7, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 2){
				asm volatile("vse16.v v8, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 3){
				asm volatile("vse16.v v9, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 4){
				asm volatile("vse16.v v10, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 5){
				asm volatile("vse16.v v11, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
		}
	}
}





//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                A2W2                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


void ulppack_conv2d32_W2_A2_vec_3x3(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

int16_t *i_ = i_ptr;
int16_t *i__;
uint16_t *f_ = f_ptr; 
uint16_t *f__;
uint16_t *o_ = o_ptr;

uint64_t const ldo = C_out * (W_in - F + 1);
uint64_t const stride_o = C_out << 1;

uint64_t ldi = C_in * W_in;

uint64_t vlen = W_in;//VLEN_A2_3x3;

uint16_t f_loop[F];

	asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(vlen));
	
	
	for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
		i__ = i_ + channels;
		
		asm volatile("vlse16.v v0, (%0), %1; add %0, %0, %2" : "+&r" (i__) : "r"(C_in), "r"(ldi));
		asm volatile("vlse16.v v2, (%0), %1"                 : "+&r" (i__) : "r"(C_in));
	
		for(int f_w = 0; f_w < F ; f_w++){
			f__ = f_ + f_w  * (C_in >> 1) + channels;
			
			for(int i = 0; i < F - 1 ; i++){
				f_loop[i] = (f__[i * F * (C_in >> 1)] >> 8) + (f__[i * F * (C_in >> 1)] << 8); //swap the two values of the filter
				// channels	 1st    2nd 
				//       □□▥▥□□▤▤
				//	      □□▤▤□□▥▥ => invert values for ULPPACK to work
			}
			
			if(f_w > 0){ 
				asm volatile("vmacc.vx v8,  %0, v0" ::"r"(f_loop[0]));
				asm volatile("vmacc.vx v10,  %0, v2" ::"r"(f_loop[0]));
			}
			else{
				asm volatile("vmul.vx v8,  v0, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v10,  v2, %0" :: "r"(f_loop[0]));
			}

			asm volatile("vmacc.vx v8,  %0, v2" ::"r"(f_loop[1]));
			
			// each plane outputs max F²(2²-1)² which with 2b prec and 3x3 tops at 81
			// but since we're adding 2 planes at a time, it adds up to 162, still below 2⁸-1
			// so we can keep using an 8b buffer for each plane and sum it on 16 after each 2 planes are processed
			
			if(f_w < F - 1){
				asm volatile("vslidedown.vi v0, v0, 1");
				asm volatile("vslidedown.vi v2, v2, 1");
			}
			
		}
		if(channels > 0){
			asm volatile("vsrl.vi v8,  v8,  8");
			asm volatile("vsrl.vi v10, v10, 8");
			
			asm volatile("vadd.vv v28, v28, v8");
			asm volatile("vadd.vv v30, v30, v10");
		}
		else{
			asm volatile("vsrl.vi v28, v8,  8");
			asm volatile("vsrl.vi v30, v10, 8");
		}
	}
	
	i_ += ((F - 1) >> 1) * W_in * C_in;
	
	for (int height = F - 1 ; height < H_in ; height += 4){
	
		asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(vlen));
				
		for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
		
			i__ = i_ + channels;
			
			asm volatile("vlse16.v v0, (%0), %1; add %0, %0, %2" : "+&r" (i__) : "r"(C_in), "r"(ldi));
			asm volatile("vlse16.v v2, (%0), %1; add %0, %0, %2" : "+&r" (i__) : "r"(C_in), "r"(ldi));
			asm volatile("vlse16.v v4, (%0), %1; add %0, %0, %2" : "+&r" (i__) : "r"(C_in), "r"(ldi));
			asm volatile("vlse16.v v6, (%0), %1"                 : "+&r" (i__) : "r"(C_in));
				
			for(int f_w = 0; f_w < F ; f_w++){
				f__ = f_ + f_w  * (C_in >> 1) + channels;
				
				for(int i = 0; i < F ; i++){
					f_loop[i] = (f__[i * F * (C_in >> 1)] >> 8) + (f__[i * F * (C_in >> 1)] << 8); 
					// swap the two values of the filter (necessary for ULPPACK)
					// channels	 1st    2nd 
					//       □□▥▥□□▤▤
					//	      □□▤▤□□▥▥ => invert values for ULPPACK to work
				}
				// Row 0
				if(f_w > 0){ 
					asm volatile("vmacc.vx v12, %0, v0" ::"r"(f_loop[0]));
					asm volatile("vmacc.vx v10, %0, v0" ::"r"(f_loop[1]));
					asm volatile("vmacc.vx v8,  %0, v0" ::"r"(f_loop[2]));
				}
				else{
					asm volatile("vmul.vx  v12, v0, %0" ::"r"(f_loop[0]));
					asm volatile("vmul.vx  v10, v0, %0" ::"r"(f_loop[1]));
					asm volatile("vmul.vx  v8,  v0, %0" ::"r"(f_loop[2]));
				}
				
				
				
				// Row 1
				if(height < H_in - 1){
					if(f_w > 0)
						asm volatile("vmacc.vx v14, %0, v2" ::"r"(f_loop[0]));
					else
						asm volatile("vmul.vx  v14, v2, %0" ::"r"(f_loop[0]));
					
					asm volatile("vmacc.vx v12, %0, v2" ::"r"(f_loop[1]));
					asm volatile("vmacc.vx v10, %0, v2" ::"r"(f_loop[2]));
				}
				
				
				// Row 2
				if(height < H_in - 2){
					if(f_w > 0)
						asm volatile("vmacc.vx v16, %0, v4" ::"r"(f_loop[0]));

					else
						asm volatile("vmul.vx  v16, v4, %0" ::"r"(f_loop[0]));

					asm volatile("vmacc.vx v14, %0, v4" ::"r"(f_loop[1]));
					asm volatile("vmacc.vx v12, %0, v4" ::"r"(f_loop[2]));
				}
				
				// Row 3
				if(height < H_in - 3){
					if(f_w > 0)
						asm volatile("vmacc.vx v18, %0, v6" ::"r"(f_loop[0]));
					else
						asm volatile("vmul.vx  v18, v6, %0" ::"r"(f_loop[0]));
					
					asm volatile("vmacc.vx v16, %0, v6" ::"r"(f_loop[1]));
					asm volatile("vmacc.vx v14, %0, v6" ::"r"(f_loop[2]));
				}
				
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					if(height < H_in - 1)
						asm volatile("vslidedown.vi v2, v2, 1");
					if(height < H_in - 2)
						asm volatile("vslidedown.vi v4, v4, 1");
					if(height < H_in - 3)
						asm volatile("vslidedown.vi v6, v6, 1");
				}
				
			}
			
			
			if(channels > 0){
				
				asm volatile("vsrl.vi v8,  v8,  8");
				if(height < H_in - 1)
					asm volatile("vsrl.vi v10, v10, 8");
				if(height < H_in - 2){
					asm volatile("vsrl.vi v12, v12, 8");
					asm volatile("vsrl.vi v14, v14, 8");
					asm volatile("vsrl.vi v16, v16, 8");
					asm volatile("vsrl.vi v18, v18, 8");
				}
				
				asm volatile("vadd.vv v20, v20, v8");
				if(height < H_in - 1)
					asm volatile("vadd.vv v22, v22, v10");
				if(height < H_in - 2){
					asm volatile("vadd.vv v24, v24, v12");
					asm volatile("vadd.vv v26, v26, v14");
					asm volatile("vadd.vv v28, v28, v16");
					asm volatile("vadd.vv v30, v30, v18");
				}
			}
			else{
				asm volatile("vsrl.vi v8,  v8,  8");
				asm volatile("vsrl.vi v10, v10, 8");
				asm volatile("vsrl.vi v24, v12, 8");
				asm volatile("vsrl.vi v26, v14, 8");
				
				asm volatile("vadd.vv v20, v28, v8");
				asm volatile("vadd.vv v22, v30, v10"); // get the value from first stage
				
				asm volatile("vsrl.vi v28, v16, 8");
				asm volatile("vsrl.vi v30, v18, 8");
				
			}
		}
		
		i_ += (4 >> 1) * W_in * C_in;
		
		asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(vlen - F + 1));
		
		asm volatile("vsse16.v v20, (%0), %1" : "+&r"(o_) : "r"(stride_o));
		o_ += ldo;
		if(height < H_in - 1){
			asm volatile("vsse16.v v22, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			o_ += ldo;
		}
		if(height < H_in - 2){
			asm volatile("vsse16.v v24, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			o_ += ldo;
		}
		if(height < H_in - 3){
			asm volatile("vsse16.v v26, (%0), %1" : "+&r"(o_) : "r"(stride_o));
			o_ += ldo;
		}
	}
}




void ulppack_conv2d32_W2_A2_vec_7x7_A1W1(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

uint64_t const ldo = W_in - F + 1;

uint64_t ldi = W_in;

uint64_t vlen;
	

uint8_t f_packed[F * F * (C_in >> 1)];
uint8_t *f_loop = f_packed;
	
int8_t *f_ = f_ptr;

	// PACKING THE FILTER
	
	for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
		
		uint64_t ldf = F * F;
		
		asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(F * F));

		asm volatile("vle8.v v1, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		asm volatile("vle8.v v0, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		
		asm volatile("vmacc.vx v0, %0, v1" ::"r"(16));
				
		asm volatile("vse8.v v0, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf));
			
	}		
	
	
	for (int width = 0 ; width < (W_in - 6) ; width += VLEN_MF2_OUT) // IF CONVOLUTION NEED TO BE TILED (C > VLEN_WIDE)
	{

		
		int8_t *i_ = i_ptr + width; 									// input pointer realtive to the tile (constant throughout the tile)
		int16_t *o_ = o_ptr + width;	
		
		if(width > W_in - VLEN_MF2) 	// if we are at the right border of the input
			vlen = (W_in - width);		 	// we set the vector length to fit the last inputs
		else
			vlen = VLEN_MF2;						// else we go full length
		
		int8_t * i__ = i_;							// inside tile pointer (change at each load)
		
		int64_t vlen_out = vlen - 6;

		for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
		
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			i__ = i_ + 2 * channels * H_in * W_in;
			
			asm volatile("vle8.v v0, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v1, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v2, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v3, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v4, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v5, (%0)"                 : "+&r" (i__));
			
			i__ += (H_in - F + 2) * W_in;
			
			asm volatile("vle8.v v6,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v7,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v8,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v9,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v10, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v11, (%0)"                 : "+&r" (i__));
			
			// PACKING 2 8b data into 1 8b register (with 4b interval)
			
			asm volatile("vmacc.vx v0, %0, v6"  ::"r"(16));
			asm volatile("vmacc.vx v1, %0, v7"  ::"r"(16));
			asm volatile("vmacc.vx v2, %0, v8"  ::"r"(16));
			asm volatile("vmacc.vx v3, %0, v9"  ::"r"(16));
			asm volatile("vmacc.vx v4, %0, v10" ::"r"(16));
			asm volatile("vmacc.vx v5, %0, v11" ::"r"(16));
		
			// Each iteration of tis loop processes one column of the filter
			for(int f_w = 0; f_w < F ; f_w++){
			
				f_loop = f_packed + f_w + channels * F * F;

				asm volatile("vmul.vx v6,  v0, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v7,  v1, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v8,  v2, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v9,  v3, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v10, v4, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v11, v5, %0" :: "r"(f_loop[0]));
				
				asm volatile("vmacc.vx v6,   %0, v1" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v7,   %0, v2" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v8,   %0, v3" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v9,   %0, v4" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v10,  %0, v5" ::"r"(f_loop[7]));
				
				asm volatile("vmacc.vx v6,   %0, v2" ::"r"(f_loop[14]));
				asm volatile("vmacc.vx v7,   %0, v3" ::"r"(f_loop[14]));
				asm volatile("vmacc.vx v8,   %0, v4" ::"r"(f_loop[14]));
				asm volatile("vmacc.vx v9,   %0, v5" ::"r"(f_loop[14]));
				
				asm volatile("vmacc.vx v6,   %0, v3" ::"r"(f_loop[21]));
				asm volatile("vmacc.vx v7,   %0, v4" ::"r"(f_loop[21]));
				asm volatile("vmacc.vx v8,   %0, v5" ::"r"(f_loop[21]));
				
				asm volatile("vmacc.vx v6,   %0, v4" ::"r"(f_loop[28]));
				asm volatile("vmacc.vx v7,   %0, v5" ::"r"(f_loop[28]));
				
				asm volatile("vmacc.vx v6,   %0, v5" ::"r"(f_loop[35]));
				
				// each plane outputs max F²(2²-1)² which with 2b prec and 3x3 tops at 81

				// but since we're adding 2 planes at a time, it adds up to 162, still below 2⁸-1
				// so we can keep using an 8b buffer for each plane and sum it on 16 after each 2 planes are processed
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v1, v1, 1");
					asm volatile("vslidedown.vi v2, v2, 1");
					asm volatile("vslidedown.vi v3, v3, 1");
					asm volatile("vslidedown.vi v4, v4, 1");
					asm volatile("vslidedown.vi v5, v5, 1");
				}
				
				// LOCAL ACCUMULATION DONE FOR EVERY KERNEL COLUMN TO PREVENT OVERFLOW
				if(channels > 0 || f_w > 0){
					asm volatile("vsrl.vi v6,  v6,  4");
					asm volatile("vsrl.vi v7,  v7,  4");
					asm volatile("vsrl.vi v8,  v8,  4");
					asm volatile("vsrl.vi v9,  v9,  4");
					asm volatile("vsrl.vi v10, v10, 4");
					asm volatile("vsrl.vi v11, v11, 4");				
					
					asm volatile("vwadd.wv v25, v25, v6");
					asm volatile("vwadd.wv v26, v26, v7");
					asm volatile("vwadd.wv v27, v27, v8");
					asm volatile("vwadd.wv v28, v28, v9");
					asm volatile("vwadd.wv v29, v29, v10");
					asm volatile("vwadd.wv v30, v30, v11");
					

				}
				else{
					
					asm volatile("vsrl.vi v26, v6,  4");
					asm volatile("vsrl.vi v27, v7,  4");
					asm volatile("vsrl.vi v28, v8,  4");
					asm volatile("vsrl.vi v29, v9,  4");
					asm volatile("vsrl.vi v30, v10, 4");
					asm volatile("vsrl.vi v31, v11, 4");
					
					asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
					
					asm volatile("vzext.vf2 v25, v26");
					asm volatile("vzext.vf2 v26, v27");
					asm volatile("vzext.vf2 v27, v28");
					asm volatile("vzext.vf2 v28, v29");
					asm volatile("vzext.vf2 v29, v30");
					asm volatile("vzext.vf2 v30, v31");
					
					asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
					
					
				}
				
			}

		}
		
		i_ += (F - 1) * W_in;
		
		for (int height = F - 1 ; height < H_in ; height += 6){
		
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
					
			for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
			
			
				i__ = i_ + 2 * channels * H_in * W_in;
				
				
				asm volatile("vle8.v v0, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 1){
					asm volatile("vle8.v v1, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
					if(height < H_in - 2){
						asm volatile("vle8.v v2, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
						if(height < H_in - 3){
							asm volatile("vle8.v v3, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
							if(height < H_in - 4){
								asm volatile("vle8.v v4, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
								if(height < H_in - 5){
									asm volatile("vle8.v v5, (%0)"                 : "+&r" (i__));
								}
							}
						}
					}
				}
				
				i__ = i_ + (2 * channels + 1) * H_in * W_in;
				
				asm volatile("vle8.v v6,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				if(height < H_in - 1){
					asm volatile("vle8.v v7,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
					if(height < H_in - 2){
						asm volatile("vle8.v v8,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
						if(height < H_in - 3){
							asm volatile("vle8.v v9,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
							if(height < H_in - 4){
								asm volatile("vle8.v v10, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
									if(height < H_in - 5){
										asm volatile("vle8.v v11, (%0)"                 : "+&r" (i__));
									}
								}
							}
						}
					}
				// PACKING 2 8b data into 1 8b register (with 4b interval)
				
				asm volatile("vmacc.vx v0, %0, v6"  ::"r"(16));
				if(height < H_in - 1){
					asm volatile("vmacc.vx v1, %0, v7"  ::"r"(16));
					if(height < H_in - 2){
						asm volatile("vmacc.vx v2, %0, v8"  ::"r"(16));
						if(height < H_in - 3){
							asm volatile("vmacc.vx v3, %0, v9"  ::"r"(16));
							if(height < H_in - 4){
								asm volatile("vmacc.vx v4, %0, v10" ::"r"(16));
								if(height < H_in - 5){
									asm volatile("vmacc.vx v5, %0, v11" ::"r"(16));
								}
							}
						}
					}
				}	
				
				for(int f_w = 0; f_w < F ; f_w++){ // go through every column
					
					f_loop = f_packed + f_w + channels * F * F;
					
					// Row 0
					asm volatile("vmul.vx  v12, v0, %0" ::"r"(f_loop[0]));
					asm volatile("vmul.vx  v11, v0, %0" ::"r"(f_loop[7]));
					asm volatile("vmul.vx  v10, v0, %0" ::"r"(f_loop[14]));
					asm volatile("vmul.vx  v9,  v0, %0" ::"r"(f_loop[21]));
					asm volatile("vmul.vx  v8,  v0, %0" ::"r"(f_loop[28]));
					asm volatile("vmul.vx  v7,  v0, %0" ::"r"(f_loop[35]));
					asm volatile("vmul.vx  v6,  v0, %0" ::"r"(f_loop[42]));
					
					
					
					// Row 1
					if(height < H_in - 1){
						asm volatile("vmul.vx  v13, v1, %0" ::"r"(f_loop[0]));
						
						asm volatile("vmacc.vx v12, %0, v1" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v11, %0, v1" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v10, %0, v1" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v9,  %0, v1" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v8,  %0, v1" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v7,  %0, v1" ::"r"(f_loop[42]));

					
					
					
						// Row 2
						if(height < H_in - 2){
							asm volatile("vmul.vx  v14, v2, %0" ::"r"(f_loop[0]));
							
							asm volatile("vmacc.vx v13, %0, v2" ::"r"(f_loop[7]));
							asm volatile("vmacc.vx v12, %0, v2" ::"r"(f_loop[14]));
							asm volatile("vmacc.vx v11, %0, v2" ::"r"(f_loop[21]));
							asm volatile("vmacc.vx v10, %0, v2" ::"r"(f_loop[28]));
							asm volatile("vmacc.vx v9,  %0, v2" ::"r"(f_loop[35]));
							asm volatile("vmacc.vx v8,  %0, v2" ::"r"(f_loop[42]));
						
						
							// Row 3
							if(height < H_in - 3){
								asm volatile("vmul.vx  v15, v3, %0" ::"r"(f_loop[0]));
								
								asm volatile("vmacc.vx v14, %0, v3" ::"r"(f_loop[7]));
								asm volatile("vmacc.vx v13, %0, v3" ::"r"(f_loop[14]));
								asm volatile("vmacc.vx v12, %0, v3" ::"r"(f_loop[21]));
								asm volatile("vmacc.vx v11, %0, v3" ::"r"(f_loop[28]));
								asm volatile("vmacc.vx v10, %0, v3" ::"r"(f_loop[35]));
								asm volatile("vmacc.vx v9,  %0, v3" ::"r"(f_loop[42]));
							
							
								// Row 4
								if(height < H_in - 4){
									asm volatile("vmul.vx  v16, v4, %0" ::"r"(f_loop[0]));
									
									asm volatile("vmacc.vx v15, %0, v4" ::"r"(f_loop[7]));
									asm volatile("vmacc.vx v14, %0, v4" ::"r"(f_loop[14]));
									asm volatile("vmacc.vx v13, %0, v4" ::"r"(f_loop[21]));
									asm volatile("vmacc.vx v12, %0, v4" ::"r"(f_loop[28]));
									asm volatile("vmacc.vx v11, %0, v4" ::"r"(f_loop[35]));
									asm volatile("vmacc.vx v10, %0, v4" ::"r"(f_loop[42]));
								
								
									// Row 5
									if(height < H_in - 5){
										asm volatile("vmul.vx  v17, v5, %0" ::"r"(f_loop[0]));
										
										asm volatile("vmacc.vx v16, %0, v5" ::"r"(f_loop[7]));
										asm volatile("vmacc.vx v15, %0, v5" ::"r"(f_loop[14]));
										asm volatile("vmacc.vx v14, %0, v5" ::"r"(f_loop[21]));
										asm volatile("vmacc.vx v13, %0, v5" ::"r"(f_loop[28]));
										asm volatile("vmacc.vx v12, %0, v5" ::"r"(f_loop[35]));
										asm volatile("vmacc.vx v11, %0, v5" ::"r"(f_loop[42]));
									}
								}
							}
						}
					}
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						if(height < H_in - 1){
							asm volatile("vslidedown.vi v1, v1, 1");
							if(height < H_in - 2){
								asm volatile("vslidedown.vi v2, v2, 1");
								if(height < H_in - 3){
									asm volatile("vslidedown.vi v3, v3, 1");
									if(height < H_in - 4){
										asm volatile("vslidedown.vi v4, v4, 1");
										if(height < H_in - 5){
											asm volatile("vslidedown.vi v5, v5, 1");
										}
									}
								}
							}
						}
					}
					
					if(channels > 0 || f_w > 0){
						asm volatile("vsrl.vi v6,  v6,  4");
						if(height < H_in - 1){
							asm volatile("vsrl.vi v7,  v7,  4");
							if(height < H_in - 2){
								asm volatile("vsrl.vi v8,  v8,  4");
								if(height < H_in - 3){
									asm volatile("vsrl.vi v9,  v9,  4");
									if(height < H_in - 4){
										asm volatile("vsrl.vi v10, v10, 4");
										if(height < H_in - 5){
											asm volatile("vsrl.vi v11, v11, 4");
											if(height < H_in - 6){
												asm volatile("vsrl.vi v12, v12, 4");
												if(height < H_in - 7){
													asm volatile("vsrl.vi v13, v13, 4");
													if(height < H_in - 8){
														asm volatile("vsrl.vi v14, v14, 4");
														if(height < H_in - 9){
															asm volatile("vsrl.vi v15, v15, 4");
															if(height < H_in - 10){
																asm volatile("vsrl.vi v16, v16, 4");
																if(height < H_in - 11){
																	asm volatile("vsrl.vi v17, v17, 4");
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
											
						asm volatile("vwadd.wv v19, v19, v6");
						if(height < H_in - 1){
							asm volatile("vwadd.wv v20, v20, v7");
							if(height < H_in - 2){
								asm volatile("vwadd.wv v21, v21, v8");
								if(height < H_in - 3){
									asm volatile("vwadd.wv v22, v22, v9");
									if(height < H_in - 4){
										asm volatile("vwadd.wv v23, v23, v10");
										if(height < H_in - 5){
											asm volatile("vwadd.wv v24, v24, v11");
											if(height < H_in - 6){
												asm volatile("vwadd.wv v25, v25, v12");
												if(height < H_in - 7){
													asm volatile("vwadd.wv v26, v26, v13");
													if(height < H_in - 8){
														asm volatile("vwadd.wv v27, v27, v14");
														if(height < H_in - 9){
															asm volatile("vwadd.wv v28, v28, v15");
															if(height < H_in - 10){
																asm volatile("vwadd.wv v29, v29, v16");
																if(height < H_in - 11){
																	asm volatile("vwadd.wv v30, v30, v17");
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
					else{
						asm volatile("vsrl.vi v6,  v6,  4");
						if(height < H_in - 1){
							asm volatile("vsrl.vi v7,  v7,  4");
							if(height < H_in - 2){
								asm volatile("vsrl.vi v8,  v8,  4");
								if(height < H_in - 3){
									asm volatile("vsrl.vi v9,  v9,  4");
									if(height < H_in - 4){
										asm volatile("vsrl.vi v10, v10, 4");
										if(height < H_in - 5){
											asm volatile("vsrl.vi v11, v11, 4");
										}
									}
								}
							}
						}
							
						asm volatile("vwadd.wv v19, v25, v6");
						if(height < H_in - 1){
							asm volatile("vwadd.wv v20, v26, v7");
							if(height < H_in - 2){
								asm volatile("vwadd.wv v21, v27, v8");
								if(height < H_in - 3){
									asm volatile("vwadd.wv v22, v28, v9");
									if(height < H_in - 4){
										asm volatile("vwadd.wv v23, v29, v10");
										if(height < H_in - 5){
											asm volatile("vwadd.wv v24, v30, v11");
											
											if(height < H_in - 6){	
												asm volatile("vsrl.vi v26, v12, 4");
												if(height < H_in - 7){
													asm volatile("vsrl.vi v27, v13, 4");
													if(height < H_in - 8){
														asm volatile("vsrl.vi v28, v14, 4");
														if(height < H_in - 9){
															asm volatile("vsrl.vi v29, v15, 4");
															if(height < H_in - 10){
																asm volatile("vsrl.vi v30, v16, 4");
																if(height < H_in - 11){
																	asm volatile("vsrl.vi v31, v17, 4");
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
								
						asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
							
						if(height < H_in - 6){
							asm volatile("vzext.vf2 v25, v26");
							if(height < H_in - 7){
								asm volatile("vzext.vf2 v26, v27");
								if(height < H_in - 8){
									asm volatile("vzext.vf2 v27, v28");
									if(height < H_in - 9){
										asm volatile("vzext.vf2 v28, v29");
										if(height < H_in - 10){
											asm volatile("vzext.vf2 v29, v30");
											if(height < H_in - 11){
												asm volatile("vzext.vf2 v30, v31");
											}
										}
									}
								}
							}
						}
										
						asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
						
					}
					
				}
			}
			
			i_ += (F - 1) * W_in;
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen_out));
			
			asm volatile("vse16.v v19, (%0)" : "+&r"(o_));
			o_ += ldo;
			if(height < H_in - 1){
				asm volatile("vse16.v v20, (%0)" : "+&r"(o_));
				o_ += ldo;
			
				if(height < H_in - 2){
					asm volatile("vse16.v v21, (%0)" : "+&r"(o_));
					o_ += ldo;
					
					if(height < H_in - 3){
						asm volatile("vse16.v v22, (%0)" : "+&r"(o_));
						o_ += ldo;
						
						if(height < H_in - 4){
							asm volatile("vse16.v v23, (%0)" : "+&r"(o_));
							o_ += ldo;
							
							if(height < H_in - 5){
								asm volatile("vse16.v v24, (%0)" : "+&r"(o_));;
								o_ += ldo;
							}
						}
					}
				}
			}
		}
	}
}


void ulppack_conv2d32_W2_A2_vec_7x7_A2W2(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

uint8_t *f_ = f_ptr; 

uint64_t const ldo = W_in - F + 1;

uint64_t ldi = W_in;

uint64_t vlen;

uint16_t f_packed[F * F * (C_in >> 1)];
uint16_t *f_loop = f_packed;
	
	// PACKING THE FILTER
	
	for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){

	
		uint64_t ldf = F * F;
		
		asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(F * F));

		asm volatile("vle8.v v1, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		asm volatile("vle8.v v2, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		
		asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(F * F));
		
		asm volatile("vzext.vf2 v0, v1");
		asm volatile("vzext.vf2 v1, v2");
		
		asm volatile("vmacc.vx v0, %0, v1" ::"r"(256));
		
		asm volatile("vse16.v v0, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf << 1));
	}	
	
	for (int width = 0 ; width < (W_in - 6) ; width += VLEN_MF2) // IF CONVOLUTION NEED TO BE TILED (C > VLEN_WIDE)
	{

	
		int8_t *i_ = i_ptr + width; 									// input pointer realtive to the tile (constant throughout the tile)
		int16_t *o_ = o_ptr + width;	
		
		if(width > W_in - VLEN_MF2) 	// if we are at the right border of the input
			vlen = (W_in - width);		 	// we set the vector length to fit the last inputs
		else
			vlen = VLEN_MF2;						// else we go full length
		
		int8_t * i__ = i_;							// inside tile pointer (change at each load)
		
		int64_t vlen_out = vlen - 6;
		
		for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
			
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
			i__ = i_ + 2 * channels * H_in * W_in;
			
			asm volatile("vle8.v v6, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v7, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v8, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v9, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v10,(%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v11,(%0)"                 : "+&r" (i__));
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
			
			asm volatile("vzext.vf2 v0, v6");
			asm volatile("vzext.vf2 v1, v7");
			asm volatile("vzext.vf2 v2, v8");
			asm volatile("vzext.vf2 v3, v9");
			asm volatile("vzext.vf2 v4, v10");
			asm volatile("vzext.vf2 v5, v11");
			
			asm volatile("vsll.vi v0, v0, 8");
			asm volatile("vsll.vi v1, v1, 8");
			asm volatile("vsll.vi v2, v2, 8");
			asm volatile("vsll.vi v3, v3, 8");
			asm volatile("vsll.vi v4, v4, 8");
			asm volatile("vsll.vi v5, v5, 8");	
			
			//asm volatile("vse16.v v5, (%0)" : "+&r"(o_));
			//print_tensor_(o_, 1, W_in, 1);
			
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
			i__ += (H_in - F + 2) * W_in;
			
			
			asm volatile("vle8.v v6,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v7,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v8,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v9,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v10, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v11, (%0)"                 : "+&r" (i__));

			asm volatile("vwadd.wv v0, v0, v6");
			asm volatile("vwadd.wv v1, v1, v7");
			asm volatile("vwadd.wv v2, v2, v8");
			asm volatile("vwadd.wv v3, v3, v9");
			asm volatile("vwadd.wv v4, v4, v10");
			asm volatile("vwadd.wv v5, v5, v11");
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
			
			//asm volatile("vse16.v v5, (%0)" : "+&r"(o_));
			//print_tensor_(o_, 1, W_in, 1);
			
		
			// Each iteration of tis loop processes one column of the filter
			for(int f_w = 0; f_w < F ; f_w++){
			
				f_loop = f_packed + f_w + channels * F * F;

				if(f_w % 2 == 0 || f_w == F - 1){
					asm volatile("vmul.vx v6,  v0, %0" :: "r"(f_loop[0]));
					asm volatile("vmul.vx v7,  v1, %0" :: "r"(f_loop[0]));
					asm volatile("vmul.vx v8,  v2, %0" :: "r"(f_loop[0]));
					asm volatile("vmul.vx v9,  v3, %0" :: "r"(f_loop[0]));
					asm volatile("vmul.vx v10, v4, %0" :: "r"(f_loop[0]));
					asm volatile("vmul.vx v11, v5, %0" :: "r"(f_loop[0]));
				}
				else{
					asm volatile("vmacc.vx v6,  %0, v0" ::"r"(f_loop[0]));
					asm volatile("vmacc.vx v7,  %0, v1" ::"r"(f_loop[0]));
					asm volatile("vmacc.vx v8,  %0, v2" ::"r"(f_loop[0]));
					asm volatile("vmacc.vx v9,  %0, v3" ::"r"(f_loop[0]));
					asm volatile("vmacc.vx v10, %0, v4" ::"r"(f_loop[0]));
					asm volatile("vmacc.vx v11, %0, v5" ::"r"(f_loop[0]));
				}	
				
				asm volatile("vmacc.vx v6,   %0, v1" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v7,   %0, v2" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v8,   %0, v3" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v9,   %0, v4" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v10,  %0, v5" ::"r"(f_loop[7]));
				
				asm volatile("vmacc.vx v6,   %0, v2" ::"r"(f_loop[14]));
				asm volatile("vmacc.vx v7,   %0, v3" ::"r"(f_loop[14]));
				asm volatile("vmacc.vx v8,   %0, v4" ::"r"(f_loop[14]));
				asm volatile("vmacc.vx v9,   %0, v5" ::"r"(f_loop[14]));
				
				asm volatile("vmacc.vx v6,   %0, v3" ::"r"(f_loop[21]));
				asm volatile("vmacc.vx v7,   %0, v4" ::"r"(f_loop[21]));
				asm volatile("vmacc.vx v8,   %0, v5" ::"r"(f_loop[21]));
				
				asm volatile("vmacc.vx v6,   %0, v4" ::"r"(f_loop[28]));
				asm volatile("vmacc.vx v7,   %0, v5" ::"r"(f_loop[28]));
				
				asm volatile("vmacc.vx v6,   %0, v5" ::"r"(f_loop[35]));
				
				// each plane outputs max F²(2²-1)² which with 2b prec and 3x3 tops at 81

				// but since we're adding 2 planes at a time, it adds up to 162, still below 2⁸-1
				// so we can keep using an 8b buffer for each plane and sum it on 16 after each 2 planes are processed
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v1, v1, 1");
					asm volatile("vslidedown.vi v2, v2, 1");
					asm volatile("vslidedown.vi v3, v3, 1");
					asm volatile("vslidedown.vi v4, v4, 1");
					asm volatile("vslidedown.vi v5, v5, 1");
				}
				
				// LOCAL ACCUMULATION DONE FOR EVERY KERNEL COLUMN TO PREVENT OVERFLOW
				if(f_w % 2 == 1 || f_w == F - 1){ 
					if(channels > 0 || f_w > 1){
						asm volatile("vsrl.vi v6,  v6,  8");
						asm volatile("vsrl.vi v7,  v7,  8");
						asm volatile("vsrl.vi v8,  v8,  8");
						asm volatile("vsrl.vi v9,  v9,  8");
						asm volatile("vsrl.vi v10, v10, 8");
						asm volatile("vsrl.vi v11, v11, 8");
						
						asm volatile("vadd.vv v25, v25, v6");
						asm volatile("vadd.vv v26, v26, v7");
						asm volatile("vadd.vv v27, v27, v8");
						asm volatile("vadd.vv v28, v28, v9");
						asm volatile("vadd.vv v29, v29, v10");
						asm volatile("vadd.vv v30, v30, v11");
						
					}
					else{
						asm volatile("vsrl.vi v25, v6,  8");
						asm volatile("vsrl.vi v26, v7,  8");
						asm volatile("vsrl.vi v27, v8,  8");
						asm volatile("vsrl.vi v28, v9,  8");
						asm volatile("vsrl.vi v29, v10, 8");
						asm volatile("vsrl.vi v30, v11, 8");

					}
				}
				
			}

		}
		
		i_ += (F - 1) * W_in;
		
		for (int height = F - 1 ; height < H_in ; height += 6){
					
			for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
				
				asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
				i__ = i_ + 2 * channels * H_in * W_in;
				
				asm volatile("vle8.v v6, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v7, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v8, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v9, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v10,(%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v11,(%0)"                 : "+&r" (i__));
				
				asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
				
				asm volatile("vzext.vf2 v0, v6");
				asm volatile("vzext.vf2 v1, v7");
				asm volatile("vzext.vf2 v2, v8");
				asm volatile("vzext.vf2 v3, v9");
				asm volatile("vzext.vf2 v4, v10");
				asm volatile("vzext.vf2 v5, v11");
				
				asm volatile("vsll.vi v0, v0, 8");
				asm volatile("vsll.vi v1, v1, 8");
				asm volatile("vsll.vi v2, v2, 8");
				asm volatile("vsll.vi v3, v3, 8");
				asm volatile("vsll.vi v4, v4, 8");
				asm volatile("vsll.vi v5, v5, 8");	
				
				//asm volatile("vse16.v v5, (%0)" : "+&r"(o_));
				//print_tensor_(o_, 1, W_in, 1);
				
				asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
				
				i__ += (H_in - F + 2) * W_in;
				
				
				asm volatile("vle8.v v6,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v7,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v8,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v9,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v10, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v11, (%0)"                 : "+&r" (i__));

				asm volatile("vwadd.wv v0, v0, v6");
				asm volatile("vwadd.wv v1, v1, v7");
				asm volatile("vwadd.wv v2, v2, v8");
				asm volatile("vwadd.wv v3, v3, v9");
				asm volatile("vwadd.wv v4, v4, v10");
				asm volatile("vwadd.wv v5, v5, v11");
				
				asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
					
				for(int f_w = 0; f_w < F ; f_w++){
					
					f_loop = f_packed + f_w + channels * F * F;
					
					// Row 0
					if(f_w % 2 == 0 || f_w == F - 1){
						asm volatile("vmul.vx  v12, v0, %0" ::"r"(f_loop[0]));
						asm volatile("vmul.vx  v11, v0, %0" ::"r"(f_loop[7]));
						asm volatile("vmul.vx  v10, v0, %0" ::"r"(f_loop[14]));
						asm volatile("vmul.vx  v9,  v0, %0" ::"r"(f_loop[21]));
						asm volatile("vmul.vx  v8,  v0, %0" ::"r"(f_loop[28]));
						asm volatile("vmul.vx  v7,  v0, %0" ::"r"(f_loop[35]));
						asm volatile("vmul.vx  v6,  v0, %0" ::"r"(f_loop[42]));
					}
					else{
						asm volatile("vmacc.vx v12, %0, v0" ::"r"(f_loop[0]));
						asm volatile("vmacc.vx v11, %0, v0" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v10, %0, v0" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v9,  %0, v0" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v8,  %0, v0" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v7,  %0, v0" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v6,  %0, v0" ::"r"(f_loop[42]));
					}
					
					
					// Row 1
					if(height < H_in - 1){
						if(f_w % 2 == 0 || f_w == F - 1)
							asm volatile("vmul.vx  v13, v1, %0" ::"r"(f_loop[0]));
						else
							asm volatile("vmacc.vx v13, %0, v1" ::"r"(f_loop[0]));
							
						asm volatile("vmacc.vx v12, %0, v1" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v11, %0, v1" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v10, %0, v1" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v9,  %0, v1" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v8,  %0, v1" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v7,  %0, v1" ::"r"(f_loop[42]));

					}
					
					
					// Row 2
					if(height < H_in - 2){
						if(f_w % 2 == 0 || f_w == F - 1)
							asm volatile("vmul.vx  v14, v2, %0" ::"r"(f_loop[0]));
						else
							asm volatile("vmacc.vx v14, %0, v2" ::"r"(f_loop[0]));
							
						asm volatile("vmacc.vx v13, %0, v2" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v12, %0, v2" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v11, %0, v2" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v10, %0, v2" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v9,  %0, v2" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v8,  %0, v2" ::"r"(f_loop[42]));
					}
					
					// Row 3
					if(height < H_in - 3){
						if(f_w % 2 == 0 || f_w == F - 1)
							asm volatile("vmul.vx  v15, v3, %0" ::"r"(f_loop[0]));
						else
							asm volatile("vmacc.vx v15, %0, v3" ::"r"(f_loop[0]));
											
						asm volatile("vmacc.vx v14, %0, v3" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v13, %0, v3" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v12, %0, v3" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v11, %0, v3" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v10, %0, v3" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v9,  %0, v3" ::"r"(f_loop[42]));
					}
					
					// Row 4
					if(height < H_in - 4){
						if(f_w % 2 == 0 || f_w == F - 1)
							asm volatile("vmul.vx  v16, v4, %0" ::"r"(f_loop[0]));
						else
							asm volatile("vmacc.vx v16, %0, v4" ::"r"(f_loop[0]));
											
						asm volatile("vmacc.vx v15, %0, v4" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v14, %0, v4" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v13, %0, v4" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v12, %0, v4" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v11, %0, v4" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v10, %0, v4" ::"r"(f_loop[42]));
					}
					
					// Row 5
					if(height < H_in - 5){
						if(f_w % 2 == 0 || f_w == F - 1)
							asm volatile("vmul.vx  v17, v5, %0" ::"r"(f_loop[0]));
						else
							asm volatile("vmacc.vx v17, %0, v5" ::"r"(f_loop[0]));
											
						asm volatile("vmacc.vx v16, %0, v5" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v15, %0, v5" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v14, %0, v5" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v13, %0, v5" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v12, %0, v5" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v11, %0, v5" ::"r"(f_loop[42]));
					}
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						if(height < H_in - 1)
							asm volatile("vslidedown.vi v1, v1, 1");
						if(height < H_in - 2)
							asm volatile("vslidedown.vi v2, v2, 1");
						if(height < H_in - 3)
							asm volatile("vslidedown.vi v3, v3, 1");
						if(height < H_in - 4)
							asm volatile("vslidedown.vi v4, v4, 1");
						if(height < H_in - 5)
							asm volatile("vslidedown.vi v5, v5, 1");
					}
					
					if(f_w % 2 == 1 || f_w == F - 1){ 
						if(channels > 0 || f_w > 1){
							asm volatile("vsrl.vi v6,  v6,  8");
							if(height < H_in - 1){
								asm volatile("vsrl.vi v7,  v7,  8");
							if(height < H_in - 2){
								asm volatile("vsrl.vi v8,  v8,  8");
							if(height < H_in - 3){
								asm volatile("vsrl.vi v9,  v9,  8");
							if(height < H_in - 4){
								asm volatile("vsrl.vi v10, v10, 8");
							if(height < H_in - 5){
								asm volatile("vsrl.vi v11, v11, 8");
								asm volatile("vsrl.vi v12, v12, 8");
								asm volatile("vsrl.vi v13, v13, 8");
								asm volatile("vsrl.vi v14, v14, 8");
								asm volatile("vsrl.vi v15, v15, 8");
								asm volatile("vsrl.vi v16, v16, 8");
								asm volatile("vsrl.vi v17, v17, 8");
							}}}}}
							
							
							asm volatile("vadd.vv v19, v19, v6");
							if(height < H_in - 1){
								asm volatile("vadd.vv v20, v20, v7");
							if(height < H_in - 2){
								asm volatile("vadd.vv v21, v21, v8");
							if(height < H_in - 3){
								asm volatile("vadd.vv v22, v22, v9");
							if(height < H_in - 4){
								asm volatile("vadd.vv v23, v23, v10");
							if(height < H_in - 5){
								asm volatile("vadd.vv v24, v24, v11");
								asm volatile("vadd.vv v25, v25, v12");
								asm volatile("vadd.vv v26, v26, v13");
								asm volatile("vadd.vv v27, v27, v14");
								asm volatile("vadd.vv v28, v28, v15");
								asm volatile("vadd.vv v29, v29, v16");
								asm volatile("vadd.vv v30, v30, v17");
								
							}}}}}
						}
						else{
							asm volatile("vsrl.vi v6,  v6,  8");
							asm volatile("vsrl.vi v7,  v7,  8");
							asm volatile("vsrl.vi v8,  v8,  8");
							asm volatile("vsrl.vi v9,  v9,  8");
							asm volatile("vsrl.vi v10, v10, 8");
							asm volatile("vsrl.vi v11, v11, 8");
							

							asm volatile("vadd.vv v19, v25, v6");
							if(height < H_in - 1){
								asm volatile("vadd.vv v20, v26, v7");
							if(height < H_in - 2){
								asm volatile("vadd.vv v21, v27, v8");
							if(height < H_in - 3){
								asm volatile("vadd.vv v22, v28, v9");
							if(height < H_in - 4){
								asm volatile("vadd.vv v23, v29, v10");
							if(height < H_in - 5){
								asm volatile("vadd.vv v24, v30, v11");
								}}}}}
							asm volatile("vsrl.vi v25, v12, 8");
							asm volatile("vsrl.vi v26, v13, 8");
							asm volatile("vsrl.vi v27, v14, 8");
							asm volatile("vsrl.vi v28, v15, 8");
							asm volatile("vsrl.vi v29, v16, 8");
							asm volatile("vsrl.vi v30, v17, 8");
								
							
							
							}
							
						}
					}
					
			}
			
			i_ += (F - 1) * W_in;
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen_out));
			
			asm volatile("vse16.v v19, (%0)" : "+&r"(o_));
			o_ += ldo;
			if(height < H_in - 1){
				asm volatile("vse16.v v20, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 2){
				asm volatile("vse16.v v21, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 3){
				asm volatile("vse16.v v22, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 4){
				asm volatile("vse16.v v23, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 5){
				asm volatile("vse16.v v24, (%0)" : "+&r"(o_));;
				o_ += ldo;
			}
		}
	}
}

void ulppack_conv2d32_W2_A2_vec_7x7_A3W3(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out){

uint8_t *f_ = f_ptr; 

uint64_t const ldo = W_in - F + 1;

uint64_t ldi = W_in;

uint64_t vlen;

uint16_t f_packed[F * F * (C_in >> 1)];
uint16_t *f_loop = f_packed;
	
	// PACKING THE FILTER
	
	for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){

	
		uint64_t ldf = F * F;
		
		asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(F * F));

		asm volatile("vle8.v v1, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		asm volatile("vle8.v v2, (%0); add %0, %0, %1" : "+&r" (f_) : "r"(ldf));
		
		asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(F * F));
		
		asm volatile("vzext.vf2 v0, v1");
		asm volatile("vzext.vf2 v1, v2");
		
		asm volatile("vmacc.vx v0, %0, v1" ::"r"(256));
		
		asm volatile("vse16.v v0, (%0); add %0, %0, %1" : "+&r"(f_loop) : "r"(ldf << 1));
	}	
	
	for (int width = 0 ; width < (W_in - 6) ; width += VLEN_MF2) // IF CONVOLUTION NEED TO BE TILED (C > VLEN_WIDE)
	{

	
		int8_t *i_ = i_ptr + width; 									// input pointer realtive to the tile (constant throughout the tile)
		int16_t *o_ = o_ptr + width;	
		
		if(width > W_in - VLEN_MF2) 	// if we are at the right border of the input
			vlen = (W_in - width);		 	// we set the vector length to fit the last inputs
		else
			vlen = VLEN_MF2;						// else we go full length
		
		int8_t * i__ = i_;							// inside tile pointer (change at each load)
		
		int64_t vlen_out = vlen - 6;
		
		for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
			
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
			i__ = i_ + 2 * channels * H_in * W_in;
			
			asm volatile("vle8.v v6, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v7, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v8, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v9, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v10,(%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v11,(%0)"                 : "+&r" (i__));
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
			
			asm volatile("vzext.vf2 v0, v6");
			asm volatile("vzext.vf2 v1, v7");
			asm volatile("vzext.vf2 v2, v8");
			asm volatile("vzext.vf2 v3, v9");
			asm volatile("vzext.vf2 v4, v10");
			asm volatile("vzext.vf2 v5, v11");
			
			asm volatile("vsll.vi v0, v0, 8");
			asm volatile("vsll.vi v1, v1, 8");
			asm volatile("vsll.vi v2, v2, 8");
			asm volatile("vsll.vi v3, v3, 8");
			asm volatile("vsll.vi v4, v4, 8");
			asm volatile("vsll.vi v5, v5, 8");	
			
			//asm volatile("vse16.v v5, (%0)" : "+&r"(o_));
			//print_tensor_(o_, 1, W_in, 1);
			
			asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
			i__ += (H_in - F + 2) * W_in;
			
			
			asm volatile("vle8.v v6,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v7,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v8,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v9,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v10, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
			asm volatile("vle8.v v11, (%0)"                 : "+&r" (i__));

			asm volatile("vwadd.wv v0, v0, v6");
			asm volatile("vwadd.wv v1, v1, v7");
			asm volatile("vwadd.wv v2, v2, v8");
			asm volatile("vwadd.wv v3, v3, v9");
			asm volatile("vwadd.wv v4, v4, v10");
			asm volatile("vwadd.wv v5, v5, v11");
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
			
			//asm volatile("vse16.v v5, (%0)" : "+&r"(o_));
			//print_tensor_(o_, 1, W_in, 1);
			
		
			// Each iteration of tis loop processes one column of the filter
			for(int f_w = 0; f_w < F ; f_w++){
			
				f_loop = f_packed + f_w + channels * F * F;

				asm volatile("vmul.vx v6,  v0, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v7,  v1, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v8,  v2, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v9,  v3, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v10, v4, %0" :: "r"(f_loop[0]));
				asm volatile("vmul.vx v11, v5, %0" :: "r"(f_loop[0]));

				asm volatile("vmacc.vx v6,   %0, v1" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v7,   %0, v2" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v8,   %0, v3" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v9,   %0, v4" ::"r"(f_loop[7]));
				asm volatile("vmacc.vx v10,  %0, v5" ::"r"(f_loop[7]));
				
				// LOCAL ACCUMULATION
				
				if(channels > 0 || f_w > 0){
					asm volatile("vsrl.vi v6,  v6,  8");
					asm volatile("vsrl.vi v7,  v7,  8");
					asm volatile("vsrl.vi v8,  v8,  8");
					asm volatile("vsrl.vi v9,  v9,  8");
					asm volatile("vsrl.vi v10, v10, 8");
					asm volatile("vsrl.vi v11, v11, 8");
					
					asm volatile("vadd.vv v25, v25, v6");
					asm volatile("vadd.vv v26, v26, v7");
					asm volatile("vadd.vv v27, v27, v8");
					asm volatile("vadd.vv v28, v28, v9");
					asm volatile("vadd.vv v29, v29, v10");
					asm volatile("vadd.vv v30, v30, v11");
						
				}
				else{
					asm volatile("vsrl.vi v25, v6,  8");
					asm volatile("vsrl.vi v26, v7,  8");
					asm volatile("vsrl.vi v27, v8,  8");
					asm volatile("vsrl.vi v28, v9,  8");
					asm volatile("vsrl.vi v29, v10, 8");
					asm volatile("vsrl.vi v30, v11, 8");

				}
				
				asm volatile("vmul.vx v6,  v2, %0" :: "r"(f_loop[14]));
				asm volatile("vmul.vx v7,  v3, %0" :: "r"(f_loop[14]));
				asm volatile("vmul.vx v8,  v4, %0" :: "r"(f_loop[14]));
				asm volatile("vmul.vx v9,  v5, %0" :: "r"(f_loop[14]));
				
				asm volatile("vmacc.vx v6,   %0, v3" ::"r"(f_loop[21]));
				asm volatile("vmacc.vx v7,   %0, v4" ::"r"(f_loop[21]));
				asm volatile("vmacc.vx v8,   %0, v5" ::"r"(f_loop[21]));
				
				// LOCAL ACCUMULATION
				
				asm volatile("vsrl.vi v6,  v6,  8");
				asm volatile("vsrl.vi v7,  v7,  8");
				asm volatile("vsrl.vi v8,  v8,  8");
				asm volatile("vsrl.vi v9,  v9,  8");
					
				asm volatile("vadd.vv v25, v25, v6");
				asm volatile("vadd.vv v26, v26, v7");
				asm volatile("vadd.vv v27, v27, v8");
				asm volatile("vadd.vv v28, v28, v9");


				asm volatile("vmul.vx v6,  v4, %0" :: "r"(f_loop[28]));
				asm volatile("vmul.vx v7,  v5, %0" :: "r"(f_loop[28]));
				
				asm volatile("vmacc.vx v6,   %0, v5" ::"r"(f_loop[35]));
				
				// LOCAL ACCUMULATION
				
				asm volatile("vsrl.vi v6,  v6,  8");
				asm volatile("vsrl.vi v7,  v7,  8");
					
				asm volatile("vadd.vv v25, v25, v6");
				asm volatile("vadd.vv v26, v26, v7");
				
				// each plane outputs max F²(2²-1)² which with 2b prec and 3x3 tops at 81

				// but since we're adding 2 planes at a time, it adds up to 162, still below 2⁸-1
				// so we can keep using an 8b buffer for each plane and sum it on 16 after each 2 planes are processed
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v1, v1, 1");
					asm volatile("vslidedown.vi v2, v2, 1");
					asm volatile("vslidedown.vi v3, v3, 1");
					asm volatile("vslidedown.vi v4, v4, 1");
					asm volatile("vslidedown.vi v5, v5, 1");
				}
				
			}

		}
		
		i_ += (F - 1) * W_in;
		
		for (int height = F - 1 ; height < H_in ; height += 6){
					
			for(int channels = 0 ; channels < (C_in >> 1) ; channels ++){
				
				asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
			
				i__ = i_ + 2 * channels * H_in * W_in;
				
				asm volatile("vle8.v v6, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v7, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v8, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v9, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v10,(%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v11,(%0)"                 : "+&r" (i__));
				
				asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
				
				asm volatile("vzext.vf2 v0, v6");
				asm volatile("vzext.vf2 v1, v7");
				asm volatile("vzext.vf2 v2, v8");
				asm volatile("vzext.vf2 v3, v9");
				asm volatile("vzext.vf2 v4, v10");
				asm volatile("vzext.vf2 v5, v11");
				
				asm volatile("vsll.vi v0, v0, 8");
				asm volatile("vsll.vi v1, v1, 8");
				asm volatile("vsll.vi v2, v2, 8");
				asm volatile("vsll.vi v3, v3, 8");
				asm volatile("vsll.vi v4, v4, 8");
				asm volatile("vsll.vi v5, v5, 8");	
				
				//asm volatile("vse16.v v5, (%0)" : "+&r"(o_));
				//print_tensor_(o_, 1, W_in, 1);
				
				asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" ::"r"(vlen));
				
				i__ += (H_in - F + 2) * W_in;
				
				
				asm volatile("vle8.v v6,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v7,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v8,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v9,  (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v10, (%0); add %0, %0, %1" : "+&r" (i__) : "r"(ldi));
				asm volatile("vle8.v v11, (%0)"                 : "+&r" (i__));

				asm volatile("vwadd.wv v0, v0, v6");
				asm volatile("vwadd.wv v1, v1, v7");
				asm volatile("vwadd.wv v2, v2, v8");
				asm volatile("vwadd.wv v3, v3, v9");
				asm volatile("vwadd.wv v4, v4, v10");
				asm volatile("vwadd.wv v5, v5, v11");
				
				asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen));
					
				for(int f_w = 0; f_w < F ; f_w++){
					
					f_loop = f_packed + f_w + channels * F * F;
					
					// Row 0
					
					asm volatile("vmul.vx  v12, v0, %0" ::"r"(f_loop[0]));
					asm volatile("vmul.vx  v11, v0, %0" ::"r"(f_loop[7]));
					asm volatile("vmul.vx  v10, v0, %0" ::"r"(f_loop[14]));
					asm volatile("vmul.vx  v9,  v0, %0" ::"r"(f_loop[21]));
					asm volatile("vmul.vx  v8,  v0, %0" ::"r"(f_loop[28]));
					asm volatile("vmul.vx  v7,  v0, %0" ::"r"(f_loop[35]));
					asm volatile("vmul.vx  v6,  v0, %0" ::"r"(f_loop[42]));

					
					// Row 1
					if(height < H_in - 1){
						asm volatile("vmul.vx  v13, v1, %0" ::"r"(f_loop[0]));
							
						asm volatile("vmacc.vx v12, %0, v1" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v11, %0, v1" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v10, %0, v1" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v9,  %0, v1" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v8,  %0, v1" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v7,  %0, v1" ::"r"(f_loop[42]));

					}
					
					if(channels > 0 || f_w > 0){
						asm volatile("vsrl.vi v6,  v6,  8");
						if(height < H_in - 1){
							asm volatile("vsrl.vi v7,  v7,  8");
						if(height < H_in - 2){
							asm volatile("vsrl.vi v8,  v8,  8");
						if(height < H_in - 3){
							asm volatile("vsrl.vi v9,  v9,  8");
						if(height < H_in - 4){
								asm volatile("vsrl.vi v10, v10, 8");
						if(height < H_in - 5){
							asm volatile("vsrl.vi v11, v11, 8");
							asm volatile("vsrl.vi v12, v12, 8");
							asm volatile("vsrl.vi v13, v13, 8");
						}}}}}
							
							
						asm volatile("vadd.vv v19, v19, v6");
						if(height < H_in - 1){
							asm volatile("vadd.vv v20, v20, v7");
						if(height < H_in - 2){
							asm volatile("vadd.vv v21, v21, v8");
						if(height < H_in - 3){
							asm volatile("vadd.vv v22, v22, v9");
						if(height < H_in - 4){
							asm volatile("vadd.vv v23, v23, v10");
						if(height < H_in - 5){
							asm volatile("vadd.vv v24, v24, v11");
							asm volatile("vadd.vv v25, v25, v12");
							asm volatile("vadd.vv v26, v26, v13");
							
						}}}}}}
					else{
						asm volatile("vsrl.vi v6,  v6,  8");
						asm volatile("vsrl.vi v7,  v7,  8");
						asm volatile("vsrl.vi v8,  v8,  8");
						asm volatile("vsrl.vi v9,  v9,  8");
						asm volatile("vsrl.vi v10, v10, 8");
						asm volatile("vsrl.vi v11, v11, 8");
						

						asm volatile("vadd.vv v19, v25, v6");
						if(height < H_in - 1){
							asm volatile("vadd.vv v20, v26, v7");
						if(height < H_in - 2){
							asm volatile("vadd.vv v21, v27, v8");
						if(height < H_in - 3){
							asm volatile("vadd.vv v22, v28, v9");
						if(height < H_in - 4){
							asm volatile("vadd.vv v23, v29, v10");
						if(height < H_in - 5){
							asm volatile("vadd.vv v24, v30, v11");
						}}}}}
						asm volatile("vsrl.vi v25, v12, 8");
						asm volatile("vsrl.vi v26, v13, 8");
								
							

							
					}
					
					
					// Row 2
					if(height < H_in - 2){
						asm volatile("vmul.vx  v14, v2, %0" ::"r"(f_loop[0]));
						asm volatile("vmul.vx  v13, v2, %0" ::"r"(f_loop[7]));
						asm volatile("vmul.vx  v12, v2, %0" ::"r"(f_loop[14]));
						asm volatile("vmul.vx  v11, v2, %0" ::"r"(f_loop[21]));
						asm volatile("vmul.vx  v10, v2, %0" ::"r"(f_loop[28]));
						asm volatile("vmul.vx  v9,  v2, %0" ::"r"(f_loop[35]));
						asm volatile("vmul.vx  v8,  v2, %0" ::"r"(f_loop[42]));
					}
					
					// Row 3
					if(height < H_in - 3){
						asm volatile("vmul.vx  v15, v3, %0" ::"r"(f_loop[0]));

						asm volatile("vmacc.vx v14, %0, v3" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v13, %0, v3" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v12, %0, v3" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v11, %0, v3" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v10, %0, v3" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v9,  %0, v3" ::"r"(f_loop[42]));
					}
					
					if(height < H_in - 2){
						asm volatile("vsrl.vi v8,  v8,  8");
						asm volatile("vsrl.vi v9,  v9,  8");
						asm volatile("vsrl.vi v10, v10, 8");
						asm volatile("vsrl.vi v11, v11, 8");
						asm volatile("vsrl.vi v12, v12, 8");
						asm volatile("vsrl.vi v13, v13, 8");
						if(channels > 0 || f_w > 0){
							asm volatile("vsrl.vi v14, v14, 8");
							asm volatile("vsrl.vi v15, v15, 8");
						}
						else{
							asm volatile("vsrl.vi v27, v14, 8");
							asm volatile("vsrl.vi v28, v15, 8");
							
						}
							
							
							
						asm volatile("vadd.vv v21, v21, v8");
						asm volatile("vadd.vv v22, v22, v9");
						asm volatile("vadd.vv v23, v23, v10");
						asm volatile("vadd.vv v24, v24, v11");
						asm volatile("vadd.vv v25, v25, v12");
						asm volatile("vadd.vv v26, v26, v13");
						if(channels > 0 || f_w > 0){
							asm volatile("vadd.vv v27, v27, v14");
							asm volatile("vadd.vv v28, v28, v15");
						}
							
					}
					
					// Row 4
					if(height < H_in - 4){
						asm volatile("vmul.vx  v16, v4, %0" ::"r"(f_loop[0]));
						asm volatile("vmul.vx  v15, v4, %0" ::"r"(f_loop[7]));
						asm volatile("vmul.vx  v14, v4, %0" ::"r"(f_loop[14]));
						asm volatile("vmul.vx  v13, v4, %0" ::"r"(f_loop[21]));
						asm volatile("vmul.vx  v12, v4, %0" ::"r"(f_loop[28]));
						asm volatile("vmul.vx  v11, v4, %0" ::"r"(f_loop[35]));
						asm volatile("vmul.vx  v10, v4, %0" ::"r"(f_loop[42]));
					}
					
					// Row 5
					if(height < H_in - 5){
						asm volatile("vmul.vx  v17, v5, %0" ::"r"(f_loop[0]));
	
						asm volatile("vmacc.vx v16, %0, v5" ::"r"(f_loop[7]));
						asm volatile("vmacc.vx v15, %0, v5" ::"r"(f_loop[14]));
						asm volatile("vmacc.vx v14, %0, v5" ::"r"(f_loop[21]));
						asm volatile("vmacc.vx v13, %0, v5" ::"r"(f_loop[28]));
						asm volatile("vmacc.vx v12, %0, v5" ::"r"(f_loop[35]));
						asm volatile("vmacc.vx v11, %0, v5" ::"r"(f_loop[42]));
					}
					
					if(height < H_in - 4){
							
							asm volatile("vsrl.vi v10, v10, 8");
							asm volatile("vsrl.vi v11, v11, 8");
							asm volatile("vsrl.vi v12, v12, 8");
							asm volatile("vsrl.vi v13, v13, 8");
							asm volatile("vsrl.vi v14, v14, 8");
							asm volatile("vsrl.vi v15, v15, 8");
							if(channels > 0 || f_w > 0){
								asm volatile("vsrl.vi v16, v16, 8");
								asm volatile("vsrl.vi v17, v17, 8");
							}
							else{
								asm volatile("vsrl.vi v29, v16, 8");
								asm volatile("vsrl.vi v30, v17, 8");
							}
							
							asm volatile("vadd.vv v23, v23, v10");
							asm volatile("vadd.vv v24, v24, v11");
							asm volatile("vadd.vv v25, v25, v12");
							asm volatile("vadd.vv v26, v26, v13");
							asm volatile("vadd.vv v27, v27, v14");
							asm volatile("vadd.vv v28, v28, v15");
							if(channels > 0 || f_w > 0){
								asm volatile("vadd.vv v29, v29, v16");
								asm volatile("vadd.vv v30, v30, v17");
							}
							
					}
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						if(height < H_in - 1)
							asm volatile("vslidedown.vi v1, v1, 1");
						if(height < H_in - 2)
							asm volatile("vslidedown.vi v2, v2, 1");
						if(height < H_in - 3)
							asm volatile("vslidedown.vi v3, v3, 1");
						if(height < H_in - 4)
							asm volatile("vslidedown.vi v4, v4, 1");
						if(height < H_in - 5)
							asm volatile("vslidedown.vi v5, v5, 1");
					}
				}
					
			}
			
			i_ += (F - 1) * W_in;
			
			asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vlen_out));
			
			asm volatile("vse16.v v19, (%0)" : "+&r"(o_));
			o_ += ldo;
			if(height < H_in - 1){
				asm volatile("vse16.v v20, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 2){
				asm volatile("vse16.v v21, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 3){
				asm volatile("vse16.v v22, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 4){
				asm volatile("vse16.v v23, (%0)" : "+&r"(o_));
				o_ += ldo;
			}
			if(height < H_in - 5){
				asm volatile("vse16.v v24, (%0)" : "+&r"(o_));;
				o_ += ldo;
			}
		}
	}
}




