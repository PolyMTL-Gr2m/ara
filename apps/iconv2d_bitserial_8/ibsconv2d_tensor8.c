#include "ibsconv2d_tensor8.h"
#include <stdio.h>

//#define DEBUG

#define MIN(a, b) ((a) < (b) ? (a) : (b))

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

#ifdef DEBUG
void print_tensor_(uint8_t *tensor, uint64_t num_rows, uint64_t num_columns, uint64_t num_depth) {
  printf("0x%8X\n", (uint64_t)tensor);
  for (uint64_t k = 0; k < num_depth; ++k) {
  	for (uint64_t i = 0; i < num_rows; ++i) {
    	for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%10d ", tensor[(i+k*num_rows) * num_columns  + j ]);
    	}
    	printf("\n");
  	 }
  	 printf("\n");
	}
}
#endif

void ibsconv2d_tensor8_3x3(int8_t *o, int8_t *i, int8_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out, int64_t precA, int64_t precW) {
	
  int8_t *i_;
  int8_t *o_;
  int8_t *f_;
  
  //helper variable
  
  for(int64_t c = 0; c < C_out; c++) {
  		// First iteration round, c = 0 for the adress of the first value of the first filter
		o_ = o + c * (H_in - F + 1) * (W_in - F + 1);      // Output is incremented 
		i_ = i;                                            // Since we aren't working on batch, we only consider one input
		f_ = f + c * F * F * C_in;

	  // Iterate over the output rows
	  

		switch (precA){
			 case 1:
			 		if(precW == 1)
						ibsconv2d8_W1_A1_vec_3x3(o_, i_, f_, H_in, W_in, C_in, F);
					else
						ibsconv2d8_W2_A1_vec_3x3(o_, i_, f_, H_in, W_in, C_in, F);
				break;

			 case 2:
			 		if(precW == 1)
						ibsconv2d8_W1_A2_vec_3x3(o_, i_, f_, H_in, W_in, C_in, F);
					else
						ibsconv2d8_W2_A2_vec_3x3(o_, i_, f_, H_in, W_in, C_in, F);
				break;
				
		}
		 
		 
	}
}



//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                1x1 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                3x3 kernel                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



void ibsconv2d8_W1_A1_vec_3x3(int8_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

#ifdef DEBUG
	uint64_t cycle_count = 0;
	uint64_t sum_count = 0;
	uint64_t call_bp = 0;
#endif

int8_t f_packed[9];

int64_t ldo = W_in - 2;

int64_t vlen;	

for (int width = 0 ; width < (W_in - 2) ; width += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t * f_ptr_ = f_ptr;

	if(width > W_in - TILE_SIZE) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE;						// else we go full length

	bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);

	int8_t *i_ = i_ptr + width; 									// input pointer realtive to the tile (constant throughout the tile)
	int8_t *o_ = o_ptr + width;									// output pointer relative to the tile	
	
	int8_t *i__ = i_ptr + width;
	int8_t *o__ = o_ptr + width;


	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_1_to_8(i__ + 2 * W_in, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
	#endif
	
	asm volatile("vmv.v.v v4, v0");
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_1_to_8(i__ + W_in, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
	#endif
	
	asm volatile("vmv.v.v v2, v0");
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_1_to_8(i__, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
	#endif

			
		// LSB
		asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v12, v2, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v4, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v16, v2, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v18, v4, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		

		// v10 = popcount(v10)
		asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		
		// Pre-Calc
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");

			
			
		asm volatile("vadd.vv v30, v10, v12");
		asm volatile("vadd.vv v28, v16, v18");
		asm volatile("vmv.v.v v26, v20");
		
		asm volatile("vadd.vv v30, v30, v14");		
		
		asm volatile("vslidedown.vi v0, v0, 1");				
		asm volatile("vslidedown.vi v2, v2, 1");				
		asm volatile("vslidedown.vi v4, v4, 1");

		
		
		// LSB
		asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v12, v2, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v4, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v16, v2, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v18, v4, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel


		// v10 = popcount(v10)
		asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		
		// Pre-Calc
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");


		asm volatile("vadd.vv v30, v30, v10");
		asm volatile("vadd.vv v28, v28, v16");
		asm volatile("vadd.vv v26, v26, v20");
			
		asm volatile("vadd.vv v30, v30, v12");
		asm volatile("vadd.vv v28, v28, v18");
			
		asm volatile("vadd.vv v30, v30, v14");		
		
		asm volatile("vslidedown.vi v0, v0, 1");				
		asm volatile("vslidedown.vi v2, v2, 1");				
		asm volatile("vslidedown.vi v4, v4, 1");
		
		
		// LSB
		asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v12, v2, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v4, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v16, v2, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v18, v4, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		

		// v10 = popcount(v10)
		asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		
		// Pre-Calc
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");

			
			
		asm volatile("vadd.vv v30, v30, v10");
		asm volatile("vadd.vv v28, v28, v16");
		asm volatile("vadd.vv v26, v26, v20");
			
		asm volatile("vadd.vv v30, v30, v12");
		asm volatile("vadd.vv v28, v28, v18");
			
		asm volatile("vadd.vv v30, v30, v14");		
		
		
		for(int channels = 8 ; channels < C_in ; channels += 8){ //can only do multiple of 8 channels
		
			f_ptr_ += 8 * F * F;
			
			i__ += 8 * H_in * W_in;
			
			bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
		
				
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_1_to_8(i__ + 2 * W_in, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
			#endif
			
			asm volatile("vmv.v.v v4, v0");
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_1_to_8(i__ + W_in, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
			#endif
			
			asm volatile("vmv.v.v v2, v0");
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_1_to_8(i__, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
			#endif
			
						
			// LSB
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v12, v2, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v14, v4, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			// Pre-Calc
			
			asm volatile("vand.vx v16, v2, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
			asm volatile("vand.vx v18, v4, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

			asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[0])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
			
			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
			// v14 = popcount(v14)
			asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
			
			// Pre-Calc
			// v16 = popcount(v16)
			asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			// v18 = popcount(v18)
			asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
			// v20 = popcount(v20)
			asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
					
				
			asm volatile("vadd.vv v30, v30, v10");
			asm volatile("vadd.vv v28, v28, v16");
			asm volatile("vadd.vv v26, v26, v20");
			
			asm volatile("vadd.vv v30, v30, v12");
			asm volatile("vadd.vv v28, v28, v18");
			
			asm volatile("vadd.vv v30, v30, v14");	
			
			asm volatile("vslidedown.vi v0, v0, 1");				
			asm volatile("vslidedown.vi v2, v2, 1");				
			asm volatile("vslidedown.vi v4, v4, 1");
				
			
			// LSB
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v12, v2, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v14, v4, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			// Pre-Calc
			
			asm volatile("vand.vx v16, v2, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v18, v4, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[1])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
			// v14 = popcount(v14)
			asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
			
			// Pre-Calc
			// v16 = popcount(v16)
			asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			// v18 = popcount(v18)
			asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
			// v20 = popcount(v20)
			asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
			
			
			asm volatile("vadd.vv v30, v30, v10");
			asm volatile("vadd.vv v28, v28, v16");
			asm volatile("vadd.vv v26, v26, v20");
			
			asm volatile("vadd.vv v30, v30, v12");
			asm volatile("vadd.vv v28, v28, v18");
			
			asm volatile("vadd.vv v30, v30, v14");
			
			asm volatile("vslidedown.vi v0, v0, 1");				
			asm volatile("vslidedown.vi v2, v2, 1");				
			asm volatile("vslidedown.vi v4, v4, 1");
			
			
			// LSB
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v12, v2, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v14, v4, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			// Pre-Calc
			
			asm volatile("vand.vx v16, v2, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v18, v4, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
			asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[2])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
			// v14 = popcount(v14)
			asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
			
			// Pre-Calc
			// v16 = popcount(v16)
			asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			// v18 = popcount(v18)
			asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
			// v20 = popcount(v20)
			asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
					
				
			asm volatile("vadd.vv v30, v30, v10");
			asm volatile("vadd.vv v28, v28, v16");
			asm volatile("vadd.vv v26, v26, v20");
			
			asm volatile("vadd.vv v30, v30, v12");
			asm volatile("vadd.vv v28, v28, v18");
			
			asm volatile("vadd.vv v30, v30, v14");	
			
	
		}
		
				
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
			
		asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
		
		i_ += 3 * W_in;	
		
		// v26 (needs 2 more lines) and v28 (needs one more line) are used to store precalculated values
		
		for (int height = 3 ; height < H_in - 2 ; height += 1){
			
			f_ptr_ = f_ptr;
			i__ = i_;
			o_ += ldo;
			
			
			if(C_in > 8)
				bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);	
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_1_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
				#endif

			
			
			
			
			// LSB
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[6])); // complete last line for v28
			
			asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[3])); // complete 1/2 line for v26 which will become v28
			
			
			// Pre-Calc
			
			asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[0])); // Pre-Calc for next iteration which will become v26
			
			

			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
			// v14 = popcount(v14)
			asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
			

			asm volatile("vslidedown.vi v0, v0, 1");
				
				
			asm volatile("vadd.vv v30, v28, v10");
			
			asm volatile("vadd.vv v28, v26, v12");		
			
			asm volatile("vmv.v.v v26, v14");		
			
			
			
			
			
			
			
			// LSB
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[7])); // complete last line for v28
			
			asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[4])); // complete 1/2 line for v26 which will become v28
			
			
			// Pre-Calc
			
			asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[1])); // Pre-Calc for next iteration which will become v26
			
			

			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
			// v14 = popcount(v14)
			asm volatile(".byte  0x57, 0x07, 0x07, 0x06");

				
			asm volatile("vslidedown.vi v0, v0, 1");
			
				
			asm volatile("vadd.vv v30, v30, v10");
			
			asm volatile("vadd.vv v28, v28, v12");		
			
			asm volatile("vadd.vv v26, v26, v14");			
			
			
			
			
			
			// LSB
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[8])); // complete last line for v28
			
			asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[5])); // complete 1/2 line for v26 which will become v28
			
			
			// Pre-Calc
			
			asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[2])); // Pre-Calc for next iteration which will become v26
			
			

			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
			// v14 = popcount(v14)
			asm volatile(".byte  0x57, 0x07, 0x07, 0x06");

				
				
			asm volatile("vadd.vv v30, v30, v10");
			
			asm volatile("vadd.vv v28, v28, v12");		
			
			asm volatile("vadd.vv v26, v26, v14");
			
			for(int channels = 8 ; channels < C_in ; channels += 8){ //can only do multiple of 8 channels
			
				f_ptr_ += 8 * F * F;
				
				i__ += 8 * H_in * W_in;
				
				bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
				
				asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

			
				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_1_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
				#endif
				
				
				// LSB
				asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[6])); // complete last line for v28
				
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[3])); // complete 1/2 line for v26 which will become v28
				
				
				// Pre-Calc
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[0])); // Pre-Calc for next iteration which will become v26
				
				

				// v10 = popcount(v10)
				asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");


				asm volatile("vslidedown.vi v0, v0, 1");
					
					
				asm volatile("vadd.vv v30, v30, v10");
				
				asm volatile("vadd.vv v28, v28, v12");		
				
				asm volatile("vadd.vv v26, v26, v14");		
				
				
				
				
				
				// LSB
				asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[7])); // complete last line for v28
				
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[4])); // complete 1/2 line for v26 which will become v28
				
				
				// Pre-Calc
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[1])); // Pre-Calc for next iteration which will become v26
				
				

				// v10 = popcount(v10)
				asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");


				asm volatile("vslidedown.vi v0, v0, 1");
					
					
				asm volatile("vadd.vv v30, v30, v10");
				
				asm volatile("vadd.vv v28, v28, v12");		
				
				asm volatile("vadd.vv v26, v26, v14");			
				
				
				
				
				
				// LSB
				asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[8])); // complete last line for v28
				
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[5])); // complete 1/2 line for v26 which will become v28
				
				
				// Pre-Calc
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[2])); // Pre-Calc for next iteration which will become v26
				
				

				// v10 = popcount(v10)
				asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");

					
					
				asm volatile("vadd.vv v30, v30, v10");
				
				asm volatile("vadd.vv v28, v28, v12");		
				
				asm volatile("vadd.vv v26, v26, v14");
			
			}			
			
			i_ += W_in;
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));	
					
		}
		
		// last 2 lines (no need for pre calculation)
		
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
			
		f_ptr_ = f_ptr;			
		i__ = i_;
		o_ += ldo;

		if(C_in > 8)
			bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
		
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

		#ifdef DEBUG
			start_timer();
		#endif
		bitpack8_vec_1_to_8(i__, H_in * W_in); 
		#ifdef DEBUG
			stop_timer();
			sum_count = get_timer();
			cycle_count += sum_count;
			call_bp ++;
		#endif
				
			
		// LSB
		asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			
			
		// v10 = popcount(v10)
		asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");

		asm volatile("vslidedown.vi v0, v0, 1");

		asm volatile("vadd.vv v30, v28, v10");
		asm volatile("vadd.vv v28, v26, v12");
		
		
		
		
		// LSB
		asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			
			
		// v10 = popcount(v10)
		asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");

		asm volatile("vslidedown.vi v0, v0, 1");

		asm volatile("vadd.vv v30, v30, v10");
		asm volatile("vadd.vv v28, v28, v12");
		
		
		
		
		// LSB
		asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			
			
		// v10 = popcount(v10)
		asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");

		asm volatile("vadd.vv v30, v30, v10");
		asm volatile("vadd.vv v28, v28, v12");	
		
		for(int channels = 8 ; channels < C_in ; channels += 8){ //can only do multiple of 8 channels
			
			f_ptr_ += 8 * F * F;
			
			i__ += 8 * H_in * W_in;
			
			bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
		
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_1_to_8(i__, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
			#endif
				
			// LSB
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				
			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");

			asm volatile("vslidedown.vi v0, v0, 1");

			asm volatile("vadd.vv v30, v30, v10");
			asm volatile("vadd.vv v28, v28, v12");
			
			
			
			
			// LSB
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[4])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				
			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");

			asm volatile("vslidedown.vi v0, v0, 1");

			asm volatile("vadd.vv v30, v30, v10");
			asm volatile("vadd.vv v28, v28, v12");
			
			
			
			
			// LSB
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
			asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[5])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				
			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");

			asm volatile("vadd.vv v30, v30, v10");
			asm volatile("vadd.vv v28, v28, v12");	
			
			
			
			}
		
			
		i_ += W_in;
		
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
			
		asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
		
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
					
					
		f_ptr_ = f_ptr;				
		i__ = i_;
		o_ += ldo;
		
		if(C_in > 8)
			bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);	
		
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));	

		#ifdef DEBUG
			start_timer();
		#endif
		bitpack8_vec_1_to_8(i__, H_in * W_in); 
		#ifdef DEBUG
			stop_timer();
			sum_count = get_timer();
			cycle_count += sum_count;
			call_bp ++;
		#endif
			
		
		// LSB (activation bit 0)
		asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
		asm volatile("vslidedown.vi v0, v0, 1");
			
		// v10 = popcount(v10)
		asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			
		// LSB (activation bit 0)
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
		asm volatile("vslidedown.vi v0, v0, 1");
		
		asm volatile("vadd.vv v30, v28, v10");
			
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		
		// LSB (activation bit 0)
		asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
			
		asm volatile("vadd.vv v30, v30, v12");
		
		// v10 = popcount(v10)
		asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
		
		asm volatile("vadd.vv v30, v30, v10");
		
		for(int channels = 8 ; channels < C_in ; channels += 8){ //can only do multiple of 8 channels
			
			f_ptr_ += 8 * F * F;
			
			i__ += 8 * H_in * W_in;
			
			bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
		
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_1_to_8(i__, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
			#endif
			
			// LSB (activation bit 0)
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
			asm volatile("vslidedown.vi v0, v0, 1");
				
			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
				
			// LSB (activation bit 0)
			asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[7])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
			asm volatile("vslidedown.vi v0, v0, 1");
			
			asm volatile("vadd.vv v30, v30, v10");
				
			// v12 = popcount(v12)
			asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
			
			// LSB (activation bit 0)
			asm volatile("vand.vx v10, v0, %0" :: "r"(f_packed[8])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
			asm volatile("vadd.vv v30, v30, v12");
			
			// v10 = popcount(v10)
			asm volatile(".byte  0x57, 0x05, 0x05, 0x06");
			
			asm volatile("vadd.vv v30, v30, v10");
			
			}

		
			
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
			
		asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
		
		
	}
	#ifdef DEBUG
		printf("The execution took %d cycles.\n", cycle_count);
	#endif
}






void ibsconv2d8_W2_A1_vec_3x3(int8_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

#ifdef DEBUG
	uint64_t cycle_count = 0;
	uint64_t sum_count = 0;
	uint64_t call_bp = 0;
#endif

int8_t f_packed[18];

int64_t ldo = W_in - 2;

int64_t vlen;

for (int width = 0 ; width < (W_in - 2) ; width += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t * f_ptr_ = f_ptr;

	if(width > W_in - TILE_SIZE) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE;						// else we go full length


	bitpack_filter8_vec_2_to_8(f_ptr, f_packed, F*F);

	int8_t *i_ = i_ptr + width; 									// input pointer realtive to the tile (constant throughout the tile)
	int8_t *o_ = o_ptr + width;									// output pointer relative to the tile	
	
	int8_t *i__ = i_ptr + width;
	int8_t *o__ = o_ptr + width;

	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_1_to_8(i__ + 2 * W_in, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
		call_bp ++;
	#endif
	
	asm volatile("vmv.v.v v4, v0");
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_1_to_8(i__ + W_in, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
		call_bp ++;
	#endif
	
	asm volatile("vmv.v.v v2, v0");
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_1_to_8(i__, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
		call_bp ++;
	#endif
	
	
	for (int f_w = 0 ; f_w < F ; f_w ++) {
	
	
		// LSB
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		// Pre-Calc
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
		// v22 = popcount(v22)
		asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");

			
		if(f_w == 0){
					
			asm volatile("vadd.vv v30, v12, v14");
			asm volatile("vadd.vv v28, v18, v20");
			asm volatile("vmv.v.v v26, v22");	
			
			asm volatile("vadd.vv v30, v30, v16");
				
		}
		else
		{
			asm volatile("vadd.vv v30, v30, v12");
			asm volatile("vadd.vv v28, v28, v18");
			asm volatile("vadd.vv v26, v26, v22");	
			
			asm volatile("vadd.vv v30, v30, v14");
			asm volatile("vadd.vv v28, v28, v20");
			
			asm volatile("vadd.vv v30, v30, v16");		
		}
			

		

		// MSB
		
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		// Pre-Calc
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
		// v22 = popcount(v22)
		asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
		
		asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
		asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
		asm volatile("vmacc.vx v26,  %0, v22" ::"r"(2));
		
		asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
		asm volatile("vmacc.vx v28,  %0, v20" ::"r"(2));
		
		asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
		
		if(f_w < F - 1){
			asm volatile("vslidedown.vi v0, v0, 1");
			asm volatile("vslidedown.vi v2, v2, 1");
			asm volatile("vslidedown.vi v4, v4, 1");
		}
				
		
		}
		
		
		
		for(int channels = 8 ; channels < C_in ; channels += 8){
		
			
			f_ptr_ += 8 * F * F;
			
			i__ += 8 * H_in * W_in;
			
			bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
		
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_1_to_8(i__ + 2 * W_in, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
			
			asm volatile("vmv.v.v v4, v0");
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_1_to_8(i__ + W_in, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
			
			asm volatile("vmv.v.v v2, v0");
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_1_to_8(i__, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
			
			
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
			
				// LSB
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// Pre-Calc
				
				asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

				asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				// Pre-Calc
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");

					
				asm volatile("vadd.vv v30, v30, v12");
				asm volatile("vadd.vv v28, v28, v18");
				asm volatile("vadd.vv v26, v26, v22");	
					
				asm volatile("vadd.vv v30, v30, v14");
				asm volatile("vadd.vv v28, v28, v20");
					
				asm volatile("vadd.vv v30, v30, v16");		
				

				// MSB
				
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// Pre-Calc
				
				asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel

				asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				// Pre-Calc
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
				
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
				asm volatile("vmacc.vx v26,  %0, v22" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v20" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
				
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v2, v2, 1");
					asm volatile("vslidedown.vi v4, v4, 1");
				}
						
				
				}
			}	
		

		
				
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
			
		asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
		
		i_ += 3 * W_in;	
		
		// v26 (needs 2 more lines) and v28 (needs one more line) are used to store precalculated values
		
		
		for (int height = 3 ; height < H_in - 2 ; height += 1){
		
			f_ptr_ = f_ptr;
		
			if(C_in > 8)
				bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
			
			i__ = i_;
			o_ += ldo;

				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_1_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif
				
				
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
				// LSB (activation bit 0)
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				if(f_w == 0){
					
					asm volatile("vadd.vv v30, v28, v12");
					
					asm volatile("vadd.vv v28, v26, v14");
					
					asm volatile("vmv.v.v v26, v16");
				
				}
				else
				{
					asm volatile("vadd.vv v30, v30, v12");
				
					asm volatile("vadd.vv v28, v28, v14");		
				
					asm volatile("vadd.vv v26, v26, v16");	
				}
				
												
						
						
				// MSB (activation bit 1)
				
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
						
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
				
				asm volatile("vmacc.vx v28,  %0, v14" ::"r"(2));
				
				asm volatile("vmacc.vx v26,  %0, v16" ::"r"(2));

						
				
				if(f_w < F - 1)
					asm volatile("vslidedown.vi v0, v0, 1");
				
			
			
				}
				
				for(int channels = 8 ; channels < C_in ; channels += 8){
				
					
					f_ptr_ += 8 * F * F;
					
					i__ += 8 * H_in * W_in;
					
					bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
					
					asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
				
					#ifdef DEBUG
						start_timer();
					#endif
					bitpack8_vec_1_to_8(i__, H_in * W_in); 
					#ifdef DEBUG
						stop_timer();
						sum_count = get_timer();
						cycle_count += sum_count;
						call_bp ++;
					#endif
					
					
					for (int f_w = 0 ; f_w < F ; f_w ++) {
					
						// LSB (activation bit 0)
						asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						
						// v12 = popcount(v12)
						asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
						// v14 = popcount(v14)
						asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
						// v16 = popcount(v16)
						asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
						
						
						
						asm volatile("vadd.vv v30, v30, v12");
						
						asm volatile("vadd.vv v28, v28, v14");		
						
						asm volatile("vadd.vv v26, v26, v16");	
										
														
								
								
						// MSB (activation bit 1)
						
						asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						
						// v12 = popcount(v12)
						asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
						// v14 = popcount(v14)
						asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
						// v16 = popcount(v16)
						asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
						
								
						asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
						
						asm volatile("vmacc.vx v28,  %0, v14" ::"r"(2));
						
						asm volatile("vmacc.vx v26,  %0, v16" ::"r"(2));

								
						
						if(f_w < F - 1)
							asm volatile("vslidedown.vi v0, v0, 1");
						
					
					
						}
				}
				
				
			i_ += W_in;
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));	
			}
			
			
			// last 2 lines (no need for pre calculation)
			
			f_ptr_ = f_ptr;
			
			if(C_in > 8)
				bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
			
			i__ = i_;
			o_ += ldo;

			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_1_to_8(i__, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
				
			asm volatile("vmv.v.v v30, v28");
			asm volatile("vmv.v.v v28, v26");
				
				
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
				// LSB (activation bit 0)
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// MSB (activation bit 1)
				asm volatile("vand.vx v20, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v22, v0, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
				
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
				}
				
				asm volatile("vadd.vv v30, v30, v12");
				asm volatile("vadd.vv v28, v28, v14");
				
				asm volatile("vmacc.vx v30,  %0, v20" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v22" ::"r"(2));

			}
			
				for(int channels = 8 ; channels < C_in ; channels += 8){
				
					f_ptr_ += 8 * F * F;
					
					i__ += 8 * H_in * W_in;
					
					bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
					
					asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
					
					#ifdef DEBUG
						start_timer();
					#endif
					bitpack8_vec_1_to_8(i__, H_in * W_in); 
					#ifdef DEBUG
						stop_timer();
						sum_count = get_timer();
						cycle_count += sum_count;
						call_bp ++;
					#endif
					
					for (int f_w = 0 ; f_w < F ; f_w ++) {
					
						// LSB (activation bit 0)
						asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						// MSB (activation bit 1)
						asm volatile("vand.vx v20, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						asm volatile("vand.vx v22, v0, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						
						// v12 = popcount(v12)
						asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
						// v14 = popcount(v14)
						asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
						// v20 = popcount(v20)
						asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
						// v22 = popcount(v22)
						asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
						
						if(f_w < F - 1){
							asm volatile("vslidedown.vi v0, v0, 1");
						}
						
						asm volatile("vadd.vv v30, v30, v12");
						asm volatile("vadd.vv v28, v28, v14");
						
						asm volatile("vmacc.vx v30,  %0, v20" ::"r"(2));
						asm volatile("vmacc.vx v28,  %0, v22" ::"r"(2));

					}
			
			}
				
			i_ += W_in;
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
	
			f_ptr_ = f_ptr;
			
			if(C_in > 8)
				bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
						
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));			
						
			i__ = i_;
			o_ += ldo;

				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_1_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif
			
			asm volatile("vmv.v.v v30, v28");
				
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
				// LSB (weights bit 0)
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				// MSB (weights bit 1)
				asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
				}
				
				asm volatile("vadd.vv v30, v30, v12");;
				asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
				
				}
				
				for(int channels = 8 ; channels < C_in ; channels += 8){
				
					f_ptr_ += 8 * F * F;
					
					i__ += 8 * H_in * W_in;
					
					bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
					
					asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));	
					
					#ifdef DEBUG
						start_timer();
					#endif
					bitpack8_vec_1_to_8(i__, H_in * W_in); 
					#ifdef DEBUG
						stop_timer();
						sum_count = get_timer();
						cycle_count += sum_count;
						call_bp ++;
					#endif
					
					for (int f_w = 0 ; f_w < F ; f_w ++) {
					
						// LSB (weights bit 0)
						asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						// MSB (weights bit 1)
						asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						
						// v12 = popcount(v12)
						asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
						// v16 = popcount(v16)
						asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
						
						if(f_w < F - 1){
							asm volatile("vslidedown.vi v0, v0, 1");
						}
						
						asm volatile("vadd.vv v30, v30, v12");;
						asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
						
					}
			}
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
			
		
	}
	#ifdef DEBUG
		printf("The execution took %d cycles.\n", cycle_count);
		printf("Bitpack operation was called %d times.\n", call_bp);
	#endif


}


void ibsconv2d8_W1_A2_vec_3x3(int8_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

#ifdef DEBUG
	uint64_t cycle_count = 0;
	uint64_t sum_count = 0;
	uint64_t call_bp = 0;
#endif

int8_t f_packed[9];

int64_t ldo = W_in - 2;

int64_t vlen;



for (int width = 0 ; width < (W_in - 2) ; width += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t * f_ptr_ = f_ptr;

	if(width > W_in - TILE_SIZE) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE;						// else we go full length


	int8_t *i_ = i_ptr + width; 									// input pointer realtive to the tile (constant throughout the tile)
	int8_t *o_ = o_ptr + width;									// output pointer relative to the tile	
	
	int8_t *i__ = i_ptr + width;
	int8_t *o__ = o_ptr + width;
	
	bitpack_filter8_vec_1_to_8(f_ptr, f_packed, F*F);

	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_2_to_8(i__ + 2 * W_in, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
		call_bp ++;
	#endif
	
	asm volatile("vmv.v.v v4, v0");
	asm volatile("vmv.v.v v10, v6");
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_2_to_8(i__ + W_in, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
		call_bp ++;
	#endif
	
	asm volatile("vmv.v.v v2, v0");
	asm volatile("vmv.v.v v8, v6");
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_2_to_8(i__, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
		call_bp ++;
	#endif
	
	
	for (int f_w = 0 ; f_w < F ; f_w ++) {
	
	
		// LSB
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		// Pre-Calc
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
		// v22 = popcount(v22)
		asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");

			
		if(f_w == 0){
					
			asm volatile("vadd.vv v30, v12, v14");
			asm volatile("vadd.vv v28, v18, v20");
			asm volatile("vmv.v.v v26, v22");	
			
			asm volatile("vadd.vv v30, v30, v16");
				
		}
		else
		{
			asm volatile("vadd.vv v30, v30, v12");
			asm volatile("vadd.vv v28, v28, v18");
			asm volatile("vadd.vv v26, v26, v22");	
			
			asm volatile("vadd.vv v30, v30, v14");
			asm volatile("vadd.vv v28, v28, v20");
			
			asm volatile("vadd.vv v30, v30, v16");		
		}				
		
		
		// LSB
		asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v16, v10, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v18, v8, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v20, v10, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v22, v10, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		

		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		// Pre-Calc
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
		// v22 = popcount(v22)
		asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
		
		asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
		asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
		asm volatile("vmacc.vx v26,  %0, v22" ::"r"(2));
		
		asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
		asm volatile("vmacc.vx v28,  %0, v20" ::"r"(2));
		
		asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
				
				
				
		if(f_w < F - 1){
			asm volatile("vslidedown.vi v0, v0, 1");
			asm volatile("vslidedown.vi v2, v2, 1");
			asm volatile("vslidedown.vi v4, v4, 1");
			
			asm volatile("vslidedown.vi v6, v6, 1");
			asm volatile("vslidedown.vi v8, v8, 1");
			asm volatile("vslidedown.vi v10, v10, 1");
		}
		
		}
		
		for(int channels = 8 ; channels < C_in ; channels += 8){
				
			f_ptr_ += 8 * F * F;
					
			i__ += 8 * H_in * W_in;
			
			bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
					
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));	
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_2_to_8(i__ + 2 * W_in, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
			
			asm volatile("vmv.v.v v4, v0");
			asm volatile("vmv.v.v v10, v6");
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_2_to_8(i__ + W_in, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
			
			asm volatile("vmv.v.v v2, v0");
			asm volatile("vmv.v.v v8, v6");
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_2_to_8(i__, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
			
			
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
			
				// LSB
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// Pre-Calc
				
				asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

				asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				// Pre-Calc
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");

				asm volatile("vadd.vv v30, v30, v12");
				asm volatile("vadd.vv v28, v28, v18");
				asm volatile("vadd.vv v26, v26, v22");	
					
				asm volatile("vadd.vv v30, v30, v14");
				asm volatile("vadd.vv v28, v28, v20");
					
				asm volatile("vadd.vv v30, v30, v16");				
				
				
				// LSB
				asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v10, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// Pre-Calc
				
				asm volatile("vand.vx v18, v8, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v20, v10, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

				asm volatile("vand.vx v22, v10, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				

				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				// Pre-Calc
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
				
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
				asm volatile("vmacc.vx v26,  %0, v22" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v20" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
						
						
						
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v2, v2, 1");
					asm volatile("vslidedown.vi v4, v4, 1");
					
					asm volatile("vslidedown.vi v6, v6, 1");
					asm volatile("vslidedown.vi v8, v8, 1");
					asm volatile("vslidedown.vi v10, v10, 1");
				}
				
			}
		}
				
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
			
		asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
		
		i_ += 3 * W_in;	
		
		// v26 (needs 2 more lines) and v28 (needs one more line) are used to store precalculated values
		
		
		for (int height = 3 ; height < H_in - 2 ; height += 1){
		
			f_ptr_ = f_ptr;
			
			if(C_in > 8)
				bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
			
			i__ = i_;
			o_ += ldo;

				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif
				
				
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
				// LSB (activation bit 0)
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				if(f_w == 0){
					
					asm volatile("vadd.vv v30, v28, v12");
					
					asm volatile("vadd.vv v28, v26, v14");
					
					asm volatile("vmv.v.v v26, v16");
				
				}
				else
				{
					asm volatile("vadd.vv v30, v30, v12");
				
					asm volatile("vadd.vv v28, v28, v14");		
				
					asm volatile("vadd.vv v26, v26, v16");	
				}
				
				
				// LSB (weights bit 0)
				
				asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				

				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			
				
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
				
				asm volatile("vmacc.vx v28,  %0, v14" ::"r"(2));
				
				asm volatile("vmacc.vx v26,  %0, v16" ::"r"(2));
				
						
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v6, v6, 1");
				}
			}
			for(int channels = 8 ; channels < C_in ; channels += 8){
					
				f_ptr_ += 8 * F * F;
						
				i__ += 8 * H_in * W_in;
				
				bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
						
				asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));	
				
				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif
				
				
				for (int f_w = 0 ; f_w < F ; f_w ++) {
				
					// LSB (activation bit 0)
					asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					
					// v12 = popcount(v12)
					asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
					// v14 = popcount(v14)
					asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
					// v16 = popcount(v16)
					asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
					
					asm volatile("vadd.vv v30, v30, v12");
					
					asm volatile("vadd.vv v28, v28, v14");		
					
					asm volatile("vadd.vv v26, v26, v16");	
										
					
					// LSB (weights bit 0)
					
					asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					

					// v12 = popcount(v12)
					asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
					// v14 = popcount(v14)
					asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
					// v16 = popcount(v16)
					asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
					
					asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
					
					asm volatile("vmacc.vx v28,  %0, v14" ::"r"(2));
					
					asm volatile("vmacc.vx v26,  %0, v16" ::"r"(2));
					
							
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						asm volatile("vslidedown.vi v6, v6, 1");
					}
				}
			}
				
			i_ += W_in;
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));	
			}
			
			
			// last 2 lines (no need for pre calculation)
			
			f_ptr_ = f_ptr;
			
			if(C_in > 8)
				bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
			
			i__ = i_;
			o_ += ldo;

				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif
				
			asm volatile("vmv.v.v v30, v28");
			asm volatile("vmv.v.v v28, v26");
				
				
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
				// LSB (activation bit 0)
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// LSB (weights bit 0)
				asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v18, v6, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");

				
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v6, v6, 1");
				}
				
				asm volatile("vadd.vv v30, v30, v12");
				asm volatile("vadd.vv v28, v28, v14");
				
				asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
				
			
			
			}
			for(int channels = 8 ; channels < C_in ; channels += 8){
					
				f_ptr_ += 8 * F * F;
						
				i__ += 8 * H_in * W_in;
				
				bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
						
				asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));	
				
				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif	
				
				for (int f_w = 0 ; f_w < F ; f_w ++) {
			
					// LSB (activation bit 0)
					asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					// LSB (weights bit 0)
					asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v18, v6, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					
					// v12 = popcount(v12)
					asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
					// v14 = popcount(v14)
					asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
					// v16 = popcount(v16)
					asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
					// v18 = popcount(v18)
					asm volatile(".byte  0x57, 0x09, 0x09, 0x06");

					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						asm volatile("vslidedown.vi v6, v6, 1");
					}
					
					asm volatile("vadd.vv v30, v30, v12");
					asm volatile("vadd.vv v28, v28, v14");
					
					asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
					asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
				}
			}
				
			i_ += W_in;
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
			
			
			
			f_ptr_ = f_ptr;
			
			if(C_in > 8)
				bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
						
			i__ = i_;
			o_ += ldo;

				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif
			
			asm volatile("vmv.v.v v30, v28");
				
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
				// LSB (activation bit 0)
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				// LSB (weights bit 0)
				asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v6, v6, 1");
				}
				
				asm volatile("vadd.vv v30, v30, v12");;
				asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
			
				}	
				
				
				for(int channels = 8 ; channels < C_in ; channels += 8){
					
					f_ptr_ += 8 * F * F;
							
					i__ += 8 * H_in * W_in;
					
					bitpack_filter8_vec_1_to_8(f_ptr_, f_packed, F*F);
							
					asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));	
					
					#ifdef DEBUG
						start_timer();
					#endif
					bitpack8_vec_2_to_8(i__, H_in * W_in); 
					#ifdef DEBUG
						stop_timer();
						sum_count = get_timer();
						cycle_count += sum_count;
						call_bp ++;
					#endif	
					
					for (int f_w = 0 ; f_w < F ; f_w ++) {
			
						// LSB (activation bit 0)
						asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						// LSB (weights bit 0)
						asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
						
						// v12 = popcount(v12)
						asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
						// v14 = popcount(v14)
						asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
						
						if(f_w < F - 1){
							asm volatile("vslidedown.vi v0, v0, 1");
							asm volatile("vslidedown.vi v6, v6, 1");
						}
						
						asm volatile("vadd.vv v30, v30, v12");;
						asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
					
						}
					}	
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
			
		
	}
	#ifdef DEBUG
		printf("The execution took %d cycles.\n", cycle_count);
		printf("Bitpack operation was called %d times.\n", call_bp);
	#endif


}



void ibsconv2d8_W2_A2_vec_3x3(int8_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

#ifdef DEBUG
	uint64_t cycle_count = 0;
	uint64_t sum_count = 0;
	uint64_t call_bp = 0;
#endif

int8_t f_packed[18];

int64_t ldo = W_in - 2;

int64_t vlen;

bitpack_filter8_vec_2_to_8(f_ptr, f_packed, F*F);

for (int width = 0 ; width < (W_in - 2) ; width += TILE_SIZE_OUT) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
{

	int8_t * f_ptr_ = f_ptr;

	if(width > W_in - TILE_SIZE) 	// if we are at the right border of the input
		vlen = W_in % TILE_SIZE;		 	// we set the vector length to fit the last inputs
	else
		vlen = TILE_SIZE;						// else we go full length


	int8_t *i_ = i_ptr + width; 									// input pointer realtive to the tile (constant throughout the tile)
	int8_t *o_ = o_ptr + width;									// output pointer relative to the tile	
	
	int8_t *i__ = i_ptr + width;
	int8_t *o__ = o_ptr + width;
	
	if(C_in > 8)
		bitpack_filter8_vec_2_to_8(f_ptr, f_packed, F*F);

	asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_2_to_8(i__ + 2 * W_in, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
		call_bp ++;
	#endif
	
	asm volatile("vmv.v.v v4, v0");
	asm volatile("vmv.v.v v10, v6");
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_2_to_8(i__ + W_in, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
		call_bp ++;
	#endif
	
	asm volatile("vmv.v.v v2, v0");
	asm volatile("vmv.v.v v8, v6");
	
	#ifdef DEBUG
		start_timer();
	#endif
	bitpack8_vec_2_to_8(i__, H_in * W_in); 
	#ifdef DEBUG
		stop_timer();
		sum_count = get_timer();
		cycle_count += sum_count;
		call_bp ++;
	#endif
	
	
	for (int f_w = 0 ; f_w < F ; f_w ++) {
	
	
		// LSB
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		// Pre-Calc
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
		// v22 = popcount(v22)
		asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");

			
		if(f_w == 0){
					
			asm volatile("vadd.vv v30, v12, v14");
			asm volatile("vadd.vv v28, v18, v20");
			asm volatile("vmv.v.v v26, v22");	
			
			asm volatile("vadd.vv v30, v30, v16");
				
		}
		else
		{
			asm volatile("vadd.vv v30, v30, v12");
			asm volatile("vadd.vv v28, v28, v18");
			asm volatile("vadd.vv v26, v26, v22");	
			
			asm volatile("vadd.vv v30, v30, v14");
			asm volatile("vadd.vv v28, v28, v20");
			
			asm volatile("vadd.vv v30, v30, v16");		
		}
			

		

		// MSB
		
		asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");

		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		// Pre-Calc
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
		// v22 = popcount(v22)
		asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
		
		asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
		asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
		asm volatile("vmacc.vx v26,  %0, v22" ::"r"(2));
		
		asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
		asm volatile("vmacc.vx v28,  %0, v20" ::"r"(2));
		
		asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
		
					
		
		
		// LSB
		asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v16, v10, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v18, v8, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v20, v10, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v22, v10, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		

		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		// Pre-Calc
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
		// v22 = popcount(v22)
		asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
		
		asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
		asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
		asm volatile("vmacc.vx v26,  %0, v22" ::"r"(2));
		
		asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
		asm volatile("vmacc.vx v28,  %0, v20" ::"r"(2));
		
		asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
				
				
		// MSB
		
		asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v16, v10, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// Pre-Calc
		
		asm volatile("vand.vx v18, v8, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		asm volatile("vand.vx v20, v10, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel

		asm volatile("vand.vx v22, v10, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
		
		// v12 = popcount(v12)
		asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
		// v14 = popcount(v14)
		asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
		// v16 = popcount(v16)
		asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
		
		// Pre-Calc
		// v18 = popcount(v18)
		asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
		// v20 = popcount(v20)
		asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
		// v22 = popcount(v22)
		asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
				
		if(f_w < F - 1){
			asm volatile("vslidedown.vi v0, v0, 1");
			asm volatile("vslidedown.vi v2, v2, 1");
			asm volatile("vslidedown.vi v4, v4, 1");
			
			asm volatile("vslidedown.vi v6, v6, 1");
			asm volatile("vslidedown.vi v8, v8, 1");
			asm volatile("vslidedown.vi v10, v10, 1");
		}
				
		asm volatile("vmacc.vx v30,  %0, v12" ::"r"(4));
		asm volatile("vmacc.vx v28,  %0, v18" ::"r"(4));
		asm volatile("vmacc.vx v26,  %0, v22" ::"r"(4));
		
		asm volatile("vmacc.vx v30,  %0, v14" ::"r"(4));
		asm volatile("vmacc.vx v28,  %0, v20" ::"r"(4));
		
		asm volatile("vmacc.vx v30,  %0, v16" ::"r"(4));
		
		}
		
		
		for(int channels = 8 ; channels < C_in ; channels += 8){
					
			f_ptr_ += 8 * F * F;
							
			i__ += 8 * H_in * W_in;
					
			bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_2_to_8(i__ + 2 * W_in, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
			
			asm volatile("vmv.v.v v4, v0");
			asm volatile("vmv.v.v v10, v6");
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_2_to_8(i__ + W_in, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
			
			asm volatile("vmv.v.v v2, v0");
			asm volatile("vmv.v.v v8, v6");
			
			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_2_to_8(i__, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
			
			
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
			
				// LSB
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// Pre-Calc
				
				asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

				asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				// Pre-Calc
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");

				asm volatile("vadd.vv v30, v30, v12");
				asm volatile("vadd.vv v28, v28, v18");
				asm volatile("vadd.vv v26, v26, v22");	
				
				asm volatile("vadd.vv v30, v30, v14");
				asm volatile("vadd.vv v28, v28, v20");
				
				asm volatile("vadd.vv v30, v30, v16");							

				

				// MSB
				
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v2, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v4, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// Pre-Calc
				
				asm volatile("vand.vx v18, v2, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v20, v4, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel

				asm volatile("vand.vx v22, v4, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				// Pre-Calc
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
				
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
				asm volatile("vmacc.vx v26,  %0, v22" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v20" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
				
							
				
				
				// LSB
				asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v10, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// Pre-Calc
				
				asm volatile("vand.vx v18, v8, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v20, v10, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel

				asm volatile("vand.vx v22, v10, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				

				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				// Pre-Calc
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
				
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
				asm volatile("vmacc.vx v26,  %0, v22" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v20" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
						
						
				// MSB
				
				asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v8, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v10, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// Pre-Calc
				
				asm volatile("vand.vx v18, v8, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v20, v10, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel

				asm volatile("vand.vx v22, v10, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				// Pre-Calc
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
						
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v2, v2, 1");
					asm volatile("vslidedown.vi v4, v4, 1");
					
					asm volatile("vslidedown.vi v6, v6, 1");
					asm volatile("vslidedown.vi v8, v8, 1");
					asm volatile("vslidedown.vi v10, v10, 1");
				}
						
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(4));
				asm volatile("vmacc.vx v28,  %0, v18" ::"r"(4));
				asm volatile("vmacc.vx v26,  %0, v22" ::"r"(4));
				
				asm volatile("vmacc.vx v30,  %0, v14" ::"r"(4));
				asm volatile("vmacc.vx v28,  %0, v20" ::"r"(4));
				
				asm volatile("vmacc.vx v30,  %0, v16" ::"r"(4));
				
				}
		}
				
		asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
			
		asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
		
		i_ += 3 * W_in;	
		
		// v26 (needs 2 more lines) and v28 (needs one more line) are used to store precalculated values
		
		
		for (int height = 3 ; height < H_in - 2 ; height += 1){
		
			f_ptr_ = f_ptr;
			
			if(C_in > 8)
				bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));

			i__ = i_;
			o_ += ldo;

			#ifdef DEBUG
				start_timer();
			#endif
			bitpack8_vec_2_to_8(i__, H_in * W_in); 
			#ifdef DEBUG
				stop_timer();
				sum_count = get_timer();
				cycle_count += sum_count;
				call_bp ++;
			#endif
				
				
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
				// LSB (activation bit 0)
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
				if(f_w == 0){
					
					asm volatile("vadd.vv v30, v28, v12");
					
					asm volatile("vadd.vv v28, v26, v14");
					
					asm volatile("vmv.v.v v26, v16");
				
				}
				else
				{
					asm volatile("vadd.vv v30, v30, v12");
				
					asm volatile("vadd.vv v28, v28, v14");		
				
					asm volatile("vadd.vv v26, v26, v16");	
				}
				
				
				// LSB (weights bit 0)
				
				asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				

				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
			
				
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
				
				asm volatile("vmacc.vx v28,  %0, v14" ::"r"(2));
				
				asm volatile("vmacc.vx v26,  %0, v16" ::"r"(2));
												
						
						
				// MSB (activation bit 1)
				
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
						
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
				
				asm volatile("vmacc.vx v28,  %0, v14" ::"r"(2));
				
				asm volatile("vmacc.vx v26,  %0, v16" ::"r"(2));
				
						
						
				// MSB (weights bit 1)
				
				asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel

				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");

						
				
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v6, v6, 1");
				}
					
						
				asm volatile("vmacc.vx v30,  %0, v12" ::"r"(4));
				
				asm volatile("vmacc.vx v28,  %0, v14" ::"r"(4));
				
				asm volatile("vmacc.vx v26,  %0, v16" ::"r"(4));
			
			
			}	
				
			for(int channels = 8 ; channels < C_in ; channels += 8){
						
				f_ptr_ += 8 * F * F;
								
				i__ += 8 * H_in * W_in;
						
				bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
				
				asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
				
				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif
				
				for (int f_w = 0 ; f_w < F ; f_w ++) {
			
					// LSB (activation bit 0)
					asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					
					// v12 = popcount(v12)
					asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
					// v14 = popcount(v14)
					asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
					// v16 = popcount(v16)
					asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
					
					asm volatile("vadd.vv v30, v30, v12");
					
					asm volatile("vadd.vv v28, v28, v14");		
					
					asm volatile("vadd.vv v26, v26, v16");	
					
					
					// LSB (weights bit 0)
					
					asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					

					// v12 = popcount(v12)
					asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
					// v14 = popcount(v14)
					asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
					// v16 = popcount(v16)
					asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				
					
					asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
					
					asm volatile("vmacc.vx v28,  %0, v14" ::"r"(2));
					
					asm volatile("vmacc.vx v26,  %0, v16" ::"r"(2));
													
							
							
					// MSB (activation bit 1)
					
					asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					
					// v12 = popcount(v12)
					asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
					// v14 = popcount(v14)
					asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
					// v16 = popcount(v16)
					asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
					
							
					asm volatile("vmacc.vx v30,  %0, v12" ::"r"(2));
					
					asm volatile("vmacc.vx v28,  %0, v14" ::"r"(2));
					
					asm volatile("vmacc.vx v26,  %0, v16" ::"r"(2));
					
							
							
					// MSB (weights bit 1)
					
					asm volatile("vand.vx v12, v6, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w + 9])); // ALL the input multiplied by the 1ST value of 1ST row kernel

					
					// v12 = popcount(v12)
					asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
					// v14 = popcount(v14)
					asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
					// v16 = popcount(v16)
					asm volatile(".byte  0x57, 0x08, 0x08, 0x06");

							
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						asm volatile("vslidedown.vi v6, v6, 1");
					}
						
							
					asm volatile("vmacc.vx v30,  %0, v12" ::"r"(4));
					
					asm volatile("vmacc.vx v28,  %0, v14" ::"r"(4));
					
					asm volatile("vmacc.vx v26,  %0, v16" ::"r"(4));				
				}
			}	
				
			
				
			i_ += W_in;
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));	
			}
			
			
			// last 2 lines (no need for pre calculation)
			
			f_ptr_ = f_ptr;
			
			if(C_in > 8)
				bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
			
			i__ = i_;
			o_ += ldo;

				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif
				
			asm volatile("vmv.v.v v30, v28");
			asm volatile("vmv.v.v v28, v26");
				
				
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
				// LSB (activation bit 0)
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// LSB (weights bit 0)
				asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v18, v6, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// MSB (activation bit 1)
				asm volatile("vand.vx v20, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v22, v0, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				// MSB (weights bit 1)
				asm volatile("vand.vx v24, v6, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				asm volatile("vand.vx v26, v6, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
				// v20 = popcount(v20)
				asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
				// v22 = popcount(v22)
				asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
				// v24 = popcount(v24)
				asm volatile(".byte  0x57, 0x0C, 0x0C, 0x06");
				// v26 = popcount(v26)
				asm volatile(".byte  0x57, 0x0D, 0x0D, 0x06");
				
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v6, v6, 1");
				}
				
				asm volatile("vadd.vv v30, v30, v12");
				asm volatile("vadd.vv v28, v28, v14");
				
				asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v20" ::"r"(2));
				asm volatile("vmacc.vx v28,  %0, v22" ::"r"(2));
				
				asm volatile("vmacc.vx v30,  %0, v24" ::"r"(4));
				asm volatile("vmacc.vx v28,  %0, v26" ::"r"(4));		
			}
			
			for(int channels = 8 ; channels < C_in ; channels += 8){
						
				f_ptr_ += 8 * F * F;
								
				i__ += 8 * H_in * W_in;
						
				bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
				
				asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
				
				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif	
				
				for (int f_w = 0 ; f_w < F ; f_w ++) {
			
					// LSB (activation bit 0)
					asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v14, v0, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					// LSB (weights bit 0)
					asm volatile("vand.vx v16, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v18, v6, %0" :: "r"(f_packed[f_w + 3])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					// MSB (activation bit 1)
					asm volatile("vand.vx v20, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v22, v0, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					// MSB (weights bit 1)
					asm volatile("vand.vx v24, v6, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					asm volatile("vand.vx v26, v6, %0" :: "r"(f_packed[f_w + 12])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					
					// v12 = popcount(v12)
					asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
					// v14 = popcount(v14)
					asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
					// v16 = popcount(v16)
					asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
					// v18 = popcount(v18)
					asm volatile(".byte  0x57, 0x09, 0x09, 0x06");
					// v20 = popcount(v20)
					asm volatile(".byte  0x57, 0x0A, 0x0A, 0x06");
					// v22 = popcount(v22)
					asm volatile(".byte  0x57, 0x0B, 0x0B, 0x06");
					// v24 = popcount(v24)
					asm volatile(".byte  0x57, 0x0C, 0x0C, 0x06");
					// v26 = popcount(v26)
					asm volatile(".byte  0x57, 0x0D, 0x0D, 0x06");
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						asm volatile("vslidedown.vi v6, v6, 1");
					}
					
					asm volatile("vadd.vv v30, v30, v12");
					asm volatile("vadd.vv v28, v28, v14");
					
					asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
					asm volatile("vmacc.vx v28,  %0, v18" ::"r"(2));
					
					asm volatile("vmacc.vx v30,  %0, v20" ::"r"(2));
					asm volatile("vmacc.vx v28,  %0, v22" ::"r"(2));
					
					asm volatile("vmacc.vx v30,  %0, v24" ::"r"(4));
					asm volatile("vmacc.vx v28,  %0, v26" ::"r"(4));
				}
				
			}
			
				
			i_ += W_in;
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
	
			f_ptr_ = f_ptr;
			
			if(C_in > 8)
				bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
						
			i__ = i_;
			o_ += ldo;

				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif
			
			asm volatile("vmv.v.v v30, v28");
				
			for (int f_w = 0 ; f_w < F ; f_w ++) {
			
				// LSB (activation bit 0)
				asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				// LSB (weights bit 0)
				asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				// MSB (activation bit 1)
				asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				// MSB (weights bit 1)				
				asm volatile("vand.vx v18, v6, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
				
				
				// v12 = popcount(v12)
				asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
				// v14 = popcount(v14)
				asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
				// v16 = popcount(v16)
				asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
				// v18 = popcount(v18)
				asm volatile(".byte  0x57, 0x09, 0x09, 0x06"); 
				
				if(f_w < F - 1){
					asm volatile("vslidedown.vi v0, v0, 1");
					asm volatile("vslidedown.vi v6, v6, 1");
				}
				
				asm volatile("vadd.vv v30, v30, v12");;
				asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
				asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
				asm volatile("vmacc.vx v30,  %0, v18" ::"r"(4));	
			
				}	
			
			for(int channels = 8 ; channels < C_in ; channels += 8){
						
				f_ptr_ += 8 * F * F;
								
				i__ += 8 * H_in * W_in;
						
				bitpack_filter8_vec_2_to_8(f_ptr_, f_packed, F*F);
				
				asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen));
				
				#ifdef DEBUG
					start_timer();
				#endif
				bitpack8_vec_2_to_8(i__, H_in * W_in); 
				#ifdef DEBUG
					stop_timer();
					sum_count = get_timer();
					cycle_count += sum_count;
					call_bp ++;
				#endif	
				
				for (int f_w = 0 ; f_w < F ; f_w ++) {
				
					// LSB (activation bit 0)
					asm volatile("vand.vx v12, v0, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					// LSB (weights bit 0)
					asm volatile("vand.vx v14, v6, %0" :: "r"(f_packed[f_w + 6])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					// MSB (activation bit 1)
					asm volatile("vand.vx v16, v0, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					// MSB (weights bit 1)				
					asm volatile("vand.vx v18, v6, %0" :: "r"(f_packed[f_w + 15])); // ALL the input multiplied by the 1ST value of 1ST row kernel
					
					
					// v12 = popcount(v12)
					asm volatile(".byte  0x57, 0x06, 0x06, 0x06");
					// v14 = popcount(v14)
					asm volatile(".byte  0x57, 0x07, 0x07, 0x06");
					// v16 = popcount(v16)
					asm volatile(".byte  0x57, 0x08, 0x08, 0x06");
					// v18 = popcount(v18)
					asm volatile(".byte  0x57, 0x09, 0x09, 0x06"); 
					
					if(f_w < F - 1){
						asm volatile("vslidedown.vi v0, v0, 1");
						asm volatile("vslidedown.vi v6, v6, 1");
					}
					
					asm volatile("vadd.vv v30, v30, v12");;
					asm volatile("vmacc.vx v30,  %0, v14" ::"r"(2));
					asm volatile("vmacc.vx v30,  %0, v16" ::"r"(2));
					asm volatile("vmacc.vx v30,  %0, v18" ::"r"(4));	
				
					}	
				}
			
			
			asm volatile("vsetvli zero, %0, e8, m2, ta, ma" ::"r"(vlen - 2));
				
			asm volatile("vse8.v v30, (%0)" : "+&r"(o_));
			
		
	}
	#ifdef DEBUG
		printf("The execution took %d cycles.\n", cycle_count);
		printf("Bitpack operation was called %d times.\n", call_bp);
	#endif
}
