#include "ipooling_tensor32.h"
#include <stdio.h>

#ifndef Spike
#include "printf.h"
#endif 

#define block_size_2x2 2
#define block_size_3x3 3


//////////////////////////////////////////////////////////////////////////////////////
//               Description : Functions pooling (average and max for now)          //		
//                                                                                  //
//    1 x C_in x H_in x W_in   &   C_in x F x F   =   1 x C_in x H_in/F x W_in/F    //			
//              input              pooling size               output                //
//                                                                                  //			
//////////////////////////////////////////////////////////////////////////////////////

// *o : tensor convolution output pointer k x C x R (dimmensions depends on the filter size and stride)
// *i : input tensor pointer 1 x W x C x R
// H_in  : number of input Rows
// W_in  : number of input Column
// C_in  : channels of the input tensor
// F  : size of the filter


// Comments :
// Only 2x2 average and max pooling are available right now
// stride has yet to be implemented


void imax_pool32(int32_t *o, int32_t *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

		switch (F){
			 case 2:
					imax_pool_vec_1xC_2x2(o, i, H_in, W_in, C_in, F, F);
				break;

			 case 3:
					imax_pool_vec_1xC_3x3(o, i, H_in, W_in, C_in, F, F);
				break;
		}


}

void iavg_pool32(int32_t *o, int32_t *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

		switch (F){
			 case 2:
					iavg_pool_vec_1xC_2x2(o, i, H_in, W_in, C_in, F, F);
				break;

			 case 3:
					iavg_pool_vec_1xC_3x3(o, i, H_in, W_in, C_in, F, F);
				break;
		}


}



void imax_pool_vec_1xC_2x2(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride) {

	int64_t ld = stride << 2;
	
	// vsetvli C (in order to load everything)
	asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(TILE_SIZE >> 1)); // we only load one of two values 
	
	for (int c = 0 ; c < C ; c += TILE_SIZE) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
		if(c > C - TILE_SIZE)
			asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"((C % TILE_SIZE) >> 1)); // we only load one of two values 

		int32_t * i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
		int32_t * o_ = o + (c >> 1);			// output pointer relative to the tile	
		
		//channel and height loop are fused to avoid branches
		for (int r = 0 ; r < R * W ; r += block_size_2x2){
			
		  	// Load F row of size C
		  	asm volatile("vlse32.v v16, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v20, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 1;
		  	// next line
		  	asm volatile("vlse32.v v24, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v28, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 1;
		 		
		 	// max function between the F rows
		 	asm volatile("vmax.vv v4,  v20,  v16");
		 	asm volatile("vmax.vv v8,  v24,  v28");
		 	asm volatile("vmax.vv v0,  v4,  v8");
		 		
		 	// store
			asm volatile("vse32.v v0, (%0)" : "+&r"(o_));
			o_ += C >> 1;
	  	 
  		}
  	}
}


void imax_pool_vec_1xC_3x3(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride) {
	
	int64_t ld = stride << 2;
	
	// vsetvli C (in order to load everything)
	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_3x3 / 3)); // we only load one of two values 
	
	for (int c = 0 ; c < C ; c += TILE_SIZE_3x3) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
		if(c > C - TILE_SIZE_3x3)
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"((C % TILE_SIZE_3x3) / 3)); // we only load one of two values 

		int32_t * i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
		int32_t * o_ = o + c / 3;			// output pointer relative to the tile	
		
		//channel and height loop are fused to avoid branches
		for (int r = 0 ; r < R * W ; r += block_size_3x3){
			
		  	// Load F row of size C
		  	asm volatile("vlse32.v v10, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v12, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v14, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 2;
		  	
		  	// next line
		  	asm volatile("vlse32.v v16, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v18, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v20, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 2;
		  	
		  	// next line
		  	asm volatile("vlse32.v v22, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v24, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v26, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 2;
		 		
		 	// max function between the F rows
		 	asm volatile("vmax.vv v2,  v12,  v10");
		 	asm volatile("vmax.vv v4,  v14,  v2");
		 	
		 	asm volatile("vmax.vv v2,  v16,  v18");
		 	asm volatile("vmax.vv v6,  v20,  v2");
		 	
		 	asm volatile("vmax.vv v2,  v22,  v24");
		 	asm volatile("vmax.vv v8,  v26,  v2");
		 	
		 	asm volatile("vmax.vv v2,  v4,  v6");
			asm volatile("vmax.vv v0,  v8,  v2");
		 		
		 	// store
			asm volatile("vse32.v v0, (%0)" : "+&r"(o_));
			o_ += C / 3;
	  	 
  		}
  	}
}


void iavg_pool_vec_1xC_2x2(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride) {
	
	// see re-use in case of stride < F (overlap of each tile)
	
	int64_t vlen;
	
	int64_t ld = stride << 2;
	
	int32_t *i_;
	int32_t *o_;
	
	int avg_div = 4;
	
	for (int c = 0 ; c < C ; c += TILE_SIZE) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
		if(c > C - TILE_SIZE)
			vlen = C % TILE_SIZE >> 1;
		else
			vlen = TILE_SIZE >> 1;

		i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
		o_ = o + (c >> 1);			// output pointer relative to the tile	
		
		// vsetvli C (in order to load everything)
		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen)); // we only load one of two values 
		
		
		//channel and height loop are fused to avoid branches
		for (int r = 0 ; r < R * W ; r += block_size_2x2){
			
		  	// Load F row of size C
		  	asm volatile("vlse32.v v16, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v20, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 1;
		  	
		  	// next line
		  	asm volatile("vlse32.v v24, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v28, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 1;
		  	
		  	// Add vectors on 16b 
		 		
		 	// max function between the F rows
		 	asm volatile("vwadd.vv v0,  v16,  v20");
		 	asm volatile("vwadd.vv v8,  v24,  v28");
		 	
		 	asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(vlen)); // we only load one of two values 
		 	
		 	asm volatile("vadd.vv v8, v0,  v8");
		 	
		 	asm volatile("vdiv.vx v8, v8, %0" : "+&r"(avg_div));
		 	
		 	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen)); // we only load one of two values 
		 	
		 	asm volatile("vnsra.wi v0, v8, 0"); // shift 8 to return 8b vector and divide EEW by 2
		   
		 	// store
			asm volatile("vse32.v v0, (%0)" : "+&r"(o_));
			o_ += C >> 1;
			
  		}
  	}
}


void iavg_pool_vec_1xC_3x3(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride) {

	
	int64_t ld = stride << 2;
	
	int64_t vlen;
	
	int32_t *i_;
	int32_t *o_;
	
	int avg_div = 9;
	
	for (int c = 0 ; c < C ; c += TILE_SIZE_3x3) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
		if(c > C - TILE_SIZE_3x3)
			vlen = C % TILE_SIZE_3x3 / 3;
		else
			vlen = TILE_SIZE_3x3 / 3;


		i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
		o_ = o + c / 3;			// output pointer relative to the tile	
		
		// vsetvli C (in order to load everything)
		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen)); // we only load one of two values 
		
		
		//channel and height loop are fused to avoid branches
		for (int r = 0 ; r < R * W ; r += block_size_3x3){
			
		  	// Load F row of size C
		  	asm volatile("vlse32.v v12, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v14, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v16, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 2;
		  	
		  	// next line
		  	asm volatile("vlse32.v v18, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v20, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v22, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 2;
		  	
		  	// next line
		  	asm volatile("vlse32.v v24, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v26, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += 1;
		  	asm volatile("vlse32.v v28, (%0), %1" : "+&r"(i_) : "r"(ld));
		  	i_ += C - 2;
		  	
		  	// Add vectors on 16b 
			  	
		  	// first two lines
		 	asm volatile("vwadd.vv v0,  v12,  v14");
		 	asm volatile("vwadd.vv v4,  v18,  v20");
		 	asm volatile("vwadd.vv v8,  v24,  v26");
		 	
		 	// add last line of kernel
		 	asm volatile("vwadd.vx v12, v16,  zero"); // we could use sign extend to do the exact same
		 	asm volatile("vwadd.vx v16, v22,  zero");
		 	asm volatile("vwadd.vx v20, v28,  zero");
 			
 			asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(vlen)); // all our results are on 16b
 			// add all the 9 registers
 			
 			asm volatile("vadd.vv v0, v4,  v0");
 			asm volatile("vadd.vv v4, v8,  v12");
 			asm volatile("vadd.vv v8, v16, v20");
 			asm volatile("vadd.vv v0, v4, v0");
 			asm volatile("vadd.vv v0, v8, v0");
 			
 			
		 	// divide by the sum of values per block which is F*F = 9
		 	asm volatile("vdiv.vx v4, v0, %0" : "+&r"(avg_div));
		 	
		 	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vlen));
		 	
		 	// narrowing arithmetic right shift by 0 (which just cut the upper byte to go from 16b to 8b)
		 	asm volatile("vnsra.wi v0, v4, 0"); 
		   
		 	// store
			asm volatile("vse32.v v0, (%0)" : "+&r"(o_));
			o_ += C / 3;
			
  		}
  	}
}


