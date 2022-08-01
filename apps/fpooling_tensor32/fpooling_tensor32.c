#include "fpooling_tensor32.h"
#include <stdio.h>

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


void fmax_pool32(float *o, float *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

		switch (F){
			 case 2:
					fmax_pool_vec_1xC_2x2(o, i, H_in, W_in, C_in, F, F);
				break;

			 case 3:
					fmax_pool_vec_1xC_3x3(o, i, H_in, W_in, C_in, F, F);
				break;
		}


}

void favg_pool32(float *o, float *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F){

		switch (F){
			 case 2:
					favg_pool_vec_1xC_2x2(o, i, H_in, W_in, C_in, F, F);
				break;

			 case 3:
					favg_pool_vec_1xC_3x3(o, i, H_in, W_in, C_in, F, F);
				break;
		}


}



void fmax_pool_vec_1xC_2x2(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride) {

	int64_t ld = stride << 2;
	
	// vsetvli C (in order to load everything)
	asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(TILE_SIZE >> 1)); // we only load one of two values 
	
	for (int c = 0 ; c < C ; c += TILE_SIZE) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
		if(c > C - TILE_SIZE)
			asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"((C % TILE_SIZE) >> 1)); // we only load one of two values 

		float * i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
		float * o_ = o + (c >> 1);			// output pointer relative to the tile	
		
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
		 	asm volatile("vfmax.vv v4,  v20,  v16");
		 	asm volatile("vfmax.vv v8,  v24,  v28");
		 	asm volatile("vfmax.vv v0,  v4,  v8");
		 		
		 	// store
			asm volatile("vse32.v v0, (%0)" : "+&r"(o_));
			o_ += C >> 1;
	  	 
  		}
  	}
}



void fmax_pool_vec_1xC_3x3(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride) {
	
	// see re-use in case of stride < F (overlap of each tile)
	int64_t ld = stride << 3;
	
	// vsetvli C (in order to load everything)
	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_3x3 / 3)); // we only load one of two values 
	
	for (int c = 0 ; c < C ; c += TILE_SIZE_3x3) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
		if(c > C - TILE_SIZE_3x3)
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"((C % TILE_SIZE_3x3) / 3)); // we only load one of two values 

		float * i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
		float * o_ = o + c / 3;			// output pointer relative to the tile	
		
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
		 	asm volatile("vfmax.vv v2,  v12,  v10");
		 	asm volatile("vfmax.vv v4,  v14,  v2");
		 	
		 	asm volatile("vfmax.vv v2,  v16,  v18");
		 	asm volatile("vfmax.vv v6,  v20,  v2");
		 	
		 	asm volatile("vfmax.vv v2,  v22,  v24");
		 	asm volatile("vfmax.vv v8,  v26,  v2");
		 	
		 	asm volatile("vfmax.vv v2,  v4,  v6");
			asm volatile("vfmax.vv v0,  v8,  v2");
		 		
		 	// store
			asm volatile("vse32.v v0, (%0)" : "+&r"(o_));
			o_ += C / 3;
	  	 
  		}
  	}
}


void favg_pool_vec_1xC_2x2(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride) {
	
	int64_t ld = stride << 2;
	
	float avg_div = 1.0/4;
	
	// vsetvli C (in order to load everything)
	asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(TILE_SIZE >> 1)); // we only load one of two values 
	
	for (int c = 0 ; c < C ; c += TILE_SIZE) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
		if(c > C - TILE_SIZE)
			asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"((C % TILE_SIZE) >> 1));

		float *i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
		float *o_ = o + (c >> 1);			// output pointer relative to the tile	
		
		
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
		 	asm volatile("vfadd.vv v0,  v16,  v20");
		 	asm volatile("vfadd.vv v8,  v24,  v28");
		 	
		 	asm volatile("vfadd.vv v8, v0,  v8");
		 	
		 	asm volatile("vfmul.vf v0, v8, %0" : "+&f"(avg_div));
		   
		 	// store
			asm volatile("vse32.v v0, (%0)" : "+&r"(o_));
			
			o_ += C >> 1;
	  	 
  		}
  	}
}


void favg_pool_vec_1xC_3x3(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride) {
	
	int64_t ld = stride << 2;
	
	float avg_div = 1.0/9;
	
	// vsetvli C (in order to load everything)
	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_3x3 / 3)); // we only load one of two values 

	
	for (int c = 0 ; c < C ; c += TILE_SIZE_3x3) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
		if(c > C - TILE_SIZE_3x3)
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"((C % TILE_SIZE_3x3) / 3));

		float *i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
		float *o_ = o + c / 3;			// output pointer relative to the tile	

		
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
		  	
			  	
		  	// first two lines
		 	asm volatile("vfadd.vv v0,  v12,  v14");
		 	asm volatile("vfadd.vv v4,  v18,  v20");
		 	asm volatile("vfadd.vv v8,  v24,  v26");
 			
 			// add all the 9 registers
 			
 			asm volatile("vfadd.vv v0, v0, v16");
 			asm volatile("vfadd.vv v4, v4, v22");
 			asm volatile("vfadd.vv v8, v8, v28");
 			
 			asm volatile("vfadd.vv v0, v4, v0");
 			asm volatile("vfadd.vv v0, v8, v0");
 			
 			
		 	// divide by the sum of values per block which is F*F = 9
		 	asm volatile("vfmul.vf v0, v0, %0" : "+&f"(avg_div));
		 	
		 	// store
			asm volatile("vse32.v v0, (%0)" : "+&r"(o_));
			
			o_ += C / 3;
	  	 
  		}
  	}
}
