#include "iconv2d_tensor16.h"
#include <stdio.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 							Description : Functions for cross-correlation between  											//		
// 																																				//
//								1 x W x C x R   *   K x W x F x F   =    K x C x R													//			
//									input						kernels				output													//
//									(16b)						(16b)					(32b)														//	
//																																					//
//											tiled to 256 width input (see header)														//
//																																					//				
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// *o : tensor convolution output pointer k x C x R
// *i : input tensor pointer 1 x W x C x R
// *f : kernel/filter tensor pointer
// R  : number of input Rows
// C  : number of input Column
// W  : Depth of the input tensor
// F  : size of the kernel/filter 
// K  : number of kernel/filter to convolve with the input tensor

void iconv2d_tensor_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {
	
	
  int16_t *i_;
  int32_t *o_;
  int16_t *f_;
  
  
  
  //helper variable
  
  for(int64_t k = 0; k < K; k++) {
  		// First iteration round, k = 0 for the adress of the first value of the first filter
		o_ = o + k * R * C;		// Output is incremented 
		i_ = i;						// Since we aren't working on batch, we only consider one input
		f_ = f + k * F * F * W;

	  // Iterate over the output rows
	  switch (F)
	  {
	   case 1:
	  		iconv2d_tensor16_vec_8xC_1x1_wide(o_, i_, f_, R, C, W, F);
	  		break;
		case 3:
			iconv2d_tensor16_vec_8xC_3x3_wide(o_, i_, f_, R, C, W, F);
			break;
		case 5:
			iconv2d_tensor16_vec_4xC_5x5_wide(o_, i_, f_, R, C, W, F);
			break;
		case 7:
			iconv2d_tensor16_vec_2xC_7x7_wide(o_, i_, f_, R, C, W, F);
			break;
		}
			
	}
}


//////////////////////////////////////////////////////////////////////////
//																								//
//											1x1 kernel										//
//							INPUT/OUTPUT MUST BE DIVISIBLE BY 8						//
//////////////////////////////////////////////////////////////////////////

// Calculate 2 output matrix rows
void iconv2d_tensor16_vec_8xC_1x1_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F) {

  // Temporary variables
  int16_t t;
  
  // Helper variables
  int64_t const ldo = C << 2; 			// increment adress on output store
  int64_t const ldi = C << 1;					// increment adress on input  loads 
  int64_t const next_kernel = 1 << 1;			// increment adress on filter loads
  int64_t const size = C % TILE_SIZE; 	// check for tiling

  int16_t *f_ = f;
  int16_t *i_ = i;
  int32_t *o_ = o;

  // Compute on C elements
  
  for (int c = 0 ; c <= C - TILE_SIZE  ; c += TILE_SIZE){ // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
  	i_ = i + c; // increase the input pointer on the width from tile n to tile n+1 
  	o_ = o + c; // same for output
  
  
	  for (int r = 0; r < R ; r += 8){
	  
			  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));
			  
			  int16_t * i__ = i_ + r * C;
			  f_ = f;
			  
			  
			  // Channel 0
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(next_kernel));  
			  
			  asm volatile("vle16.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v17, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v19, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
			  
			  // MUL operation is used on channel 0 to reset the registers values
			  // On channel > 0, we use multiplication-accumulate
			  // the loop on the channel is unrolled to avoid a conditionnal branch (if ch == 0) at each iteration
				  
			  asm volatile("vwmul.vx v0,  v16, %0" ::"r"(t));
			  asm volatile("vwmul.vx v2,  v17, %0" ::"r"(t));
			  asm volatile("vwmul.vx v4,  v18, %0" ::"r"(t)); 
			  asm volatile("vwmul.vx v6,  v19, %0" ::"r"(t));
			  asm volatile("vwmul.vx v8,  v20, %0" ::"r"(t));
			  asm volatile("vwmul.vx v10, v21, %0" ::"r"(t));
			  asm volatile("vwmul.vx v12, v22, %0" ::"r"(t));
			  asm volatile("vwmul.vx v14, v23, %0" ::"r"(t));
	  
	  		
	  			// From channel 1 -> W
	  		
		  	  for (int ch = 1; ch < W ; ch++){
		  	  	  
		  	  	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(next_kernel));  
				  
				  asm volatile("vle16.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v17, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v19, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
					  
				  asm volatile("vwmacc.vx v0,  %0, v16" ::"r"(t));
				  asm volatile("vwmacc.vx v2,  %0, v17" ::"r"(t));
				  asm volatile("vwmacc.vx v4,  %0, v18" ::"r"(t));
				  asm volatile("vwmacc.vx v6,  %0, v19" ::"r"(t));
				  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t));
				  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t));
				  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t));
				  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t));
				  
		}
			// STORE THE CALCULATED REGISTERS OUTPUT
		  
		  	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE)); 
			
			asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v12, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v14, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	  
	  }
  }
  
  // LAST GROUP (in case C % TILE_SIZE != 0)
  
  if(size != 0) {
  		
  	  o_ = o + C - size; // pointer to fill the gaps that weren't calculated at tiling step. if no tiling (C < TILE_SIZE) then o_ = o
  	  i_ = i + C - size; // pointer to the input that weren't used in the tiling step.
  
	  for (int r = 0; r < R ; r += 8){
		  
			  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size));
			  
			  int16_t * i__ = i_ + r * C; // increment the input 
			  f_ = f;
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(next_kernel));  
			  
			  asm volatile("vle16.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v17, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v19, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
					  
			  asm volatile("vwmul.vx v0,  v16, %0" ::"r"(t));
			  asm volatile("vwmul.vx v2,  v17, %0" ::"r"(t));
			  asm volatile("vwmul.vx v4,  v18, %0" ::"r"(t)); 
			  asm volatile("vwmul.vx v6,  v19, %0" ::"r"(t));
			  asm volatile("vwmul.vx v8,  v20, %0" ::"r"(t));
			  asm volatile("vwmul.vx v10, v21, %0" ::"r"(t));
			  asm volatile("vwmul.vx v12, v22, %0" ::"r"(t));
			  asm volatile("vwmul.vx v14, v23, %0" ::"r"(t));
	 
	  		
		  	  for (int ch = 1; ch < W ; ch++){
		  	  	  
		  	  	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(next_kernel));  
				  
				  asm volatile("vle16.v v16, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v17, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v18, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v19, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
				  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));	
					  
				  asm volatile("vwmacc.vx v0,  %0, v16" ::"r"(t));
				  asm volatile("vwmacc.vx v2,  %0, v17" ::"r"(t));
				  asm volatile("vwmacc.vx v4,  %0, v18" ::"r"(t));
				  asm volatile("vwmacc.vx v6,  %0, v19" ::"r"(t));
				  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t));
				  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t));
				  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t));
				  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t));
	 
				  
			}
			
			// STORE THE CALCULATED REGISTERS OUTPUT
		  
		  	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
			
			asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v12, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
			asm volatile("vse32.v  v14, (%0); add %0, %0, %1" : "+&r"(o_) : "r"(ldo));
	  
	  }
	 
	}
}

//////////////////////////////////////////////////////////////////////////
//																								//
//											3x3 kernel										//	
//							OUPUT SIZE MUST BE DIVISIBLE BY 8 						//
//////////////////////////////////////////////////////////////////////////

// Calculate 2 output matrix rows
void iconv2d_tensor16_vec_8xC_3x3_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F) {

  // Temporary variables
  int16_t t0, t1, t2;

  // Helper variables
  int64_t const ldo = C << 2;
  int64_t const ldi = (C + F - 1) << 1;
  int64_t const ldf = F << 1;
  int64_t next_column = (F * (F - 1) - 1) << 1;
  int64_t const next_kernel = 1 << 1;
  
  int64_t r_loads;
  
  int64_t const size = C % TILE_SIZE_OUT; 	// check for tiling

  int16_t *f_ = f;
  int16_t *i_ = i;
  int32_t *o_ = o; 
  
  for (int c = 0 ; c <= (C - TILE_SIZE_OUT) ; c += TILE_SIZE_OUT){ // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)

	  i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
	  int16_t * i__ = i_;	// inside tile pointer (change at each load)
	  
	  o_ = o + c;			// output pointer relative to the tile
	  int32_t * o__ = o_;// inside tile output pointer 
  	  	
  	  f_ = f;				// filter pointer 
  	  	
	  // FIRST 8 output rows (from first plane)
	  // Loops are unrolled to avoid branching on conditions (ex: use of vmul on first plane and vmacc on others)
	  
	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));
	  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
	  
	  r_loads = 8; // 8 loads have been done
	  
	  //Fetch first column of filter
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));   
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));  
	  
	  asm volatile("vwmul.vx v0,  v20, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v2,  v21, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v4,  v22, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v6,  v23, %0" ::"r"(t0)); 
	  asm volatile("vwmul.vx v8,  v24, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v10, v25, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v12, v26, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v14, v27, %0" ::"r"(t0));

	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
		  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
		  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	 
	  
		 //we go from 2nd plane to others
	  for (int ch = 1; ch < W ; ch++){ 
		  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));

		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column)); 
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel)); 
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
	  }
	  
		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_OUT)); 
		
		asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		
		asm volatile("vmv.v.v v0, v12");
		asm volatile("vmv.v.v v2, v14");
	  
	  for (int r = 8; r < (R + F - 1 - block_size_3x3) ; r += block_size_3x3){
	  // this part does all the block between the first 8 output rows and last 8 output rows
	  	
	  	   asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));
	  	
	  		i__ = i_ + r * ( C + F - 1 );
	  		f_ = f;
	  		o__ = o_ + (r - F + 1) * C;

	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));  
		  
		  asm volatile("vwmul.vx v4,  v20, %0" ::"r"(t0)); 
		  asm volatile("vwmul.vx v6,  v21, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v8,  v22, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v10, v23, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v12, v24, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v14, v25, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v16, v26, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v18, v27, %0" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
		  
			 //we go from 2nd plane to others
		  for (int ch = 1; ch < W ; ch++){ 
		  
			  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
			  
			  //Fetch first column of filter
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
			  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
			  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
				  
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v24, v24, 1");
			  asm volatile("vslidedown.vi v25, v25, 1");
			  asm volatile("vslidedown.vi v26, v26, 1");
			  asm volatile("vslidedown.vi v27, v27, 1");

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
			  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
			  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
			  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
				  
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v24, v24, 1");
			  asm volatile("vslidedown.vi v25, v25, 1");
			  asm volatile("vslidedown.vi v26, v26, 1");
			  asm volatile("vslidedown.vi v27, v27, 1");

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
			  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
			  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
			  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));

			  
		  }
		  
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_OUT)); 
			
			asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v12, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v14, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			
			r_loads += block_size_3x3;
			
			asm volatile("vmv.v.v v0, v16");
			asm volatile("vmv.v.v v2, v18");
		}
	  
	  
	  // LAST 8 OUPTUT
	  		
			i__ = i_ + r_loads * ( C + F - 1 ); //R + F - 1 - block_size_o  here it's 8
	  		f_ = f;
	  		o__ = o_ + (r_loads - F + 1) * C;

	  	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));
	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));  
		  
		  asm volatile("vwmul.vx v4,  v20, %0" ::"r"(t0)); 
		  asm volatile("vwmul.vx v6,  v21, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v8,  v22, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v10, v23, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v12, v24, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v14, v25, %0" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
		  
			 //we go from 2nd plane to others
		  for (int ch = 1; ch < W ; ch++){ 
		  
			  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
			  
			  //Fetch first column of filter
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
				  
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v24, v24, 1");
			  asm volatile("vslidedown.vi v25, v25, 1");
			  asm volatile("vslidedown.vi v26, v26, 1");
			  asm volatile("vslidedown.vi v27, v27, 1");

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
			  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
				  
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v24, v24, 1");
			  asm volatile("vslidedown.vi v25, v25, 1");
			  asm volatile("vslidedown.vi v26, v26, 1");
			  asm volatile("vslidedown.vi v27, v27, 1");

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
			  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));

			  
		  }
		  
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_OUT)); 
			
			asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v12, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v14, (%0)" : "+&r"(o__));
	}
	
	// LAST ITERATION OF TILING (OR ONLY ITERATION IF C < TILE_SIZE)
	
	if(size != 0) {
  	  
	  o_ = o + C - size; // pointer to fill the gaps that weren't calculated at tiling step. if no tiling (C < TILE_SIZE) then o_ = o
  	  i_ = i + C - size; // pointer to the input that weren't used in the tiling step.
	  f_ = f;
	  
	  int32_t * o__ = o_; // pointer to fill the gaps that weren't calculated at tiling step. if no tiling (C < TILE_SIZE) then o_ = o
  	  int16_t *  i__ = i_;
	  
	  // FIRST 8 output rows (from first plane)
	  // Loops are unrolled to avoid branching on conditions (ex: use of vmul on first plane and vmacc on others)
	  
	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size + F - 1));
	  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
	  
	  r_loads = 8;
	  
	  //Fetch first column of filter
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));  
	  
	  asm volatile("vwmul.vx v0,  v20, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v2,  v21, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v4,  v22, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v6,  v23, %0" ::"r"(t0)); 
	  asm volatile("vwmul.vx v8,  v24, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v10, v25, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v12, v26, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v14, v27, %0" ::"r"(t0));

	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
		  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
		  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	 
	  
		 //we go from 2nd plane to others
	  for (int ch = 1; ch < W ; ch++){ 
		  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));

		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column)); 
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel)); 
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
	  }
	  
		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
		
		asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		
		asm volatile("vmv.v.v v0, v12");
		asm volatile("vmv.v.v v2, v14");
	  
	  for (int r = 8; r < (R + F - 1 - block_size_3x3) ; r += block_size_3x3){
	  // this part does all the block between the first 8 output rows and last 8 output rows
	  	
	  		i__ = i_ + r * ( C + F - 1 );
	  		f_ = f;
	  		o__ = o_ + (r - F + 1) * C;
	  		
	  	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size + F - 1));
	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));  
		  
		  asm volatile("vwmul.vx v4,  v20, %0" ::"r"(t0)); 
		  asm volatile("vwmul.vx v6,  v21, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v8,  v22, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v10, v23, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v12, v24, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v14, v25, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v16, v26, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v18, v27, %0" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
		  
			 //we go from 2nd plane to others
		  for (int ch = 1; ch < W ; ch++){ 
		  
			  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
			  
			  //Fetch first column of filter
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
			  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
			  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
				  
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v24, v24, 1");
			  asm volatile("vslidedown.vi v25, v25, 1");
			  asm volatile("vslidedown.vi v26, v26, 1");
			  asm volatile("vslidedown.vi v27, v27, 1");

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
			  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
			  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
			  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
				  
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v24, v24, 1");
			  asm volatile("vslidedown.vi v25, v25, 1");
			  asm volatile("vslidedown.vi v26, v26, 1");
			  asm volatile("vslidedown.vi v27, v27, 1");

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
			  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  asm volatile("vwmacc.vx v16, %0, v26" ::"r"(t0));
			  asm volatile("vwmacc.vx v18, %0, v27" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));
			  asm volatile("vwmacc.vx v16, %0, v27" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));

			  
		  }
		  
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
			
			asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v12, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v14, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			
			r_loads += block_size_3x3;
			
			asm volatile("vmv.v.v v0, v16");
			asm volatile("vmv.v.v v2, v18");
		}
	  
	  
	  // LAST 8 OUPTUT
	  		
			i__ = i_ + r_loads * ( C + F - 1 ); //R + F - 1 - block_size_o  here it's 8
	  		f_ = f;
	  		o__ = o_ + (r_loads - F + 1) * C;

	  	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size + F - 1));
	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));  
		  
		  asm volatile("vwmul.vx v4,  v20, %0" ::"r"(t0)); 
		  asm volatile("vwmul.vx v6,  v21, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v8,  v22, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v10, v23, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v12, v24, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v14, v25, %0" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
			  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
		  
			 //we go from 2nd plane to others
		  for (int ch = 1; ch < W ; ch++){ 
		  
			  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
			  
			  //Fetch first column of filter
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
		  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
				  
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v24, v24, 1");
			  asm volatile("vslidedown.vi v25, v25, 1");
			  asm volatile("vslidedown.vi v26, v26, 1");
			  asm volatile("vslidedown.vi v27, v27, 1");

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
			  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
				  
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v24, v24, 1");
			  asm volatile("vslidedown.vi v25, v25, 1");
			  asm volatile("vslidedown.vi v26, v26, 1");
			  asm volatile("vslidedown.vi v27, v27, 1");

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf)); 
			  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t0));
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t0));
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t0));
			  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
			  
			  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t1));
			  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t1));
			  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t1));
			  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t1));
			  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t1));

			  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t2));
			  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t2));
			  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t2));
			  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t2));
			  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t2));
			  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t2));

			  
		  }
		  
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
			
			asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v12, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v14, (%0)" : "+&r"(o__));
	
	
	}

}



//////////////////////////////////////////////////////////////////////////
//																								//
//											5x5 kernel										//
//							OUPUT SIZE MUST BE DIVISIBLE BY 4 						//
//////////////////////////////////////////////////////////////////////////

// Calculate 4 output matrix rows at each iteration
void iconv2d_tensor16_vec_4xC_5x5_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F) {

  // Temporary variables
  int16_t t0, t1, t2, t3, t4;

  // Helper variables
  int64_t const ldo = C << 2;
  int64_t const ldi = (C + F - 1) << 1;
  int64_t const ldf = F << 1;
  int64_t next_column = (F * (F - 1) - 1) << 1;
  int64_t next_kernel = 1 << 1;

  int64_t r_loads;
  
  int64_t const size = C % TILE_SIZE_OUT; 	// check for tiling

  int16_t *f_ = f;
  int16_t *i_ = i;
  int32_t *o_ = o; 
  
  for (int c = 0 ; c <= (C - TILE_SIZE_OUT) ; c += TILE_SIZE_OUT){ // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)

	  i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
	  int16_t * i__ = i_;	// inside tile pointer (change at each load)
	  
	  o_ = o + c;			// output pointer relative to the tile
	  int32_t * o__ = o_;// inside tile output pointer 
  	  	
  	  f_ = f;				// filter pointer 
  	  
  // We load the 8b values on 16b register to go to a 32b result easier
  
  // FIRST 4 output rows (from first plane)
  // Loops are unrolled to avoid branching on conditions (ex: use of vmul on first plane and vmacc on others)
  
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));
  
  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
  
  r_loads = 8; // 8 rows have been loaded
  
  //Fetch first column of filter
  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
  
  asm volatile("vwmul.vx v0,  v20, %0" ::"r"(t0));
  asm volatile("vwmul.vx v2,  v21, %0" ::"r"(t0));
  asm volatile("vwmul.vx v4,  v22, %0" ::"r"(t0));
  asm volatile("vwmul.vx v6,  v23, %0" ::"r"(t0));
  asm volatile("vwmul.vx v8,  v24, %0" ::"r"(t0));
  asm volatile("vwmul.vx v10, v25, %0" ::"r"(t0));
  asm volatile("vwmul.vx v12, v26, %0" ::"r"(t0));
  asm volatile("vwmul.vx v14, v27, %0" ::"r"(t0)); 
  
  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
  
  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
  
  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
	 
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vslidedown.vi v21, v21, 1");
  asm volatile("vslidedown.vi v22, v22, 1");
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vslidedown.vi v24, v24, 1");
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vslidedown.vi v27, v27, 1");
  
  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
  
  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
  
  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
  
  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
	 
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vslidedown.vi v21, v21, 1");
  asm volatile("vslidedown.vi v22, v22, 1");
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vslidedown.vi v24, v24, 1");
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vslidedown.vi v27, v27, 1");
  
  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
  
  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
  
  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
  
  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
	 
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vslidedown.vi v21, v21, 1");
  asm volatile("vslidedown.vi v22, v22, 1");
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vslidedown.vi v24, v24, 1");
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vslidedown.vi v27, v27, 1");
  
  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
  
  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
  
  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
  
  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel));
	 
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vslidedown.vi v21, v21, 1");
  asm volatile("vslidedown.vi v22, v22, 1");
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vslidedown.vi v24, v24, 1");
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vslidedown.vi v27, v27, 1");
  
  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
  
  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
  
  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
  
  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));

  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
  
  
  
    //we go from 2nd plane to others
  for (int ch = 1; ch < W ; ch++){ 
  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
	  //Fetch first column of filter
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		 
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		 
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		 
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");

	  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel));
		 
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));

	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  

  }
  
	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_OUT)); 
	
	asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	
	asm volatile("vmv.v.v v0, v8");
	asm volatile("vmv.v.v v2, v10");
	asm volatile("vmv.v.v v4, v12");
	asm volatile("vmv.v.v v6, v14");
  
  for (int r = 8; r < (R + F - 1 - block_size_5x5) ; r += block_size_5x5){ //only 8 product input right now
  
  //for (int r = 10; r < (R + F - 9 - 6) ; r += 8){
  // - 1 - 8 = - 9 -> correspond to the 8 last group of 8 values that can be calculated in the same block
  // the last block will calculate 6, 4 or 2 output rows (+ the last block of 8)
  // this way, any input matrix can be any size (with even values only)
  
  // this part does all the block between the first 4 output rows and last 4 output rows
  	
	  	i__ = i_ + r * ( C + F - 1 );
	  	f_ = f;
	  	o__ = o_ + (r - F + 1) * C;
  		
  		
  	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));
	  
	  // For now
	  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));	   
	  
	  //Fetch first column of filter
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  
	  asm volatile("vwmul.vx v8,  v20, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v10, v21, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v12, v22, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v14, v23, %0" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
	  // SECOND COLUMN
	  
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");

	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
	  // THIRD COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");

	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
	  //FOURTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");

	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
	  // FIFTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");

	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
  
		 //we go from 2nd plane to others
	  for (int ch = 1; ch < W ; ch++){ 
		  
		  // For now
	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));	  
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vslidedown.vi v22, v22, 1"); 
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  asm volatile("vslidedown.vi v23, v23, 1");
		  
		  // SECOND COLUMN

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vslidedown.vi v22, v22, 1"); 
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  asm volatile("vslidedown.vi v23, v23, 1");
		  
		  // THIRD COLUMN

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vslidedown.vi v22, v22, 1"); 
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  asm volatile("vslidedown.vi v23, v23, 1");
		  
		  //FOURTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vslidedown.vi v22, v22, 1"); 
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  asm volatile("vslidedown.vi v23, v23, 1");
		  
		  // FIFTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
		 
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  
	  }
	  
		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_OUT)); 
		
		asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		
		r_loads += block_size_5x5;
		
		asm volatile("vmv.v.v v0, v8");
		asm volatile("vmv.v.v v2, v10");
		asm volatile("vmv.v.v v4, v12");
		asm volatile("vmv.v.v v6, v14");
  		

	}
	
	// LAST 4 output rows (no need to precalculate the value of the next block)
	
		i__ = i_ + r_loads * ( C + F - 1 ); 
		f_ = f + F; 								// we don't use the first row of filter for the last block
		o__ = o_ + (r_loads - F + 1) * C;
  		
  		int64_t next_column_jump_1st_row = (3 * F - 1) << 1;
  		int64_t next_kernel_jump_1st_row = (F + 1) << 1;
  		
  	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));
  	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row)); 
	  
	  // For now
	  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));	    
	  
	  //Fetch first column of filter
	  
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
	  
  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));   	  	  
	 

	  // SECOND COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
	  
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
  	  asm volatile("vslidedown.vi v20, v20, 1");
	  
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
	  
  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
	  
	  // THIRD COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
	  
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
  	  asm volatile("vslidedown.vi v20, v20, 1");
	  
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
	  
  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
	  
	  // FOURTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
	  
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
  	  asm volatile("vslidedown.vi v20, v20, 1");
	  
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
	  
  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
	  
	  // FIFTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel_jump_1st_row));
	  
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
  	  asm volatile("vslidedown.vi v20, v20, 1");
	  
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
	  
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
	  
  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
	  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
  
		 //we go from 2nd plane to others
	  for (int ch = 1; ch < W ; ch++){ 
		  
			  // For now
		   
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row)); 
		  
		  // For now

		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));	    
		  
		  //Fetch first column of filter
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));   	  	  
		 

		  // SECOND COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
		  
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
	  	  asm volatile("vslidedown.vi v20, v20, 1");
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
		  
		  // THIRD COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
		  
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
	  	  asm volatile("vslidedown.vi v20, v20, 1");
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
		  
		  // FOURTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
		  
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
	  	  asm volatile("vslidedown.vi v20, v20, 1");
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
		  
		  // FIFTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel_jump_1st_row));
		  
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
	  	  asm volatile("vslidedown.vi v20, v20, 1");
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));
		  
		  
	  }
	  
		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_OUT)); 
		
		asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v6,  (%0)" : "+&r"(o__));
		
		}
		
		
			// LAST ITERATION OF TILING (OR ONLY ITERATION IF C < TILE_SIZE)
	
	if(size != 0) {
  	  
	  o_ = o + C - size; // pointer to fill the gaps that weren't calculated at tiling step. if no tiling (C < TILE_SIZE) then o_ = o
  	  i_ = i + C - size; // pointer to the input that weren't used in the tiling step.
	  f_ = f;
	  
	  int32_t * o__ = o_; // pointer to fill the gaps that weren't calculated at tiling step. if no tiling (C < TILE_SIZE) then o_ = o
  	  int16_t *  i__ = i_;
  	  
	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size + F - 1));
	  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
	  
	  r_loads = 8; // 8 rows have been loaded
	  
	  //Fetch first column of filter
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));

	  
	  asm volatile("vwmul.vx v0,  v20, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v2,  v21, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v4,  v22, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v6,  v23, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v8,  v24, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v10, v25, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v12, v26, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v14, v27, %0" ::"r"(t0)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		 
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		 
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		 
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel));
		 
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");
	  
	  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));

	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  
	  
		 //we go from 2nd plane to others
	  for (int ch = 1; ch < W ; ch++){ 
	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		  
		  
		  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
		  
		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
		  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
		  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
		  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
		  
		  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
		  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
			 
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");
		  
		  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
		  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
		  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
		  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
		  
		  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
		  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
			 
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");
		  
		  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
		  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
		  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
		  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
		  
		  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
		  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
			 
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");

		  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
		  
		  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
		  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
		  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
		  
		  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
		  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel));
			 
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vslidedown.vi v25, v25, 1");
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vslidedown.vi v27, v27, 1");
		  
		  asm volatile("vwmacc.vx v0,   %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,   %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,   %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,   %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,   %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10,  %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12,  %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14,  %0, v27" ::"r"(t0));
		  
		  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
		  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
		  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));

		  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
		  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
		  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));

		  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
		  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
		  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
		  
		  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
		  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
		  

	  }
	  
		asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
		
		asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		
		asm volatile("vmv.v.v v0, v8");
		asm volatile("vmv.v.v v2, v10");
		asm volatile("vmv.v.v v4, v12");
		asm volatile("vmv.v.v v6, v14");
	  
	  for (int r = 8; r < (R + F - 1 - block_size_5x5) ; r += block_size_5x5){ //only 8 product input right now
	  
	  //for (int r = 10; r < (R + F - 9 - 6) ; r += 8){
	  // - 1 - 8 = - 9 -> correspond to the 8 last group of 8 values that can be calculated in the same block
	  // the last block will calculate 6, 4 or 2 output rows (+ the last block of 8)
	  // this way, any input matrix can be any size (with even values only)
	  
	  // this part does all the block between the first 4 output rows and last 4 output rows
	  	
		  	i__ = i_ + r * ( C + F - 1 );
		  	f_ = f;
		  	o__ = o_ + (r - F + 1) * C;
	  		
	  		
	  	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size + F - 1));
		  
		  // For now
		  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));	   
		  
		  //Fetch first column of filter  

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));   
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  

		  asm volatile("vwmul.vx v8,  v20, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v10, v21, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v12, v22, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v14, v23, %0" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  // SECOND COLUMN
		  
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");


		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 

		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  // THIRD COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 

		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  //FOURTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));

		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  // FIFTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1");

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  

	  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
	  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4)); 
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
	  
			 //we go from 2nd plane to others
		  for (int ch = 1; ch < W ; ch++){ 
			  
			  // For now
		  
			  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));	   
			  
			  //Fetch first column of filter

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  
			  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  
			  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));

		  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
			  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
			  
			  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vslidedown.vi v22, v22, 1"); 
			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
			  asm volatile("vslidedown.vi v23, v23, 1");
			  
			  // SECOND COLUMN

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  
			  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 

		  
			  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
		  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
			  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
			  
			  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vslidedown.vi v22, v22, 1"); 
			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));

			  asm volatile("vslidedown.vi v23, v23, 1");
			  
			  // THIRD COLUMN

			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  
			  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  
			  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
		  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));

			  
			  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 

			  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
			  
			  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vslidedown.vi v22, v22, 1"); 
			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
			  asm volatile("vslidedown.vi v23, v23, 1");
			  
			  //FOURTH COLUMN
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  
			  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  

			  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
		  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column));
			  
			  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
			  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
			  
			  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
			  asm volatile("vslidedown.vi v20, v20, 1");
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
			  asm volatile("vslidedown.vi v21, v21, 1");
			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vslidedown.vi v22, v22, 1"); 
			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
			  asm volatile("vslidedown.vi v23, v23, 1");

			  
			  // FIFTH COLUMN
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  
			  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t0));
			  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t0)); 
			  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t0));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  
			  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t1));
			  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t1)); 
			  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t1));
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
			  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t2));
			  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t2));
		  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel));
			  
			  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
			  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t3));
			 
			  
			  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4)); 
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));

			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
			  
			  
		  }
		  
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
			
			asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			
			r_loads += block_size_5x5;
			
			asm volatile("vmv.v.v v0, v8");
			asm volatile("vmv.v.v v2, v10");
			asm volatile("vmv.v.v v4, v12");
			asm volatile("vmv.v.v v6, v14");
	  		

		}
		
		// LAST 4 output rows (no need to precalculate the value of the next block)
		
			i__ = i_ + r_loads * ( C + F - 1 ); 
			f_ = f + F; 								// we don't use the first row of filter for the last block
			o__ = o_ + (r_loads - F + 1) * C;
	  		
	  		int64_t next_column_jump_1st_row = (3 * F - 1) << 1;
	  		int64_t next_kernel_jump_1st_row = (F + 1) << 1;
	  		
	  	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size + F - 1));
	  	  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row)); 
		  
		  // For now
		  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));	   
		  
		  //Fetch first column of filter
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));   	  	  
		 


		  // SECOND COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
		  
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
	  	  asm volatile("vslidedown.vi v20, v20, 1");
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  

		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 

		  

		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));

		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));

		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
		  
		  // THIRD COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
		  
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
	  	  asm volatile("vslidedown.vi v20, v20, 1");
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
		  
		  // FOURTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
		  
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");

		  asm volatile("vslidedown.vi v21, v21, 1");
	  	  asm volatile("vslidedown.vi v20, v20, 1");
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
		  
		  // FIFTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel_jump_1st_row));
		  
		  asm volatile("vslidedown.vi v23, v23, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
	  	  asm volatile("vslidedown.vi v20, v20, 1");
		  
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
		  
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
		  
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
		  
	  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
	  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
	  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
		  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
	  
			 //we go from 2nd plane to others

		  for (int ch = 1; ch < W ; ch++){ 
			  
				  // For now
				
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row)); 
			  
			  // For now
			  
			  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
			  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));	   
			  
			  //Fetch first column of filter
			  
			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
			  
			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
			  
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
			  
		  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
		  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));   	  	  

			 

			  // SECOND COLUMN
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
			  
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
		  	  asm volatile("vslidedown.vi v20, v20, 1");
			  
			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
			  
			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
			  
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
			  
		  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
		  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
			  
			  // THIRD COLUMN
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
			  
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
		  	  asm volatile("vslidedown.vi v20, v20, 1");
			  

			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
			  
			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
			  
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
			  
		  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
		  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
			  
			  // FOURTH COLUMN
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
			  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_column_jump_1st_row));
			  
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
		  	  asm volatile("vslidedown.vi v20, v20, 1");
			  
			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
			  
			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
			  
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
			  
		  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
		  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));  
			  
			  // FIFTH COLUMN
			  
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf));
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
			  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(next_kernel_jump_1st_row));

			  
			  asm volatile("vslidedown.vi v23, v23, 1");
			  asm volatile("vslidedown.vi v22, v22, 1");
			  asm volatile("vslidedown.vi v21, v21, 1");
		  	  asm volatile("vslidedown.vi v20, v20, 1");
			  
			  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t4));
			  
			  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t4));
			  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t3)); 
			  
			  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t4));
			  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t3));
			  asm volatile("vwmacc.vx v6, %0, v21" ::"r"(t2));
			  
		  	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t4));
		  	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t3)); 
		  	  asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2)); 
			  asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t1));
			  
			  
		  }
		  
			asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
			
			asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
			asm volatile("vse32.v  v6,  (%0)" : "+&r"(o__));
		
	  }
		
	
}


//////////////////////////////////////////////////////////////////////////
//																								//
//											7x7 kernel										//
//																								//
//////////////////////////////////////////////////////////////////////////

void iconv2d_tensor16_vec_2xC_7x7_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F){

  // Temporary variables
  int16_t t0, t1, t2, t3, t4, t5, t6;

  // Helper variables
  int64_t const ldo = C << 2;
  int64_t const ldi = (C + F - 1) << 1;
  int64_t const ldf = F << 1;
  int64_t next_column = (F * (F - 1) - 1) << 1;
  int64_t next_kernel = 1 << 1;
  
  int64_t r_loads;
  
  int64_t const size = C % TILE_SIZE_OUT; 	// check for tiling

  int16_t *f_ = f;
  int16_t *i_ = i;
  int32_t *o_ = o; 
  
  for (int c = 0 ; c <= (C - TILE_SIZE_OUT) ; c += TILE_SIZE_OUT){ // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)

	  i_ = i + c; 			// input pointer realtive to the tile (constant throughout the tile)
	  int16_t * i__ = i_;	// inside tile pointer (change at each load)
	  
	  o_ = o + c;			// output pointer relative to the tile
	  int32_t * o__ = o_;// inside tile output pointer 
  	  	
  	  f_ = f;				// filter pointer 
  	  
  // We load the 8b values on 16b register to go to a 32b result easier
  
  // FIRST 4 output rows (from first plane)
  // Loops are unrolled to avoid branching on conditions (ex: use of vmul on first plane and vmacc on others)
  
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));

  for (int ch = 0; ch < W ; ch ++){
  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
	  
	  r_loads = 8;
	  
	  //Fetch first column of filter
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  // FIRST COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		 
	  if (ch == 0){
	  
		  asm volatile("vwmul.vx v0,  v20, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v2,  v21, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v4,  v22, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v6,  v23, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v8,  v24, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v10, v25, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v12, v26, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v14, v27, %0" ::"r"(t0));
		  
		}
		else {
		
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
		}
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
	  // SECOND COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
	  // THIRD COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
	  // FOURTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
		// FIFTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
		// SIXTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
	  // SEVENTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_kernel));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
  
  }
  
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_OUT)); 
		
  asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
  asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		
  asm volatile("vmv.v.v v0, v4");
  asm volatile("vmv.v.v v2, v6");
  asm volatile("vmv.v.v v4, v8");
  asm volatile("vmv.v.v v6, v10");
  asm volatile("vmv.v.v v8, v12");
  asm volatile("vmv.v.v v10,v14");
  
    
  for (int r = r_loads; r < (R + F - 1 - 8) ; r += block_size_7x7){ // - 8 to make sure we do the last step outside this loop
  	
  	//we need to do the last 8 differently
  	
	  	i__ = i_ + r * ( C + F - 1 );
	  	f_ = f;
	  	o__ = o_ + (r - F + 1) * C;
	  
	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));
	  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));
	  
	  //Fetch first column of filter
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  // FIRST COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  
	  asm volatile("vwmul.vx v12, v20, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v14, v21, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v16, v22, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v18, v23, %0" ::"r"(t0));

	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  
	  // SECOND COLUMN
	  
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // THIRD COLUMN

	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // FOURTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // FIFTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // SIXTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // SEVENTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_kernel));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  
	  
	  for (int ch = 1; ch < W ; ch ++){

	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  // FIRST COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1");  
		  
		  // SECOND COLUMN
		  
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // THIRD COLUMN

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // FOURTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // FIFTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // SIXTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // SEVENTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		
	  }
	  
	  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_OUT)); 
			
	  asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  
	  r_loads += block_size_7x7;
			
	  asm volatile("vmv.v.v v0,  v8");
	  asm volatile("vmv.v.v v2,  v10");
	  asm volatile("vmv.v.v v4,  v12");
	  asm volatile("vmv.v.v v6,  v14");
	  asm volatile("vmv.v.v v8,  v16");
	  asm volatile("vmv.v.v v10, v18");
  
  
  }
  
	  i__ = i_ + r_loads * ( C + F - 1 ); 
	  f_ = f; 								// we don't use the first row of filter for the last block
	  o__ = o_ + (r_loads - F + 1) * C;
	  
  		
  	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(TILE_SIZE));
	  
	  for (int ch = 0; ch < W ; ch ++){

	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  // FIRST COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  if(ch == 0) {
		  asm volatile("vwmul.vx v12, v20, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v14, v21, %0" ::"r"(t0));
		  }
		  else{
		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));

		  }
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // SECOND COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // THIRD COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // FOURTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // FIFTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // SIXTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // SEVENTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		
	  }
	  
	  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(TILE_SIZE_OUT)); 
			
	  asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v12, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v14, (%0)" : "+&r"(o__));
	  
	}
	
	if (size != 0) {
	
	  o_ = o + C - size; // pointer to fill the gaps that weren't calculated at tiling step. if no tiling (C < TILE_SIZE) then o_ = o
  	  i_ = i + C - size; // pointer to the input that weren't used in the tiling step.
	  f_ = f;
	  
	  int32_t * o__ = o_; // pointer to fill the gaps that weren't calculated at tiling step. if no tiling (C < TILE_SIZE) then o_ = o
  	  int16_t *  i__ = i_;
	
	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size + F - 1));

  	  for (int ch = 0; ch < W ; ch ++){
  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
	  
	  r_loads = 8;
	  
	  //Fetch first column of filter
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  // FIRST COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		 
	  if (ch == 0){
	  
		  asm volatile("vwmul.vx v0,  v20, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v2,  v21, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v4,  v22, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v6,  v23, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v8,  v24, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v10, v25, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v12, v26, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v14, v27, %0" ::"r"(t0));
		  
		}
		else {
		
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
		}
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
	  // SECOND COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
	  // THIRD COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
	  // FOURTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
		// FIFTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
		// SIXTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
	  
	  // SEVENTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vslidedown.vi v25, v25, 1");
	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vslidedown.vi v27, v27, 1");   
	  asm volatile("vslidedown.vi v28, v28, 1");
	  asm volatile("vslidedown.vi v29, v29, 1");
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		
	  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t0));
	  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t0));
	  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t0));
	  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t0));
		
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v2,  %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v4,  %0, v23" ::"r"(t1));
	  asm volatile("vwmacc.vx v6,  %0, v24" ::"r"(t1));
	  asm volatile("vwmacc.vx v8,  %0, v25" ::"r"(t1));
	  asm volatile("vwmacc.vx v10, %0, v26" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v27" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v2,  %0, v23" ::"r"(t2));
	  asm volatile("vwmacc.vx v4,  %0, v24" ::"r"(t2));
	  asm volatile("vwmacc.vx v6,  %0, v25" ::"r"(t2));
	  asm volatile("vwmacc.vx v8,  %0, v26" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v27" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
	  
	  asm volatile("vwmacc.vx v0,  %0, v23" ::"r"(t3));
	  asm volatile("vwmacc.vx v2,  %0, v24" ::"r"(t3));
	  asm volatile("vwmacc.vx v4,  %0, v25" ::"r"(t3));
	  asm volatile("vwmacc.vx v6,  %0, v26" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v27" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v0,  %0, v24" ::"r"(t4));
	  asm volatile("vwmacc.vx v2,  %0, v25" ::"r"(t4));
	  asm volatile("vwmacc.vx v4,  %0, v26" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v27" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_kernel));
	  
	  asm volatile("vwmacc.vx v0,  %0, v25" ::"r"(t5));
	  asm volatile("vwmacc.vx v2,  %0, v26" ::"r"(t5));
	  asm volatile("vwmacc.vx v4,  %0, v27" ::"r"(t5));
	  
	  asm volatile("vwmacc.vx v0,  %0, v26" ::"r"(t6));
	  asm volatile("vwmacc.vx v2,  %0, v27" ::"r"(t6));
  
  }
  
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
		
  asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
  asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
		
  asm volatile("vmv.v.v v0, v4");
  asm volatile("vmv.v.v v2, v6");
  asm volatile("vmv.v.v v4, v8");
  asm volatile("vmv.v.v v6, v10");
  asm volatile("vmv.v.v v8, v12");
  asm volatile("vmv.v.v v10,v14");
  
    
  for (int r = r_loads; r < (R + F - 1 - 8) ; r += block_size_7x7){ // - 8 to make sure we do the last step outside this loop
  	
  	//we need to do the last 8 differently
  	
	  	i__ = i_ + r * ( C + F - 1 );
	  	f_ = f;
	  	o__ = o_ + (r - F + 1) * C;
	  
	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size + F - 1));
	  
	  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
	  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));
	  
	  //Fetch first column of filter
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
	  
	  // FIRST COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  
	  asm volatile("vwmul.vx v12, v20, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v14, v21, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v16, v22, %0" ::"r"(t0));
	  asm volatile("vwmul.vx v18, v23, %0" ::"r"(t0));

	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));

	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1");  
	  
	  // SECOND COLUMN
	  
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // THIRD COLUMN

	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // FOURTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // FIFTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // SIXTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vslidedown.vi v21, v21, 1");
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  asm volatile("vslidedown.vi v23, v23, 1"); 
	  
	  // SEVENTH COLUMN
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

	  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
	  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
	  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
	  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		
	  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
	  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
	  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
	  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
	  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
	  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
	  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
	  
	  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
	  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
	  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
	  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
	  
	  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
	  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
	  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
	  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
	  
	  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_kernel));
	  
	  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
	  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
	  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
	  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
	  
	  
	  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
	  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
	  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
	  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
	  
	  
	  for (int ch = 1; ch < W ; ch ++){

	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_4));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  // FIRST COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1");  
		  
		  // SECOND COLUMN
		  
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // THIRD COLUMN

		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // FOURTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // FIFTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // SIXTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); //next column
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  // SEVENTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  asm volatile("vwmacc.vx v16, %0, v22" ::"r"(t0));
		  asm volatile("vwmacc.vx v18, %0, v23" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  asm volatile("vwmacc.vx v16, %0, v23" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4, %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6, %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8, %0, v23" ::"r"(t5));
		  
		  
		  asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2, %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4, %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6, %0, v23" ::"r"(t6));
		
	  }
	  
	  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
			
	  asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  
	  r_loads += block_size_7x7;
			
	  asm volatile("vmv.v.v v0,  v8");
	  asm volatile("vmv.v.v v2,  v10");
	  asm volatile("vmv.v.v v4,  v12");
	  asm volatile("vmv.v.v v6,  v14");
	  asm volatile("vmv.v.v v8,  v16");
	  asm volatile("vmv.v.v v10, v18");
  
  
  }
  
	  i__ = i_ + r_loads * ( C + F - 1 ); 
	  f_ = f; 								// we don't use the first row of filter for the last block
	  o__ = o_ + (r_loads - F + 1) * C;
	  
  		
  	  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(size + F - 1));
	  
	  for (int ch = 0; ch < W ; ch ++){

	  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(ldi));
		  asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i__) : "r"(next_plane_8));
		  
		  //Fetch first column of filter
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));    
		  
		  // FIRST COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  
		  if(ch == 0) {
		  asm volatile("vwmul.vx v12, v20, %0" ::"r"(t0));
		  asm volatile("vwmul.vx v14, v21, %0" ::"r"(t0));
		  }
		  else{
		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));

		  }
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf));  
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column)); 
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // SECOND COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // THIRD COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // FOURTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // FIFTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // SIXTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_column));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		  
		  // SEVENTH COLUMN
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));  
		  
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("vslidedown.vi v21, v21, 1");
		  asm volatile("vslidedown.vi v22, v22, 1");
		  asm volatile("vslidedown.vi v23, v23, 1"); 
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));

		  asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t0));
		  asm volatile("vwmacc.vx v14, %0, v21" ::"r"(t0));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v24, v24, 1");
			
		  asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t1));
		  asm volatile("vwmacc.vx v12, %0, v21" ::"r"(t1));
		  asm volatile("vwmacc.vx v14, %0, v22" ::"r"(t1));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t3) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v25, v25, 1"); 
		  
		  asm volatile("vwmacc.vx v8,  %0, v20" ::"r"(t2));
		  asm volatile("vwmacc.vx v10, %0, v21" ::"r"(t2));
		  asm volatile("vwmacc.vx v12, %0, v22" ::"r"(t2));
		  asm volatile("vwmacc.vx v14, %0, v23" ::"r"(t2));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t4) : "r"(ldf));
		  asm volatile("vslidedown.vi v26, v26, 1"); 
		  
		  asm volatile("vwmacc.vx v6,  %0, v20" ::"r"(t3));
		  asm volatile("vwmacc.vx v8,  %0, v21" ::"r"(t3));
		  asm volatile("vwmacc.vx v10, %0, v22" ::"r"(t3));
		  asm volatile("vwmacc.vx v12, %0, v23" ::"r"(t3));
		  asm volatile("vwmacc.vx v14, %0, v24" ::"r"(t3));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t5) : "r"(ldf)); 
		  asm volatile("vslidedown.vi v27, v27, 1"); 
		  
		  asm volatile("vwmacc.vx v4,  %0, v20" ::"r"(t4));
		  asm volatile("vwmacc.vx v6,  %0, v21" ::"r"(t4));
		  asm volatile("vwmacc.vx v8,  %0, v22" ::"r"(t4));
		  asm volatile("vwmacc.vx v10, %0, v23" ::"r"(t4));
		  asm volatile("vwmacc.vx v12, %0, v24" ::"r"(t4));
		  asm volatile("vwmacc.vx v14, %0, v25" ::"r"(t4));
		  
		  asm volatile("lh %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t6) : "r"(next_kernel));
		  
		  asm volatile("vwmacc.vx v2,  %0, v20" ::"r"(t5));
		  asm volatile("vwmacc.vx v4,  %0, v21" ::"r"(t5));
		  asm volatile("vwmacc.vx v6,  %0, v22" ::"r"(t5));
		  asm volatile("vwmacc.vx v8,  %0, v23" ::"r"(t5));
		  asm volatile("vwmacc.vx v10, %0, v24" ::"r"(t5));
		  asm volatile("vwmacc.vx v12, %0, v25" ::"r"(t5));
		  asm volatile("vwmacc.vx v14, %0, v26" ::"r"(t5));
		  
		  asm volatile("vwmacc.vx v0,  %0, v20" ::"r"(t6));
		  asm volatile("vwmacc.vx v2,  %0, v21" ::"r"(t6));
		  asm volatile("vwmacc.vx v4,  %0, v22" ::"r"(t6));
		  asm volatile("vwmacc.vx v6,  %0, v23" ::"r"(t6));
		  asm volatile("vwmacc.vx v8,  %0, v24" ::"r"(t6));
		  asm volatile("vwmacc.vx v10, %0, v25" ::"r"(t6));
		  asm volatile("vwmacc.vx v12, %0, v26" ::"r"(t6));
		  asm volatile("vwmacc.vx v14, %0, v27" ::"r"(t6));
		
	  }
	  
	  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(size)); 
			
	  asm volatile("vse32.v  v0,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v2,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v4,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v6,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v8,  (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v10, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v12, (%0); add %0, %0, %1" : "+&r"(o__) : "r"(ldo));
	  asm volatile("vse32.v  v14, (%0)" : "+&r"(o__));
	
	
	}  

}
