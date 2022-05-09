#include "iconv2d_tensor8.h"
#include <stdio.h>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 							Description : Functions for cross-correlation between  											//		
// 																																				//
//								1 x W x C x R   *   K x W x F x F   =    K x C x R													//			
//									input						kernels				output													//	
//																																					//
//				limited to 512 x 512 matrix input (Cmax = Rmax = 512) for now and 3 x 3 kernel matrices (F = 3)		//
//																																					//				
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// *o : tensor convolution output pointer k x C x R
// *i : input tensor pointer 1 x W x C x R
// *f : kernel/filter tensor pointer
// R  : number of input Rows
// C  : number of input Column
// W  : Depth of the input tensor
// F  : size of the kernel/filter 
// K  : number of kernel/filter to convolve with the input tensor


void iconv2d_tensor8(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {
  // We work on 4 rows of the output matrix at once
  int64_t block_size_o = 16;
  // We work on block_size_o + F - 1 rows of the input tensor at once
	
	
  int8_t *i_;
  int8_t *o_;
  int8_t *f_;
  
  //helper variable
  
  for(int64_t k = 0; k < K; k++) {
  		// First iteration round, k = 0 for the adress of the first value of the first filter
		o_ = o + k * R * C;		// Output is incremented 
		i_ = i;						// Since we aren't working on batch, we only consider one input
		f_ = f + k * F * F * W;

	  // Iterate over the output rows
	  

		switch (F){
			 case 1:
			 	  for (int64_t r = 0; r < R; r += block_size_o) {
					i_ = i + r * C;
					o_ = o + C * ( k * R + r );
					iconv2d_tensor8_vec_16xC_1x1(o_, i_, f_, R, C, W, F);
					}
				break;

			 case 3:
			 	  for (int64_t r = 0; r < R; r += block_size_o) {
					i_ = i + r * ( C + 2 );
					o_ = o + C * ( k * R + r );
					iconv2d_tensor8_vec_16xC_3x3(o_, i_, f_, R, C, W, F);
					}
				break;
				
			 case 5:
			 	  for (int64_t r = 0; r < R; r += block_size_o) {
					i_ = i + r * ( C + 4 );
					o_ = o + C * ( k * R + r );
					iconv2d_tensor8_vec_16xC_5x5(o_, i_, f_, R, C, W, F);
					}
				break;
		}
		 
		 
	}
}



//////////////////////////////////////////////////////////////////////////
//																								//
//											1x1 kernel										//
//																								//
//////////////////////////////////////////////////////////////////////////

// 1x1 convolution is straight forward
// Each line (per block of 16) is multiplied by the kernel value
// Then we switch to the next plane (regarding depth) and do the same
// ...
// When every plane has been computed, we go back to the first one and do the next 16 block

/*

	 🡕 depth direction 
	▣▣▣▣...▣▣▣▣		On the first plane, we load 16 rows and multiply them by the kernel
	...
	▣▣▣▣...▣▣▣▣	
	▣▣▣▣...▣▣▣▣			
	▢▢▢▢...▢▢▢▢		Then when we get back to the first plane, we start at this line
	▢▢▢▢...▢▢▢▢
	▢▢▢▢...▢▢▢▢
	...
	▢▢▢▢...▢▢▢▢↑ row direction (R)
	⟶ column (C)
	
*/

// Calculate 16 output matrix rows
void iconv2d_tensor8_vec_16xC_1x1(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F) {

  // Temporary variables
  int8_t t;

  // Helper variables
  int64_t ldo = C;
  int64_t ldi = (C + F - 1);
  int64_t next_plane = (R + F - 16)*(C + F - 1); 
  int64_t ldf = F;
  int8_t *f_;

  f_ = f;

  // Compute on C + F - 1 elements
  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(C + F - 1)); 
    
  // Set all vector registers to 0 (we use only MAC in the depth for loop
  // So when we change plane, we add the contribution of the next one to the previous one
  
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v1,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v3,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v5,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v7,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v9,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v11, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v13, 0");
  asm volatile("vmv.v.i v14, 0");
  asm volatile("vmv.v.i v15, 0");
  
  for (int depth = 0; depth < W; depth ++){

  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));	// Load the 1x1 kernel
  
  // Inout rows load and macc operations are alternated in order to avoid data conflicts
  
  asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vle8.v v17, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v0,  %0, v16" ::"r"(t));
  
  asm volatile("vle8.v v19, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v1,  %0, v17" ::"r"(t));
  
  asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v2,  %0, v18" ::"r"(t));
  
  asm volatile("vle8.v v21, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v3,  %0, v19" ::"r"(t));
  
  asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v4,  %0, v20" ::"r"(t));
  
  asm volatile("vle8.v v23, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v5,  %0, v21" ::"r"(t));
  
  asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v6,  %0, v22" ::"r"(t));
  
  asm volatile("vle8.v v25, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v7,  %0, v23" ::"r"(t));
  
  asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v8,  %0, v24" ::"r"(t));
  
  asm volatile("vle8.v v27, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v9,  %0, v25" ::"r"(t));
  
  asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v10, %0, v26" ::"r"(t));
  
  asm volatile("vle8.v v29, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v11, %0, v27" ::"r"(t));
  
  asm volatile("vle8.v v30, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v12, %0, v28" ::"r"(t));
  
  asm volatile("vle8.v v31, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane));
  asm volatile("vmacc.vx v13, %0, v29" ::"r"(t));
  asm volatile("vmacc.vx v14, %0, v30" ::"r"(t));
  asm volatile("vmacc.vx v15, %0, v31" ::"r"(t));

  }
  
  // Store the output values
  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(C)); 
  
  asm volatile("vse8.v  v0,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v1,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v2,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v3,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v4,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v5,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v6,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v7,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v8,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v9,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v10, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v11, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v12, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v13, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v14, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v15, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
 
  
}


//////////////////////////////////////////////////////////////////////////
//																								//
//											3x3 kernel										//
//																								//
//////////////////////////////////////////////////////////////////////////

// Calculate 4 output matrix rows
void iconv2d_tensor8_vec_16xC_3x3(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F) {

  // Temporary variables
  int8_t t0, t1, t2, t3, t4, t5, t6, t7, t8; // values of the 3x3 kernel

  // Helper variables
  int64_t ldo = C;
  int64_t ldi = (C + F - 1);
  int64_t next_plane = (R + F - 18)*(C + F - 1); 
  int64_t ldf = F;
  int64_t next_column = (F * (F - 1) - 1);
  int64_t next_kernel = 1;

  // Compute on C elements
  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(C + F - 1)); 
  
  // Set all vector registers to 0 (we use only MAC in the depth for loop
  // So when we change plane, we add the contribution of the next one to the previous one
  
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v1,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v3,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v5,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v7,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v9,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v11, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v13, 0");
  asm volatile("vmv.v.i v14, 0");
  asm volatile("vmv.v.i v15, 0");
  
  
  
  //Fetch first column of filter
  for (int depth = 0; depth < W ; depth++){
  
  // Inout rows load and macc operations are alternated in order to avoid data conflicts
  
  // First column of the filter
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(t0) : "r"(ldf));
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(t1) : "r"(ldf));
  asm volatile("lb %1, (%0); sub %0, %0, %2" : "+&r"(f), "=&r"(t2) : "r"(next_column));
  
  asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));

  asm volatile("vle8.v v17, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi)); 
  asm volatile("vmacc.vx v0,  %0, v16" ::"r"(t0));
  
  asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v1,  %0, v17" ::"r"(t0));
  asm volatile("vmacc.vx v0,  %0, v17" ::"r"(t1));
  
  asm volatile("vle8.v v19, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v2,  %0, v18" ::"r"(t0));
  asm volatile("vmacc.vx v1,  %0, v18" ::"r"(t1));
  asm volatile("vmacc.vx v0,  %0, v18" ::"r"(t2));
  
  asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v3,  %0, v19" ::"r"(t0));
  asm volatile("vmacc.vx v2,  %0, v19" ::"r"(t1));
  asm volatile("vmacc.vx v1,  %0, v19" ::"r"(t2));
  
  asm volatile("vle8.v v21, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi)); 
  asm volatile("vmacc.vx v4,  %0, v20" ::"r"(t0));
  asm volatile("vmacc.vx v3,  %0, v20" ::"r"(t1));
  asm volatile("vmacc.vx v2,  %0, v20" ::"r"(t2));
  
  asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v5,  %0, v21" ::"r"(t0));
  asm volatile("vmacc.vx v4,  %0, v21" ::"r"(t1));
  asm volatile("vmacc.vx v3,  %0, v21" ::"r"(t2));
  
  asm volatile("vle8.v v23, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v6,  %0, v22" ::"r"(t0));
  asm volatile("vmacc.vx v5,  %0, v22" ::"r"(t1));
  asm volatile("vmacc.vx v4,  %0, v22" ::"r"(t2));

  asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v7,  %0, v23" ::"r"(t0));
  asm volatile("vmacc.vx v6,  %0, v23" ::"r"(t1));
  asm volatile("vmacc.vx v5,  %0, v23" ::"r"(t2));
  
  asm volatile("vle8.v v25, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));  
  asm volatile("vmacc.vx v8,  %0, v24" ::"r"(t0));
  asm volatile("vmacc.vx v7,  %0, v24" ::"r"(t1));
  asm volatile("vmacc.vx v6,  %0, v24" ::"r"(t2));
  
  asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v9,  %0, v25" ::"r"(t0));
  asm volatile("vmacc.vx v8,  %0, v25" ::"r"(t1));
  asm volatile("vmacc.vx v7,  %0, v25" ::"r"(t2));
  
  asm volatile("vle8.v v27, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v10, %0, v26" ::"r"(t0));
  asm volatile("vmacc.vx v9, %0, v26" ::"r"(t1));
  asm volatile("vmacc.vx v8, %0, v26" ::"r"(t2));
  
  asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v11, %0, v27" ::"r"(t0));
  asm volatile("vmacc.vx v10, %0, v27" ::"r"(t1));
  asm volatile("vmacc.vx v9, %0, v27" ::"r"(t2));
  
  asm volatile("vle8.v v29, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v12, %0, v28" ::"r"(t0));
  asm volatile("vmacc.vx v11, %0, v28" ::"r"(t1));
  asm volatile("vmacc.vx v10, %0, v28" ::"r"(t2));
  
  asm volatile("vle8.v v30, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));  
  asm volatile("vmacc.vx v13, %0, v29" ::"r"(t0));
  asm volatile("vmacc.vx v12, %0, v29" ::"r"(t1));
  asm volatile("vmacc.vx v11, %0, v29" ::"r"(t2));
  
  asm volatile("vle8.v v31, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v14, %0, v30" ::"r"(t0));
  asm volatile("vmacc.vx v13, %0, v30" ::"r"(t1));
  asm volatile("vmacc.vx v12, %0, v30" ::"r"(t2));
 
  asm volatile("vmacc.vx v15, %0, v31" ::"r"(t0));
  asm volatile("vmacc.vx v14, %0, v31" ::"r"(t1));
  asm volatile("vmacc.vx v13, %0, v31" ::"r"(t2));
  
  asm volatile("vslidedown.vi v16, v16, 1");
  
  // Second column of the filter
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(t3) : "r"(ldf));
  asm volatile("vslidedown.vi v17, v17, 1");
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(t4) : "r"(ldf));
  asm volatile("vslidedown.vi v18, v18, 1");
  
  asm volatile("lb %1, (%0); sub %0, %0, %2" : "+&r"(f), "=&r"(t5) : "r"(next_column));

  asm volatile("vslidedown.vi v19, v19, 1");
  asm volatile("vmacc.vx v0,  %0, v16" ::"r"(t3));
  
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vmacc.vx v1,  %0, v17" ::"r"(t3));
  asm volatile("vmacc.vx v0,  %0, v17" ::"r"(t4));
  
  asm volatile("vslidedown.vi v21, v21, 1");  
  asm volatile("vmacc.vx v2,  %0, v18" ::"r"(t3));
  asm volatile("vmacc.vx v1,  %0, v18" ::"r"(t4));
  asm volatile("vmacc.vx v0,  %0, v18" ::"r"(t5));
  
  asm volatile("vslidedown.vi v22, v22, 1");  
  asm volatile("vmacc.vx v3,  %0, v19" ::"r"(t3));
  asm volatile("vmacc.vx v2,  %0, v19" ::"r"(t4));
  asm volatile("vmacc.vx v1,  %0, v19" ::"r"(t5));
  
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vmacc.vx v4,  %0, v20" ::"r"(t3));
  asm volatile("vmacc.vx v3,  %0, v20" ::"r"(t4));
  asm volatile("vmacc.vx v2,  %0, v20" ::"r"(t5));

  asm volatile("vslidedown.vi v24, v24, 1");  
  asm volatile("vmacc.vx v5,  %0, v21" ::"r"(t3));
  asm volatile("vmacc.vx v4,  %0, v21" ::"r"(t4));
  asm volatile("vmacc.vx v3,  %0, v21" ::"r"(t5));
  
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vmacc.vx v6,  %0, v22" ::"r"(t3));
  asm volatile("vmacc.vx v5,  %0, v22" ::"r"(t4));
  asm volatile("vmacc.vx v4,  %0, v22" ::"r"(t5));
  
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vmacc.vx v7,  %0, v23" ::"r"(t3));
  asm volatile("vmacc.vx v6,  %0, v23" ::"r"(t4));
  asm volatile("vmacc.vx v5,  %0, v23" ::"r"(t5));
  
  asm volatile("vslidedown.vi v27, v27, 1");
  asm volatile("vmacc.vx v8,  %0, v24" ::"r"(t3));
  asm volatile("vmacc.vx v7,  %0, v24" ::"r"(t4));
  asm volatile("vmacc.vx v6,  %0, v24" ::"r"(t5));
  
  asm volatile("vslidedown.vi v28, v28, 1");
  asm volatile("vmacc.vx v9,  %0, v25" ::"r"(t3));
  asm volatile("vmacc.vx v8,  %0, v25" ::"r"(t4));
  asm volatile("vmacc.vx v7,  %0, v25" ::"r"(t5));
  
  asm volatile("vslidedown.vi v29, v29, 1");
  asm volatile("vmacc.vx v10, %0, v26" ::"r"(t3));
  asm volatile("vmacc.vx v9, %0, v26" ::"r"(t4));
  asm volatile("vmacc.vx v8, %0, v26" ::"r"(t5));
  
  asm volatile("vslidedown.vi v30, v30, 1");
  asm volatile("vmacc.vx v11, %0, v27" ::"r"(t3));
  asm volatile("vmacc.vx v10, %0, v27" ::"r"(t4));
  asm volatile("vmacc.vx v9, %0, v27" ::"r"(t5));
  
  asm volatile("vslidedown.vi v31, v31, 1");
  asm volatile("vmacc.vx v12, %0, v28" ::"r"(t3));
  asm volatile("vmacc.vx v11, %0, v28" ::"r"(t4));
  asm volatile("vmacc.vx v10, %0, v28" ::"r"(t5));
  
  asm volatile("vmacc.vx v13, %0, v29" ::"r"(t3));
  asm volatile("vmacc.vx v12, %0, v29" ::"r"(t4));
  asm volatile("vmacc.vx v11, %0, v29" ::"r"(t5));
  
  asm volatile("vmacc.vx v14, %0, v30" ::"r"(t3));
  asm volatile("vmacc.vx v13, %0, v30" ::"r"(t4));
  asm volatile("vmacc.vx v12, %0, v30" ::"r"(t5));
  
  asm volatile("vmacc.vx v15, %0, v31" ::"r"(t3));
  asm volatile("vmacc.vx v14, %0, v31" ::"r"(t4));
  asm volatile("vmacc.vx v13, %0, v31" ::"r"(t5));
  
  asm volatile("vslidedown.vi v16, v16, 1");
  
  // Third column of the filter

  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(t6) : "r"(ldf));
  asm volatile("vslidedown.vi v17, v17, 1");
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(t7) : "r"(ldf));
  asm volatile("vslidedown.vi v18, v18, 1");
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(t8) : "r"(next_kernel));
  asm volatile("vslidedown.vi v19, v19, 1");
  
  asm volatile("vmacc.vx v0,  %0, v16" ::"r"(t6));
  
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vmacc.vx v1,  %0, v17" ::"r"(t6));
  asm volatile("vmacc.vx v0,  %0, v17" ::"r"(t7));
  
  asm volatile("vslidedown.vi v21, v21, 1");
  asm volatile("vmacc.vx v2,  %0, v18" ::"r"(t6));
  asm volatile("vmacc.vx v1,  %0, v18" ::"r"(t7));
  asm volatile("vmacc.vx v0,  %0, v18" ::"r"(t8));
  
  asm volatile("vslidedown.vi v22, v22, 1");
  asm volatile("vmacc.vx v3,  %0, v19" ::"r"(t6));
  asm volatile("vmacc.vx v2,  %0, v19" ::"r"(t7));
  asm volatile("vmacc.vx v1,  %0, v19" ::"r"(t8));
  
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vmacc.vx v4,  %0, v20" ::"r"(t6));
  asm volatile("vmacc.vx v3,  %0, v20" ::"r"(t7));
  asm volatile("vmacc.vx v2,  %0, v20" ::"r"(t8));
  
  asm volatile("vslidedown.vi v24, v24, 1");
  asm volatile("vmacc.vx v5,  %0, v21" ::"r"(t6));
  asm volatile("vmacc.vx v4,  %0, v21" ::"r"(t7));
  asm volatile("vmacc.vx v3,  %0, v21" ::"r"(t8));
  
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vmacc.vx v6,  %0, v22" ::"r"(t6));
  asm volatile("vmacc.vx v5,  %0, v22" ::"r"(t7));
  asm volatile("vmacc.vx v4,  %0, v22" ::"r"(t8));
  
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vmacc.vx v7,  %0, v23" ::"r"(t6));
  asm volatile("vmacc.vx v6,  %0, v23" ::"r"(t7));
  asm volatile("vmacc.vx v5,  %0, v23" ::"r"(t8));
  
  asm volatile("vslidedown.vi v27, v27, 1");
  asm volatile("vmacc.vx v8,  %0, v24" ::"r"(t6));
  asm volatile("vmacc.vx v7,  %0, v24" ::"r"(t7));
  asm volatile("vmacc.vx v6,  %0, v24" ::"r"(t8));
  
  asm volatile("vslidedown.vi v28, v28, 1");
  asm volatile("vmacc.vx v9,  %0, v25" ::"r"(t6));
  asm volatile("vmacc.vx v8,  %0, v25" ::"r"(t7));
  asm volatile("vmacc.vx v7,  %0, v25" ::"r"(t8));
  
  asm volatile("vslidedown.vi v29, v29, 1");
  asm volatile("vmacc.vx v10, %0, v26" ::"r"(t6));
  asm volatile("vmacc.vx v9, %0, v26" ::"r"(t7));
  asm volatile("vmacc.vx v8, %0, v26" ::"r"(t8));
  
  asm volatile("vslidedown.vi v30, v30, 1");
  asm volatile("vmacc.vx v11, %0, v27" ::"r"(t6));
  asm volatile("vmacc.vx v10, %0, v27" ::"r"(t7));
  asm volatile("vmacc.vx v9, %0, v27" ::"r"(t8));
  
  asm volatile("vslidedown.vi v31, v31, 1");
  asm volatile("vmacc.vx v12, %0, v28" ::"r"(t6));
  asm volatile("vmacc.vx v11, %0, v28" ::"r"(t7));
  asm volatile("vmacc.vx v10, %0, v28" ::"r"(t8));
  
  asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v13, %0, v29" ::"r"(t6));
  asm volatile("vmacc.vx v12, %0, v29" ::"r"(t7));
  asm volatile("vmacc.vx v11, %0, v29" ::"r"(t8));
  
  asm volatile("vle8.v v17, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane));
  asm volatile("vmacc.vx v14, %0, v30" ::"r"(t6));
  asm volatile("vmacc.vx v13, %0, v30" ::"r"(t7));
  asm volatile("vmacc.vx v12, %0, v30" ::"r"(t8));
  
  asm volatile("vmacc.vx v15, %0, v31" ::"r"(t6));
  asm volatile("vmacc.vx v14, %0, v31" ::"r"(t7));
  asm volatile("vmacc.vx v13, %0, v31" ::"r"(t8));
  
  // Since we want to have 16 output at a time, we need 16 + F - 1 rows to calculate the contribution
  // We used 2 more loads to calculate the output values of (v15 and v14)
  // Those vectors will be loaded a second time when we get back to this plane
  
  asm volatile("vmacc.vx v15, %0, v16" ::"r"(t1));
  asm volatile("vmacc.vx v14, %0, v16" ::"r"(t2));
  asm volatile("vslidedown.vi v16, v16, 1");
  
  asm volatile("vmacc.vx v15, %0, v16" ::"r"(t4));
  asm volatile("vmacc.vx v14, %0, v16" ::"r"(t5));
  asm volatile("vslidedown.vi v16, v16, 1");
  
  asm volatile("vmacc.vx v15, %0, v16" ::"r"(t7));
  asm volatile("vmacc.vx v14, %0, v16" ::"r"(t8));

  asm volatile("vmacc.vx v15, %0, v17" ::"r"(t2));
  asm volatile("vslidedown.vi v17, v17, 1");
  
  asm volatile("vmacc.vx v15, %0, v17" ::"r"(t5));
  asm volatile("vslidedown.vi v17, v17, 1");
  
  asm volatile("vmacc.vx v15, %0, v17" ::"r"(t8));
  }
  
  // We store the C values per vector as result
   
  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(C)); 
  
  asm volatile("vse8.v  v0,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v1,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v2,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v3,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v4,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v5,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v6,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v7,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v8,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v9,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v10, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v11, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v12, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v13, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v14, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v15, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));

}


//////////////////////////////////////////////////////////////////////////
//																								//
//											5x5 kernel										//
//																								//
//////////////////////////////////////////////////////////////////////////

// Calculate 16 output matrix rows
void iconv2d_tensor8_vec_16xC_5x5(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F) {

  // Temporary variables
  int8_t k[F*F]; // We use an array here to avoid declaring 25 variables
  
  /*	▣▣▣▣▣		 0	1	2	3	4
	  	▣▣▣▣▣		5	6	7	8	9	
	  	▣▣▣▣▣		10	11	12	13	14	
  		▣▣▣▣▣		15	16	17	18	19
		▣▣▣▣▣		20	21	22	23	24
*/	
		
		
  // Helper variables
  int64_t ldo = C;
  int64_t ldi = (C + F - 1);
  int64_t next_plane = (R + F - 20)*(C + F - 1); 
  int64_t ldf = F; //F << 3
  int64_t next_column = (F * (F - 1) - 1);
  int64_t next_kernel = 1;

  // Compute on C elements
  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(C + F - 1)); 
  
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v1,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v3,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v5,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v7,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v9,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v11, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v13, 0");
  asm volatile("vmv.v.i v14, 0");
  asm volatile("vmv.v.i v15, 0");
  
  
  for (int depth = 0; depth < W ; depth++){
  
  // First column of the filter
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[0]) : "r"(ldf));
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[5]) : "r"(ldf));
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[10]) : "r"(ldf));
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[15]) : "r"(ldf));
  asm volatile("lb %1, (%0); sub %0, %0, %2" : "+&r"(f), "=&r"(k[20]) : "r"(next_column));
  
  
  asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));

  asm volatile("vle8.v v17, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi)); 
  asm volatile("vmacc.vx v0,  %0, v16" ::"r"(k[0]));
  
  asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v1,  %0, v17" ::"r"(k[0]));
  asm volatile("vmacc.vx v0,  %0, v17" ::"r"(k[5]));
  
  asm volatile("vle8.v v19, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v2,  %0, v18" ::"r"(k[0]));
  asm volatile("vmacc.vx v1,  %0, v18" ::"r"(k[5]));
  asm volatile("vmacc.vx v0,  %0, v18" ::"r"(k[10]));
  
  asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v3,  %0, v19" ::"r"(k[0]));
  asm volatile("vmacc.vx v2,  %0, v19" ::"r"(k[5]));
  asm volatile("vmacc.vx v1,  %0, v19" ::"r"(k[10]));
  asm volatile("vmacc.vx v0,  %0, v19" ::"r"(k[15]));
  
  asm volatile("vle8.v v21, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi)); 
  asm volatile("vmacc.vx v4,  %0, v20" ::"r"(k[0]));
  asm volatile("vmacc.vx v3,  %0, v20" ::"r"(k[5]));
  asm volatile("vmacc.vx v2,  %0, v20" ::"r"(k[10]));
  asm volatile("vmacc.vx v1,  %0, v20" ::"r"(k[15]));
  asm volatile("vmacc.vx v0,  %0, v20" ::"r"(k[20]));
  
  asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v5,  %0, v21" ::"r"(k[0]));
  asm volatile("vmacc.vx v4,  %0, v21" ::"r"(k[5]));
  asm volatile("vmacc.vx v3,  %0, v21" ::"r"(k[10]));
  asm volatile("vmacc.vx v2,  %0, v21" ::"r"(k[15]));
  asm volatile("vmacc.vx v1,  %0, v21" ::"r"(k[20]));
  
  asm volatile("vle8.v v23, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v6,  %0, v22" ::"r"(k[0]));
  asm volatile("vmacc.vx v5,  %0, v22" ::"r"(k[5]));
  asm volatile("vmacc.vx v4,  %0, v22" ::"r"(k[10]));
  asm volatile("vmacc.vx v3,  %0, v22" ::"r"(k[15]));
  asm volatile("vmacc.vx v2,  %0, v22" ::"r"(k[20]));

  asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v7,  %0, v23" ::"r"(k[0]));
  asm volatile("vmacc.vx v6,  %0, v23" ::"r"(k[5]));
  asm volatile("vmacc.vx v5,  %0, v23" ::"r"(k[10]));
  asm volatile("vmacc.vx v4,  %0, v23" ::"r"(k[15]));
  asm volatile("vmacc.vx v3,  %0, v23" ::"r"(k[20]));

  asm volatile("vle8.v v25, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));  
  asm volatile("vmacc.vx v8,  %0, v24" ::"r"(k[0]));
  asm volatile("vmacc.vx v7,  %0, v24" ::"r"(k[5]));
  asm volatile("vmacc.vx v6,  %0, v24" ::"r"(k[10]));
  asm volatile("vmacc.vx v5,  %0, v24" ::"r"(k[15]));
  asm volatile("vmacc.vx v4,  %0, v24" ::"r"(k[20]));
  
  asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v9,  %0, v25" ::"r"(k[0]));
  asm volatile("vmacc.vx v8,  %0, v25" ::"r"(k[5]));
  asm volatile("vmacc.vx v7,  %0, v25" ::"r"(k[10]));
  asm volatile("vmacc.vx v6,  %0, v25" ::"r"(k[15]));
  asm volatile("vmacc.vx v5,  %0, v25" ::"r"(k[20]));
  
  asm volatile("vle8.v v27, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v10, %0, v26" ::"r"(k[0]));
  asm volatile("vmacc.vx v9,  %0, v26" ::"r"(k[5]));
  asm volatile("vmacc.vx v8,  %0, v26" ::"r"(k[10]));
  asm volatile("vmacc.vx v7,  %0, v26" ::"r"(k[15]));
  asm volatile("vmacc.vx v6,  %0, v26" ::"r"(k[20]));
  
  asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v11, %0, v27" ::"r"(k[0]));
  asm volatile("vmacc.vx v10, %0, v27" ::"r"(k[5]));
  asm volatile("vmacc.vx v9,  %0, v27" ::"r"(k[10]));
  asm volatile("vmacc.vx v8,  %0, v27" ::"r"(k[15]));
  asm volatile("vmacc.vx v7,  %0, v27" ::"r"(k[20]));
  
  asm volatile("vle8.v v29, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v12, %0, v28" ::"r"(k[0]));
  asm volatile("vmacc.vx v11, %0, v28" ::"r"(k[5]));
  asm volatile("vmacc.vx v10, %0, v28" ::"r"(k[10]));
  asm volatile("vmacc.vx v9,  %0, v28" ::"r"(k[15]));
  asm volatile("vmacc.vx v8,  %0, v28" ::"r"(k[20]));
  
  asm volatile("vle8.v v30, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));  
  asm volatile("vmacc.vx v13, %0, v29" ::"r"(k[0]));
  asm volatile("vmacc.vx v12, %0, v29" ::"r"(k[5]));
  asm volatile("vmacc.vx v11, %0, v29" ::"r"(k[10]));
  asm volatile("vmacc.vx v10, %0, v29" ::"r"(k[15]));
  asm volatile("vmacc.vx v9,  %0, v29" ::"r"(k[20]));
  
  asm volatile("vle8.v v31, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v14, %0, v30" ::"r"(k[0]));
  asm volatile("vmacc.vx v13, %0, v30" ::"r"(k[5]));
  asm volatile("vmacc.vx v12, %0, v30" ::"r"(k[10]));
  asm volatile("vmacc.vx v11, %0, v30" ::"r"(k[15]));
  asm volatile("vmacc.vx v10, %0, v30" ::"r"(k[20]));
 
  asm volatile("vmacc.vx v15, %0, v31" ::"r"(k[0]));
  asm volatile("vmacc.vx v14, %0, v31" ::"r"(k[5]));
  asm volatile("vmacc.vx v13, %0, v31" ::"r"(k[10]));
  asm volatile("vmacc.vx v12, %0, v31" ::"r"(k[15]));
  asm volatile("vmacc.vx v11, %0, v31" ::"r"(k[20]));
  
  // Second column of the filter
  
  asm volatile("vslidedown.vi v16, v16, 1");
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[1]) : "r"(ldf));
  asm volatile("vslidedown.vi v17, v17, 1");
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[6]) : "r"(ldf));
  asm volatile("vslidedown.vi v18, v18, 1");

  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[11]) : "r"(ldf));
  asm volatile("vslidedown.vi v19, v19, 1");
  asm volatile("vmacc.vx v0,  %0, v16" ::"r"(k[1]));
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[16]) : "r"(ldf));
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vmacc.vx v1,  %0, v17" ::"r"(k[1]));
  asm volatile("vmacc.vx v0,  %0, v17" ::"r"(k[6]));
  
  asm volatile("lb %1, (%0); sub %0, %0, %2" : "+&r"(f), "=&r"(k[21]) : "r"(next_column));
  asm volatile("vslidedown.vi v21, v21, 1");  
  asm volatile("vmacc.vx v2,  %0, v18" ::"r"(k[1]));
  asm volatile("vmacc.vx v1,  %0, v18" ::"r"(k[6]));
  asm volatile("vmacc.vx v0,  %0, v18" ::"r"(k[11]));
  
  asm volatile("vslidedown.vi v22, v22, 1");  
  asm volatile("vmacc.vx v3,  %0, v19" ::"r"(k[1]));
  asm volatile("vmacc.vx v2,  %0, v19" ::"r"(k[6]));
  asm volatile("vmacc.vx v1,  %0, v19" ::"r"(k[11]));
  asm volatile("vmacc.vx v0,  %0, v19" ::"r"(k[16]));
  
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vmacc.vx v4,  %0, v20" ::"r"(k[1]));
  asm volatile("vmacc.vx v3,  %0, v20" ::"r"(k[6]));
  asm volatile("vmacc.vx v2,  %0, v20" ::"r"(k[11]));
  asm volatile("vmacc.vx v1,  %0, v20" ::"r"(k[16]));
  asm volatile("vmacc.vx v0,  %0, v20" ::"r"(k[21]));

  asm volatile("vslidedown.vi v24, v24, 1");  
  asm volatile("vmacc.vx v5,  %0, v21" ::"r"(k[1]));
  asm volatile("vmacc.vx v4,  %0, v21" ::"r"(k[6]));
  asm volatile("vmacc.vx v3,  %0, v21" ::"r"(k[11]));
  asm volatile("vmacc.vx v2,  %0, v21" ::"r"(k[16]));
  asm volatile("vmacc.vx v1,  %0, v21" ::"r"(k[21]));
  
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vmacc.vx v6,  %0, v22" ::"r"(k[1]));
  asm volatile("vmacc.vx v5,  %0, v22" ::"r"(k[6]));
  asm volatile("vmacc.vx v4,  %0, v22" ::"r"(k[11]));
  asm volatile("vmacc.vx v3,  %0, v22" ::"r"(k[16]));
  asm volatile("vmacc.vx v2,  %0, v22" ::"r"(k[21]));
  
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vmacc.vx v7,  %0, v23" ::"r"(k[1]));
  asm volatile("vmacc.vx v6,  %0, v23" ::"r"(k[6]));
  asm volatile("vmacc.vx v5,  %0, v23" ::"r"(k[11]));
  asm volatile("vmacc.vx v4,  %0, v23" ::"r"(k[16]));
  asm volatile("vmacc.vx v3,  %0, v23" ::"r"(k[21]));
  
  asm volatile("vslidedown.vi v27, v27, 1");
  asm volatile("vmacc.vx v8,  %0, v24" ::"r"(k[1]));
  asm volatile("vmacc.vx v7,  %0, v24" ::"r"(k[6]));
  asm volatile("vmacc.vx v6,  %0, v24" ::"r"(k[11]));
  asm volatile("vmacc.vx v5,  %0, v24" ::"r"(k[16]));
  asm volatile("vmacc.vx v4,  %0, v24" ::"r"(k[21]));
    
  asm volatile("vslidedown.vi v28, v28, 1");
  asm volatile("vmacc.vx v9,  %0, v25" ::"r"(k[1]));
  asm volatile("vmacc.vx v8,  %0, v25" ::"r"(k[6]));
  asm volatile("vmacc.vx v7,  %0, v25" ::"r"(k[11]));
  asm volatile("vmacc.vx v6,  %0, v25" ::"r"(k[16]));
  asm volatile("vmacc.vx v5,  %0, v25" ::"r"(k[21]));
  
  asm volatile("vslidedown.vi v29, v29, 1");
  asm volatile("vmacc.vx v10, %0, v26" ::"r"(k[1]));
  asm volatile("vmacc.vx v9,  %0, v26" ::"r"(k[6]));
  asm volatile("vmacc.vx v8,  %0, v26" ::"r"(k[11]));
  asm volatile("vmacc.vx v7,  %0, v26" ::"r"(k[16]));
  asm volatile("vmacc.vx v6,  %0, v26" ::"r"(k[21]));
  
  asm volatile("vslidedown.vi v30, v30, 1");
  asm volatile("vmacc.vx v11, %0, v27" ::"r"(k[1]));
  asm volatile("vmacc.vx v10, %0, v27" ::"r"(k[6]));
  asm volatile("vmacc.vx v9,  %0, v27" ::"r"(k[11]));
  asm volatile("vmacc.vx v8,  %0, v27" ::"r"(k[16]));
  asm volatile("vmacc.vx v7,  %0, v27" ::"r"(k[21]));
  
  asm volatile("vslidedown.vi v31, v31, 1");
  asm volatile("vmacc.vx v12, %0, v28" ::"r"(k[1]));
  asm volatile("vmacc.vx v11, %0, v28" ::"r"(k[6]));
  asm volatile("vmacc.vx v10, %0, v28" ::"r"(k[11]));
  asm volatile("vmacc.vx v9,  %0, v28" ::"r"(k[16]));
  asm volatile("vmacc.vx v8,  %0, v28" ::"r"(k[21]));
  
  asm volatile("vmacc.vx v13, %0, v29" ::"r"(k[1]));
  asm volatile("vmacc.vx v12, %0, v29" ::"r"(k[6]));
  asm volatile("vmacc.vx v11, %0, v29" ::"r"(k[11]));
  asm volatile("vmacc.vx v10, %0, v29" ::"r"(k[16]));
  asm volatile("vmacc.vx v9,  %0, v29" ::"r"(k[21]));
  
  asm volatile("vmacc.vx v14, %0, v30" ::"r"(k[1]));
  asm volatile("vmacc.vx v13, %0, v30" ::"r"(k[6]));
  asm volatile("vmacc.vx v12, %0, v30" ::"r"(k[11]));
  asm volatile("vmacc.vx v11, %0, v30" ::"r"(k[16]));
  asm volatile("vmacc.vx v10, %0, v30" ::"r"(k[21]));
  
  asm volatile("vmacc.vx v15, %0, v31" ::"r"(k[1]));
  asm volatile("vmacc.vx v14, %0, v31" ::"r"(k[6]));
  asm volatile("vmacc.vx v13, %0, v31" ::"r"(k[11]));
  asm volatile("vmacc.vx v12, %0, v31" ::"r"(k[16]));
  asm volatile("vmacc.vx v11, %0, v31" ::"r"(k[21]));
  
  // Third column of the filter
  
  asm volatile("vslidedown.vi v16, v16, 1");
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[2]) : "r"(ldf));
  asm volatile("vslidedown.vi v17, v17, 1");
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[7]) : "r"(ldf));
  asm volatile("vslidedown.vi v18, v18, 1");

  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[12]) : "r"(ldf));
  asm volatile("vslidedown.vi v19, v19, 1");
  asm volatile("vmacc.vx v0,  %0, v16" ::"r"(k[2]));
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[17]) : "r"(ldf));
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vmacc.vx v1,  %0, v17" ::"r"(k[2]));
  asm volatile("vmacc.vx v0,  %0, v17" ::"r"(k[7]));
  
  asm volatile("lb %1, (%0); sub %0, %0, %2" : "+&r"(f), "=&r"(k[22]) : "r"(next_column));
  asm volatile("vslidedown.vi v21, v21, 1");  
  asm volatile("vmacc.vx v2,  %0, v18" ::"r"(k[2]));
  asm volatile("vmacc.vx v1,  %0, v18" ::"r"(k[7]));
  asm volatile("vmacc.vx v0,  %0, v18" ::"r"(k[12]));
  
  asm volatile("vslidedown.vi v22, v22, 1");  
  asm volatile("vmacc.vx v3,  %0, v19" ::"r"(k[2]));
  asm volatile("vmacc.vx v2,  %0, v19" ::"r"(k[7]));
  asm volatile("vmacc.vx v1,  %0, v19" ::"r"(k[12]));
  asm volatile("vmacc.vx v0,  %0, v19" ::"r"(k[17]));
  
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vmacc.vx v4,  %0, v20" ::"r"(k[2]));
  asm volatile("vmacc.vx v3,  %0, v20" ::"r"(k[7]));
  asm volatile("vmacc.vx v2,  %0, v20" ::"r"(k[12]));
  asm volatile("vmacc.vx v1,  %0, v20" ::"r"(k[17]));
  asm volatile("vmacc.vx v0,  %0, v20" ::"r"(k[22]));

  asm volatile("vslidedown.vi v24, v24, 1");  
  asm volatile("vmacc.vx v5,  %0, v21" ::"r"(k[2]));
  asm volatile("vmacc.vx v4,  %0, v21" ::"r"(k[7]));
  asm volatile("vmacc.vx v3,  %0, v21" ::"r"(k[12]));
  asm volatile("vmacc.vx v2,  %0, v21" ::"r"(k[17]));
  asm volatile("vmacc.vx v1,  %0, v21" ::"r"(k[22]));
  
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vmacc.vx v6,  %0, v22" ::"r"(k[2]));
  asm volatile("vmacc.vx v5,  %0, v22" ::"r"(k[7]));
  asm volatile("vmacc.vx v4,  %0, v22" ::"r"(k[12]));
  asm volatile("vmacc.vx v3,  %0, v22" ::"r"(k[17]));
  asm volatile("vmacc.vx v2,  %0, v22" ::"r"(k[22]));
  
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vmacc.vx v7,  %0, v23" ::"r"(k[2]));
  asm volatile("vmacc.vx v6,  %0, v23" ::"r"(k[7]));
  asm volatile("vmacc.vx v5,  %0, v23" ::"r"(k[12]));
  asm volatile("vmacc.vx v4,  %0, v23" ::"r"(k[17]));
  asm volatile("vmacc.vx v3,  %0, v23" ::"r"(k[22]));
  
  asm volatile("vslidedown.vi v27, v27, 1");
  asm volatile("vmacc.vx v8,  %0, v24" ::"r"(k[2]));
  asm volatile("vmacc.vx v7,  %0, v24" ::"r"(k[7]));
  asm volatile("vmacc.vx v6,  %0, v24" ::"r"(k[12]));
  asm volatile("vmacc.vx v5,  %0, v24" ::"r"(k[17]));
  asm volatile("vmacc.vx v4,  %0, v24" ::"r"(k[22]));
    
  asm volatile("vslidedown.vi v28, v28, 1");
  asm volatile("vmacc.vx v9,  %0, v25" ::"r"(k[2]));
  asm volatile("vmacc.vx v8,  %0, v25" ::"r"(k[7]));
  asm volatile("vmacc.vx v7,  %0, v25" ::"r"(k[12]));
  asm volatile("vmacc.vx v6,  %0, v25" ::"r"(k[17]));
  asm volatile("vmacc.vx v5,  %0, v25" ::"r"(k[22]));
  
  asm volatile("vslidedown.vi v29, v29, 1");
  asm volatile("vmacc.vx v10, %0, v26" ::"r"(k[2]));
  asm volatile("vmacc.vx v9,  %0, v26" ::"r"(k[7]));
  asm volatile("vmacc.vx v8,  %0, v26" ::"r"(k[12]));
  asm volatile("vmacc.vx v7,  %0, v26" ::"r"(k[17]));
  asm volatile("vmacc.vx v6,  %0, v26" ::"r"(k[22]));
  
  asm volatile("vslidedown.vi v30, v30, 1");
  asm volatile("vmacc.vx v11, %0, v27" ::"r"(k[2]));
  asm volatile("vmacc.vx v10, %0, v27" ::"r"(k[7]));
  asm volatile("vmacc.vx v9,  %0, v27" ::"r"(k[12]));
  asm volatile("vmacc.vx v8,  %0, v27" ::"r"(k[17]));
  asm volatile("vmacc.vx v7,  %0, v27" ::"r"(k[22]));
  
  asm volatile("vslidedown.vi v31, v31, 1");
  asm volatile("vmacc.vx v12, %0, v28" ::"r"(k[2]));
  asm volatile("vmacc.vx v11, %0, v28" ::"r"(k[7]));
  asm volatile("vmacc.vx v10, %0, v28" ::"r"(k[12]));
  asm volatile("vmacc.vx v9,  %0, v28" ::"r"(k[17]));
  asm volatile("vmacc.vx v8,  %0, v28" ::"r"(k[22]));
  
  asm volatile("vmacc.vx v13, %0, v29" ::"r"(k[2]));
  asm volatile("vmacc.vx v12, %0, v29" ::"r"(k[7]));
  asm volatile("vmacc.vx v11, %0, v29" ::"r"(k[12]));
  asm volatile("vmacc.vx v10, %0, v29" ::"r"(k[17]));
  asm volatile("vmacc.vx v9,  %0, v29" ::"r"(k[22]));
  
  asm volatile("vmacc.vx v14, %0, v30" ::"r"(k[2]));
  asm volatile("vmacc.vx v13, %0, v30" ::"r"(k[7]));
  asm volatile("vmacc.vx v12, %0, v30" ::"r"(k[12]));
  asm volatile("vmacc.vx v11, %0, v30" ::"r"(k[17]));
  asm volatile("vmacc.vx v10, %0, v30" ::"r"(k[22]));
  
  asm volatile("vmacc.vx v15, %0, v31" ::"r"(k[2]));
  asm volatile("vmacc.vx v14, %0, v31" ::"r"(k[7]));
  asm volatile("vmacc.vx v13, %0, v31" ::"r"(k[12]));
  asm volatile("vmacc.vx v12, %0, v31" ::"r"(k[17]));
  asm volatile("vmacc.vx v11, %0, v31" ::"r"(k[22]));
  
  // Fourth column of the filter
  
  asm volatile("vslidedown.vi v16, v16, 1");
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[3]) : "r"(ldf));
  asm volatile("vslidedown.vi v17, v17, 1");
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[8]) : "r"(ldf));
  asm volatile("vslidedown.vi v18, v18, 1");

  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[13]) : "r"(ldf));
  asm volatile("vslidedown.vi v19, v19, 1");
  asm volatile("vmacc.vx v0,  %0, v16" ::"r"(k[3]));
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[18]) : "r"(ldf));
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vmacc.vx v1,  %0, v17" ::"r"(k[3]));
  asm volatile("vmacc.vx v0,  %0, v17" ::"r"(k[8]));
  
  asm volatile("lb %1, (%0); sub %0, %0, %2" : "+&r"(f), "=&r"(k[23]) : "r"(next_column));
  asm volatile("vslidedown.vi v21, v21, 1");  
  asm volatile("vmacc.vx v2,  %0, v18" ::"r"(k[3]));
  asm volatile("vmacc.vx v1,  %0, v18" ::"r"(k[8]));
  asm volatile("vmacc.vx v0,  %0, v18" ::"r"(k[13]));
  
  asm volatile("vslidedown.vi v22, v22, 1");  
  asm volatile("vmacc.vx v3,  %0, v19" ::"r"(k[3]));
  asm volatile("vmacc.vx v2,  %0, v19" ::"r"(k[8]));
  asm volatile("vmacc.vx v1,  %0, v19" ::"r"(k[13]));
  asm volatile("vmacc.vx v0,  %0, v19" ::"r"(k[18]));
  
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vmacc.vx v4,  %0, v20" ::"r"(k[3]));
  asm volatile("vmacc.vx v3,  %0, v20" ::"r"(k[8]));
  asm volatile("vmacc.vx v2,  %0, v20" ::"r"(k[13]));
  asm volatile("vmacc.vx v1,  %0, v20" ::"r"(k[18]));
  asm volatile("vmacc.vx v0,  %0, v20" ::"r"(k[23]));

  asm volatile("vslidedown.vi v24, v24, 1");  
  asm volatile("vmacc.vx v5,  %0, v21" ::"r"(k[3]));
  asm volatile("vmacc.vx v4,  %0, v21" ::"r"(k[8]));
  asm volatile("vmacc.vx v3,  %0, v21" ::"r"(k[13]));
  asm volatile("vmacc.vx v2,  %0, v21" ::"r"(k[18]));
  asm volatile("vmacc.vx v1,  %0, v21" ::"r"(k[23]));
  
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vmacc.vx v6,  %0, v22" ::"r"(k[3]));
  asm volatile("vmacc.vx v5,  %0, v22" ::"r"(k[8]));
  asm volatile("vmacc.vx v4,  %0, v22" ::"r"(k[13]));
  asm volatile("vmacc.vx v3,  %0, v22" ::"r"(k[18]));
  asm volatile("vmacc.vx v2,  %0, v22" ::"r"(k[23]));
  
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vmacc.vx v7,  %0, v23" ::"r"(k[3]));
  asm volatile("vmacc.vx v6,  %0, v23" ::"r"(k[8]));
  asm volatile("vmacc.vx v5,  %0, v23" ::"r"(k[13]));
  asm volatile("vmacc.vx v4,  %0, v23" ::"r"(k[18]));
  asm volatile("vmacc.vx v3,  %0, v23" ::"r"(k[23]));
  
  asm volatile("vslidedown.vi v27, v27, 1");
  asm volatile("vmacc.vx v8,  %0, v24" ::"r"(k[3]));
  asm volatile("vmacc.vx v7,  %0, v24" ::"r"(k[8]));
  asm volatile("vmacc.vx v6,  %0, v24" ::"r"(k[13]));
  asm volatile("vmacc.vx v5,  %0, v24" ::"r"(k[18]));
  asm volatile("vmacc.vx v4,  %0, v24" ::"r"(k[23]));
    
  asm volatile("vslidedown.vi v28, v28, 1");
  asm volatile("vmacc.vx v9,  %0, v25" ::"r"(k[3]));
  asm volatile("vmacc.vx v8,  %0, v25" ::"r"(k[8]));
  asm volatile("vmacc.vx v7,  %0, v25" ::"r"(k[13]));
  asm volatile("vmacc.vx v6,  %0, v25" ::"r"(k[18]));
  asm volatile("vmacc.vx v5,  %0, v25" ::"r"(k[23]));
  
  asm volatile("vslidedown.vi v29, v29, 1");
  asm volatile("vmacc.vx v10, %0, v26" ::"r"(k[3]));
  asm volatile("vmacc.vx v9,  %0, v26" ::"r"(k[8]));
  asm volatile("vmacc.vx v8,  %0, v26" ::"r"(k[13]));
  asm volatile("vmacc.vx v7,  %0, v26" ::"r"(k[18]));
  asm volatile("vmacc.vx v6,  %0, v26" ::"r"(k[23]));
  
  asm volatile("vslidedown.vi v30, v30, 1");
  asm volatile("vmacc.vx v11, %0, v27" ::"r"(k[3]));
  asm volatile("vmacc.vx v10, %0, v27" ::"r"(k[8]));
  asm volatile("vmacc.vx v9,  %0, v27" ::"r"(k[13]));
  asm volatile("vmacc.vx v8,  %0, v27" ::"r"(k[18]));
  asm volatile("vmacc.vx v7,  %0, v27" ::"r"(k[23]));
  
  asm volatile("vslidedown.vi v31, v31, 1");
  asm volatile("vmacc.vx v12, %0, v28" ::"r"(k[3]));
  asm volatile("vmacc.vx v11, %0, v28" ::"r"(k[8]));
  asm volatile("vmacc.vx v10, %0, v28" ::"r"(k[13]));
  asm volatile("vmacc.vx v9,  %0, v28" ::"r"(k[18]));
  asm volatile("vmacc.vx v8,  %0, v28" ::"r"(k[23]));
  
  asm volatile("vmacc.vx v13, %0, v29" ::"r"(k[3]));
  asm volatile("vmacc.vx v12, %0, v29" ::"r"(k[8]));
  asm volatile("vmacc.vx v11, %0, v29" ::"r"(k[13]));
  asm volatile("vmacc.vx v10, %0, v29" ::"r"(k[18]));
  asm volatile("vmacc.vx v9,  %0, v29" ::"r"(k[23]));
  
  asm volatile("vmacc.vx v14, %0, v30" ::"r"(k[3]));
  asm volatile("vmacc.vx v13, %0, v30" ::"r"(k[8]));
  asm volatile("vmacc.vx v12, %0, v30" ::"r"(k[13]));
  asm volatile("vmacc.vx v11, %0, v30" ::"r"(k[18]));
  asm volatile("vmacc.vx v10, %0, v30" ::"r"(k[23]));
  
  asm volatile("vmacc.vx v15, %0, v31" ::"r"(k[3]));
  asm volatile("vmacc.vx v14, %0, v31" ::"r"(k[8]));
  asm volatile("vmacc.vx v13, %0, v31" ::"r"(k[13]));
  asm volatile("vmacc.vx v12, %0, v31" ::"r"(k[18]));
  asm volatile("vmacc.vx v11, %0, v31" ::"r"(k[23]));
  
  // Fifth column of the filter
  
  asm volatile("vslidedown.vi v16, v16, 1");
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[4]) : "r"(ldf));
  asm volatile("vslidedown.vi v17, v17, 1");
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[9]) : "r"(ldf));
  asm volatile("vslidedown.vi v18, v18, 1");

  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[14]) : "r"(ldf));
  asm volatile("vslidedown.vi v19, v19, 1");
  asm volatile("vmacc.vx v0,  %0, v16" ::"r"(k[4]));
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[19]) : "r"(ldf));
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vmacc.vx v1,  %0, v17" ::"r"(k[4]));
  asm volatile("vmacc.vx v0,  %0, v17" ::"r"(k[9]));
  
  asm volatile("lb %1, (%0); add %0, %0, %2" : "+&r"(f), "=&r"(k[24]) : "r"(next_kernel));
  asm volatile("vslidedown.vi v21, v21, 1");  
  asm volatile("vmacc.vx v2,  %0, v18" ::"r"(k[4]));
  asm volatile("vmacc.vx v1,  %0, v18" ::"r"(k[9]));
  asm volatile("vmacc.vx v0,  %0, v18" ::"r"(k[14]));
  
  asm volatile("vslidedown.vi v22, v22, 1");  
  asm volatile("vmacc.vx v3,  %0, v19" ::"r"(k[4]));
  asm volatile("vmacc.vx v2,  %0, v19" ::"r"(k[9]));
  asm volatile("vmacc.vx v1,  %0, v19" ::"r"(k[14]));
  asm volatile("vmacc.vx v0,  %0, v19" ::"r"(k[19]));
  
  asm volatile("vslidedown.vi v23, v23, 1");
  asm volatile("vmacc.vx v4,  %0, v20" ::"r"(k[4]));
  asm volatile("vmacc.vx v3,  %0, v20" ::"r"(k[9]));
  asm volatile("vmacc.vx v2,  %0, v20" ::"r"(k[14]));
  asm volatile("vmacc.vx v1,  %0, v20" ::"r"(k[19]));
  asm volatile("vmacc.vx v0,  %0, v20" ::"r"(k[24]));

  asm volatile("vslidedown.vi v24, v24, 1");  
  asm volatile("vmacc.vx v5,  %0, v21" ::"r"(k[4]));
  asm volatile("vmacc.vx v4,  %0, v21" ::"r"(k[9]));
  asm volatile("vmacc.vx v3,  %0, v21" ::"r"(k[14]));
  asm volatile("vmacc.vx v2,  %0, v21" ::"r"(k[19]));
  asm volatile("vmacc.vx v1,  %0, v21" ::"r"(k[24]));
  
  asm volatile("vslidedown.vi v25, v25, 1");
  asm volatile("vmacc.vx v6,  %0, v22" ::"r"(k[4]));
  asm volatile("vmacc.vx v5,  %0, v22" ::"r"(k[9]));
  asm volatile("vmacc.vx v4,  %0, v22" ::"r"(k[14]));
  asm volatile("vmacc.vx v3,  %0, v22" ::"r"(k[19]));
  asm volatile("vmacc.vx v2,  %0, v22" ::"r"(k[24]));
  
  asm volatile("vslidedown.vi v26, v26, 1");
  asm volatile("vmacc.vx v7,  %0, v23" ::"r"(k[4]));
  asm volatile("vmacc.vx v6,  %0, v23" ::"r"(k[9]));
  asm volatile("vmacc.vx v5,  %0, v23" ::"r"(k[14]));
  asm volatile("vmacc.vx v4,  %0, v23" ::"r"(k[19]));
  asm volatile("vmacc.vx v3,  %0, v23" ::"r"(k[24]));
  
  asm volatile("vslidedown.vi v27, v27, 1");
  asm volatile("vmacc.vx v8,  %0, v24" ::"r"(k[4]));
  asm volatile("vmacc.vx v7,  %0, v24" ::"r"(k[9]));
  asm volatile("vmacc.vx v6,  %0, v24" ::"r"(k[14]));
  asm volatile("vmacc.vx v5,  %0, v24" ::"r"(k[19]));
  asm volatile("vmacc.vx v4,  %0, v24" ::"r"(k[24]));
    
  asm volatile("vslidedown.vi v28, v28, 1");
  asm volatile("vmacc.vx v9,  %0, v25" ::"r"(k[4]));
  asm volatile("vmacc.vx v8,  %0, v25" ::"r"(k[9]));
  asm volatile("vmacc.vx v7,  %0, v25" ::"r"(k[14]));
  asm volatile("vmacc.vx v6,  %0, v25" ::"r"(k[19]));
  asm volatile("vmacc.vx v5,  %0, v25" ::"r"(k[24]));
  
  asm volatile("vslidedown.vi v29, v29, 1");
  asm volatile("vmacc.vx v10, %0, v26" ::"r"(k[4]));
  asm volatile("vmacc.vx v9,  %0, v26" ::"r"(k[9]));
  asm volatile("vmacc.vx v8,  %0, v26" ::"r"(k[14]));
  asm volatile("vmacc.vx v7,  %0, v26" ::"r"(k[19]));
  asm volatile("vmacc.vx v6,  %0, v26" ::"r"(k[24]));
  
  asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vslidedown.vi v30, v30, 1");
  asm volatile("vmacc.vx v11, %0, v27" ::"r"(k[4]));
  asm volatile("vmacc.vx v10, %0, v27" ::"r"(k[9]));
  asm volatile("vmacc.vx v9,  %0, v27" ::"r"(k[14]));
  asm volatile("vmacc.vx v8,  %0, v27" ::"r"(k[19]));
  asm volatile("vmacc.vx v7,  %0, v27" ::"r"(k[24]));
  
  asm volatile("vle8.v v17, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vslidedown.vi v31, v31, 1");
  asm volatile("vmacc.vx v12, %0, v28" ::"r"(k[4]));
  asm volatile("vmacc.vx v11, %0, v28" ::"r"(k[9]));
  asm volatile("vmacc.vx v10, %0, v28" ::"r"(k[14]));
  asm volatile("vmacc.vx v9,  %0, v28" ::"r"(k[19]));
  asm volatile("vmacc.vx v8,  %0, v28" ::"r"(k[24]));

  asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v13, %0, v29" ::"r"(k[4]));
  asm volatile("vmacc.vx v12, %0, v29" ::"r"(k[9]));
  asm volatile("vmacc.vx v11, %0, v29" ::"r"(k[14]));
  asm volatile("vmacc.vx v10, %0, v29" ::"r"(k[19]));
  asm volatile("vmacc.vx v9,  %0, v29" ::"r"(k[24]));
  
  asm volatile("vle8.v v19, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane));
  asm volatile("vmacc.vx v14, %0, v30" ::"r"(k[4]));
  asm volatile("vmacc.vx v13, %0, v30" ::"r"(k[9]));
  asm volatile("vmacc.vx v12, %0, v30" ::"r"(k[14]));
  asm volatile("vmacc.vx v11, %0, v30" ::"r"(k[19]));
  asm volatile("vmacc.vx v10, %0, v30" ::"r"(k[24]));
  
  
  asm volatile("vmacc.vx v15, %0, v31" ::"r"(k[4]));
  asm volatile("vmacc.vx v14, %0, v31" ::"r"(k[9]));
  asm volatile("vmacc.vx v13, %0, v31" ::"r"(k[14]));
  asm volatile("vmacc.vx v12, %0, v31" ::"r"(k[19]));
  asm volatile("vmacc.vx v11, %0, v31" ::"r"(k[24]));
  
  
  // Since we want to have 16 output at a time, we need 16 + F - 1 rows to calculate the contribution
  // We used 4 more loads to calculate the output values
  // Those vectors will be loaded a second time when we get back to this plane
  
  asm volatile("vmacc.vx v15, %0, v16" ::"r"(k[5]));
  asm volatile("vmacc.vx v14, %0, v16" ::"r"(k[10]));
  asm volatile("vmacc.vx v13, %0, v16" ::"r"(k[15]));
  asm volatile("vmacc.vx v12, %0, v16" ::"r"(k[20]));
  
  asm volatile("vslidedown.vi v16, v16, 1");
  asm volatile("vmacc.vx v15, %0, v17" ::"r"(k[10]));
  asm volatile("vmacc.vx v14, %0, v17" ::"r"(k[15]));
  asm volatile("vmacc.vx v13, %0, v17" ::"r"(k[20]));
  
  asm volatile("vslidedown.vi v17, v17, 1");
  asm volatile("vmacc.vx v15, %0, v18" ::"r"(k[15]));
  asm volatile("vmacc.vx v14, %0, v18" ::"r"(k[20]));
  
  asm volatile("vslidedown.vi v18, v18, 1");
  asm volatile("vmacc.vx v15, %0, v19" ::"r"(k[20]));
  
  
  
  asm volatile("vslidedown.vi v19, v19, 1");
  asm volatile("vmacc.vx v15, %0, v16" ::"r"(k[6]));
  asm volatile("vmacc.vx v14, %0, v16" ::"r"(k[11]));
  asm volatile("vmacc.vx v13, %0, v16" ::"r"(k[16]));
  asm volatile("vmacc.vx v12, %0, v16" ::"r"(k[21]));
  
  asm volatile("vslidedown.vi v16, v16, 1");
  asm volatile("vmacc.vx v15, %0, v17" ::"r"(k[11]));
  asm volatile("vmacc.vx v14, %0, v17" ::"r"(k[16]));
  asm volatile("vmacc.vx v13, %0, v17" ::"r"(k[21]));
  
  asm volatile("vslidedown.vi v17, v17, 1");
  asm volatile("vmacc.vx v15, %0, v18" ::"r"(k[16]));
  asm volatile("vmacc.vx v14, %0, v18" ::"r"(k[21]));
  
  asm volatile("vslidedown.vi v18, v18, 1");
  asm volatile("vmacc.vx v15, %0, v19" ::"r"(k[21]));
  
  
  
  asm volatile("vslidedown.vi v19, v19, 1");
  asm volatile("vmacc.vx v15, %0, v16" ::"r"(k[7]));
  asm volatile("vmacc.vx v14, %0, v16" ::"r"(k[12]));
  asm volatile("vmacc.vx v13, %0, v16" ::"r"(k[17]));
  asm volatile("vmacc.vx v12, %0, v16" ::"r"(k[22]));
  
  asm volatile("vslidedown.vi v16, v16, 1");
  asm volatile("vmacc.vx v15, %0, v17" ::"r"(k[12]));
  asm volatile("vmacc.vx v14, %0, v17" ::"r"(k[17]));
  asm volatile("vmacc.vx v13, %0, v17" ::"r"(k[22]));
  
  asm volatile("vslidedown.vi v17, v17, 1");
  asm volatile("vmacc.vx v15, %0, v18" ::"r"(k[17]));
  asm volatile("vmacc.vx v14, %0, v18" ::"r"(k[22]));
  
  asm volatile("vslidedown.vi v18, v18, 1");
  asm volatile("vmacc.vx v15, %0, v19" ::"r"(k[22]));
  
  
  
  
  asm volatile("vslidedown.vi v19, v19, 1");
  asm volatile("vmacc.vx v15, %0, v16" ::"r"(k[8]));
  asm volatile("vmacc.vx v14, %0, v16" ::"r"(k[13]));
  asm volatile("vmacc.vx v13, %0, v16" ::"r"(k[18]));
  asm volatile("vmacc.vx v12, %0, v16" ::"r"(k[23]));
  
  asm volatile("vslidedown.vi v16, v16, 1");
  asm volatile("vmacc.vx v15, %0, v17" ::"r"(k[13]));
  asm volatile("vmacc.vx v14, %0, v17" ::"r"(k[18]));
  asm volatile("vmacc.vx v13, %0, v17" ::"r"(k[23]));
  
  asm volatile("vslidedown.vi v17, v17, 1");
  asm volatile("vmacc.vx v15, %0, v18" ::"r"(k[18]));
  asm volatile("vmacc.vx v14, %0, v18" ::"r"(k[23]));
  
  asm volatile("vslidedown.vi v18, v18, 1");
  asm volatile("vmacc.vx v15, %0, v19" ::"r"(k[23]));
  
  
  
  asm volatile("vslidedown.vi v19, v19, 1");  
  asm volatile("vmacc.vx v15, %0, v16" ::"r"(k[9]));
  asm volatile("vmacc.vx v14, %0, v16" ::"r"(k[14]));
  asm volatile("vmacc.vx v13, %0, v16" ::"r"(k[19]));
  asm volatile("vmacc.vx v12, %0, v16" ::"r"(k[24]));
  
  asm volatile("vmacc.vx v15, %0, v17" ::"r"(k[14]));
  asm volatile("vmacc.vx v14, %0, v17" ::"r"(k[19]));
  asm volatile("vmacc.vx v13, %0, v17" ::"r"(k[24]));
  
  asm volatile("vmacc.vx v15, %0, v18" ::"r"(k[19]));
  asm volatile("vmacc.vx v14, %0, v18" ::"r"(k[24]));

  asm volatile("vmacc.vx v15, %0, v19" ::"r"(k[24]));
  
  }
  
  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(C)); 
  
  // We store the 16 output values
  
  asm volatile("vse8.v  v0,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v1,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v2,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v3,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v4,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v5,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v6,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v7,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v8,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v9,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v10, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v11, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v12, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v13, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v14, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse8.v  v15, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
 
  
}


//////////////////////////////////////////////////////////////////////////
//																								//
//											7x7 kernel										//
//																								//
//////////////////////////////////////////////////////////////////////////

// TODO


