#include "iconv2d_tensor.h"
#include <stdio.h>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Description : Functions for cross-correlation between  
// 			
//				1 x W x C x R   *   K x W x F x F   =    K x C x R
//					input						kernels				output
//
//				limited to 3 x 256 x 256 matrix input (Cmax = Rmax = 256) for now and 3 x 3 kernel matrices (F = 3)
//																																									
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// *o : tensor convolution output pointer k x C x R
// *i : input tensor pointer 1 x W x C x R
// *f : kernel/filter tensor pointer
// R  : number of input Rows
// C  : number of input Column
// W  : Depth of the input tensor
// F  : size of the kernel/filter tensor FxF
// k  : number of kernel/filter tensor to convolve with the input tensor

void iconv2d_tensor_3x3(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W,
                int64_t F, int64_t K) {
  // We work on 4 rows of the output matrix at once
  int64_t block_size_o = 4;
  // We work on block_size_o + F - 1 rows of the input matrix at once
	
	
  int16_t *i_;
  int16_t *o_;
  int16_t *f_;
  
  //helper variable
  
  for(int64_t k = 0; k < K; k++) {
  		// First iteration round, r = 0
		o_ = o + k * R * C;
		i_ = i;
		f_ = f + F*F*W * k;

	  // For simplicity, compute over the padding rows as well
	  //conv2d_vec_4xC_slice_init_3x3(o_, C);
	  // Preload the first two input rows -> This is not needed in the other rounds
	  iconv2d_tensor_vec_4xC_slice_preload_3x3(i_, C, F);
	  // The first F-1 rows have already been loaded by
	  // conv2d_vec_4xC_slice_preload_3x3()
	  int16_t *i__ = i_ + (F - 1) * (C + F - 1);
	  iconv2d_tensor_vec_4xC_3x3(o_, i__, f_, C, W, F);
	  
	  // Re-use some of the already-loaded input rows
	  //conv2d_vec_4xC_slice_move_3x3(C, F);

	  // Iterate over the output rows
	  for (int64_t r = block_size_o; r < R; r += block_size_o) {
		 i_ = i + r * ( C + F - 1 );
		 o_ = o + C * ( k * R + r );

		 // For simplicity, compute over the padding rows as well
		 //conv2d_vec_4xC_slice_init_3x3(o_, C);
		 // The first F-1 rows have already been loaded by
		 i__ = i_ + (F - 1) * (C + F - 1);
		 iconv2d_tensor_vec_4xC_3x3(o_, i__, f_, C, W, F);
		 
		 // Re-use some of the already-loaded input rows
		 //conv2d_vec_4xC_slice_move_3x3(C, F);
	  }
	}
}

// Load 4 rows of the output matrix
/*void iconv2d_tensor_vec_4xC_slice_init_3x3(int64_t *o, int64_t C) {
  // Helper variables
  int64_t ldo = C << 3;

  // Set the vector configuration
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(C));
  // Fetch 4 output rows
  asm volatile("vmv.v.i v0,  0");// add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vmv.v.i v2,  0");//  add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vmv.v.i v4,  0");//  add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vmv.v.i v6,  0");//  : "+r"(o));
}*/

// Load 4 rows of the output matrix
void iconv2d_tensor_vec_4xC_slice_preload_3x3(int16_t *i, int64_t C, int64_t F) {
  // Helper variables
  int64_t ldi = (C + F - 1) << 1;
  int64_t next_plane = (C + F - 2)*ldi;

  // Set the vector configuration
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(C + F - 1));
  
  // Fetch the first floor(F/2) + 1 input rows
  asm volatile("vle16.v v8,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vle16.v v10, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane)); 
  
  asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi)); //second plabe
  asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane));
	  
  asm volatile("vle16.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi)); //third plane
  asm volatile("vle16.v v30, (%0); add %0, %0, %1" : "+r"(i)); 
}

// Calculate 4 output matrix rows
void iconv2d_tensor_vec_4xC_3x3(int16_t *o, int16_t *i, int16_t *f, int64_t C, int64_t W,
                        int64_t F) {

  // Temporary variables
  int16_t t0, t1, t2;

  // Helper variables
  int64_t ldo = C << 1;
  int64_t ldi = (C + F - 1) << 1;
  int64_t next_plane = (C + F - 4)*(C + F - 1) << 1; 
  int64_t ldf = F << 1; //F << 3
  int64_t next_column = (F * (F - 1) - 1) << 1;
  int64_t next_kernel = 1 << 1;
  int16_t *f_;
  
  // Compute on C elements
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(C + F - 1)); 
  
  asm volatile("vmv.v.i v0,  0");//  add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vmv.v.i v2,  0");//  add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vmv.v.i v4,  0");//  add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vmv.v.i v6,  0");//  : "+r"(o));
  
  f_ = f;
  //Fetch first column of filter
  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
  asm volatile("lhu %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column)); 
  
  // Fetch 4 + F - 1 - 2 rows of the input matrix for each depth plane
  // Fetch the first column of the filter, and start calculating its
  // contribution on the four output rows (v0, v1, v2, v3)

  asm volatile("vle16.v v12, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v0, %0, v8"  ::"r"(t0));  
  asm volatile("vmacc.vx v2, %0, v10" ::"r"(t0));
  
  asm volatile("vle16.v v14, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v4, %0, v12" ::"r"(t0));
  asm volatile("vmacc.vx v6, %0, v14" ::"r"(t0));
  
  asm volatile("vle16.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vmacc.vx v0, %0, v10" ::"r"(t1));
  asm volatile("vmacc.vx v2, %0, v12" ::"r"(t1));
  asm volatile("vmacc.vx v4, %0, v14" ::"r"(t1));
  asm volatile("vmacc.vx v6, %0, v16" ::"r"(t1));
  
  asm volatile("vle16.v v18, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane)); 
  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t2));
  asm volatile("vmacc.vx v2, %0, v14" ::"r"(t2));
  asm volatile("vmacc.vx v4, %0, v16" ::"r"(t2));
  asm volatile("vmacc.vx v6, %0, v18" ::"r"(t2));
  
  // Fetch the middle column of the filter, and start calculating its
  // contributions on the output rows To do so, slide down the input rows by one
  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
  asm volatile("lhu %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
  
  asm volatile("vslidedown.vi v8, v8, 1");
  asm volatile("vmacc.vx v0, %0, v8"  ::"r"(t0));

  asm volatile("vslidedown.vi v10, v10, 1");
  asm volatile("vmacc.vx v0, %0, v10" ::"r"(t1));
  asm volatile("vmacc.vx v2, %0, v10" ::"r"(t0));
  
  asm volatile("vslidedown.vi v12, v12, 1");
  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t2));  
  asm volatile("vmacc.vx v2, %0, v12" ::"r"(t1)); 
  asm volatile("vmacc.vx v4, %0, v12" ::"r"(t0));
  
  asm volatile("vslidedown.vi v14, v14, 1");
  asm volatile("vmacc.vx v2, %0, v14" ::"r"(t2));
  asm volatile("vmacc.vx v4, %0, v14" ::"r"(t1));
  asm volatile("vmacc.vx v6, %0, v14" ::"r"(t0));
  

  // KEEP v16 & v18 untouched to write them back after the block is computed
  asm volatile("vslidedown.vi v20, v16, 1");
  asm volatile("vmacc.vx v4, %0, v20" ::"r"(t2));
  asm volatile("vmacc.vx v6, %0, v20" ::"r"(t1));
  
  asm volatile("vslidedown.vi v22, v18, 1");
  asm volatile("vmacc.vx v6, %0, v22" ::"r"(t2));
  
  // Fetch the third column of the filter, and start calculating its
  // contributions on the output rows To do so, slide down the input rows by one
  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
  
  asm volatile("vslidedown.vi v8, v8, 1");
  asm volatile("vmacc.vx v0, %0, v8" ::"r"(t0));
  
  asm volatile("vslidedown.vi v10, v10, 1");
  asm volatile("vmacc.vx v0, %0, v10" ::"r"(t1));
  asm volatile("vmacc.vx v2, %0, v10" ::"r"(t0));

  asm volatile("vslidedown.vi v12, v12, 1");
  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t2));
  asm volatile("vmacc.vx v2, %0, v12" ::"r"(t1));
  asm volatile("vmacc.vx v4, %0, v12" ::"r"(t0));
  
  asm volatile("vslidedown.vi v14, v14, 1");
  asm volatile("vmacc.vx v2, %0, v14" ::"r"(t2));
  asm volatile("vmacc.vx v4, %0, v14" ::"r"(t1));
  asm volatile("vmacc.vx v6, %0, v14" ::"r"(t0));
  
  asm volatile("vslidedown.vi v20, v20, 1");
  asm volatile("vmacc.vx v4, %0, v20" ::"r"(t2));
  asm volatile("vmacc.vx v6, %0, v20" ::"r"(t1));
  
  asm volatile("vslidedown.vi v22, v22, 1");
  asm volatile("vmacc.vx v6, %0, v22" ::"r"(t2));
  
  
  // same as previous for the second and third plane of the kernel following depth, and start calculating its
  // contributions on the output rows To do so, slide down the input rows by one

 if (W >= 2){
 
	  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
	  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lhu %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column)); 


	  asm volatile("vle16.v v12, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  asm volatile("vmacc.vx v0, %0, v24" ::"r"(t0));  
	  asm volatile("vmacc.vx v2, %0, v26" ::"r"(t0));

	  asm volatile("vle16.v v14, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  asm volatile("vmacc.vx v4, %0, v12" ::"r"(t0));
	  asm volatile("vmacc.vx v6, %0, v14" ::"r"(t0));
	  
	  asm volatile("vle16.v v8,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  asm volatile("vmacc.vx v0, %0, v26" ::"r"(t1));  
	  asm volatile("vmacc.vx v2, %0, v12" ::"r"(t1));
	  asm volatile("vmacc.vx v4, %0, v14" ::"r"(t1));
	  asm volatile("vmacc.vx v6, %0, v8"  ::"r"(t1));

	  asm volatile("vle16.v v10, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane)); 
	  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t2));  
	  asm volatile("vmacc.vx v2, %0, v14" ::"r"(t2));
	  asm volatile("vmacc.vx v4, %0, v8"  ::"r"(t2));
	  asm volatile("vmacc.vx v6, %0, v10" ::"r"(t2));
	  
	  
	  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); 
	  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lhu %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
	  
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vmacc.vx v0, %0, v24" ::"r"(t0));

	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vmacc.vx v0, %0, v26" ::"r"(t1));
	  asm volatile("vmacc.vx v2, %0, v26" ::"r"(t0));

	  asm volatile("vslidedown.vi v12, v12, 1");
	  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t2));  
	  asm volatile("vmacc.vx v2, %0, v12" ::"r"(t1)); 
	  asm volatile("vmacc.vx v4, %0, v12" ::"r"(t0));

	  asm volatile("vslidedown.vi v14, v14, 1");
	  asm volatile("vmacc.vx v2, %0, v14" ::"r"(t2));  
	  asm volatile("vmacc.vx v4, %0, v14" ::"r"(t1)); 
	  asm volatile("vmacc.vx v6, %0, v14" ::"r"(t0));
	  // KEEP v8 & v10 untouched to write them back after the block is computed
	  asm volatile("vslidedown.vi v20, v8, 1");
	  asm volatile("vmacc.vx v4, %0, v20" ::"r"(t2));
	  asm volatile("vmacc.vx v6, %0, v20" ::"r"(t1));

	  asm volatile("vslidedown.vi v22, v10, 1");
	  asm volatile("vmacc.vx v6, %0, v22" ::"r"(t2));



	  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
	  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
	  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_kernel));
	   
	  asm volatile("vslidedown.vi v24, v24, 1");
	  asm volatile("vmacc.vx v0, %0, v24" ::"r"(t0));

	  asm volatile("vslidedown.vi v26, v26, 1");
	  asm volatile("vmacc.vx v0, %0, v26" ::"r"(t1));
	  asm volatile("vmacc.vx v2, %0, v26" ::"r"(t0));

	  asm volatile("vslidedown.vi v12, v12, 1");
	  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t2));  
	  asm volatile("vmacc.vx v2, %0, v12" ::"r"(t1)); 
	  asm volatile("vmacc.vx v4, %0, v12" ::"r"(t0));

	  asm volatile("vslidedown.vi v14, v14, 1");
	  asm volatile("vmacc.vx v2, %0, v14" ::"r"(t2));  
	  asm volatile("vmacc.vx v4, %0, v14" ::"r"(t1));
	  asm volatile("vmacc.vx v6, %0, v14" ::"r"(t0));

	  asm volatile("vslidedown.vi v20, v20, 1");
	  asm volatile("vmacc.vx v4, %0, v20" ::"r"(t2));
	  asm volatile("vmacc.vx v6, %0, v20" ::"r"(t1));

	  asm volatile("vslidedown.vi v22, v22, 1");
	  asm volatile("vmacc.vx v6, %0, v22" ::"r"(t2));

 	if (W == 3){ 
 		  //third plane
 		  
		  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf));
		  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lhu %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vle16.v v12, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
		  asm volatile("vmacc.vx v0, %0, v28" ::"r"(t0));  
		  asm volatile("vmacc.vx v2, %0, v30" ::"r"(t0));
		  
		  asm volatile("vle16.v v14, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
		  asm volatile("vmacc.vx v4, %0, v12" ::"r"(t0));
		  asm volatile("vmacc.vx v6, %0, v14" ::"r"(t0));
		  
		  asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi)); 
		  asm volatile("vmacc.vx v0, %0, v30" ::"r"(t1));  
		  asm volatile("vmacc.vx v2, %0, v12" ::"r"(t1));
		  asm volatile("vmacc.vx v4, %0, v14" ::"r"(t1));
		  asm volatile("vmacc.vx v6, %0, v20" ::"r"(t1));

		  asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
		  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t2));  
		  asm volatile("vmacc.vx v2, %0, v14" ::"r"(t2));
		  asm volatile("vmacc.vx v4, %0, v20" ::"r"(t2));
		  asm volatile("vmacc.vx v6, %0, v22" ::"r"(t2));


		  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); 
		  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lhu %1, (%0); sub %0, %0, %2" : "+&r"(f_), "=&r"(t2) : "r"(next_column));
		  
		  asm volatile("vslidedown.vi v28, v28, 1");
		  asm volatile("vmacc.vx v0, %0, v28" ::"r"(t0));  

		  asm volatile("vslidedown.vi v30, v30, 1");
		  asm volatile("vmacc.vx v0, %0, v30" ::"r"(t1));
		  asm volatile("vmacc.vx v2, %0, v30" ::"r"(t0));

		  asm volatile("vslidedown.vi v12, v12, 1");
		  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t2)); 
		  asm volatile("vmacc.vx v2, %0, v12" ::"r"(t1));
		  asm volatile("vmacc.vx v4, %0, v12" ::"r"(t0));

		  asm volatile("vslidedown.vi v14, v14, 1");
		  asm volatile("vmacc.vx v2, %0, v14" ::"r"(t2)); 
		  asm volatile("vmacc.vx v4, %0, v14" ::"r"(t1));
		  asm volatile("vmacc.vx v6, %0, v14" ::"r"(t0));
		 
		  // KEEP v20 & v22 untouched to write them back after the block is computed
		  asm volatile("vslidedown.vi v24, v20, 1");
		  asm volatile("vmacc.vx v4, %0, v24" ::"r"(t2));
		  asm volatile("vmacc.vx v6, %0, v24" ::"r"(t1));
		  
		  asm volatile("vslidedown.vi v26, v22, 1");
		  asm volatile("vmacc.vx v6, %0, v26" ::"r"(t2));

		  
		  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t0) : "r"(ldf)); 
		  asm volatile("lhu %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t1) : "r"(ldf));
		  asm volatile("lhu %1, (%0)" : "+&r"(f_), "=&r"(t2));
		  
		  asm volatile("vslidedown.vi v28, v28, 1");
		  asm volatile("vmacc.vx v0, %0, v28" ::"r"(t0));
		  
		  asm volatile("vslidedown.vi v30, v30, 1");
		  asm volatile("vmacc.vx v0, %0, v30" ::"r"(t1));
		  asm volatile("vmacc.vx v2, %0, v30" ::"r"(t0));
		 
		  asm volatile("vslidedown.vi v12, v12, 1");
		  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t2));
		  asm volatile("vmacc.vx v2, %0, v12" ::"r"(t1));
		  asm volatile("vmacc.vx v4, %0, v12" ::"r"(t0));
		  
		  asm volatile("vslidedown.vi v14, v14, 1");
		  asm volatile("vmacc.vx v2, %0, v14" ::"r"(t2)); 
		  asm volatile("vmacc.vx v4, %0, v14" ::"r"(t1));
		  asm volatile("vmacc.vx v6, %0, v14" ::"r"(t0));
		  
		  asm volatile("vslidedown.vi v24, v24, 1");
		  asm volatile("vmacc.vx v4, %0, v24" ::"r"(t2));
		  asm volatile("vmacc.vx v6, %0, v24" ::"r"(t1));
		  
		  asm volatile("vslidedown.vi v26, v26, 1");
		  asm volatile("vmacc.vx v6, %0, v26" ::"r"(t2));
  		}
  	}

  //store back into output tensor pointer on only C elements
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(C)); 
  
  asm volatile("vse16.v  v0, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse16.v  v2, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse16.v  v4, (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  asm volatile("vse16.v  v6, (%0);" : "+r"(o));
  
  // Move the last floor(F/2) + 1 input rows
  //this is done to reduce memory load since the next first vectors have already been loaded
  //order of instruction does matter
  
  asm volatile("vmv.v.v v28, v20");
  asm volatile("vmv.v.v v30, v22");
	  
  asm volatile("vmv.v.v v24, v8");
  asm volatile("vmv.v.v v26, v10");

  asm volatile("vmv.v.v v8, v16");
  asm volatile("vmv.v.v v10, v18");
 
  
}

/*void iconv2d_tensor_vec_4xC_slice_move_3x3(int64_t C, int64_t F) { //could be implemented direclty in main convolution function conv2d_vec
  // Move C+F-1 elements
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(C + F - 1));
  
  

	  asm volatile("vmv.v.v v28, v20");
	  asm volatile("vmv.v.v v30, v22");
	  
	  asm volatile("vmv.v.v v24, v8");
	  asm volatile("vmv.v.v v26, v10");

  	  asm volatile("vmv.v.v v8, v16");
  	  asm volatile("vmv.v.v v10, v18");
}*/


