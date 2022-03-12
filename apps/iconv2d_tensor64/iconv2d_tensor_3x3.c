#include "iconv2d_tensor.h"
#include <stdio.h>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 							Description : Functions for cross-correlation between  											 //
// 																																				 //
//								1 x W x C x R   *   K x W x F x F   =    K x C x R													 //
//									input						kernels				output													 //	
//																																					 //
//		limited to 128 x 128 matrix input (Cmax = Rmax = 128) for now and 3 x 3 kernel matrices (F = 3)				 //
//																																					 //				
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// *o : tensor convolution output pointer k x C x R
// *i : input tensor pointer 1 x W x C x R
// *f : kernel/filter tensor pointer
// R  : number of input Rows
// C  : number of input Column
// W  : Depth of the input tensor
// F  : size of the kernel/filter tensor FxF
// k  : number of kernel/filter tensor to convolve with the input tensor

void iconv2d_tensor_3x3(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C, int64_t W,
                int64_t F, int64_t K) {
  // We work on 4 rows of the output matrix at once
  int64_t block_size_o = 1;
  // We work on block_size_o + F - 1 rows of the input matrix at once
	
	
  int64_t *i_;
  int64_t *o_;
  int64_t *f_;
  
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
	  int64_t *i__ = i_ + (F - 1) * (C + F - 1);
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
void iconv2d_tensor_vec_4xC_slice_preload_3x3(int64_t *i, int64_t C, int64_t F) {
  // Helper variables
  int64_t ldi = (C + F - 1) << 3;
  int64_t next_plane = (C + F - 2)*ldi;

  // Set the vector configuration
  asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(C + F - 1));
  
  // Fetch the first floor(F/2) + 1 input rows
  asm volatile("vle64.v v4,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  asm volatile("vle64.v v8,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane));
  
  asm volatile("vle64.v v12, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi)); //second plabe
  asm volatile("vle64.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane)); //second plabe
	  
  asm volatile("vle64.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi)); //third plane
  asm volatile("vle64.v v24, (%0)" : "+&r"(i)); //third plane
  
}

// Calculate 4 output matrix rows
void iconv2d_tensor_vec_4xC_3x3(int64_t *o, int64_t *i, int64_t *f, int64_t C, int64_t W,
                        int64_t F) {

  // Temporary variables
  int64_t t;

  // Helper variables
  int64_t next_plane = (C + F - 1)*(C + F - 1)  << 3; 
  int64_t ldf = 1 << 3;
  int64_t *f_;

  // Compute on C elements
  asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(C + F - 1)); 
  
  f_ = f;
  //Fetch first column of filter
  
  // Fetch 4 + F - 1 - 2 rows of the input matrix for each depth plane
  // Fetch the first column of the filter, and start calculating its
  // contribution on the four output rows (v0, v1, v2, v3)
  
  asm volatile("vmv.v.i v0,  0");
  
  asm volatile("vle64.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane));
  
  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
  asm volatile("vmacc.vx v0, %0, v4"  ::"r"(t));
  asm volatile("vslidedown.vi v4, v4, 1");
  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
  asm volatile("vmacc.vx v0, %0, v4"  ::"r"(t));
  asm volatile("vslidedown.vi v4, v4, 1");
  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
  asm volatile("vmacc.vx v0, %0, v4"  ::"r"(t));
  
  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
  asm volatile("vmacc.vx v0, %0, v8"  ::"r"(t));
  asm volatile("vslidedown.vi v4, v8, 1");
  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
  asm volatile("vmacc.vx v0, %0, v4"  ::"r"(t));
  asm volatile("vslidedown.vi v4, v4, 1");
  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
  asm volatile("vmacc.vx v0, %0, v4"  ::"r"(t));
  
  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf)); 
  asm volatile("vmacc.vx v0, %0, v28" ::"r"(t));
  asm volatile("vslidedown.vi v4, v28, 1");
  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
  asm volatile("vmacc.vx v0, %0, v4" ::"r"(t));
  asm volatile("vslidedown.vi v4, v4, 1");
  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf)); 
  asm volatile("vmacc.vx v0, %0, v4" ::"r"(t));
  
  asm volatile("vmv.v.v v4, v8");
  asm volatile("vmv.v.v v8, v28");
  
 if (W >= 2){
 
 	  asm volatile("vle64.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(next_plane));
 
	  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
	  asm volatile("vmacc.vx v0, %0, v12"  ::"r"(t));
	  asm volatile("vslidedown.vi v12, v12, 1");
	  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
	  asm volatile("vmacc.vx v0, %0, v12"  ::"r"(t));
	  asm volatile("vslidedown.vi v12, v12, 1");
	  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
	  asm volatile("vmacc.vx v0, %0, v12"  ::"r"(t));
	  
	  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
	  asm volatile("vmacc.vx v0, %0, v16"  ::"r"(t));
	  asm volatile("vslidedown.vi v12, v16, 1");
	  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
	  asm volatile("vmacc.vx v0, %0, v12"  ::"r"(t));
	  asm volatile("vslidedown.vi v12, v12, 1");
	  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
	  asm volatile("vmacc.vx v0, %0, v12"  ::"r"(t));
	  
	  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf)); 
	  asm volatile("vmacc.vx v0, %0, v28" ::"r"(t));
	  asm volatile("vslidedown.vi v12, v28, 1");
	  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
	  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t));
	  asm volatile("vslidedown.vi v12, v12, 1");
	  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf)); 
	  asm volatile("vmacc.vx v0, %0, v12" ::"r"(t));
	  
	  asm volatile("vmv.v.v v12, v16");
     asm volatile("vmv.v.v v16, v28");

 	if (W == 3){ 
 	
 		  asm volatile("vle64.v v28, (%0)" : "+&r"(i));
 	
		  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
		  asm volatile("vmacc.vx v0, %0, v20"  ::"r"(t));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
		  asm volatile("vmacc.vx v0, %0, v20"  ::"r"(t));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
		  asm volatile("vmacc.vx v0, %0, v20"  ::"r"(t));
		  
		  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
		  asm volatile("vmacc.vx v0, %0, v24"  ::"r"(t));
		  asm volatile("vslidedown.vi v20, v24, 1");
		  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
		  asm volatile("vmacc.vx v0, %0, v20"  ::"r"(t));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
		  asm volatile("vmacc.vx v0, %0, v20"  ::"r"(t));
		  
		  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf)); 
		  asm volatile("vmacc.vx v0, %0, v28" ::"r"(t));
		  asm volatile("vslidedown.vi v20, v28, 1");
		  asm volatile("ld %1, (%0); add %0, %0, %2" : "+&r"(f_), "=&r"(t) : "r"(ldf));
		  asm volatile("vmacc.vx v0, %0, v20" ::"r"(t));
		  asm volatile("vslidedown.vi v20, v20, 1");
		  asm volatile("ld %1, (%0)" : "+&r"(f_), "=&r"(t)); 
		  asm volatile("vmacc.vx v0, %0, v20" ::"r"(t));
		  
		  asm volatile("vmv.v.v v20, v24");
  		  asm volatile("vmv.v.v v24, v28");
  		}
  	}

   //store back into output tensor pointer
  asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(C)); 
  
  asm volatile("vse64.v  v0, (%0)" : "+&r"(o));
  

 
  
}



