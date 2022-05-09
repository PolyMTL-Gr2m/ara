#include "iconv2d_tensor8.h"
#include <stdio.h>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Description : NWHC Functions for cross-correlation between  
// 			
//				1 x W x C x R   *   K x W x F x F   =    K x C x R
//					input						kernels				output
//
//				limited to 512 channels (W = 512)
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

void iconv2d_tensor8_1x1_NWHC(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {
	
//helper variable
  int8_t *i_;
  int8_t *o_;
  int8_t *f_;
  

  
  for(int64_t k = 0; k < K; k++) {
		o_ = o + k * R * C;
		i_ = i;
		f_ = f + k * F * F;
		
		
		iconv2d_tensor8_filter_load_1x1(f_, W); 
		// 9 registers are dedicated to the values of filters

	  // Iterate over the output rows
	  for (int64_t r = 0; r < R; r ++) {
		 i_ = i + r * W * (C + F - 1);
		 o_ = o + r * C; //output, in this application we consider that we convolve over the padding as well (size of input would be (C + F - 1)*(R + F - 1) for a C * R output)

		 iconv2d_tensor8_vec_4xW_1x1(o_, i_, C, W);
	  }
	}
}



void iconv2d_tensor8_filter_load_1x1(int8_t *f, int64_t W) {
  
/*		▣ v31 */
  	
  	
  	//LOAD FILTER
  	
  	asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(W));
  	asm volatile("vle8.v v31, (%0)" : "+&r"(f));
  	
  	
  }




void iconv2d_tensor8_vec_4xW_1x1(int8_t *o, int8_t *i, int64_t C, int64_t W) {

  // Helper variables
  int64_t ldo = 1;
  int64_t ldi = W;
  uint64_t const block_size_width = 16;

	 
	/*
	 
	 🡕 depth direction (W up to VLEN = 4096 / Size = 8b / LMUL = 1 = 512)
	▣▣▣...▣▣▣▣▣▣▣▣▣		block_size of 16 output at each iteration
	▢▢▢...▢▢▢▢▢▢▢▢▢		
	▢▢▢...▢▢▢▢▢▢▢▢▢		
	▢▢▢...▢▢▢▢▢▢▢▢▢
	▢▢▢...▢▢▢▢▢▢▢▢▢
	▢▢▢...▢▢▢▢▢▢▢▢▢
	▢▢▢...▢▢▢▢▢▢▢▢▢
	........................
	▢▢▢...▢▢▢▢▢▢▢▢▢↑ row direction (R unlimited)
	⟶ column (C unlimited)
	
	*/
	
	
  	for (int c = 0 ; c < C ; c += block_size_width){ 
  	
  		asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(W));
  		
  		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v17, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v19, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v21, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
  		asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v23, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v25, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	   asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v27, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	  	asm volatile("vle8.v v29, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
		asm volatile("vle8.v v30, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
		
		asm volatile("vmul.vv v0,  v16,  v31"); 
		asm volatile("vmul.vv v1,  v17,  v31"); 
		asm volatile("vmul.vv v2,  v18,  v31"); 
		asm volatile("vmul.vv v3,  v19,  v31"); 
		asm volatile("vmul.vv v4,  v20,  v31"); 
		asm volatile("vmul.vv v5,  v21,  v31"); 
		asm volatile("vmul.vv v6,  v22,  v31"); 
		asm volatile("vmul.vv v7,  v23,  v31"); 
		asm volatile("vmul.vv v8,  v24,  v31"); 
		asm volatile("vmul.vv v9,  v25,  v31"); 
		asm volatile("vmul.vv v10, v26,  v31"); 
		asm volatile("vmul.vv v11, v27,  v31"); 
		asm volatile("vmul.vv v12, v28,  v31"); 
		asm volatile("vmul.vv v13, v29,  v31"); 
		asm volatile("vmul.vv v14, v30,  v31"); 
		
		asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
		
		asm volatile("vmul.vv v15, v16,  v31");
		
		// use v16 for the 16th output contribution
		// use v17 as a "zero" vector for sum reduction instruction
		
		asm volatile("vmv.v.i v17, 0");
  		
  		asm volatile("vredsum.vs v0,  v0, v17");
  		asm volatile("vredsum.vs v1,  v0, v17");
  		asm volatile("vredsum.vs v2,  v0, v17");
  		asm volatile("vredsum.vs v3,  v0, v17");
  		asm volatile("vredsum.vs v4,  v0, v17");
  		asm volatile("vredsum.vs v5,  v0, v17");
  		asm volatile("vredsum.vs v6,  v0, v17");
  		asm volatile("vredsum.vs v7,  v0, v17");
  		asm volatile("vredsum.vs v8,  v0, v17");
  		asm volatile("vredsum.vs v9,  v0, v17");
  		asm volatile("vredsum.vs v10, v0, v17");
  		asm volatile("vredsum.vs v11, v0, v17");
  		asm volatile("vredsum.vs v12, v0, v17");
  		asm volatile("vredsum.vs v13, v0, v17");
  		asm volatile("vredsum.vs v14, v0, v17");
  		asm volatile("vredsum.vs v15, v0, v17");
  		
  		
  		asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(ldo)); 
  
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
  		 
  		
  		// FIRST BLOCK_SIZE OF FIRST LINE DONE (adress of vectors continues to complete line)
  }
}
