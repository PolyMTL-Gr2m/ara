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

void iconv2d_tensor8_3x3_NWHC(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {
	
//helper variable
  int8_t *i_;
  int8_t *o_;
  int8_t *f_;
  

  
  for(int64_t k = 0; k < K; k++) {
		o_ = o + k * R * C;
		i_ = i;
		f_ = f + k * F * F;
		
		
		iconv2d_tensor8_filter_load_3x3(f_, W); 
		// 9 registers are dedicated to the values of filters

	  // Iterate over the output rows
	  for (int64_t r = 0; r < R; r ++) {
		 i_ = i + r * W * (C + F - 1);
		 o_ = o + r * C; //output, in this application we consider that we convolve over the padding as well (size of input would be (C + F - 1)*(R + F - 1) for a C * R output)

		 iconv2d_tensor8_vec_4xW_3x3(o_, i_, C, W);
	  }
	}
}



void iconv2d_tensor8_filter_load_3x3(int8_t *f, int64_t W) {

uint64_t ldf = W; 
  
/*		▣▣▣		22 23 24
	  	▣▣▣		25 26 27
	  	▣▣▣		28 29 30 */
  	
  	
  	//LOAD FILTER
  	
  	asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(W));
  	
  	asm volatile("vle8.v v22, (%0); add %0, %0, %1" : "+&r"(f) : "r"(ldf));
  	asm volatile("vle8.v v23, (%0); add %0, %0, %1" : "+&r"(f) : "r"(ldf));
  	asm volatile("vle8.v v24, (%0); add %0, %0, %1" : "+&r"(f) : "r"(ldf));
  	asm volatile("vle8.v v25, (%0); add %0, %0, %1" : "+&r"(f) : "r"(ldf));
  	asm volatile("vle8.v v26, (%0); add %0, %0, %1" : "+&r"(f) : "r"(ldf));
  	asm volatile("vle8.v v27, (%0); add %0, %0, %1" : "+&r"(f) : "r"(ldf));
  	asm volatile("vle8.v v28, (%0); add %0, %0, %1" : "+&r"(f) : "r"(ldf));
  	asm volatile("vle8.v v29, (%0); add %0, %0, %1" : "+&r"(f) : "r"(ldf));
  	asm volatile("vle8.v v30, (%0)" : "+&r"(f));
  	
  	
  }




void iconv2d_tensor8_vec_4xW_3x3(int8_t *o, int8_t *i, int64_t C, int64_t W) {

  // Helper variables
  int64_t ldo = 1;
  int64_t ldi = W;
  uint64_t const block_size_width = 4;
  uint64_t const next_line_init = (C + 2 - 1) * W;
  uint64_t const next_line = (C + 2 - block_size_width + 1) * W;
  
  int8_t * i_;
  
  i_ = i;

  // Compute on C elements
  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(W)); 
  asm volatile("vmv.v.i v31,  0");

	 
	/*
	 
	 🡕 depth direction (W up to VLEN = 4096 / Size = 8b / LMUL = 1 = 512)
	▣▣▣▣▣▣▢▢▢▢▢▢		4  5  6  7  8  9 
	▣▣▣▣▣▣▢▢▢▢▢▢		10	11 12 13 14 15
	▣▣▣▣▣▣▢▢▢▢▢▢		16 17 18 19 20 21
	▢▢▢▢▢▢▢▢▢▢▢▢
	▢▢▢▢▢▢▢▢▢▢▢▢
	▢▢▢▢▢▢▢▢▢▢▢▢
	▢▢▢▢▢▢▢▢▢▢▢▢
	▢▢▢▢▢▢▢▢▢▢▢▢↑ row direction (R unlimited)
	⟶ column (C unlimited)
	
	for 4 output at a time
	
	Regisers -> 32 - 9 for filters - 18 for input - 4 for output = 1 left
	
	*/
	
	  	asm volatile("vle8.v v4,  (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v5,  (%0); add %0, %0, %1" : "+&r"(i_) : "r"(next_line_init));
	  	asm volatile("vle8.v v10, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v11, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(next_line_init));
	  	asm volatile("vle8.v v16, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v17, (%0)" : "+&r"(i_));

	
  	for (int c = 0 ; c < C ; c += block_size_width){ 
  	
  		asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(W));
  	
  		i_ = i + (c + 2) * W; // get back to adress of "6" 
  		// + 2 corresponds to the 2 values of each line that we don't need to reload 
  	
  		asm volatile("vle8.v v6,  (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v7,  (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v8,  (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v9,  (%0); add %0, %0, %1" : "+&r"(i_) : "r"(next_line));
	   asm volatile("vle8.v v12, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v13, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v14, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v15, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(next_line));
		asm volatile("vle8.v v18, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v19, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v20, (%0); add %0, %0, %1" : "+&r"(i_) : "r"(ldi));
	  	asm volatile("vle8.v v21, (%0)" : "+&r"(i_));
  		
  		// FIRST OUTPUT VECTOR REGARDING DEPTH
  		
  		
  		asm volatile("vmul.vv v0,  v4,  v22");    
  		
  		asm volatile("vmul.vv v1,  v5,  v22");
  		asm volatile("vmacc.vv v0,  v5,  v23");
  		
  		asm volatile("vmul.vv v2,  v6,  v22");
  		asm volatile("vmacc.vv v1,  v6,  v23");
  		asm volatile("vmacc.vv v0,  v6,  v24");
  		
  		asm volatile("vmul.vv v3,  v7,  v22");
  		asm volatile("vmacc.vv v2,  v7,  v23");
  		asm volatile("vmacc.vv v1,  v7,  v24");
  		asm volatile("vmacc.vv v0,  v10, v25");
  		
  		asm volatile("vmacc.vv v3,  v8,  v23");
  		asm volatile("vmacc.vv v2,  v8,  v24");
  		asm volatile("vmacc.vv v1,  v11, v25");
  		asm volatile("vmacc.vv v0,  v11, v26");
  		
  		asm volatile("vmacc.vv v3,  v9,  v24");
  		asm volatile("vmacc.vv v2,  v12, v25");
  		asm volatile("vmacc.vv v1,  v12, v26");
  		asm volatile("vmacc.vv v0,  v12, v27");
  		
  		asm volatile("vmacc.vv v3,  v13, v25");
  		asm volatile("vmacc.vv v2,  v13, v26");
  		asm volatile("vmacc.vv v1,  v13, v27");
  		asm volatile("vmacc.vv v0,  v16, v28");
  		
  		asm volatile("vmacc.vv v3,  v14, v26");
  		asm volatile("vmacc.vv v2,  v14, v27");
  		asm volatile("vmacc.vv v1,  v17, v28");
  		asm volatile("vmacc.vv v0,  v17, v29");
  		
  		asm volatile("vmacc.vv v3,  v15, v27");
  		asm volatile("vmacc.vv v2,  v18, v28");
  		asm volatile("vmacc.vv v1,  v18, v29");
  		asm volatile("vmacc.vv v0,  v18, v30");
  		asm volatile("vredsum.vs v0, v0, v31");  		
  		
  		asm volatile("vmacc.vv v3,  v19, v28");
  		asm volatile("vmacc.vv v2,  v19, v29");
  		asm volatile("vmacc.vv v1,  v19, v30");
  		
  		asm volatile("vmacc.vv v3,  v20, v29");
  		asm volatile("vmacc.vv v2,  v20, v30");
  		//asm volatile("vredsum.vs v1, v1, v31"); //Uncomment this line to make the scalar alu busy and break the code
  		
		asm volatile("vmacc.vv v3,  v21, v30");
		//asm volatile("vredsum.vs v2, v2, v31");
		
		  		
  		asm volatile("vmv.v.v v4,  v8");
	  	asm volatile("vmv.v.v v5,  v9");
  		
  		asm volatile("vmv.v.v v10, v14");
	  	asm volatile("vmv.v.v v11, v15");
	  	
	  	asm volatile("vmv.v.v v16, v20");
	  	asm volatile("vmv.v.v v17, v21");
  	
  		asm volatile("vmv.v.i v31, 0"); // this line is optional but the code won't work without it

  		asm volatile("vredsum.vs v3, v3, v31");
  		asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(ldo));
  
	   asm volatile("vse8.v  v0,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
	   asm volatile("vse8.v  v1,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
	   asm volatile("vse8.v  v2,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
	   asm volatile("vse8.v  v3,  (%0); add %0, %0, %1" : "+&r"(o) : "r"(ldo));
  		 
  		
  		// FIRST BLOCK_SIZE OF FIRST LINE DONE (adress of vectors continues to complete line)
/*  		
  	🡕 depth direction (W up to VLEN = 4096 / Size = 8b / LMUL = 1 = 512)
	▢▢▢▢▨▨▣▣▣▣▢▢		4  5  6  7  8  9 
	▢▢▢▢▨▨▣▣▣▣▢▢		10	11 12 13 14 15
	▢▢▢▢▨▨▣▣▣▣▢▢		16 17 18 19 20 21
	▢▢▢▢▢▢▢▢▢▢▢▢
	▢▢▢▢▢▢▢▢▢▢▢▢
	▢▢▢▢▢▢▢▢▢▢▢▢
	▢▢▢▢▢▢▢▢▢▢▢▢
	▢▢▢▢▢▢▢▢▢▢▢▢↑ row direction (R unlimited)
	⟶ column (C unlimited)
	
	We have to load ▣, whereas ▨ are already loaded from previous block.
	Thus, we just mv the vectors
*/
  }
  // We still have to reload the vectors when we move to the next block of rows
}
