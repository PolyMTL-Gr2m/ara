#include "NCHW_to_NHWC.h"
#include <stdio.h>



void NCHW_to_NHWC_tensor64(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W)
 {
  
//helper variable
  int64_t *i_;
  int64_t *o_;

		o_ = o;
		i_ = i;

	  // Iterate over the output rows
	  for (int depth = 0; depth < W; depth ++){
			i_ = i + depth * C * R;
			o_ = o + depth;
			NCHW_to_NHWC_tensor64_vec_8xC(o_, i_, R, C, W);
			
	}
}


//fetch the filter value 
void NCHW_to_NHWC_tensor64_vec_8xC(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W) {

	uint64_t ldo = (W*C) << 3;			// Jump depth x column nb at each store
	uint64_t stride = W << 3;			// scalar values of each vectorn is stored C adress appart
	uint64_t ldi = C << 3;				// 
	int64_t block_size_o = 8; //16 input and 16 output at each iteration


  
for (int64_t r = 0; r < R; r += block_size_o) {

	asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(C));
	
	// Fetch 32 input vectors
	
	asm volatile("vle64.v v0,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle64.v v4,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle64.v v8,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle64.v v12, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle64.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle64.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle64.v v24, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle64.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));

	
	// Store each vector with the right stride 

	asm volatile("vsse64.v v0,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse64.v v4,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse64.v v8,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse64.v v12, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse64.v v16, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse64.v v20, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse64.v v24, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse64.v v28, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));

	}
	

}

