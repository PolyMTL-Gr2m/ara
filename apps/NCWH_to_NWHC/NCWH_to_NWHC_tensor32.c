#include "NCWH_to_NWHC.h"
#include <stdio.h>



void NCWH_to_NWHC_tensor32(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W)
 {
  
//helper variable
  int32_t *i_;
  int32_t *o_;

		o_ = o;
		i_ = i;

	  // Iterate over the output rows
	  for (int depth = 0; depth < W; depth ++){
			i_ = i + depth * C * R;
			o_ = o + depth;
			NCWH_to_NWHC_tensor32_vec_16xC(o_, i_, R, C, W);
			
	}
}


//fetch the filter value 
void NCWH_to_NWHC_tensor32_vec_16xC(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W) {

	uint64_t ldo = (W*C) << 2;			// Jump depth x column nb at each store
	uint64_t stride = W << 2;			// scalar values of each vectorn is stored C adress appart
	uint64_t ldi = C << 2;				// 
	int64_t block_size_o = 16; //16 input and 16 output at each iteration


  
for (int64_t r = 0; r < R; r += block_size_o) {

	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(C));
	
	// Fetch 32 input vectors
	
	asm volatile("vle32.v v0,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v2,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v4,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v6,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v8,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v10, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v12, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v14, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	
	// Store each vector with the right stride 

	asm volatile("vsse32.v v0,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v2,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v4,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v6,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v8,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v10, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v12, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v14, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v16, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v18, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v20, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v22, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v24, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v26, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v28, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v30, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	}
	

}

