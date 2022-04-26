#include "NCWH_to_NWHC.h"
#include <stdio.h>



void NCWH_to_NWHC_tensor16(int16_t *o, int16_t *i, int64_t R, int64_t C, int64_t W)
 {
  
//helper variable
  int16_t *i_;
  int16_t *o_;

		o_ = o;
		i_ = i;

	  // Iterate over the output rows
	  for (int depth = 0; depth < W; depth ++){
			i_ = i + depth * C * R;
			o_ = o + depth;
			NCWH_to_NWHC_tensor16_vec_32xC(o_, i_, R, C, W);
			
	}
}


//fetch the filter value 
void NCWH_to_NWHC_tensor16_vec_32xC(int16_t *o, int16_t *i, int64_t R, int64_t C, int64_t W) {

	uint64_t ldo = (W*C) << 1;			// Jump depth x column nb at each store
	uint64_t stride = W << 1;			// scalar values of each vectorn is stored C adress appart
	uint64_t ldi = C << 1;				// 
	int64_t block_size_o = 32; //16 input and 16 output at each iteration


  
for (int64_t r = 0; r < R; r += block_size_o) {

	asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(C));
	
	// Fetch 32 input vectors
	
	asm volatile("vle16.v v0,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v1,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v2,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v3,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v4,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v5,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v6,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v7,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v8,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v9,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v10, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v11, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v12, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v13, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v14, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v15, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v17, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v18, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v19, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v21, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v22, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v23, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v24, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v25, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v26, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v27, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v29, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v30, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle16.v v31, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	
	// Store each vector with the right stride 

	asm volatile("vsse16.v v0,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v1,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v2,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v3,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v4,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v5,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v6,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v7,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v8,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v9,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v10, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v11, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v12, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v13, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v14, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v15, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v16, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v17, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v18, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v19, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v20, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v21, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v22, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v23, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v24, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v25, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v26, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v27, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v28, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v29, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v30, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse16.v v31, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));

	}
	

}

