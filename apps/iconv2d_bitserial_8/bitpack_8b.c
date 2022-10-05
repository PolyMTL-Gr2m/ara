#include "ibsconv2d_tensor8.h"
#include <stdio.h>



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                           Description : Functions for cross-correlation between                             //		
//                                                                                                             //
//                      1 x Cin x Hin x Win  * Cout x Cin x F x F   =    Cout x Hout x Wout                    //			
//                          input              kernels                       output                            //	
//																																					//
//																																					//				
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// *o    : tensor convolution output pointer 
// *i    : input tensor pointer
// *f    : kernel/filter tensor pointer
// Hin   : number of input cows
// Win   : number of input column
// Cin   : number of input channels 
// F     : size of the kernel/filter 
// Cout  : number of kernel/filter corresponding to the number of output channels

/////////////////////////////////////////////////////////////////
//
//                      1 Bits precision packing
//
/////////////////////////////////////////////////////////////////

void bitpack8_vec_1_to_8(uint8_t * tensor, uint64_t size){

    // tensor       : input tensor
    // packed_data  : output packed data
    // DATA_WIDTH   : encoding size used to packed data
    // size : size of one plane
    // len  : len of input vector needed to be packed
    // bit_prec     : precision of values 
    
    uint8_t *i = tensor;
    
	 for (int data_pos = 0 ; data_pos < 8 ; data_pos ++){
	 	 
		 asm volatile("vle8.v v8, (%0)" : "+&r" (i));
		 i += size;
		 
		 if(data_pos > 0)
		 	asm volatile("vmacc.vx v0,  %0, v8" ::"r"(0x80 >> data_pos));
		 else
		 	asm volatile("vmul.vx v0, v8, %0" :: "r"(0x80));
	 }
	 
	 // WE DO NOT STORE IT, WE WANT TO USE THE REGISTER RIGHT AWAY
}



void bitpack_filter8_vec_1_to_8(uint8_t* tensor, uint8_t* packed_data, uint64_t len){

    // tensor       : input tensor
    // packed_data  : output packed data
    // DATA_WIDTH   : encoding size used to packed data
    // input_height : height of input tensor
    // input_width  : width of input tensor
    // bit_prec     : precision of values 
    
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(len));
    
    uint8_t *i = tensor;
    uint8_t *o = packed_data;
    
    asm volatile("vmv.v.i v0, 0");
    
    
	 for (int data_pos = 0 ; data_pos < 8 ; data_pos ++){
	 	 
		 asm volatile("vle8.v v8, (%0)" : "+&r" (i));
		 i += len;
		 
		 asm volatile("vmacc.vx v0,  %0, v8" ::"r"(0x80 >> data_pos));
	 }

    asm volatile("vse8.v v0, (%0)" : "+&r"(o));
}











/////////////////////////////////////////////////////////////////
//
//                      2 Bits precision packing
//
/////////////////////////////////////////////////////////////////





void bitpack8_vec_2_to_8(uint8_t * tensor, uint64_t size){

    // tensor       : input tensor
    // packed_data  : output packed data
    // DATA_WIDTH   : encoding size used to packed data
    // size : size of one plane
    // len  : len of input vector needed to be packed
    // bit_prec     : precision of values 
    
    uint8_t *i = tensor;
    
	 for (int data_pos = 0 ; data_pos < 8 ; data_pos ++){
	 	 
		 asm volatile("vle8.v v20, (%0)" : "+&r" (i));
		 i += size;
		 
		 
		 asm volatile("vsrl.vi v16, v20, 1");
		 asm volatile("vand.vi v20, v20, 1");
		 
		 if(data_pos > 0){
			 asm volatile("vmacc.vx v0,  %0, v20" ::"r"(0x80 >> data_pos));
			 asm volatile("vmacc.vx v6,  %0, v16" ::"r"(0x80 >> data_pos));
		 }
		 else{
		 	 asm volatile("vmul.vx v0, v20, %0" :: "r"(0x80));
		 	 asm volatile("vmul.vx v6, v16, %0" :: "r"(0x80));
		 }
		
	 }
	 
	 // WE DO NOT STORE IT, WE WANT TO USE THE REGISTER RIGHT AWAY
}


void bitpack_filter8_vec_2_to_8(uint8_t* tensor, uint8_t* packed_data, uint64_t len){

    // tensor       : input tensor
    // packed_data  : output packed data
    // DATA_WIDTH   : encoding size used to packed data
    // input_height : height of input tensor
    // input_width  : width of input tensor
    // bit_prec     : precision of values 
    
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(len));
    
    uint8_t *i = tensor;
    uint8_t *o = packed_data; 
    
	 for (int8_t data_pos = 0 ; data_pos < 8 ; data_pos ++){
	 	 
		 asm volatile("vle8.v v8, (%0)" : "+&r" (i));
		 i += len;
		 
		 
		 asm volatile("vsrl.vi v16, v8, 1");
		 asm volatile("vand.vi v8, v8, 1");
		 
		 if(data_pos > 0){
			 asm volatile("vmacc.vx v0,  %0, v8" ::"r"(0x80 >> data_pos));
			 asm volatile("vmacc.vx v6,  %0, v16" ::"r"(0x80 >> data_pos));
		 }
		 else{
		 	 asm volatile("vmul.vx v0, v8, %0" :: "r"(0x80));
		 	 asm volatile("vmul.vx v6, v16, %0" :: "r"(0x80));
		 }
		 
		 
		
	 }

    asm volatile("vse8.v v0, (%0)" : "+&r"(o));
    o += len;
    asm volatile("vse8.v v6, (%0)" : "+&r"(o));
}





