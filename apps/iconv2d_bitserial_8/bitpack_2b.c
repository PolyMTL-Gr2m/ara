#include "iconv2d_tensor2.h"
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

void vbitpack_1(uint8_t * tensor, uint64_t size){

    // tensor       : input tensor
    // packed_data  : output packed data
    // DATA_WIDTH   : encoding size used to packed data
    // size : size of one plane
    // len  : len of input vector needed to be packed
    // bit_prec     : precision of values 
    
    
    uint8_t *i = tensor;
    
	 for (int data_pos = 7 ; data_pos >= 0 ; data_pos --){
	 	 
		 asm volatile("vle8.v v8, (%0)" : "+&r" (i));
		 i += size;
 	
		 asm volatile("vsll.vx v8, v8, %0"  :: "r"(data_pos)); 
		 
		 if(data_pos < 7){
		 
			 asm volatile("vand.vx v8, v8, %0" :: "r"(0x01 << data_pos));
		 
			 asm volatile("vadd.vv v0, v0, v8");
			 
		 }
		 else{		 
		 
			 asm volatile("vand.vx v0, v8, %0" :: "r"(0x80));
			 
		 }
		
	 }
	 
	 // WE DO NOT STORE IT, WE WANT TO USE THE REGISTER RIGHT AWAY
}



void bitpack_filter_1(uint8_t* tensor, uint8_t* packed_data, uint64_t len){

    // tensor       : input tensor
    // packed_data  : output packed data
    // DATA_WIDTH   : encoding size used to packed data
    // input_height : height of input tensor
    // input_width  : width of input tensor
    // bit_prec     : precision of values 
    
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(len));
    
    uint8_t *i = tensor;
    uint8_t *o = packed_data;
    
	 for (int data_pos = 7 ; data_pos >= 0 ; data_pos --){
	 	 
		 asm volatile("vle8.v v0, (%0)" : "+&r" (i)); 
		 i += len; // F*F for each channel
		 
		 asm volatile("vsll.vx v10, v0, %0"  :: "r"(data_pos)); 
		 
		 

		 
		 if(data_pos < 7){
			 asm volatile("vand.vx v10, v10, %0" :: "r"(0x01 << data_pos));
			  
			 asm volatile("vadd.vv v8, v8, v10");
		 }
		 else{		 
			 asm volatile("vand.vx v8, v10, %0" :: "r"(0x80));
		 }
		
	 }

    asm volatile("vse8.v v8, (%0)" : "+&r"(o));
}











/////////////////////////////////////////////////////////////////
//
//                      2 Bits precision packing
//
/////////////////////////////////////////////////////////////////





void vbitpack(uint8_t * tensor, uint64_t size){

    // tensor       : input tensor
    // packed_data  : output packed data
    // DATA_WIDTH   : encoding size used to packed data
    // size : size of one plane
    // len  : len of input vector needed to be packed
    // bit_prec     : precision of values 
    
    uint8_t *i = tensor;
    
	 for (int data_pos = 7 ; data_pos >= 0 ; data_pos --){
	 	 
		 asm volatile("vle8.v v8, (%0)" : "+&r" (i));
		 i += size;
		 
		 if(data_pos > 0){
		 	
		 	 asm volatile("vsll.vx v16, v8, %0"  :: "r"(data_pos - 1));
		 	 asm volatile("vsll.vx v8, v8, %0"  :: "r"(data_pos)); 
		 	 
		 }
		 else{
		 
		 	 asm volatile("vsrl.vi v16, v8, 1");
		 	 
		 }
		  
		 
		 if(data_pos < 7){
		 
			 asm volatile("vand.vx v8, v8, %0" :: "r"(0x01 << data_pos));
			 asm volatile("vand.vx v16, v16, %0" :: "r"(0x01 << data_pos));
		 
			 asm volatile("vadd.vv v0, v0, v8");
			 asm volatile("vadd.vv v6, v6, v16");
			 
		 }
		 else{		 
		 
			 asm volatile("vand.vx v0, v8, %0" :: "r"(0x80));
			 asm volatile("vand.vx v6, v16, %0" :: "r"(0x80));
			 
		 }
		
	 }
	 
	 // WE DO NOT STORE IT, WE WANT TO USE THE REGISTER RIGHT AWAY
}



void bitpack_filter(uint8_t* tensor, uint8_t* packed_data, uint64_t len){

    // tensor       : input tensor
    // packed_data  : output packed data
    // DATA_WIDTH   : encoding size used to packed data
    // input_height : height of input tensor
    // input_width  : width of input tensor
    // bit_prec     : precision of values 
    
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(len));
    
    uint8_t *i = tensor;
    uint8_t *o = packed_data;
    
	 for (int data_pos = 7 ; data_pos >= 0 ; data_pos --){
	 	 
		 asm volatile("vle8.v v0, (%0)" : "+&r" (i)); 
		 i += len; // F*F for each channel
		 
		 
		 if(data_pos > 0){
		 	asm volatile("vsll.vx v10, v0, %0"  :: "r"(data_pos)); 
		 	asm volatile("vsll.vx v11, v0, %0"  :: "r"(data_pos - 1)); 
		 }
		 else{
		 	asm volatile("vsrl.vx v10, v0, %0"  :: "r"(data_pos)); 
		 	asm volatile("vsrl.vx v11, v0, %0"  :: "r"(data_pos + 1)); 
		 }
		 
		 

		 
		 if(data_pos < 7){
			 asm volatile("vand.vx v10, v10, %0" :: "r"(0x01 << data_pos));
			 asm volatile("vand.vx v11, v11, %0" :: "r"(0x01 << data_pos));
			  
			 asm volatile("vadd.vv v8, v8, v10");
			 asm volatile("vadd.vv v9, v9, v11");
		 }
		 else{		 
			 asm volatile("vand.vx v8, v10, %0" :: "r"(0x80));
		 	 asm volatile("vand.vx v9, v11, %0" :: "r"(0x80)); 
		 }
		
	 }

    asm volatile("vse8.v v8, (%0)" : "+&r"(o));
    o += len;
    asm volatile("vse8.v v9, (%0)" : "+&r"(o));
}



