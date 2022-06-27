#!/usr/bin/env python3

import numpy as np
import scipy.signal
import sys


def rand_tensor(O, L, N, M, seed):
	return np.arange(seed, seed+O*N*M*L, dtype=np.float64).reshape(O, L, N, M) * 3.141

def emit(name, array, alignment='3'):
	print(".global %s" % name)
	print(".align " + alignment)
	print("%s:" % name)
	bs = array.tobytes()
	while ((len(bs) % 4) != 0):
		bs += bytes([0]) 
	for i in range(0, len(bs), 4):
		s = ""
		for n in range(4):
			s += "%02x" % bs[i+3-n] 
		print("    .word 0x%s" % s)


#ARGUMENTS

if sys.argv[1] == "wide":
	output_size = "wide"
else:
	output_size = "same"
	
if sys.argv[2] == "NHWC":
	memlayout = "NHWC"
else:
	memlayout = "NCHW"
	
if sys.argv[3] != "null":
	filter_size = int(sys.argv[3])
	assert(filter_size % 2 == 1), "The filter size must be an odd integer number"
else:
	filter_size = 3 # filter is 3x3 by default
	
if sys.argv[4] != "null":
	num_filter = int(sys.argv[4])
else:
	num_filter = 1


# Define the filter size
if len(sys.argv) > 1:
	filter_size = int(sys.argv[3])
	# Filter size must be odd
	assert(filter_size % 2 == 1), "The filter size must be an odd integer number"
else:
	filter_size = 1
num_filter = 1

# Input image
M = 506
N = 506
L = 3
padding = int(filter_size/2)
M_pad = M + 2*padding
N_pad = N + 2*padding
L_pad = L + 0*padding
#assert(M % 8 == 0), "Output image dimension must be divisible by 4, pad the input image accordingly"
#assert(N % 8 == 0), "Output image dimension must be divisible by 4, pad the input image accordingly"
#assert(L / filter_size != 1), "Depth of tensor must be same depth as the filter"

# Generate a random int64 input padded image
tensor = np.around(rand_tensor(1, L_pad, M_pad, N_pad, 0)).astype(np.int16)
np.random.shuffle(tensor.flat)

# Generate a random int64 filter
gen_filter = np.around(rand_tensor(num_filter, L, filter_size, filter_size, 0)).astype(np.int16)
np.random.shuffle(gen_filter.flat)



# Calculate the output matrix
# if wide input (8b) -> output (32b)
if output_size == "wide":

	#transform to 16b for convolution (otherwise, it will lock at 8b and cause overflow)
	tensor_conv = tensor.astype(np.int32)
	gen_filter_conv = gen_filter.astype(np.int32)

	# Create the empty o matrix
	empty_o = np.zeros((num_filter, M, N)).astype(np.int32)

	result = np.zeros((num_filter, M, N)).astype(np.int32) #Num_filter x M x N
	for num in range(num_filter):
		for plane in range(L):
			result[num,:,:] += np.around(scipy.signal.convolve2d(np.flip(gen_filter_conv[num,plane,:,:]), tensor_conv[0,plane,:,:], 'valid')).astype(np.int32) # https://stackoverflow.com/questions/41613155/what-does-scipy-signal-convolve2d-calculate


# if wide input (8b) -> output (8b)

else:
	tensor_conv = tensor.astype(np.int16)
	gen_filter_conv = gen_filter.astype(np.int16)
	
	# Create the empty o matrix
	empty_o = np.zeros((num_filter, M, N)).astype(np.int16)
	
	result = np.zeros((num_filter, M, N)).astype(np.int16) #Num_filter x M x N
	for num in range(num_filter):
		for plane in range(L):
			result[num,:,:] += np.around(scipy.signal.convolve2d(np.flip(gen_filter_conv[num,plane,:,:]), tensor_conv[0,plane,:,:], 'valid')).astype(np.int16) # https://stackoverflow.com/questions/41613155/what-does-scipy-signal-convolve2d-calculate
	
	# reshape into NWHC format
if memlayout == "NHWC":
	result_NWHC = np.zeros((M, N, num_filter)).astype(np.int16) 
	tensor_NWHC = np.zeros((1, M_pad, N_pad, L)).astype(np.int16) 
	gen_filter_NWHC = np.zeros((num_filter,filter_size, filter_size, L_pad)).astype(np.int16) 

	for i in range(0, num_filter):
		for j in range(0, M):
			for k in range(0, N):
				
				result_NWHC[k,j,i] = result[i,k,j]
				
	for i in range(0, L):
		for j in range(0, M_pad):
			for k in range(0, N_pad):
				
				tensor_NWHC[0,k,j,i] = tensor[0,i,k,j]
		
	for index_filter in range(0, num_filter):		
		for i in range(0, L):
			for j in range(0, filter_size):
				for k in range(0, filter_size):
					
					gen_filter_NWHC[index_filter,k,j,i] = gen_filter[index_filter,i,k,j]
			

# Calculate a checksum
checksum = np.sum(result, dtype=np.uint64)

original_stdout = sys.stdout
sys.stdout = sys.stderr # Redirect the standard output to the standard error.

# Print information on display
#vhex = np.vectorize(hex)
print("\n----------------------")
if memlayout == "NHWC":
	print("NHWC memory layout")
else:
	print("NCHW memory layout")
print("----------------------\n")

print("Image: dim = 1 x {0} x {1} x {2} \n".format(L, N_pad, M_pad))
print(tensor)
print("\n")
print("Filter: dim = {0} x {1} x {2} x {3}\n".format(num_filter, L_pad, filter_size, filter_size))
print(gen_filter)
print("\n")
print("Results: dim = {0} x {1} x {2} \n".format(num_filter, M, N))
print(result)
print("\n")
print(checksum)

sys.stdout = original_stdout # Reset the standard output to its original value

# Print information on file
print(".section .data,\"aw\",@progbits")
emit("M", np.array(M, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("L", np.array(L, dtype=np.uint64))
emit("F", np.array(filter_size, dtype=np.uint64))
emit("K", np.array(num_filter, dtype=np.uint64))
emit("i", tensor, 'NR_LANES*4')
emit("f", gen_filter, 'NR_LANES*4')
emit("o", empty_o, 'NR_LANES*4')
emit("golden_o", result, 'NR_LANES*4')
emit("o_checksum", checksum)


