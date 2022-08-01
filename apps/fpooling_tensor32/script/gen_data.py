#!/usr/bin/env python3

import numpy as np
import scipy.signal
import sys
import skimage.measure


def rand_tensor(L, N, M, seed):
	return np.arange(seed, seed+N*M*L, dtype=np.float64).reshape(L, N, M) * 3.141

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


# input image
M = 36
N = 36
L = 3


# Generate a random int64 input padded image
tensor = rand_tensor(L, M, N, -128).astype(np.float32) / 2**16
np.random.shuffle(tensor.flat)

# Create the empty o matrix
empty_o = np.zeros((L, M, N)).astype(np.float32)

M_o = int(M / 2)
N_o = int(N / 2)

result_max_2 = np.zeros((L, M_o, N_o)).astype(np.float32) #Num_filter x M x N
result_avg_2 = np.zeros((L, M_o, N_o)).astype(np.float32) #Num_filter x M x N

#stride is equal to filter_size
for channel in range (L):
	result_max_2[channel,:,:] = skimage.measure.block_reduce(tensor[channel,:,:], (2, 2), np.max)
	result_avg_2[channel,:,:] = skimage.measure.block_reduce(tensor[channel,:,:], (2, 2), np.average)
	
M_o = int(M / 3)
N_o = int(N / 3)

result_max_3 = np.zeros((L, M_o, N_o)).astype(np.float32) #Num_filter x M x N
result_avg_3 = np.zeros((L, M_o, N_o)).astype(np.float32) #Num_filter x M x N
	
for channel in range (L):
	result_max_3[channel,:,:] = skimage.measure.block_reduce(tensor[channel,:,:], (3, 3), np.max)
	result_avg_3[channel,:,:] = skimage.measure.block_reduce(tensor[channel,:,:], (3, 3), np.average)
	
"""	# reshape into NWHC format
if memlayout == "NHWC":
	result_NWHC = np.zeros((M, N, num_filter)).astype(np.float32) 
	tensor_NWHC = np.zeros((1, M_pad, N_pad, L)).astype(np.float32) 
	gen_filter_NWHC = np.zeros((num_filter,filter_size, filter_size, L_pad)).astype(np.float32) 

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
			
"""

tensor = np.float32(tensor)

# Calculate a checksum
checksum = np.sum(result_max_2, dtype=np.uint64)

original_stdout = sys.stdout
sys.stdout = sys.stderr # Redirect the standard output to the standard error.

# Print information on display
#vhex = np.vectorize(hex)
print("\n----------------------")
print("NCHW memory layout")
print("----------------------\n")


print("Image: dim = 1 x {0} x {1} x {2} \n".format(L, M, N))
print(tensor)
print("\n")
print("Results: dim = {0} x {1} x {2} \n".format(L, M_o, N_o))
print("MAX\n")
print(result_max_2)
print("AVG\n")
print(result_avg_3)
print("\n")
print(checksum)

sys.stdout = original_stdout # Reset the standard output to its original value

# Print information on file
print(".section .data,\"aw\",@progbits")
emit("M", np.array(M, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("L", np.array(L, dtype=np.uint64))
#emit("F", np.array(filter_size, dtype=np.uint64))
emit("i", tensor, 'NR_LANES*4')
emit("o", empty_o, 'NR_LANES*4')
emit("golden_o_max_2", result_max_2, 'NR_LANES*4')
emit("golden_o_avg_2", result_avg_2, 'NR_LANES*4')
emit("golden_o_max_3", result_max_3, 'NR_LANES*4')
emit("golden_o_avg_3", result_avg_3, 'NR_LANES*4')
emit("o_checksum", checksum)


