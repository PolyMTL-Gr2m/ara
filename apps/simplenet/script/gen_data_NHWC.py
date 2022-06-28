#!/usr/bin/env python3

import numpy as np
import scipy.signal
import sys


def rand_tensor(O, L, N, M, seed):
	return np.arange(seed, seed+O*N*M*L, dtype=np.float64).reshape(O, L, N, M) * 3.141

def emit_8b(name, array, alignment='3'):
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
		
def emit_64b(name, array, alignment='3'):
	print(".global %s" % name)
	print(".align " + alignment)
	print("%s:" % name)
	bs = array.tobytes()
	for i in range(0, len(bs), 4):
		s = ""
		for n in range(4):
			s += "%02x" % bs[i+3-n]
		print("    .word 0x%s" % s)

# Define the filter size
if len(sys.argv) > 1:
	filter_size = int(sys.argv[1])
	# Filter size must be odd
	assert(filter_size % 2 == 1), "The filter size must be an odd integer number"
else:
	filter_size = 1
	num_filter = 1

# Input image
M = 32
N = 32
L = 128
padding = int(filter_size/2)
M_pad = M + 2*padding
N_pad = N + 2*padding
L_pad = L + 0*padding
assert(M % 4 == 0), "Output image dimension must be divisible by 16, pad the input image accordingly"
assert(N % 4 == 0), "Output image dimension must be divisible by 16, pad the input image accordingly"
#assert(L / filter_size != 1), "Depth of tensor must be same depth as the filter"

# Generate a random int64 input padded image
tensor = np.around(rand_tensor(1, L_pad, M_pad, N_pad,1)).astype(np.uint8)>>6
np.random.shuffle(tensor.flat)

# Generate a random int64 filter
gen_filter = np.around(rand_tensor(num_filter, L, filter_size, filter_size, 0)).astype(np.uint8)>>4
np.random.shuffle(gen_filter.flat)

# Create the empty o matrix
empty_o = np.zeros((L, M, N)).astype(np.uint8)




# Calculate the output matrix
result = np.zeros((num_filter, M, N)).astype(np.uint8) #Num_filter x M x N
for num in range(num_filter):
	for plane in range(L):
		result[num,:,:] += np.around(scipy.signal.convolve2d(np.flip(gen_filter[num,plane,:,:]), tensor[0,plane,:,:], 'valid')).astype(np.uint8) # https://stackoverflow.com/questions/41613155/what-does-scipy-signal-convolve2d-calculate
		

#reshape into NWHC format

result_NWHC = np.zeros((M, N, num_filter)).astype(np.uint8) 
tensor_NWHC = np.zeros((1, M_pad, N_pad, L)).astype(np.uint8) 
gen_filter_NWHC = np.zeros((num_filter,filter_size, filter_size, L_pad)).astype(np.uint8) 

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
checksum = np.sum(result, dtype=np.uint8)

original_stdout = sys.stdout
sys.stdout = sys.stderr # Redirect the standard output to the standard error.

# Print information on display
print("----------------------------------------")
print(" Tensors before transposition into NHWC ")
print("----------------------------------------")

vhex = np.vectorize(hex)
print("Image: dim = 1 x {0} x {1} x {2} \n".format(L_pad, N_pad, M_pad))
print(tensor)
print("\n")
print("Filter: dim = {0} x {1} x {2} x {3}\n".format(num_filter, L_pad, filter_size, filter_size))
print(gen_filter)
print("\n")
print("Results: dim = {0} x {1} x {2} \n".format(num_filter, M, N))
print(result)
print("\n")



print("---------------------------------------")
print(" Tensors after transposition into NHWC ")
print("---------------------------------------")

print("Image: dim = 1 x {0} x {1} x {2} \n".format(N_pad, M_pad, L_pad))
print(tensor_NWHC)
print("\n")
print("Filter: dim = {0} x {1} x {2} x {3}\n".format(num_filter, filter_size, filter_size, L_pad))
print(gen_filter_NWHC)
print("\n")
print("Results: dim = {0} x {1} x {2} \n".format(M, N, num_filter))
print(result_NWHC)
print("\n")

sys.stdout = original_stdout # Reset the standard output to its original value

# Print information on file
print(".section .data,\"aw\",@progbits")
emit_8b("M", np.array(M, dtype=np.uint64))
emit_8b("N", np.array(N, dtype=np.uint64))
emit_8b("L", np.array(L, dtype=np.uint64))
emit_8b("F", np.array(filter_size, dtype=np.uint64))
emit_8b("K", np.array(num_filter, dtype=np.uint64))
emit_8b("i", tensor_NWHC, 'NR_LANES*4')
emit_8b("f", gen_filter_NWHC, 'NR_LANES*4')
emit_8b("o", empty_o, 'NR_LANES*4')
emit_8b("golden_o", result_NWHC, 'NR_LANES*4')
emit_8b("o_checksum", checksum)


