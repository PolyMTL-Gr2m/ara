#!/usr/bin/env python3

import numpy as np
import scipy.signal
import sys


def rand_tensor(O, L, N, M, seed):
	return np.arange(seed, seed+O*N*M*L, dtype=np.float64).reshape(O, L, N, M) * 3.141

def emit_16b(name, array, alignment='3'):
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
	filter_size = 3
	num_filter = 1

# Input image
M = 224
N = 224
L = 3
padding = int(filter_size/2)
M_pad = M + 2*padding
N_pad = N + 2*padding
L_pad = L + 0*padding
assert(M % 4 == 0), "Output image dimension must be divisible by 4, pad the input image accordingly"
assert(N % 4 == 0), "Output image dimension must be divisible by 4, pad the input image accordingly"
#assert(L / filter_size != 1), "Depth of tensor must be same depth as the filter"

# Generate a random int64 input padded image
tensor = np.around(rand_tensor(1, L_pad, M_pad, N_pad,1)).astype(np.uint16)>>6
np.random.shuffle(tensor.flat)

# Generate a random int64 filter
gen_filter = np.around(rand_tensor(num_filter, L, filter_size, filter_size, 0)).astype(np.uint16)>>6
np.random.shuffle(gen_filter.flat)

# Create the empty o matrix
empty_o = np.zeros((1, M, N)).astype(np.uint16)


# TESTS#############################################################################
#for num in range(num_filter):
	#gen_filter[num,0,:,:] = np.ones((1, filter_size, filter_size)).astype(np.int64)
	#gen_filter[num,1,:,:] = np.ones((1, filter_size, filter_size)).astype(np.int64)
	#gen_filter[num,2,:,:] = np.ones((1, filter_size, filter_size)).astype(np.int64)
	
#gen_filter[0,0,:,:] = np.zeros((1, filter_size, filter_size)).astype(np.int64)
#gen_filter[0,0,:,:] = np.zeros((1, filter_size, filter_size)).astype(np.int64)
#gen_filter[0,2,:,:] = np.zeros((1, filter_size, filter_size)).astype(np.int64)
	
#for line in range(0, L ):
	#gen_filter[0,0,line,:] = line*np.ones((1, L)).astype(np.int64)
	#gen_filter[0,1,line,:] = line*np.ones((1, L)).astype(np.int64)
	#gen_filter[0,2,line,:] = line*np.ones((1, L)).astype(np.int64)

#tensor[0,0,0,:] = np.zeros((1, N_pad)).astype(np.int64)
#tensor[0,0,1,:] = np.zeros((1, N_pad)).astype(np.int64)
#tensor[0,0,2,:] = np.zeros((1, N_pad)).astype(np.int64)

#tensor[0,1,0,:] = np.zeros((1, N_pad)).astype(np.int64)
#tensor[0,1,1,:] = np.zeros((1, N_pad)).astype(np.int64)
#tensor[0,1,2,:] = np.zeros((1, N_pad)).astype(np.int64)

#tensor[0,2,0,:] = np.zeros((1, N_pad)).astype(np.int64)
#tensor[0,2,1,:] = np.zeros((1, N_pad)).astype(np.int64)
#tensor[0,2,2,:] = np.zeros((1, N_pad)).astype(np.int64)
#tensor[0,2,:,:] = np.zeros((1, M_pad, N_pad)).astype(np.int64)
#tensor[0,2,:,:] = np.zeros((1, M_pad, N_pad)).astype(np.int64)
#for line in range(N_pad):
	#tensor[0,0,line,:] = np.zeros((1, N_pad)).astype(np.int64)
	#tensor[0,1,line,:] = np.zeros((1, N_pad)).astype(np.int64)
	#tensor[0,2,line,:] = np.zeros((1, N_pad)).astype(np.int64)
	
#for line in range(2,int(N_pad) ):
	#tensor[0,0,line,:] = np.zeros((1, N_pad)).astype(np.int64)
	#tensor[0,1,line,:] = np.zeros((1, N_pad)).astype(np.int64)
	#tensor[0,2,line,:] = np.zeros((1, N_pad)).astype(np.int64)
###################################################################################


# Calculate the output matrix
result = np.zeros((num_filter, M, N)).astype(np.uint16) #Num_filter x M x N
for num in range(num_filter):
	for plane in range(L):
		result[num,:,:] += np.around(scipy.signal.convolve2d(np.flip(gen_filter[num,plane,:,:]), tensor[0,plane,:,:], 'valid')).astype(np.uint16) # https://stackoverflow.com/questions/41613155/what-does-scipy-signal-convolve2d-calculate

# Calculate a checksum
checksum = np.sum(result, dtype=np.uint16)

# Print information on display
vhex = np.vectorize(hex)
print("Image: dim = 1 x {0} x {1} x {2} \n".format(filter_size, N_pad, M_pad))
print(tensor)
print("\n")
print("Filter: dim = {0} x {1} x {2} x {3}\n".format(num_filter, L_pad, filter_size, filter_size))
print(gen_filter)
print("\n")
print("Results: dim = {0} x {1} x {2} \n".format(num_filter, M, N))
print(result)
print("\n")
print(checksum)

# Print information on file
print(".section .data,\"aw\",@progbits")
emit_16b("M", np.array(M, dtype=np.uint64))
emit_16b("N", np.array(N, dtype=np.uint64))
emit_16b("L", np.array(L, dtype=np.uint64))
emit_16b("F", np.array(filter_size, dtype=np.uint64))
emit_16b("K", np.array(num_filter, dtype=np.uint64))
emit_16b("i", tensor, 'NR_LANES*4')
emit_16b("f", gen_filter, 'NR_LANES*4')
emit_16b("o", empty_o, 'NR_LANES*4')
emit_16b("golden_o", result, 'NR_LANES*4')
emit_16b("o_checksum", checksum)


