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
		
def emit_32b(name, array, alignment='3'):
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

# Generate a random int64 input padded image
tensor8  = np.around(rand_tensor(1, L, M, N, 1)).astype(np.uint8)
tensor16 = np.around(rand_tensor(1, L, M, N, 1)).astype(np.uint16)
tensor32 = np.around(rand_tensor(1, L, M, N, 1)).astype(np.uint32)
tensor64 = np.around(rand_tensor(1, L, M, N, 1)).astype(np.uint64)


np.random.shuffle(tensor8.flat)
np.random.shuffle(tensor16.flat)
np.random.shuffle(tensor32.flat)
np.random.shuffle(tensor64.flat)

# Create the empty o tensors
empty_o8  = np.zeros((L, M, N)).astype(np.uint8)
empty_o16 = np.zeros((L, M, N)).astype(np.uint16)
empty_o32 = np.zeros((L, M, N)).astype(np.uint32)
empty_o64 = np.zeros((L, M, N)).astype(np.uint64)


# convert into NWHC format
result8  = np.zeros((M, N, L)).astype(np.uint8) #L x M x N
result16 = np.zeros((M, N, L)).astype(np.uint16) #L x M x N
result32 = np.zeros((M, N, L)).astype(np.uint32) #L x M x N
result64 = np.zeros((M, N, L)).astype(np.uint64) #L x M x N

for i in range(0, L):
	for j in range(0, M):
		for k in range(0, N):
		
			result8[k,j,i] = tensor8[0,i,k,j]
			
for i in range(0, L):
	for j in range(0, M):
		for k in range(0, N):
		
			result16[k,j,i] = tensor16[0,i,k,j]

for i in range(0, L):
	for j in range(0, M):
		for k in range(0, N):
		
			result32[k,j,i] = tensor32[0,i,k,j]

for i in range(0, L):
	for j in range(0, M):
		for k in range(0, N):
		
			result64[k,j,i] = tensor64[0,i,k,j]

# Calculate a checksum
checksum = np.sum(result64, dtype=np.uint64)

# Print information on display
#vhex = np.vectorize(hex)
#print("Image: dim = 1 x {0} x {1} x {2} \n".format(L, M, N))
#print(tensor8)
#print("\n")
#print("Results: dim = {0} x {1} x {2} \n".format(L, M, N))
#print(result8)
#print("\n")
#print("\n")
#print("\n")
#print("Image: dim = 1 x {0} x {1} x {2} \n".format(L, M, N))
#print(tensor16)
#print("\n")
#print("Results: dim = {0} x {1} x {2} \n".format(L, M, N))
#print(result16)
#print("\n")
#print("\n")
#print("\n")
#print("Image: dim = 1 x {0} x {1} x {2} \n".format(L, M, N))
#print(tensor32)
#print("\n")
#print("Results: dim = {0} x {1} x {2} \n".format(L, M, N))
#print(result32)
#print("\n")
#print("\n")
#print("\n")
#print("Image: dim = 1 x {0} x {1} x {2} \n".format(L, M, N))
#print(tensor64)
#print("\n")
#print("Results: dim = {0} x {1} x {2} \n".format(L, M, N))
#print(result64)
#print("\n")
#print(checksum)

# Print information on file
print(".section .data,\"aw\",@progbits")
emit_8b("M", np.array(M, dtype=np.uint64))
emit_8b("N", np.array(N, dtype=np.uint64))
emit_8b("L", np.array(L, dtype=np.uint64))
emit_8b("i8", tensor8, 'NR_LANES*4')
emit_16b("i16", tensor16, 'NR_LANES*4')
emit_32b("i32", tensor32, 'NR_LANES*4')
emit_64b("i64", tensor64, 'NR_LANES*4')
emit_8b("o8", empty_o8, 'NR_LANES*4')
emit_16b("o16", empty_o16, 'NR_LANES*4')
emit_32b("o32", empty_o32, 'NR_LANES*4')
emit_64b("o64", empty_o64, 'NR_LANES*4')
emit_8b("golden_o8", result8, 'NR_LANES*4')
emit_16b("golden_o16", result16, 'NR_LANES*4')
emit_32b("golden_o32", result32, 'NR_LANES*4')
emit_64b("golden_o64", result64, 'NR_LANES*4')
emit_64b("o_checksum", checksum)


