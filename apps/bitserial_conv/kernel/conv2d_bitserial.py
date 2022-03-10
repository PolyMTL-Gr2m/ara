import math
import numpy as np
import torch
import torch.nn as nn
from texttable import Texttable

class SimpleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, groups, dilation, weights):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)
        self.conv1.weight.data = torch.from_numpy(weights)
    def forward(self, x):
        out = self.conv1(x)
        return out

def check_result(output, golden):
    if (output.shape != golden.shape): 
        print("Output size does not match golden output")
        return False
    else:
        cnt = 0
        for val,gold in zip(output.flatten(), golden.flatten()):
            if val!=gold:
                print("output[{}]:{} does not macth golden[{}]:{}".format(cnt, val, cnt, gold))
                return False
            cnt += 1
    return True

def bitpack(matrix, DATA_WIDTH, dlen, prec):
    if DATA_WIDTH==64:
        packed_data = np.zeros(math.ceil(len(matrix)/DATA_WIDTH)*prec).astype(np.int64)
    elif DATA_WIDTH==32:
        packed_data = np.zeros(math.ceil(len(matrix)/DATA_WIDTH)*prec).astype(np.int32)
    elif DATA_WIDTH==16:
        packed_data = np.zeros(math.ceil(len(matrix)/DATA_WIDTH)*prec).astype(np.int16)
    elif DATA_WIDTH==8:
        packed_data = np.zeros(math.ceil(len(matrix)/DATA_WIDTH)*prec).astype(np.int8)
    elif DATA_WIDTH==4:
        packed_data = np.zeros(math.ceil(len(matrix)/DATA_WIDTH)*prec).astype(np.int8)
    else:
        print("Unsupported element length: {} bits".format(DATA_WIDTH))
        return None
    p_ptr = 0
    # make sure input is integer
    matrix = [int(val) for val in matrix]
    for i in range(0,dlen, DATA_WIDTH):
        for el in range(0, DATA_WIDTH):
            for bit_pos in range(0, prec):
                bit_idx = p_ptr+bit_pos
                if i+el >=dlen:
                    break
                data = (matrix[i+el] >> bit_pos) & 0x1
                packed_data[bit_idx] <<= 1
                packed_data[bit_idx] = (packed_data[bit_idx] | data)
        p_ptr += prec

    if DATA_WIDTH==64:
        return packed_data.astype(np.uint64)
    elif DATA_WIDTH==32:
        return packed_data.astype(np.uint32)
    elif DATA_WIDTH==16:
        return packed_data.astype(np.uint16)
    elif DATA_WIDTH==8:
        return packed_data.astype(np.uint8)
    elif DATA_WIDTH==4:
        return packed_data.astype(np.uint8)

def im2col_get_pixel(im,  height,  width, row, col, channel, pad):
    global count
    row -= pad
    col -= pad
    if (row < 0 or col < 0 or row >= height or col >= width):
         return 0
    return im[col + width*(row + height*channel)]

def im2col(data_im, channels, height, width, ksize, stride, pad):
    c,h,w = 0,0,0
    height_col = int((height + 2*pad - ksize) / stride + 1)
    width_col = int((width + 2*pad - ksize) / stride + 1)
    channels_col = int(channels * ksize * ksize)
    output = np.zeros((1,channels_col*height_col*width_col))
    for c in  range(channels_col):
        w_offset = int(c % ksize)
        h_offset = int((c / ksize) % ksize)
        c_im = int(c / ksize / ksize)
        for h in range(height_col):
            for w in range(width_col):
                im_row = h_offset + h * stride
                im_col = w_offset + w * stride
                col_index = (c * height_col + h) * width_col + w
                output[0,col_index] = im2col_get_pixel(data_im, height, width, im_row, im_col, c_im, pad)
    return output

def popcnt(a):
    return bin(a).count("1")

def bitserial_gemm(inputs, weights, packed_row_len, wprec, aprec, oshape, wshape, DATA_WIDTH):
    '''
        Example: 
            - activation:[16x9]
            - weight: [1x9]
            - DATA_WIDTH: 4
            - aprec: 2
            - wprec: 2
        Weight: 
            [0. 0. 2. 1. 2. 3. 3. 2. 0.]
        Activation:
            [0. 0. 0. 0. 0. 3. 0. 3. 3.]
            [0. 0. 0. 0. 3. 1. 3. 3. 3.]
            [0. 0. 0. 3. 1. 0. 3. 3. 3.]
            [0. 0. 0. 1. 0. 0. 3. 3. 0.]
            [0. 0. 3. 0. 3. 3. 0. 1. 3.]
            [0. 3. 1. 3. 3. 3. 1. 3. 1.]
            [3. 1. 0. 3. 3. 3. 3. 1. 2.]
            [1. 0. 0. 3. 3. 0. 1. 2. 0.]
            [0. 3. 3. 0. 1. 3. 0. 0. 3.]
            [3. 3. 3. 1. 3. 1. 0. 3. 2.]
            [3. 3. 3. 3. 1. 2. 3. 2. 0.]
            [3. 3. 0. 1. 2. 0. 2. 0. 0.]
            [0. 1. 3. 0. 0. 3. 0. 0. 0.]
            [1. 3. 1. 0. 3. 2. 0. 0. 0.]
            [3. 1. 2. 3. 2. 0. 0. 0. 0.]
            [1. 2. 0. 2. 0. 0. 0. 0. 0.]
        - First output element:
            - Using GEMM:
                [0. 0. 2. 1. 2. 3. 3. 2. 0.]
                [0. 0. 0. 0. 3. 1. 3. 3. 3.]
                = 6+3+9+6
            - Using bit serial:
                +---+---------------+-----------------+------------------+
                |idx|    INPUT      |   WEIGHT        | Partial Result   |
                +---+---------------+-----------------+------------------+
                | 0 | [   0]: 0000  |   [   0]: 0001  | 0                |
                | 1 | [   1]: 0000  |   [   1]: 0010  |                  |
                +---+---------------+-----------------+------------------+
                | 2 | [   2]: 0101  |   [   2]: 0110  | 1+4+2+8 = 15     |
                | 3 | [   3]: 0101  |   [   3]: 1111  |                  |
                +---+---------------+-----------------+------------------+
                | 4 | [   4]: 1000  |   [   4]: 0000  | 0                |
                | 5 | [   5]: 1000  |   [   5]: 0000  |                  |
                +---+---------------+-----------------+------------------+
                | 6 | [   6]: 0111  |   [   0]: 0001  | 5                |
                | 7 | [   7]: 0101  |   [   1]: 0010  |                  |
                +---+---------------+-----------------+------------------+
                | 8 | [   8]: 1100  |   [   2]: 0110  | 1+4+2+8=15       |
                | 9 | [   9]: 1100  |   [   3]: 1111  |                  |
                +---+---------------+-----------------+------------------+
                |10 | [  10]: 0110  |   [   4]: 0000  | 0                |
                |11 | [  11]: 0100  |   [   5]: 0000  |                  |
                +---+---------------+-----------------+------------------+
    '''
    
    out_ch,ow,oh = oshape
    _, in_ch, ksize, ksize = wshape
    output = np.zeros(out_ch*ow*oh)
    # import ipdb as pdb; pdb.set_trace()   
    optr = 0
    weights = weights.reshape(out_ch, wprec*math.ceil(in_ch*ksize*ksize/DATA_WIDTH))
    num_mul = 0
    num_add = 0
    num_popcnt = 0
    num_shifts = 0
    num_mem_ops = 0
    for oc in range(out_ch):
        weight_oc = weights[oc]
        for i in range(0, len(inputs), aprec*packed_row_len):
            # for j in range(0, len(weights), wprec*packed_row_len):
                o_acc = 0
                for k in range(0, packed_row_len):
                    for ap in range(0, aprec):
                        for wp in range(0, wprec):
                            # import ipdb as pdb; pdb.set_trace()
                            aidx = ap+k*aprec + i
                            widx = wp+k*wprec
                            # print("i:{}, oc:{}, k:{}, ap:{}, wp:{},  a[{}] w[{}]".format(i,oc,k,ap,wp,aidx,widx))
                            # print("{:04b}".format(inputs[i*k+ap]))
                            # print("{:04b}".format(weights[j*k+wp]))
                            o_acc += popcnt((inputs[aidx]&weight_oc[widx])) << (ap+wp)
                            num_add += 2
                            num_popcnt += 1
                            num_shifts += 1
                            num_mem_ops += 2
                # print("[{:4d}]: {:4d}".format(optr, o_acc))
                # import ipdb as pdb; pdb.set_trace()
                # print("{}\n".format(o_acc))
                output[optr] = o_acc
                optr += 1
        # import ipdb as pdb; pdb.set_trace()
        # pass
    return output, [num_mul,num_add,num_popcnt,num_shifts,num_mem_ops]

def print_packed_data(packed_data):
    for id, val in enumerate(packed_data):
        if DATA_WIDTH==64:
            pos_val = val if val>0 else (np.uint64(val)&np.uint64(0xffffffffffffffff))
            print('[{0:4d}]: {1:064b}'.format(id, pos_val))
        elif DATA_WIDTH==32:
            pos_val = val if val>0 else (np.uint32(val)&np.uint32(0xffffffff))
            print('[{0:4d}]: {1:032b}'.format(id, pos_val))
        elif DATA_WIDTH==16:
            pos_val = val if val>0 else (np.uint16(val)&np.uint16(0xffff))
            print('[{0:4d}]: {1:016b}'.format(id, pos_val))
        elif DATA_WIDTH==8:
            pos_val = val if val>0 else (np.uint8(val)&np.uint8(0xff))
            print('[{0:4d}]: {1:08b}'.format(id, pos_val))
        elif DATA_WIDTH==4:
            pos_val = val if val>0 else (np.uint8(val)&np.uint8(0xff))
            print('[{0:4d}]: {1:04b}'.format(id, pos_val))

if __name__ == '__main__':
    np.random.seed(seed=0)
    batch = 1
    iw = 28
    ih = 28
    in_ch = 32
    out_ch = 64
    ksize  = 3
    stride = 1
    padding = 1
    groups = 1
    dilation = 1
    prec = 4
    DATA_WIDTH = 8
    VLEN = 1
    aprec = prec
    wprec = prec

    # Initialize input and weight
    max_int = (2**prec) - 1 
    inputs = np.random.randint(max_int+1, size=(batch, in_ch, iw, ih)).astype(np.float32)
    weights = np.random.randint(max_int+1, size=(out_ch, in_ch, ksize, ksize)).astype(np.float32)
    torch_conv = SimpleConv(in_ch, out_ch, ksize, stride, padding, groups, dilation, weights)

    golden_output = torch_conv(torch.from_numpy(inputs))
    #================================================================
    # Computing Convolution with im2col method with normal GEMM:
    #================================================================
    inputs_flatten = inputs.flatten()
    inputs_im2col = im2col(inputs_flatten, in_ch, ih, iw, ksize, stride, padding)
    inputs_im2col = inputs_im2col.reshape(in_ch*ksize*ksize,iw*ih)
    weights_mat = weights.reshape(out_ch,in_ch*ksize*ksize)
    output_im2col = np.dot(weights_mat, inputs_im2col)
    num_mul = np.prod(output_im2col.shape) * weights_mat.shape[1]
    num_add = np.prod(output_im2col.shape) * weights_mat.shape[1]
    im2col_stats = [num_mul, num_add, 0, 0, "?"]
    
    ow = int((iw + 2*padding - ksize) / stride + 1)
    oh = int((ih + 2*padding - ksize) / stride + 1)
    output = output_im2col.reshape(1, out_ch, ow, oh)
    if check_result(output, golden_output):
        print("im2col conv matches pytorch conv")
    else:
        print("im2col conv does not match pytorch conv")

    #================================================================
    # Computing Convolution with im2col method with bitserial GEMM:
    #================================================================
    # import ipdb as pdb; pdb.set_trace()
    # inputs_im2col = inputs_im2col.reshape(ksize*ksize,16).transpose()
    inputs_im2col = inputs_im2col.reshape(in_ch*ksize*ksize,iw*ih).transpose()
    if (in_ch*ksize*ksize)%DATA_WIDTH != 0:
        i_org_shape = inputs_im2col.shape
        inputs_im2col_fixed = np.zeros((i_org_shape[0], i_org_shape[1]+DATA_WIDTH-((in_ch*ksize*ksize)%DATA_WIDTH)))
        for id, row in enumerate(inputs_im2col):
            inputs_im2col_fixed[id][0:i_org_shape[1]] = row
        w_org_shape = weights_mat.shape
        weights_mat_fixed = np.zeros((w_org_shape[0], w_org_shape[1]+DATA_WIDTH-((in_ch*ksize*ksize)%DATA_WIDTH)))
        for id, row in enumerate(weights_mat):
            weights_mat_fixed[id][0:w_org_shape[1]] = row
        inputs_im2col = inputs_im2col_fixed
        weights_mat = weights_mat_fixed
    # import ipdb as pdb; pdb.set_trace()
    bit_packed_inputs = bitpack(inputs_im2col.flatten(), DATA_WIDTH, len(inputs_im2col.flatten()), aprec)
    bit_packed_weights = bitpack(weights_mat.flatten(), DATA_WIDTH, len(weights_mat.flatten()), wprec)

    # import ipdb as pdb; pdb.set_trace()
    packed_row_len = int(inputs_im2col.shape[1]/DATA_WIDTH)

    # print_packed_data(bit_packed_inputs)
    # print_packed_data(bit_packed_weights)
    # print("\n")
    # # import ipdb as pdb; pdb.set_trace()
    # print(inputs_im2col)
    # print(weights_mat)
    # print(golden_output)
    # import ipdb as pdb; pdb.set_trace()
    output_bitserial, bit_serial_stats = bitserial_gemm(bit_packed_inputs, bit_packed_weights, packed_row_len, wprec, aprec, [out_ch,ow,oh], [out_ch, in_ch, ksize, ksize], DATA_WIDTH)
    output = output_bitserial.reshape(1, out_ch, ow, oh)
    t = Texttable(max_width=160)
    t.add_row(['Type', 'Num Mul', 'Num Add', 'Num Popcnt', 'Num Shifts', 'Num MemOps'])
    t.add_row(["Bit-Serial", *bit_serial_stats])
    t.add_row(["Im2Col GEMM", *im2col_stats])

    if check_result(output, golden_output):
        print("bitserial conv matches pytorch conv")
    else:
        print("bitserial conv does not match pytorch conv")
    #================================================================
    # Printing Results:
    #================================================================
    print("Computation cost for:")
    print("\t Input Shape: {}x{}x{}".format(in_ch, iw, ih))
    print("\t Weight Shape: {}x{}x{}x{}".format(out_ch, in_ch, ksize, ksize))
    print("\t Data Width: {}".format(DATA_WIDTH))
    print("\t Vector Length: {}".format(VLEN))
    print("\t aprec: {}".format(aprec))
    print("\t wprec: {}".format(wprec))
    print(t.draw())
