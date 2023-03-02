def encode_instruction(opcode, vm, vs2, vs1, funct3, vd, category):
    # Pack the fields into a single 32-bit word
    instruction = (opcode << 26) | (vm << 25) | (vs2 << 20) | (vs1 << 15) | (funct3 << 12) | (vd << 7) | category

    # Split the 32-bit word into 4 separate 8-bit bytes
    b0 = (instruction >> 24) & 0xff
    b1 = (instruction >> 16) & 0xff
    b2 = (instruction >> 8) & 0xff
    b3 = instruction & 0xff

    # Return the hexadecimal encoding of the instruction in the format used by the .byte directive
    print("asm volatile (\".byte 0x{:02x}, 0x{:02x}, 0x{:02x}, 0x{:02x}\");".format(b3, b2, b1, b0))
    
    
############################################################


TYPE = "OPMVX" # or OPMVV for vector-vector instruction

vs1 = 1 # we use v2 as shifting values
rs1 = 5 # To be changed if you want to use scalar registers (use the value of x_ /!\ not t_ or a_ or s_)
vs2 = 2 # we use v1 as the shifted vector
vd  = 0 # we write back the result in v0


#############################################################
# NO NEED TO MODIFY BELOW THIS LINE

# func6 of vshac specified in ara_dispatcher.sv 
opcode = 0b101110

vm = 0b1;

if(TYPE == "OPMVV"):
	funct3 = 0b010
else:
	funct3 = 0b110
		
category = 0b1010111

	
if(TYPE == "OPMVV"):
	print("//vshac.vv v{0}, v{1}, v{2}".format(vd, vs2, vs1))
	encode_instruction(opcode, vm, vs2, vs1, funct3, vd, category)
else:
	print("//vshac.vx v{0}, v{1}, x{2}".format(vd, vs2, rs1))
	encode_instruction(opcode, vm, vs2, rs1, funct3, vd, category)
