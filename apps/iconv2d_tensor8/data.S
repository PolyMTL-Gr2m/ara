.section .data,"aw",@progbits
.global M
.align 3
M:
    .word 0x00000008
    .word 0x00000000
.global N
.align 3
N:
    .word 0x00000008
    .word 0x00000000
.global L
.align 3
L:
    .word 0x00000001
    .word 0x00000000
.global F
.align 3
F:
    .word 0x00000003
    .word 0x00000000
.global K
.align 3
K:
    .word 0x00000001
    .word 0x00000000
.global i
.align NR_LANES*4
i:
    .word 0x80000080
    .word 0x80808000
    .word 0x80808080
    .word 0x00000080
    .word 0x80008080
    .word 0x00000080
    .word 0x00800000
    .word 0x80000000
    .word 0x80000000
    .word 0x00008080
    .word 0x00808080
    .word 0x80008000
    .word 0x00808000
    .word 0x00008000
    .word 0x80800000
    .word 0x80800080
    .word 0x00000000
    .word 0x80000080
    .word 0x80800000
    .word 0x80008080
    .word 0x80808000
    .word 0x00000000
    .word 0x80000080
    .word 0x00000080
    .word 0x80000000
.global f
.align NR_LANES*4
f:
    .word 0x776e847b
    .word 0x71817e87
    .word 0x00000074
.global o
.align NR_LANES*4
o:
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
.global golden_o
.align NR_LANES*4
golden_o:
    .word 0x80800000
    .word 0x00008000
    .word 0x00800080
    .word 0x00008080
    .word 0x00808000
    .word 0x00008000
    .word 0x80008080
    .word 0x80000080
    .word 0x00000080
    .word 0x80008000
    .word 0x80000080
    .word 0x00000000
    .word 0x80808080
    .word 0x00000000
    .word 0x80808000
    .word 0x00808080
.global o_checksum
.align 3
o_checksum:
    .word 0xfffff100
    .word 0xffffffff
