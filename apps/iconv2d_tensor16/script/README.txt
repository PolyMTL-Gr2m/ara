Python script to generate a memory file

Command to use and arguments

python3 gen_data.py <wide> <Memory_layout> <Filter_Size> <Number_Filter>  >../data.S




# Exemple for memory file with widening output (8b input -> 16b output) with 2 3x3 filter on NCHW memory layout #

python3 gen_data.py wide NCHW 3 2 >../data.S


If default parameters want to be used, pass "null" as an arguments

