## Generator (most times) function to return a range of values with a float step
def frange(start, stop, step):
    if start == stop or stop == 0 or step == 0:
        yield start
    r = start
    while r < stop:
        yield r
        r += step

## Function to uncomment a block of code from the given start line to the first time a blank line is found
def uncomment_block(start_line_num, output_lines):
    for line_num in range(start_line_num,len(output_lines)):
        com_line = output_lines[line_num]
        if com_line == '\n':
            return output_lines
        output_lines[line_num] = com_line[2:]
    return output_lines
