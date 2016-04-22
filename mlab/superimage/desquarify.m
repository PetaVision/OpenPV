# Convert matrix to vector by row:
# 
# a b c
# d e f  -->  a b c d e f g h i
# g h i 

function out = desquarify(in)
out = [];
for i = 1:size(in)(1)
  out = [out in(i,:)];
endfor
endfunction