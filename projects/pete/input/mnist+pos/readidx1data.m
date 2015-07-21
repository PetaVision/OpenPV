function out = readidx1data(in)
% out = readixd1data(in)
% in is the char array of data that should be in the idx1 data format
% out is the uint8 array after verifying and removing the header
% It exits with an error if in does not satisfy the idx1 data format

assert(all(in(1:4)==[0,0,8,1]));
assert(in(5:8)*256.^(3:-1:0)'==numel(in)-8);
out = uint8(in(9:end));
