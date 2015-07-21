function out = readidx3data(in)
% out = readixd1data(in)
% in is the char array of data that should be in the idx1 data format
% out is the uint8 array after verifying and removing the header
% It exits with an error if in does not satisfy the idx1 data format

assert(all(in(1:4)==[0,0,8,3]));
m = in(9:12)*256.^(3:-1:0)';
n = in(13:16)*256.^(3:-1:0)';
p = (numel(in)-16)/(m*n);
assert(in(5:8)*256.^(3:-1:0)'==p);
out = uint8(reshape(in(17:end),m,n,p));
