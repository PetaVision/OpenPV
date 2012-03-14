function ssi = shannonsparsityindex(x)
% ssi = shannonsparsityindex(x)
% If x is a column vector, ssi is a measure of sparsity derived from the
% Shannon diversity index.  If x is all zeros except for a single
% element, ssi = 1.  If all x are equal, ssi = 0.
% If H is the Shannon diversity index of an N-vector x, ssi = 1 - H/log(N).
%
% If x is a matrix, ssi is the row vector whose elements are the Shannon
% sparsity index of the columns of x.
N = size(x,1);
S = sum(x,1);
p = bsxfun(@rdivide,x,S);
plogp = p.*log(p);
plogp(isnan(plogp))=0;
sdi = sum(-plogp,1);
ssi = 1 - sdi/log(N);