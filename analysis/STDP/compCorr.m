function v = compCorr(x,m)
% computes autocorrelation sequence

if size(x,1) == 1 % make sure x is a column vector
    x = x';
end
size(x)

av = mean(x);
fprintf('average = %f\n',av);
x = x - av;

n = length(x);
fprintf('length(x) = %d\n',n);

m = min([n,m]);

v = zeros(m,1);

for k=0:(m-1)
    v(k+1) = (x(1:(n-k))' * x((1+k):n) ) / (n-k);
end

v = v ./ v(1);