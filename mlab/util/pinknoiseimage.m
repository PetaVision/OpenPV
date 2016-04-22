function [Yre,Yim] = pinknoiseimage(m,n)
% [Yre,Yim] = pinknoiseimage(m,n)
% Yre and Yim will be independent n-by-m pinknoise images
M = ceil(m/2)*2;
N = ceil(n/2)*2;

assert(mod(M,2)==0 && mod(N,2)==0);


X = repmat((-N/2:N/2-1),M,1);
Y = repmat((-M/2:M/2-1)',1,N);
rsq = X.^2 + Y.^2;

randphase = exp(2i*pi*rand(M,N));

oneoverf = rsq.^(-1);
oneoverf(rsq==0)=0;
fhat = oneoverf .* randphase;

f = ifft2(fhat);

chkbd = repmat([1 -1; -1 1],M/2,N/2);
Yre = real(f).*chkbd;
Yre = Yre(1:m,1:n);
if nargout > 1
    Yim = imag(f).*chkbd;
    Yim = Yim(1:m,1:n);
end


