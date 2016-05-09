function [reconstruction,kernelft,kernelinverseft] = deconvolvemirrorbc(convolvedimage, kernel, reg)
% A function to invert a transformation obtained by convolving a 2-D image
% with a kernel using mirror boundary conditions.
%
% [reconstruction,kernelft,kernelinverseft] = invertconvmirrorbc(convolvedimage, kernel, reg)
%
% convolvedimage is an m-by-n image
% kernel is a p-by-q kernel
% reg is a regularization parameter for computing the inverse transform:
%    1/x is replaced by 1/sqrt(x^2+reg^2)*sign(x)
%
% reconstruction is the solution to convolvedimage = reconstruction*kernel,
%     where '*' is the convolution operator.
% kernelft is the 2m-by-2n representation of the kernel in fourier space.
%     The doubling of the dimension is because the function uses
%     reflections to make mirror boundary conditions equivalent to circular
%     boundary conditions.
% kernelinverseft is the deconvolution in fourier space.  It is generally
%     1./kernelft, but where kernelft==0, kernelinverseft is also zero.

if ~exist('reg','var')
    reg = 0;
end

dogconvrefl = [convolvedimage convolvedimage(:,end:-1:1); convolvedimage(end:-1:1,:) convolvedimage(end:-1:1, end:-1:1)];
[gy, gx] = size(dogconvrefl);

[ysize, xsize] = size(kernel);
xctr = (xsize+1)/2;
yctr = (ysize+1)/2;

kernelfull = zeros(gy,gx);
kernelfull([gy-yctr+2:gy 1:yctr],[gx-xctr+2:gx 1:xctr])=kernel;
kernelft = real(fft2(kernelfull));

% fmy = exp(2*pi*1i/gy*(0:gy-1)');
% 
% kernelft = zeros(gy, gx);
% for n=(1:xsize)-xctr
%     for m=(1:ysize)-yctr
%         kernelft = kernelft + kernel(m+yctr,n+yctr) * fmy.^m * fmx.^n;
%     end
% end
% kernelft = real(kernelft);

if reg==0
    kernelinverseft = 1./kernelft;
else
    kernelinverseft = 1./sqrt(kernelft.^2 + reg^2).*sign(kernelft);
end

kernelinverseft(kernelft==0) = 0;

reconstructionrefl = real(ifft2(fft2(dogconvrefl).*kernelinverseft));
reconstruction = reconstructionrefl(1:gy/2, 1:gx/2);
%reconstruction = reconstructionrefl(1:gx/2, 1:gy/2);
