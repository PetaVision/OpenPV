function [reconstruction,kernelft,kernelinverseft] = deconvolvemirrorbc(convolvedimage, kernel)
% A function to invert a transformation obtained by convolving a 2-D image
% with a kernel using mirror boundary conditions.
%
% [reconstruction,kernelft,kernelinverseft] = invertconvmirrorbc(convolvedimage, kernel)
%
% convolvedimage is an m-by-n image
% kernel is a p-by-q kernel
%
% reconstruction is the solution to convolvedimage = reconstruction*kernel,
%     where '*' is the convolution operator.
% kernelft is the 2m-by-2n representation of the kernel in fourier space.
%     The doubling of the dimension is because the function uses
%     reflections to make mirror boundary conditions equivalent to circular
%     boundary conditions.
% kernelinverseft is the deconvolution in fourier space.  It is generally
%     1./kernelft, but where kernelft==0, kernelinverseft is also zero.

dogconvrefl = [convolvedimage convolvedimage(:,end:-1:1); convolvedimage(end:-1:1,:) convolvedimage(end:-1:1, end:-1:1)];
[gy, gx] = size(dogconvrefl);

[ysize, xsize] = size(kernel);
xctr = (xsize+1)/2;
yctr = (ysize+1)/2;

fmx = exp(2*pi*1i/gx*(0:gx-1));
fmy = exp(2*pi*1i/gy*(0:gy-1)');

kernelft = zeros(gy, gx);
for n=(1:xsize)-xctr
    for m=(1:ysize)-yctr
        kernelft = kernelft + kernel(m+yctr,n+yctr) * fmy.^m * fmx.^n;
    end
end
kernelft = real(kernelft);

kernelinverseft = 1./kernelft;
kernelinverseft(kernelft==0) = 0;

reconstructionrefl = real(ifft2(fft2(dogconvrefl).*kernelinverseft));
reconstruction = reconstructionrefl(1:gx/2, 1:gy/2);
