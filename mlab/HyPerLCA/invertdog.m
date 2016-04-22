function reconstruction = invertdog(dogconv, dogkernel)

[gy, gx] = size(dogconv);

[ysize, xsize] = size(dogkernel);
xctr = (xsize+1)/2;
yctr = (ysize+1)/2;

fmx = exp(2*pi*1i/gx*(0:gx-1));
fmy = exp(2*pi*1i/gy*(0:gy-1)');

kernel = zeros(gy, gx);
for n=(1:xsize)-xctr
    for m=(1:ysize)-yctr
        kernel = kernel + dogkernel(m+yctr,n+yctr) * fmy.^m * fmx.^n;
    end
end

kernelinverse = real(1./kernel);
kernelinverse(1) = 0;

reconstruction = real(ifft2(fft2(dogconv).*kernelinverse));
