function [background_struct] = fInvKernel(fInv_struct)

  fInv_image = zeros(fInv_struct.image_rect_size);
  fInv_freqs = cell(2, 1);
  num_freqs = zeros(2,1);
  num_freqs(1) = fInv_struct.image_rect_size(1);
  num_freqs(2) = fInv_struct.image_rect_size(2);
  fInv_freqs{1} = [0 : num_freqs(1)-1] / num_freqs(1);
  fInv_freqs{2} = [0 : num_freqs(2)-1] / num_freqs(2);
  fInv_amp = cell(2, 1);
  fInv_phase = exp(i * 2 * pi * rand(fInv_struct.image_rect_size));
  fInv_fft2 = ...
      fInv_phase .* ...
      1 ./ (sqrt( repmat(fInv_freqs{1}.', 1, num_freqs(2)).^2 + repmat(fInv_freqs{2}, num_freqs(1), 1).^2 ));
  fInv_fft2(1,1) = 0;
  fInv_ifft2 = real(ifft2(fInv_fft2));
  fInv_mean = mean(fInv_ifft2(:));
  fInv_std = std(fInv_ifft2(:));
  a_factor = fInv_struct.background_amp * 255 / (fInv_std + (fInv_std==0));
  b_factor = 128 - a_factor * fInv_mean;
%% fix mean and std to gray and user specificed values, respectively
  fInv_image = a_factor * fInv_ifft2 + b_factor;  
  background_struct = struct;
  background_struct.background_image = fInv_image;
  background_struct.num_freqs = num_freqs;
  background_struct.fInv_freqs = fInv_freqs;
  