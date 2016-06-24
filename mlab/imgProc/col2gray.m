function ret = col2gray(im)
%apply the luminance equation to the image

  if size(im,3) == 3
    ret = .2989*im(:,:,1) ...
        +.5870*im(:,:,2) ...
        +.1140*im(:,:,3);
  else
    ret = im;
  endif

end
