function [ target, clutter ] = pv_parseBMP( filename )
% Return a list of the pixels corresponding to target and clutter in a color-coded BMP file
% of a gray-scale image.
% black (0,0,0) is background
% red  > 0 denotes target pixel
% green value encodes target index 
% (as integer multiple of 255/num_targets)
% blue > 0 denotes clutter pixel. 
%  

global N NX NY 


if exist(filename, 'file')
    imshow(BMP_path);
    pixels = imread(filename);
    [NX NY num_colors] = size(pixels);
    N = NX * NY;
else
    target = find(ones(NY, NX));
    clutter = [];
    return
end

if num_colors == 3
   clutter = find( (pixels(:,:,3) > 0) & (pixels(:,:,1) == 0) );
   for i_target = 1:num_targets
       target{i_target} = find( ( pixels(:,:,1) > 0 ) & ( floor( pixels(:,:,3) * num_targets / 255 ) == ( i_target - 1 ) ) );
   end
else
    clutter = [];
    target = find(pixels');
end
