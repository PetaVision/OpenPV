function [ target, clutter ] = pvp_parseBMP( filename, plot_input_image )
% Return a list of the pixels corresponding to target and clutter in a color-coded BMP file
% of a gray-scale image.
% black (0,0,0) is background
% red  > 0 denotes target pixel
% green value encodes target index 
% (as integer multiple of 255/num_targets)
% blue > 0 denotes clutter pixel. 
%  

global N_image NROWS_image NCOLS_image 
global num_targets 

if nargin < 2
    plot_input_image = 1;
end

if exist(filename, 'file')
    pixels = imread(filename);
    [NROWS_image NCOLS_image num_colors] = size(pixels);
    N_image = NROWS_image * NCOLS_image;
else
    target = find(ones(NROWS_image, NCOLS_image));
    clutter = [];
    return
end

target = cell(num_targets,1);
if num_colors == 3
   clutter = find( ( pixels(:,:,3) > 0 ) .* ( pixels(:,:,1) == 0 ) );
   for i_target = 1:num_targets
       target{i_target} = find( ( pixels(:,:,1) > 0 ) & ( pixels(:,:,3) == 0 ) & ( floor( pixels(:,:,3) * num_targets / 255 ) == ( i_target - 1 ) ) );
   end
else
    clutter = [];
    target = find(pixels');
end


if plot_input_image
    figure('Name', 'input image');
    imagesc(pixels);
end