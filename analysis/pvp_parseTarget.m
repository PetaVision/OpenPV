function [ target, clutter, image, fh ] = ...
    pvp_parseTarget( image_filename, ...
    target_filename, ...
    invert_image_flag, ...
    plot_input_image )
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

if nargin < 3
    invert_image_flag = 0;
end%%if
if nargin < 4
    plot_input_image = 0;
end%%if

num_targets = 0;
target = []; %find(ones(NROWS_image, NCOLS_image));
clutter = [];

if exist(image_filename, 'file')
    image_pixels = imread(image_filename);
    [NROWS_image NCOLS_image num_colors] = size(image_pixels);
    disp(['size(image_pixels) = ', num2str(size(image_pixels))]);
    N_image = NROWS_image * NCOLS_image;
else
    disp(['exist(', image_filename, ') = ', ...
        num2str( exist( image_filename, 'file' ) ) ]);
end%%if

if invert_image_flag
    image_pixels = ...
        max(image_pixels(:)) - image_pixels;
end%%if

if size(image_pixels, 3) > 1
    image_pixels = squeeze( mean( image_pixels, 3 ) );
end%%if

image = find(image_pixels(:));
disp(['size(image) = ', num2str(size(image))]);

if ~isempty(target_filename)
    num_targets = length(target_filename);
else
    return
end%%if

target = cell(num_targets,1);
union_target = [];
for i_target = 1 : num_targets
    if exist(target_filename{i_target}, 'file')
        target_pixels = imread(target_filename{i_target});
        [NROWS_target NCOLS_target num_colors_target] = size(target_pixels);
        if size(target_pixels, 1) ~= size(image_pixels, 1) || size(target_pixels, 2) ~= size(image_pixels, 2)
            disp(['size(target_pixels{', num2str(i_target), '}) = ', ...
                num2str(size(target_pixels)), ...
                ' ~= size(image_pixels)' ]);
            return
        end%%if
        if invert_image_flag
            target_pixels = ...
                max(target_pixels(:)) - target_pixels;
        end%%if
        if size( target_pixels, 3 ) > 1
            target_pixels = ...
                squeeze( mean(target_pixels, 3) );
        end%%if
        target{i_target} = ...
            find( target_pixels(:) );
        union_target = union( union_target, target{i_target} );
        union_target = unique(union_target);
    else
        disp(['exist(', ...
            target_filename, ...
            '{', ...
            num2str(i_target), ...
            '}) = ', ...
            num2str( exist( target_filename{i_target}, 'file' ) ) ]);
        num_targets = i_target - 1;
        break
    end%%if
end%%for
disp(['size(union_target) = ', num2str(size(union_target))]);

clutter = setdiff( image, union_target );
disp(['size(clutter) = ', num2str(size(clutter))]);


if plot_input_image
    fh = figure('Name', 'input image');
    imagesc(image_pixels);
    colormap('gray');
else
  fh = 0;
end