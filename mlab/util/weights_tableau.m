function weight_patch_array = weights_tableau(weight_vals, weights_order)
% weight_patch_array = weights_tableau(weight_vals, weights_order)
%
% weight_vals   : a 4-dimensional array of weight patches, with dimensions
%                 (number of rows)-by-(number of columns)-by-(number of features)-by-(dictionary size)
% weights_order : an ordering of the indices from 1 to the dictionary size.
%                 The default ordering is [1; 2; ...; (dictionary size)]
%
% weight_patch_array is 3-dimensional uint8 array, showing a tiling of the weight patches.
% The number of rows of patches is obtained by taking the square root of the dictionary size and
% rounding down. The number of columns is then the number necessary to hold all the patches.
% The third dimension is the number of features.

    num_patches = size(weight_vals, 4);
    if ~exist('weights_order', 'var') || isempty(weights_order), weights_order = (1:num_patches)'; end
    num_patches_rows = floor(sqrt(num_patches));
    num_patches_columns = ceil(num_patches / num_patches_rows);
    num_colors = size(weight_vals,3);

    weight_patch_array_height = num_patches_rows*size(weight_vals,1);
    weight_patch_array_width = num_patches_columns*size(weight_vals,2);

    weight_patch_array = uint8(zeros(weight_patch_array_height, weight_patch_array_width, num_colors));
    for j_patch = 1  : num_patches
       i_patch = weights_order(j_patch);
       patch_tmp = weight_vals(:,:,:,i_patch);
       min_patch = min(patch_tmp(:));
       max_patch = max(patch_tmp(:));
       patch_tmp = (patch_tmp - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
       patch_tmp = uint8(permute(patch_tmp, [2,1,3]));
       col_index = 1 + mod(j_patch-1, num_patches_columns);
       row_index = 1 + floor((j_patch-1) / num_patches_columns);
       weight_patch_array(((row_index-1)*size(patch_tmp,1)+1):row_index*size(patch_tmp,1), ...
       ((col_index-1)*size(patch_tmp,2)+1):col_index*size(patch_tmp,2),:) = patch_tmp;
    end  %% j_patch
