function pixels = pv_read_bin_image(file_name )
% Reads a binary image file.
%   

fid = fopen(file_name, 'r');
pixels = fread(fid,[64,64],'float32');


end

