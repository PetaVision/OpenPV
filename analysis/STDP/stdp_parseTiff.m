function [target, X, Y]  = stdp_parseTiff( filename )
% Return a list of the objects in the colored TIFF file.
% Image dimensions come from the file. The color are hard-coded
% here, namely: black (0) is background, white (255) is foreground,
% and anything else is a target. We could use this to
% support multi targets.

% For now, just sum the RGB values. TODO
%pixels = sum(imread(filename),3);
pixels = imread(filename);
% NOTE: The return value pixels is an array containing the image data. If 
% the file contains a grayscale image, A is an M-by-N array. If the file 
% contains a truecolor image, A is an M-by-N-by-3 array.
if ndims(pixels) == 3 % color file
   %clutter = find(pixels(:,:,3) == 255);
   target = find(pixels(:,:,1) == 255);
else
    
    target = find(pixels);
    % find locates all nonzero elements of array pixels, and returns the linear
    % indices of those elements in vector ind.
    [Y, X] = find(pixels);
    % returns the row and column indices of the nonzero entries in the 
    % matrix pixels.
end
