function [ target, clutter ] = pv_parseTiff( filename )
% Return a list of the objects in the colored TIFF file.
% Image dimensions come from the file. The color are hard-coded
% here, namely: black (0) is background, white (255) is foreground,
% and anything else is a target. We could use this to
% support multi targets.

% For now, just sum the RGB values. TODO
%pixels = sum(imread(filename),3);
pixels = imread(filename);

if ndims(pixels) == 3
   clutter = find(pixels(:,:,3) == 255);
   target = find(pixels(:,:,1) == 255);
else
    clutter = [];
    target = find(pixels');

end
