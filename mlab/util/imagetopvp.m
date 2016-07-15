function pvpdata = imagetopvp(infile, outfile)
% pvpdata = imagetopvp(infile, outfile)
%
% Reads in the file given by infile as an image using imread, and
% converts it to pvp activity file format.
%
% The return value is a data structure of the same type as returned by
% readpvpfile when given a nonsparse activity pvp file with one frame
% and a timestamp of zero.
%
% Hence, it can be used as input to writepvpactivityfile.
%
% If the second input argument is present, the pvpdata is saved to the
% path given by outfile.

im = imread(infile);
if isequal(class(im), 'logical'), conversionfactor = 1; end
if isequal(class(im), 'uint8'), conversionfactor = 255; end
if isequal(class(im), 'uint16'), conversionfactor = 65535; end
im = double(im)/conversionfactor;
im = permute(im, [2 1 3]);

pvpdata{1}.time = 0;
pvpdata{1}.values = im;

if nargin > 1
    writepvpactivityfile(outfile, pvpdata);
end
