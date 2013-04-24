function im = loadImage(filename)
%LOADIMAGE      Load an image file.
%
% DESCRIPTION:
%       loadImage loads an external image file using imread and returns a
%       two-dimensional image matrix scaled between 0 and 1. If a colour
%       image is loaded, the colour channels are summed before scaling. If
%       no input is given for filename, a selection dialog is invoked.
%
% USAGE:
%       im = loadImage()
%       im = loadImage(filename)
%
% OPTIONAL INPUTS:
%       filename    - filename (and pathname if not in the same directory)
%                     of the image to load
%
% OUTPUTS:
%       im          - scaled image matrix
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 30th June 2009
%       last update - 17th July 2009
%       
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox
%
% See also imread, resize

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% 
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
% more details. 
% 
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>.

% load the file if not selected
if nargin == 0
    [filename pathname] = getFilename();
    filename = strcat(pathname, filename);
end

% extract the filetype
split_filename = regexp(filename, '\.', 'split');
filetype = lower(split_filename{2});

% check the filetype
if ~strcmp(filetype, 'jpg') && ~strcmp(filetype, 'bmp') && ~strcmp(filetype, 'gif')  && ~strcmp(filetype, 'png')
    disp('WARNING: image filetype may not be supported - see loadImage.m');
end
 
% load the image
im = imread(filename);

% if there are multiple colour channels, sum them up
if numDim(im) > 2
    im = squeeze(double(im(:, :, 1)) + double(im(:, :, 2)) + double(im(:, :, 3)));
else
    im = double(im);
end

% scale pixel values from 0 -> 1
im = max(im(:)) - im;
im = im .* (1/max(im(:)));