% CANNY - Canny edge detection
%
% Function to perform Canny edge detection. Code uses modifications as
% suggested by Fleck (IEEE PAMI No. 3, Vol. 14. March 1992. pp 337-345)
%
% Usage: [gradient or] = canny(im, sigma)
%
% Arguments:   im    - image to be procesed
%              sigma - standard deviation of Gaussian smoothing filter
%                      (typically 1)
%
% Returns:     gradient - edge strength image (gradient amplitude)
%              or       - orientation image (in degrees 0-180, positive
%                         anti-clockwise)
%
% See also:  NONMAXSUP, HYSTHRESH

% Copyright (c) 1999-2003 Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% http://www.csse.uwa.edu.au/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% April 1999    Original version
% January 2003  Error in calculation of d2 corrected

function [gradient, or] = canny(im, sigma)

[rows, cols] = size(im);
im = double(im);         % Ensure double

hsize = [6*sigma+1, 6*sigma+1];   % The filter size.

gaussian = fspecial('gaussian',hsize,sigma);
im = filter2(gaussian,im);        % Smoothed image.

h =  [  im(:,2:cols)  zeros(rows,1) ] - [  zeros(rows,1)  im(:,1:cols-1)  ];
v =  [  im(2:rows,:); zeros(1,cols) ] - [  zeros(1,cols); im(1:rows-1,:)  ];
d1 = [  im(2:rows,2:cols) zeros(rows-1,1); zeros(1,cols) ] - ...
                               [ zeros(1,cols); zeros(rows-1,1) im(1:rows-1,1:cols-1)  ];
d2 = [  zeros(1,cols); im(1:rows-1,2:cols) zeros(rows-1,1);  ] - ...
                               [ zeros(rows-1,1) im(2:rows,1:cols-1); zeros(1,cols)   ];

X = h + (d1 + d2)/2.0;
Y = v + (d1 - d2)/2.0;

gradient = sqrt(X.*X + Y.*Y); % Gradient amplitude.

or = atan2(-Y, X);            % Angles -pi to + pi.
neg = or<0;                   % Map angles to 0-pi.
or = or.*~neg + (or+pi).*neg; 
or = or*180/pi;               % Convert to degrees.
