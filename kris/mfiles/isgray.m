function y = isgray(x)
%ISGRAY True for intensity images.
%   ISGRAY(A) returns 1 if A is a 2-D uint8 array or if A is a
%   2-D double array that contains values between 0.0 and 1.0.
%
%   See also ISIND, ISBW.

%   Clay M. Thompson 2-25-93
%   Copyright (c) 1993-1996 by The MathWorks, Inc.
%   $Revision: 5.4 $  $Date: 1996/09/18 21:57:30 $

if isa(x, 'uint8')
   if ndims(x)==2,
      y = 1;
   else       % Most likely it's a mxnx3 RGB image
      y = 0;  
   end
else
    y = min(x(:))>=0 & max(x(:))<=1;
end    

y = logical(double(y));    % Just make sure