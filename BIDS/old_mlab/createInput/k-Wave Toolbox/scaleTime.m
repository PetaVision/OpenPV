function time = scaleTime(seconds)
%SCALETIME   Convert seconds to hours, minutes, and seconds.
%
% DESCRIPTION:
%       scaleTime converts an integer number of seconds into hours,
%       minutes, and seconds, and returns a string with this information.
%
% USAGE:
%       time = scaleTime(seconds)
%
% INPUTS:
%       seconds         - number of seconds
%
% OUTPUTS:
%       time            - string of scaled time   
%
% ABOUT:
%       author          - Bradley Treeby
%       date            - 16th July 2009
%       last update     - 3rd December 2009
%       
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox
%
% See also scaleSI

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

hours = floor(seconds / (60*60));
seconds = seconds - hours*60*60;
minutes = floor( seconds / 60 );
seconds = seconds - minutes*60;

if hours > 0
    time = [num2str(hours) 'hours ' num2str(minutes) 'min ' num2str(seconds) 's'];
elseif minutes > 0
    time = [num2str(minutes) 'min ' num2str(seconds) 's']; 
else
    time = [num2str(seconds) 's']; 
end