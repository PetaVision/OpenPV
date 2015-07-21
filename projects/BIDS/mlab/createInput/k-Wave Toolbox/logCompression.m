function signal = logCompression(signal, a, normalise)
%LOGCOMPRESSION   Log compress an input signal.
%
% DESCRIPTION:
%       logCompression compresses the input signal using the expression
%       signal = log10(1 + a*signal)./log10(1 + a) 
%
% USAGE:
%       signal = logCompression(signal, a)
%       signal = logCompression(signal, a, normalise)
%
% INPUTS:
%       signal      - input signal
%       a           - compression factor
%
% OPTIONAL INPUTS
%       normalise   - boolean controlling whether the maximum of the input
%                     signal is normalised to unity before compression
%                     (default = false). If set to true, the original
%                     magnitude is restored after compression.
%
% OUTPUTS:
%       signal      - log compressed signal
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 24th February 2011
%       last update - 24th February 2011
%
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox

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

% check for optional normalise input
if nargin == 2
    normalise = false;
end

% compress signal
if normalise
    mx = max(signal(:));
    signal = mx*(log10(1 + a*signal./mx)./log10(1 + a));
else 
    signal = log10(1 + a*signal)./log10(1 + a);
end