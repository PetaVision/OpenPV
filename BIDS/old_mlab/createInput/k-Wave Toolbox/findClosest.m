function [value, index] = findClosest(A, a)
%FINDCLOSEST    Return the closest value in a matrix.
%
% DESCRIPTION:
%       findClosest returns the value and index of the item in A that is
%       closest to the value a. For vectors, value and index correspond to
%       the closest element in A. For matrices, value and index are row
%       vectors corresponding to the closest element from each column. For
%       N-D arrays, the function finds the closest value along the first
%       non-singleton dimension. If there is more than one element with the
%       closest value, the index of the first one is returned.
%
% USAGE:
%       value = findClosest(A, a)
%       [value, index] = findClosest(A, a)
%
% INPUTS:
%       A               - matrix to search
%       a               - value to find
%
% OUTPUTS:
%       value           - value in A that is closest to a
%       index           - the index of a within A
%       
% ABOUT:
%       author          - Bradley Treeby
%       date            - 19th November 2009
%       last update     - 19th November 2009
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

[value, index] = min(abs(A - a));
value = value + a;