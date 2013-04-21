function [zi, del_tri] = gridDataFast(x, y, z, xi, yi, del_tri)
% DESCRIPTION:
%       Stripped down version of GRIDDATA that removes inbuilt data
%       checking and allows input and output of the delaunay triangulation
%       for use on subsequent calls with the exact same set of data
%       coordinates 
%
% USAGE:
%       griddata_inbuilt(x, y, z, xi, yi)
%       griddata_inbuilt(x, y, z, xi, yi, del_tri)
%
% INPUT/OUTPUT
%       x, y, z, xi, yi, zi are the same as griddata
%       del_tri is the Delaunay triangulation
%
% ABOUT:
%       author      - John Wilkin
%       date        - October 2002
%       modified by - Bradley Treeby
%       last update - 6th September 2010
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

% enforce x,y and z to be column vectors
sz = numel(x);
x = reshape(x,sz,1);
y = reshape(y,sz,1);
z = reshape(z,sz,1);
siz = size(xi);
xi = xi(:); 
yi = yi(:);
x = x(:); 
y = y(:);

if nargin < 6
      
    % triangulize the data
    tri = delaunayn([x y]);
    
    % catch trinagulation error
    if isempty(tri)    
        error('Data cannot be triangulated.');
    end
  
    % find the nearest triangle (t)
	t = tsearch(x,y,tri,xi,yi);
      
	% only keep the relevant triangles.
	out = find(isnan(t));
    if ~isempty(out)
        t(out) = ones(size(out));
    end  
	tri = tri(t,:);

    % save triangulation data
    del_tri.tri = tri;
    del_tri.out = out;

else
    % use the triangulation from del_tri
    tri = del_tri.tri;
    out = del_tri.out;
end

% compute Barycentric coordinates
del = (x(tri(:,2))-x(tri(:,1))) .* (y(tri(:,3))-y(tri(:,1)))...
    - (x(tri(:,3))-x(tri(:,1))) .* (y(tri(:,2))-y(tri(:,1)));
w(:,3) = ((x(tri(:,1))-xi).*(y(tri(:,2))-yi)...
    - (x(tri(:,2))-xi).*(y(tri(:,1))-yi)) ./ del;
w(:,2) = ((x(tri(:,3))-xi).*(y(tri(:,1))-yi)...
    - (x(tri(:,1))-xi).*(y(tri(:,3))-yi)) ./ del;
w(:,1) = ((x(tri(:,2))-xi).*(y(tri(:,3))-yi)...
    - (x(tri(:,3))-xi).*(y(tri(:,2))-yi)) ./ del;
w(out,:) = zeros(length(out),3);

% treat z as a row so that code below involving z(tri) works even when tri
% is 1-by-3. 
z = z(:).'; 
zi = sum(z(tri) .* w,2);
zi = reshape(zi,siz);
if ~isempty(out)
  zi(out) = NaN; 
end