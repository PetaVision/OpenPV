function sphere = makeSphere(Nx, Ny, Nz, radius, plot_sphere)
%MAKESPHERE     Create a binary map of a sphere within a 3D grid.
%
% DESCRIPTION:
%       makeSphere creates a binary map of a spherical shell (using an
%       extension of the midpoint circle algorithm) within a
%       three-dimensional grid (the sphere position is denoted by 1's in
%       the matrix with 0's elsewhere).
%
% USAGE:
%       sphere = makeSphere(Nx, Ny, Nz, radius)
%       sphere = makeSphere(Nx, Ny, Nz, radius, plot_sphere)
%
% INPUTS:
%       Nx, Ny, Nz      - size of the 3D grid [grid points]
%       radius          - sphere radius [grid points]
%
% OPTIONAL INPUTS:
%       plot_sphere     - Boolean controlling whether the sphere is
%                         plotted using voxelPlot (default = false)
% OUTPUTS:
%       sphere          - 3D binary map of a sphere
%
% ABOUT:
%       author          - Bradley Treeby
%       date            - 16th June 2009
%       last update     - 27th October 2011
%       
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox
%
% See also makeBall, makeCartSphere, makeCircle

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

% check for plot_sphere input
if nargin < 5
    plot_sphere = false;
end

% enforce a centered sphere
cx = Nx/2;
cy = Ny/2;
cz = Nz/2;

% preallocate the storage variable
sphere = zeros(Nx, Ny, Nz);

% create a guide circle from which the individal radii can be extracted
guide_circle = makeCircle(Ny, Nx, cy, cx, radius);

% step through the guide circle points and create partially filled discs
centerpoints = (cx - radius):cx;
reflection_offset = length(centerpoints):-1:2;
for centerpoint_index = 1:length(centerpoints)
   
    % extract the current row from the guide circle
    row_data = guide_circle(:, centerpoints(centerpoint_index));

    % add an index to the pixels in the current row
    row_index = row_data.*(1:length(row_data)).';
    
    % calculate the radius 
    swept_radius = (max(row_index) - min(row_index(row_index ~= 0)))/2;
    
    % create a circle to add to the sphere
    circle = makeCircle(Ny, Nz, cy, cz, swept_radius);

    % make an empty fill matrix
    circle_fill = zeros(Nx, Ny);
    
    % fill in the circle line by line
    fill_centerpoints = (cz - swept_radius):(cz + swept_radius);
    for fill_centerpoint_index = 1:length(fill_centerpoints)
       
        % extract the first row
        row_data = circle(:, fill_centerpoints(fill_centerpoint_index));
        
        % add an index to the pixels in the current row
        row_index = row_data.*(1:length(row_data)).';
        
        % calculate the diameter
        start_index = min(row_index(row_index ~= 0));
        stop_index = max(row_index);
        
        % count how many points on the line
        num_points = sum(row_data);
        
        % fill in the line
        if start_index ~= stop_index && (stop_index - start_index) >= num_points
            circle_fill(start_index + num_points/2:stop_index - num_points /2, fill_centerpoints(fill_centerpoint_index)) = 1;
        end
    end
        
    % remove points from the filled circle that existed in the previous
    % layer
    if centerpoint_index == 1
        sphere(centerpoints(centerpoint_index), :, :) = circle + circle_fill;
        prev_circle = circle + circle_fill;
    else
        prev_circle_alt = circle + circle_fill;
        circle_fill = circle_fill - prev_circle;
        circle_fill(circle_fill < 0) = 0;
        sphere(centerpoints(centerpoint_index), :, :) = circle + circle_fill;
        prev_circle = prev_circle_alt;
    end
    
    % create the other half of the sphere at the same time
    if centerpoint_index ~= length(centerpoints)
        sphere(cx + reflection_offset(centerpoint_index) - 1, :, :) = sphere(centerpoints(centerpoint_index), :, :);
    end
end

% plot results
if plot_sphere
    voxelPlot(sphere);
end