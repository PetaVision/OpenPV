function unmasked_sensor_data = unmaskSensorData(sensor_data, sensor_mask, kgrid)
%UNMASKSENSORDATA   Reorder data recorded using a binary sensor mask.
%
% DESCRIPTION:
%       unmaskSensorData restores the grid position of the data recorded at
%       a single time-step within the time-series data returned by
%       kspaceFirstOrder1D, kspaceFirstOrder2D, or kspaceFirstOrder3D when
%       using a binary sensor mask.
%
% USAGE:
%       unmasked_sensor_data = unmaskSensorData(sensor_data, sensor_mask, kgrid)
%
% INPUTS:
%       sensor_data - sensor data (returned by the first order simulation
%                     functions) at a single time-step ordered using
%                     MATLAB's standard column-wise linear matrix indexing 
%       sensor_mask - binary sensor mask used in the simulation
%       kgrid       - k-Wave grid structure returned by makeGrid
%
% OUTPUTS:
%       unmasked_sensor_data - Grid with the sensor data reordered to its
%                              original position on the sensor mask
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 6th April 2009
%       last update - 5th December 2011
%       
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox
%
% See also makeGrid, kspaceFirstOrder1D, kspaceFirstOrder2D,
% kspaceFirstOrder3D

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

% create an empty matrix
switch numDim(kgrid.k)
    case 1
        unmasked_sensor_data = zeros(kgrid.Nx, 1);
    case 2
        unmasked_sensor_data = zeros(kgrid.Nx, kgrid.Ny);
    case 3
        unmasked_sensor_data = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
end

% reorder input data
unmasked_sensor_data(sensor_mask ~= 0) = sensor_data;