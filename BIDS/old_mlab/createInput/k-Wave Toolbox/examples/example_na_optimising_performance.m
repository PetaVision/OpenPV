% Optimising k-Wave Performance Example
%
% This example demonstrates how to increase the computational performance
% of k-Wave using optional input parameters and data casting. A separate
% standardised benchmarking script benchmark is also included within the
% k-Wave toolbox to allow computational times to be compared across
% different computers and GPUs. 
%
% author: Bradley Treeby
% date: 15th July 2009
% last update: 19th December 2011
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

clear all;

example_number = 1;
% 1: default input options
% 2: nearest neighbour Cartesian interpolation and plotting switched off
% 3: as above with 'DataCast' set to 'single'
% 4: as aobve with 'DataCast' set to 'gsingle'

% change scale to 2 to increase the computational time
scale = 1;

% =========================================================================
% SIMULATION
% =========================================================================

% assign the grid size and create the computational grid
Nx = 256*scale;     % number of grid points in the x (row) direction
Nz = 256*scale;     % number of grid points in the y (column) direction
x = 10e-3;          % grid size in the x direction [m]
z = 10e-3;          % grid size in the y direction [m]
dx = x/Nx;          % grid point spacing in the x direction [m]
dz = z/Nz;          % grid point spacing in the y direction [m]
kgrid = makeGrid(Nx, dx, Nz, dz);

% load the initial pressure distribution from an image and scale
p0_magnitude = 2;   % [au]
source.p0 = p0_magnitude*loadImage('EXAMPLE_source_two.bmp');
source.p0 = resize(source.p0, [Nz, Nx]);

% define the properties of the propagation medium
medium.sound_speed = 1500;	% [m/s]

% define a centered Cartesian circular sensor
sensor_radius = 4.5e-3;     % [m]
num_sensor_points = 100;
sensor.mask = makeCartCircle(sensor_radius, num_sensor_points);

% create the time array
[kgrid.t_array dt] = makeTime(kgrid, medium.sound_speed);

% set the input options
switch example_number
    case 1
        input_args = {};
    case 2
        input_args = {'CartInterp', 'nearest', 'PlotSim', false};
    case 3
        input_args = {'CartInterp', 'nearest', 'PlotSim', false, 'DataCast', 'single'};
    case 4
        input_args = {'CartInterp', 'nearest', 'PlotSim', false, 'DataCast', 'gsingle'};
        
        % run GPU warmup
        tic; fft2(gsingle(rand(128, 128))); toc;
        tic; fft2(gsingle(rand(128, 128))); toc;
        tic; fft2(gsingle(rand(128, 128))); toc;
end

% run the simulation
kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});