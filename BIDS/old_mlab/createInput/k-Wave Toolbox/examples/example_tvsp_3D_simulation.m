% Simulations In Three Dimensions Example
%
% This example provides a simple demonstration of using k-Wave for the
% simulation and detection of a time varying pressure source within a
% three-dimensional heterogeneous propagation medium. It builds on the
% Monopole Point Source In A Homogeneous Propagation Medium Example and
% Simulations In Three Dimensions examples.    
%
% author: Bradley Treeby
% date: 20th January 2010
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

% =========================================================================
% SIMULATION
% =========================================================================

% create the computational grid
Nx = 64;            % number of grid points in the x direction
Ny = 64;            % number of grid points in the y direction
Nz = 64;            % number of grid points in the z direction
dx = 0.1e-3;        % grid point spacing in the x direction [m]
dy = 0.1e-3;        % grid point spacing in the y direction [m]
dz = 0.1e-3;        % grid point spacing in the z direction [m]
kgrid = makeGrid(Nx, dx, Ny, dy, Nz, dz);

% define the properties of the propagation medium
medium.sound_speed = 1500*ones(Nx, Ny, Nz);	% [m/s]
medium.sound_speed(1:Nx/2, :, :) = 1800;    % [m/s]
medium.density = 1000*ones(Nx, Ny, Nz);     % [kg/m^3]
medium.density(:, Ny/4:end, :) = 1200;      % [kg/m^3]

% create the time array
[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed);

% define a square source element
source_radius = 5;  % [grid points]
source.p_mask = zeros(Nx, Ny, Nz);
source.p_mask(Nx/4, Ny/2 - source_radius:Ny/2 + source_radius, Nz/2 - source_radius:Nz/2 + source_radius) = 1;

% define a time varying sinusoidal source
source_freq = 2e6;  % [Hz]
source_mag = 1;     % [au]
source.p = source_mag*sin(2*pi*source_freq*kgrid.t_array);

% smooth the source
source.p = filterTimeSeries(kgrid, medium, source.p);

% define a series of Cartesian points to collect the data
x = (-22:2:22)*dx;          % [m]
y = 22*dy*ones(size(x));    % [m]
z = (-22:2:22)*dz;          % [m]
sensor.mask = [x; y; z];

% input arguments
input_args = {'DisplayMask', source.p_mask, 'DataCast', 'single'};

% run the simulation
[sensor_data, p_final] = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});

% =========================================================================
% VISUALISATION
% =========================================================================

% view final pressure field slice by slice
flyThrough(p_final);

% plot the final pressure field in the x-y plane
figure;
p_final(source.p_mask ~= 0) = 1;
[x_sc scale prefix] = scaleSI(max([kgrid.x_vec, kgrid.y_vec])); 
imagesc(kgrid.y_vec*scale, kgrid.x_vec*scale, squeeze(p_final(:, :, kgrid.Nz/2)), [-1, 1]);
colormap(getColorMap);
xlabel(['y [' prefix 'm]']);
ylabel(['x [' prefix 'm]']);
axis image;
colorbar;

% plot the simulated sensor data
figure;
imagesc(sensor_data, [-1, 1]);
colormap(getColorMap);
ylabel('Sensor Position');
xlabel('Time Step');
colorbar;