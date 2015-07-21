% Monopole Point Source In A Homogeneous Propagation Medium Example
%
% This example provides a simple demonstration of using k-Wave for the
% simulation and detection of a time varying pressure source within a
% two-dimensional homogeneous propagation medium. It builds on the
% Homogeneous Propagation Medium Example.   
%
% author: Bradley Treeby
% date: 2nd December 2009
% last update: 20th October 2011
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
Nx = 128;           % number of grid points in the x (row) direction
Ny = 128;           % number of grid points in the y (column) direction
dx = 50e-3/Nx;    	% grid point spacing in the x direction [m]
dy = dx;            % grid point spacing in the y direction [m]
kgrid = makeGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium
medium.sound_speed = 1500;  % [m/s]
medium.alpha_coeff = 0.75;  % [dB/(MHz^y cm)]
medium.alpha_power = 1.5; 

% create the time array
[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed);

% define a single source element
source.p_mask = zeros(Nx, Ny);
source.p_mask(end - Nx/4, Ny/2) = 1;

% define a time varying sinusoidal source
source_freq = 0.25e6;   % [Hz]
source_mag = 2;         % [au]
source.p = source_mag*sin(2*pi*source_freq*kgrid.t_array);

% smooth the source
source.p = filterTimeSeries(kgrid, medium, source.p);

% define a single sensor element
sensor.mask = zeros(Nx, Ny);
sensor.mask(Nx/4, Ny/2) = 1;

% run the simulation
[sensor_data, field_data] = kspaceFirstOrder2D(kgrid, medium, source, sensor);

% =========================================================================
% VISUALISATION
% =========================================================================

% plot the final wave-field
figure;
imagesc(kgrid.y_vec*1e3, kgrid.x_vec*1e3, field_data + source.p_mask + sensor.mask, [-1 1]);
colormap(getColorMap);
ylabel('x-position [mm]');
xlabel('y-position [mm]');
axis image;

% plot the simulated sensor data
figure;
[t_sc, scale, prefix] = scaleSI(max(kgrid.t_array(:)));

subplot(2, 1, 1), plot(kgrid.t_array*scale, source.p, 'k-');
xlabel(['Time [' prefix 's]']);
ylabel('Signal Amplitude');
axis tight;
title('Input Pressure Signal');

subplot(2, 1, 2), plot(kgrid.t_array*scale, sensor_data, 'r-');
xlabel(['Time [' prefix 's]']);
ylabel('Signal Amplitude');
axis tight;
title('Sensor Pressure Signal');
