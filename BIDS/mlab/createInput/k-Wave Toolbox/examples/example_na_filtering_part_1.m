% Filtering A Delta Function Input Signal Example Part 1
%
% This example illustrates the numerical aliasing that results from
% applying a temporal delta function pressure pulse without filtering or
% smoothing.
%
% author: Bradley Treeby
% date: 18th January 2010
% last update: 2nd November 2011
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

clear all

% =========================================================================
% SIMULATION
% =========================================================================

% create the computational grid
Nx = 256;           % number of grid points in the x (row) direction
dx = 10e-3/Nx;      % grid point spacing in the x direction [m]
kgrid = makeGrid(Nx, dx);

% define the properties of the propagation medium
medium.sound_speed = 1500;  % [m/s]

% create a time array
num_time_steps = 1024;
[kgrid.t_array dt] = makeTime(kgrid, medium.sound_speed);
kgrid.t_array = 0:dt:dt*(num_time_steps - 1);

% define a single element source
source_offset = 50;
source.p_mask = zeros(Nx, 1);
source.p_mask(1 + source_offset, 1) = 1;

% define a delta function input pulse
temporal_offset = 100;      % [time steps]
source_magnitude = 2;       % [au]
source.p = zeros(size(kgrid.t_array));
source.p(temporal_offset) = source_magnitude;

% define a single element sensor
sensor.mask = zeros(Nx, 1);
sensor.mask(end - source_offset, 1) = 1;

% run the simulation
sensor_data = kspaceFirstOrder1D(kgrid, medium, source, sensor, 'PMLSize', 30);

% compute the amplitude spectra of the input and recorded time
% series
[f, input_as] = spectrum(source.p, 1/dt);
[f, output_as] = spectrum(sensor_data, 1/dt);

% extract the maximum frequency supported by the grid (two points per
% wavelength)
f_max = kgrid.k_max * min(medium.sound_speed(:)) / (2*pi);

% =========================================================================
% VISUALISATION
% =========================================================================

% plot the input and recorded time series
figure;
[t_sc scale prefix] = scaleSI(max(kgrid.t_array(:)));
plot(kgrid.t_array*scale, source.p, 'k-', kgrid.t_array*scale, sensor_data, 'b-');
xlabel(['Time [' prefix 's]']);
ylabel('Pressure [au]');
legend('input pulse', 'recorded pulse');

% plot the amplitude spectra
[f_sc scale prefix] = scaleSI(max(f));
figure;
plot(f*scale, input_as, 'k-', f*scale, output_as, 'b-');
xlabel(['Frequency [' prefix 'Hz]']);
ylabel('Amplitude [au]');

% plot the maximum frequency supported by the grid
ylim = get(gca, 'YLim');
hold on;
line([f_max*scale, f_max*scale], [0, ylim(2)], 'LineStyle','--', 'Color', 'k');
legend('amplitude spectrum of input pulse', 'amplitude spectrum of recorded pulse', 'maximum frequency supported by grid');