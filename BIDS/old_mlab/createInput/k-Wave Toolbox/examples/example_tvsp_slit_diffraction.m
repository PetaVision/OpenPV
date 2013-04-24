% Diffraction Through A Slit Example Example
%
% This example illustrates the diffraction of a plane acoustic wave through
% a slit. It builds on the Monopole Point Source In A Homogeneous
% Propagation Medium and Simulating Transducer Field Patterns examples.  
%
% author: Bradley Treeby
% date: 1st November 2010
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

example_number = 1;
% 1: single slit with the source wavelength = slit size
% 2: single slit with the source wavelength = slit size / 4
% 3: double slit with the source wavelength = slit size

% =========================================================================
% SIMULATION
% =========================================================================

% create the computational grid and define the bulk medium properties
scale = 1;                    % change to 2 to produce higher resolution images
PML_size = 10;                % size of the perfectly matched layer [grid points]
Nx = scale*128 - 2*PML_size;  % number of grid points in the x (row) direction
Ny = Nx;                      % number of grid points in the y (column) direction
dx = 50e-3/Nx;                % grid point spacing in the x direction [m]
dy = dx;                      % grid point spacing in the y direction [m]
c0 = 1500;                    % [m/s]
rho0 = 1000;                  % [kg/m^3]
kgrid = makeGrid(Nx, dx, Ny, dy);

% define the ratio between the barrier and background sound speed and
% density
barrier_scale = 20;

% create the time array using the barrier sound speed
t_end = 40e-6;                % [s]
CFL = 0.5;                    % Courant–Friedrichs–Lewy number
kgrid.t_array = makeTime(kgrid, c0*barrier_scale, CFL, t_end);

% define the barrier and the source wavelength depending on the example
switch example_number
    case 1
        
        % create a mask of a barrier with a slit
        slit_thickness = scale*2;               % [grid points]
        slit_width = scale*10;                  % [grid points]
        slit_x_pos = Nx - Nx/4;                 % [grid points]
        slit_offset = Ny/2 - slit_width/2 - 1;  % [grid points]
        slit_mask = zeros(Nx, Ny);
        slit_mask(slit_x_pos:slit_x_pos + slit_thickness, 1:1 + slit_offset) = 1;
        slit_mask(slit_x_pos:slit_x_pos + slit_thickness, end - slit_offset:end) = 1;
        
        % define the source wavelength to be the same as the slit size
        source_wavelength = slit_width*dx;      % [m]
        
    case 2
        
        % create a mask of a barrier with a slit
        slit_thickness = scale*2;               % [grid points]
        slit_width = scale*30;                  % [grid points]
        slit_x_pos = Nx - Nx/4;                 % [grid points]
        slit_offset = Ny/2 - slit_width/2 - 1;  % [grid points]
        slit_mask = zeros(Nx, Ny);
        slit_mask(slit_x_pos:slit_x_pos + slit_thickness, 1:1 + slit_offset) = 1;
        slit_mask(slit_x_pos:slit_x_pos + slit_thickness, end - slit_offset:end) = 1;
        
        % define the source wavelength to be a quarter of the slit size
        source_wavelength = 0.25*slit_width*dx; % [m]
        
    case 3
        
        % create a mask of a barrier with a double slit
        slit_thickness = scale*2;               % [grid points]
        slit_width = scale*10;                  % [grid points]
        slit_spacing = scale*20;                % [grid points]
        slit_x_pos = Nx - Nx/4;                 % [grid points]
        slit_offset = Ny/2 - slit_width - slit_spacing/2 - 1; 
        slit_mask = zeros(Nx, Ny);
        slit_mask(slit_x_pos:slit_x_pos + slit_thickness, 1:1 + slit_offset) = 1;
        slit_mask(slit_x_pos:slit_x_pos + slit_thickness, slit_offset + slit_width + 2:slit_offset + slit_width + slit_spacing + 1) = 1;
        slit_mask(slit_x_pos:slit_x_pos + slit_thickness, end - slit_offset:end) = 1;
        
        % define the source wavelength to be the same as the slit size
        source_wavelength = slit_width*dx;      % [m]
end

% assign the slit to the properties of the propagation medium
medium.sound_speed = c0*ones(Nx, Ny);
medium.density = rho0*ones(Nx, Ny);
medium.sound_speed(slit_mask == 1) = barrier_scale*c0;
medium.density(slit_mask == 1) = barrier_scale*rho0;

% create a source mask of a single line
source.p_mask = zeros(Nx, Ny);
source.p_mask(end, :) = 1;

% create and smooth the time varying sinusoidal source
source_mag = 2;
source_freq = c0/source_wavelength;
source.p = source_mag*sin(2*pi*source_freq*kgrid.t_array);
source.p = filterTimeSeries(kgrid, medium, source.p);

% set the input options and run simulation leaving the sensor input empty
input_args = {'ReturnVelocity', true, 'PMLInside', false, 'PMLSize', PML_size,...
    'DisplayMask', slit_mask, 'DataCast', 'single', 'PlotPML', false};
[sensor_data, field_data] = kspaceFirstOrder2D(kgrid, medium, source, [], input_args{:});

% =========================================================================
% VISUALISATION
% =========================================================================

% plot the final wave-field
figure;
mx = max(abs(field_data.p(:)));
field_data.p(slit_mask == 1) = mx;
imagesc(kgrid.y_vec*1e3, kgrid.x_vec*1e3, field_data.p, [-mx, mx]);
colormap(getColorMap);
ylabel('x-position [mm]');
xlabel('y-position [mm]');
axis image;
title('p');

% plot the final wave-field
figure;
mx = max(abs(field_data.ux(:)));
field_data.ux(slit_mask == 1) = mx;
imagesc(kgrid.y_vec*1e3, kgrid.x_vec*1e3, field_data.ux, [-mx, mx]);
colormap(getColorMap);
ylabel('x-position [mm]');
xlabel('y-position [mm]');
axis image;
title('ux');

% plot the final wave-field
figure;
mx = max(abs(field_data.uy(:)));
field_data.uy(slit_mask == 1) = mx;
imagesc(kgrid.y_vec*1e3, kgrid.x_vec*1e3, field_data.uy, [-mx, mx]);
colormap(getColorMap);
ylabel('x-position [mm]');
xlabel('y-position [mm]');
axis image;
title('uy');