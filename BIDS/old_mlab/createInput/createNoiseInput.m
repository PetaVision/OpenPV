addpath('~/MATLAB/k-Wave Toolbox');
clear all;

% =========================================================================
% SIMULATION
% =========================================================================

% create the computational grid
Nx = 128;           % number of grid points in the x (row) direction
Ny = 128;           % number of grid points in the y (column) direction
dx = 0.1e-3;        % grid point spacing in the x direction [m]
dy = 0.1e-3;        % grid point spacing in the y direction [m]
kgrid = makeGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium
medium.sound_speed = 1500;  % [m/s]
medium.alpha_coeff = 0.75;  % [dB/(MHz^y cm)]
medium.alpha_power = 1.5; 

[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed);

source.p_mask = ones(Nx, Ny);

for i = 1:Nx
   for j = 1:Ny
      if(mod((i + j), 2) == 1)
         source.p_mask(i, j) = 0;
      end
   end
end

noise_freq = .25e6;
noise_mag = 2;
source.p = noise_mag * sin(2*pi*noise_freq*kgrid.t_array);

source.p = addNoise(source.p, 20);

% create initial pressure distribution using makeDisc
%disc_magnitude = 5; % [au]
%disc_x_pos = 50;    % [grid points]
%disc_y_pos = 50;    % [grid points]
%disc_radius = 8;    % [grid points]
%disc_1 = disc_magnitude*makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

%medium = addNoise(medium, 20);

%source.p0 = disc_1;

sensor = [];

% run the simulation
input_args = {'RecordMovie', true, 'MovieType', 'image', 'MovieName', '~/example_movie'}
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
