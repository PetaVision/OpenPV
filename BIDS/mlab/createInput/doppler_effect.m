clear all; close all;
addpath('./k-Wave Toolbox');
MOVIE_NAME = '~/plot';
% =========================================================================
% SIMULATION
% =========================================================================

Nx = 256;           % number of grid points in the x (row) direction
Ny = Nx;            % number of grid points in the y (column) direction
dx = 1.5625e-4;   % grid point spacing in the y direction [m/px]
dy = dx;            % grid point spacing in the x direction [m/px]
pml_size = 20;      % [pixels]
kgrid = makeGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium
medium.sound_speed = 1500;      % [m/s]
medium.alpha_coeff = 0;      % [dB/(MHz^y cm)]
medium.alpha_power = 1.5; 

% set the velocity of the moving source
source_vel = 40;               % [m/s]

% manually create the time array
dt = 20e-9;                     % [s]
t_end = (Ny - 2*pml_size - 2)*dy / source_vel; % [s]
kgrid.t_array = 0:dt:t_end;
length(kgrid.t_array)

% define a single time varying sinusoidal source
source_freq = 0.75e6;           % [Hz]
source_mag = 3;                 % [au]
source_pressure = source_mag*sin(2*pi*source_freq*kgrid.t_array);

% smooth the source
source_pressure = filterTimeSeries(kgrid, medium, source_pressure);

% define a line of source elements
source_x_pos = Nx/2-pml_size;               % [grid points]
source.p_mask = zeros(Nx, Ny);
source.p_mask(end - pml_size - source_x_pos, 1 + pml_size:end - pml_size) = 1;

% preallocate an empty pressure source matrix
num_source_positions = sum(source.p_mask(:));
source.p = zeros(num_source_positions, length(kgrid.t_array));

% move the source along the source mask by interpolating the pressure
% series between the source elements
sensor_index = 1;
t_index = 1;
while t_index < length(kgrid.t_array) && sensor_index < num_source_positions - 1
    
    % check if the source has moved to the next pair of grid points
    if kgrid.t_array(t_index) > (sensor_index*dy/source_vel)
        sensor_index = sensor_index + 1;
    end    
    
    % calculate the position of source in between the two current grid
    % points
    exact_pos = (source_vel*kgrid.t_array(t_index));
    discrete_pos = sensor_index*dy;
    pos_ratio = (discrete_pos - exact_pos) ./ dy;
    
    % update the pressure at the two current grid points using linear
    % interpolation
    source.p(sensor_index, t_index) = pos_ratio*source_pressure(t_index);
    source.p(sensor_index + 1, t_index) = (1 - pos_ratio)*source_pressure(t_index);
    
    % update the time index
    t_index = t_index + 1;
end

% define a single sensor element
sensor.mask = zeros(Nx, Ny);

% run the simulation
%input_args = {'RecordMovie', true, 'MovieType', 'image', 'MovieName', MOVIE_NAME, 'PlotFreq', 1,'PlotPML',false,'PlotSim',true}; %%To plot movie
[sensor_data, field_data] = kspaceFirstOrder2D(kgrid, medium, source, sensor,'PlotPML',false);

figure
imagesc(source.p_mask)
colorbar

figure
imagesc(source.p)
colorbar
