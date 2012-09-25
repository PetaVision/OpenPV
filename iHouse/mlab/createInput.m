addpath('/Users/slundquist/Documents/MATLAB');
startup;

Nx = 512;
Ny = 512;
dx = .1e-3;
dy = .1e-3;

kgrid = makeGrid(Nx, dx, Ny, dy);

%medium properties
medium.sound_speed = 1500;
medium.alpha_coeff = .75;
medium.alpha_power = 1.5;

%initial pressure distribution
disc_magnitude = 5;
disc_x_pos = 50;
disc_y_pos = 50;
disc_radius = 8;
disc_1 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

disc_magnitude = 3;
disc_x_pos = 80;
disc_y_pos = 60;
disc_radius = 5;
disc_2 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

source.p0 = disc_1 + disc_2;

%Sensor mask
sensor_radius = 4e-3;
num_sensor_points = 50;
sensor.mask = makeCartCircle(sensor_radius, num_sensor_points);

%input arguments for movie recording
input_args = {'RecordMovie', true, 'MovieType', 'frame', 'MovieName', '~/Desktop/Movie Image/example_movie'}
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
%sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor);

