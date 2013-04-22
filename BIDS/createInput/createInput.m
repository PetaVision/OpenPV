%Path to k-wave toolkit
addpath('/Users/dpaiton/Documents/Work/LANL/workspace/BIDS/createInput/k-Wave Toolbox');

%Set up kgrid world
Nx = DIM(1);
Ny = DIM(2);
dx = .25;
dy = dx;
kgrid = makeGrid(Nx, dx, Ny, dy);

%medium properties
medium.sound_speed = 1500;
medium.alpha_coeff = .75;
medium.alpha_power = 1.5;

%time array
%time step is a 16th of a period
dt = 1/(TS_PER_PERIOD*WAVE_FREQUENCY);
%1/8 second in time
kgrid.t_array = [0:dt:.125];

%t_end = DIM(3);
%dt = t_end/DIM(3);
%kgrid.t_array = 0:dt:t_end;
%[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed);
t_end = length(kgrid.t_array);
time_diff = t_end/2;

%Pressure differences
if DROP_WAVE == 1
   source.p = zeros([1 t_end]);
   source.p(time_diff:t_end) = WAVE_STRENGTH*sin(2*pi*WAVE_FREQUENCY*kgrid.t_array(time_diff:t_end));
else
   source.p = zeros([1 t_end]);
   source.p(time_diff:(t_end-time_diff)/NUM_DROPS:t_end) = DROP_STRENGTH;
end

source.p = filterTimeSeries(kgrid, medium, source.p);

%Pressure mask, or drop position
source.p_mask = makeDisc(DIM(2), DIM(1), DROP_POS(2), DROP_POS(1), DROP_RADIUS);
orig_drop = source.p_mask;
%source.p_mask = zeros([Ny, Nx]);
%source.p_mask(DROP_POS(2), DROP_POS(1)) = 1;

%Sensor mask
sensor = [];

%input arguments for movie recording
input_args = {'RecordMovie', true, 'MovieType', 'image', 'MovieName', MOVIE_NAME, 'PlotFreq', 1};
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
all_wave = sensor_data.p_plots_all;
count = sensor_data.count;
