clear all;
close all;
addpath('./k-Wave Toolbox');

DIM = [1024, 1024, 0];   % [X, Y, t=0] This will be a 1280x1280m grid for the simulation (0.4 px/m).
dx = 0.625;              % [m]
dy = dx;                 % [m]
BETA = -1;               % 0 is gaussian white, -1 is pink, -2 is Brownian

%Sine wave params
% peak vehicle frequency
WAVE_FREQUENCY = 125;    %Hz
WAVE_STRENGTH = 0.11;    %Pa (75dB SBL)

%Time properties
SOURCE_VEL  = 8.9408;    % [m/s] = 20 mph
TIME_LENGTH = 120;       % [s]
dt = 10e-3;              % [s] - 1ms

%Medium properties
% pure tone through air at 20 deg C, 30 perc humidity, 4000ft elevation (0.8755 bars, 0.864 ATM)
medium.sound_speed = 348.9; % [m/s]
medium.alpha_coeff = 404;   % [dB/MHz^y cm]
medium.alpha_power = 1.9;     % y

%Source mask params
DROP_RADIUS = 1; %~2.5m radius
DROP_POS = [1, DIM(2)/2+1]; %[X, Y] - NOTE: 1 indexed

MOVIE_NAME = '~/plot';
OUTPUT_DIR = '~/wave_stimulus';
NOISE_SCALE = .1; %1 - NOISE_SCALE = SNR (i.e. 80% is 0.2)

if ne(exist(OUTPUT_DIR),7)
   mkdir(OUTPUT_DIR);
end

disp('MasterScript: Creating wave input...')

%Create input wave
createInput;

%Remove orig drop from matrix
orig_drop = 1 - orig_drop;
all_wave = bsxfun(@times, orig_drop, all_wave); %%TODO: This might be cheating - check with Gar

[Y, X, Z] = size(all_wave);
DIM(3) = Z*2;

disp('MasterScript: Creating noise...');
all_noise = spatialPattern(DIM, BETA);

%Scale noise
std_noise = std(all_noise(:));
new_noise = all_noise ./ std_noise;
new_noise = new_noise .* NOISE_SCALE;

%Scale wave
range_wave = max(all_wave(:)) - min(all_wave(:));
new_wave = all_wave ./ range_wave;
long_wave = zeros(DIM);
long_wave(:,:,DIM(3)/2:end) = new_wave;
new_input = long_wave + new_noise;

%Scale input for imwrite
scale = (max([abs(max(new_input(:))) abs(min(new_input(:)))]) * 2);
scaled_input = new_input ./ scale;
scaled_input = scaled_input .* 255; 
scaled_input = scaled_input + 128;
scaled_input = uint8(floor(scaled_input));


disp('MasterScript: Writing files...');
h = waitbar(0, 'Writing files...');
for i = 1:DIM(3)
   %imwrite(scaled_input(:,:,i),[OUTPUT_DIR,'/scaled_input_',num2str(i),'.jpg']);
   frame_str = sprintf('%03d',i);

   imwrite(abs(scaled_input(:, :, i)),[OUTPUT_DIR,'/input_',frame_str,'.jpg']);
   waitbar(i/DIM(3));
end
close(h);

disp('MasterScript: Done.');
