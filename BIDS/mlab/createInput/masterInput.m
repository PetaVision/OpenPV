clear all;
close all;
addpath('./k-Wave Toolbox');

SIMULATION_FILENAME1 = './simulation_output.mat';
SIMULATION_FILENAME2 = './modified_simulation_output.mat';
NOISE_FILENAME       = './noise_output.mat';

DIM  = [512, 512, 0];   % [X, Y, t=0] This will be a 2560x2560m (640*4) grid for the simulation (2.5 m/px).
dx   = 1.25;             % [m]
dy   = dx;              % [m]
BETA = -1;              % 0 is gaussian white, -1 is pink, -2 is Brownian

%Sine wave params
% peak vehicle frequency
WAVE_FREQUENCY = 125;   %Hz
WAVE_STRENGTH  = 0.11;  %Pa (75dB SBL)

%Time properties
SOURCE_VEL  = 8.9408;   % [m/s] = 20 mph
TIME_LENGTH = 60;      % [s]
dt          = 10e-3;    % [s] - 1ms

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
disp('Saving simulation output...')
save(SIMULATION_FILENAME1,'all_wave','-v7.3');
clearvars -except DIM BETA NOISE_SCALE NOISE_FILENAME OUTPUT_DIR

all_wave  = matfile(SIMULATION_FILENAME1);
range_wave(i) = zeros(DIM(3));
for i = 1:DIM(3)
    %Scale wave
    range_wave(i) = max(all_wave(:,:,i)) - min(all_wave(:,:,i));
end
new_wave = all_wave ./ (max(range_wave(:))-min(range_wave(:)));
long_wave = zeros(DIM);
long_wave(:,:,DIM(3)/2:end) = new_wave;
save(SIMULATION_FILENAME2,'long_wave','-v7.3');
clearvars -except DIM BETA NOISE_SCALE NOISE_FILENAME OUTPUT_DIR

disp('MasterScript: Creating noise...');
all_noise = spatialPattern(DIM, BETA);

%Scale noise
std_noise = std(all_noise(:));
new_noise = all_noise ./ std_noise;
new_noise = new_noise .* NOISE_SCALE;
save(NOISE_FILENAME,'new_noise','-v7.3');
clearvars -except DIM OUTPUT_DIR

long_wave  = matfile(SIMULATION_FILENAME2);
new_noise = matfile(NOISE_FILENAME);

disp('MasterScript: Scaling input...');

scale_time = zeros(DIM(3));
for i = 1:DIM(3)
    new_input = long_wave(:,:,i) + new_noise(:,:,i);
    scale_time(i) = (max([abs(max(new_input(:))) abs(min(new_input(:)))]) * 2);
end

disp('MasterScript: Writing files...');
h = waitbar(0, 'Writing files...');
for i = 1:DIM(3)
   
    %Add simulation output to noise
    new_input = long_wave(:,:,i) + new_noise(:,:,i);

    %Scale input for imwrite
    scaled_input = new_input ./ scale(i);
    scaled_input = scaled_input .* 255; 
    scaled_input = scaled_input + 128;
    scaled_input = uint8(floor(scaled_input));

    %imwrite(scaled_input(:,:,i),[OUTPUT_DIR,'/scaled_input_',num2str(i),'.jpg']);
    frame_str = sprintf('%03d',i);

    imwrite(abs(scaled_input(:, :)),[OUTPUT_DIR,'/input_',frame_str,'.jpg']);
    waitbar(i/DIM(3));
end
close(h);

disp('MasterScript: Done.');
