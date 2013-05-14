%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulate non-attenuating wave propagating through space
%%
%%    To achieve reported units - 
%%       All time parameters are multiplied by alpha = 0.75e6/125 = 6000
%%       All spatial (distance) parameters are multiplied by beta = (350/1500)*(6000) = 1400
%%    
%%    Desired units:
%%       Grid size:          60m x 60m
%%       Time step:          0.12e-3 s
%%       Wave freq:          125 Hz
%%       Wave speed:         350 m/s
%%       Object speed:       8.9408 m/s (20mph)
%%       Medium Attenuation: none
%%       Wave amplitude:     arbitrary - scaled with respect to noise levels to fix SNR
%%
%%    The following script generates png files that can be interpreted (using the above units)
%%    as a pure tone sine wave representation of a vehicle's noise spectrum. The predominant
%%    frequency for a vehicle is at about 125 Hz, which is found in Hillquist et al, 1975.
%%
%%  D M Paiton, G T Kenyon, S Y Lundquist
%%  2013
%%  Los Alamos National Laboratory, New Mexico Institute of Mining and Technology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
close all;
addpath('./k-Wave Toolbox');

%World properties
DIM  = [256, 256, 0];      % [X, Y, t=0]
dx   = 1.5625e-4;          % [m/px]
dy   = dx;                 % [m/px]

%Sine wave params
% peak vehicle frequency
WAVE_FREQUENCY = 0.75e6;   % [Hz]
WAVE_STRENGTH  = 3;        % [au]

%Time properties
SOURCE_VEL  = 38.31771;    % [m/s]
dt          = 20e-9;       % [s]

%Medium properties
medium.sound_speed = 1500; % [m/s]
medium.alpha_coeff = 0;    % [dB/MHz^y cm]
medium.alpha_power = 1.01; % y

%Radius for removing center dot
SOURCE_RAD = 10;           % [px]

%Noise properties
BETA = -1;                 % 0 is gaussian white, -1 is pink, -2 is Brownian
NOISE_SCALE = .01;         % (1 - NOISE_SCALE) = SNR (i.e. 80% is 0.2)

%File Locations
SIMULATION_FILENAME = './tiny_simulation_output.mat';
NOISE_FILENAME      = './noise_output_90.mat';
OUTPUT_DIR          = '~/wave_stimulus_90';

%Clobbering preferences
CLOBBER_SIMULATION = 0;
CLOBBER_NOISE      = 0;

%%%%%%%%%%%%%
%Main code
%%%%%%%%%%%%%

if ne(exist(OUTPUT_DIR),7)
   mkdir(OUTPUT_DIR);
end

if ~exist(SIMULATION_FILENAME) || CLOBBER_SIMULATION
    disp('masterInput: Creating wave input...')

    %Create input wave
    createInput;

    %Remove orig drop from matrix
    %orig_drop = 1 - orig_drop;
    %all_wave = bsxfun(@times, orig_drop, all_wave); %%TODO: This might be cheating - check with Gar

    %Save
    disp('Saving simulation output...')
    save(SIMULATION_FILENAME,'all_wave','-v7.3');
else
    disp('masterInput: Loading wave input...')
    load(SIMULATION_FILENAME);
end

[Y, X, Z] = size(all_wave);
DIM(3) = 2*Z; %Need an equal set of noisy frames without the stimulus
disp('masterInput: Scaling simulation...')

%Scale wave
range_wave = max(all_wave(:)) - min(all_wave(:));
assert(range_wave>0);
new_wave   = all_wave ./ range_wave;
long_wave  = zeros(DIM);
long_wave(:,:,DIM(3)/2+1:end) = new_wave;

clearvars -except long_wave DIM BETA OUTPUT_DIR NOISE_SCALE NOISE_FILENAME CLOBBER_NOISE

if ~exist(NOISE_FILENAME) || CLOBBER_NOISE
    %Generate noise
    disp('masterInput: Creating noise...');
    all_noise = spatialPattern(DIM, BETA);

    %Scale noise
    std_noise = std(all_noise(:));
    new_noise = all_noise ./ std_noise;
    new_noise = new_noise .* NOISE_SCALE;
    disp('masterInput: Saving noise output...')
    save(NOISE_FILENAME,'new_noise','-v7.3');
else
    disp('masterInput: Loading noise...');
    load(NOISE_FILENAME);
end

disp('masterInput: Combining noise with simulation output...')
%Scale input for imwrite
new_input    = long_wave + new_noise;
clearvars -except new_input DIM OUTPUT_DIR

scale        = (max([abs(max(new_input(:))) abs(min(new_input(:)))]) * 2);
assert(ne(scale,0));

scaled_input = new_input ./ scale;
scaled_input = scaled_input .* 255; 
scaled_input = scaled_input + 128;
scaled_input = uint8(floor(scaled_input));

disp('masterInput: Writing files...');
%h = waitbar(0, 'Writing files...');
for i = 1:DIM(3)
    frame_str = sprintf('%05d',i);
    imwrite(abs(scaled_input(:,:,i)),[OUTPUT_DIR,'/input_',frame_str,'.jpg']);
    %waitbar(i/DIM(3));
end
%close(h);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clearvars -except new_input
disp('masterInput: Done.');
