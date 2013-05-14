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
DIL_RAD = 5;              % [px]

%Noise properties
BETA = -1;                 % 0 is gaussian white, -1 is pink, -2 is Brownian
SNR  = [1.0 0.8 0.6 0.4 0.2 0.1 0.05 0.025];      % Must be between 0 and 1, inclusive. 0.8 is 80% SNR

%File Locations
SIMULATION_FILENAME = 'tiny_simulation_output';
NOISE_FILENAME      = 'noise_output';
OUTPUT_DIR          = '~/wave_stimulus';

%Clobbering preferences
CLOBBER_SIMULATION = 0;
CLOBBER_NOISE      = 0;

%%%%%%%%%%%%%
%Main code
%%%%%%%%%%%%%

if ~exist(['./',SIMULATION_FILENAME,'.mat']) || CLOBBER_SIMULATION
    disp('masterInput: Creating wave input...')

    %Create input wave
    createInput;
    close all;
    fclose('all');

    %Save
    disp('masterInput: Saving simulation output...')
    save(['./',SIMULATION_FILENAME,'.mat'],'all_wave','-v7.3');
    save(['./',SIMULATION_FILENAME,'_source.mat'],'source','-v7.3');
else
    disp('masterInput: Loading wave input...')
    load(['./',SIMULATION_FILENAME,'.mat']);
    %load(['./',SIMULATION_FILENAME,'_source.mat']);
end

all_wave = small_wave;
clear small_wave;

[Y, X, Z] = size(all_wave);
DIM(3)    = Z; %Need an equal set of noisy frames without the stimulus

disp('masterInput: Scaling simulation...')
%Remove orig drop from matrix
wave_mask = ones(DIM(2),DIM(1),length([1:100:DIM(3)]));
wave_t = 1;
for t = 1:100:DIM(3)
    indi_wave = all_wave(:,:,t);
    [val ind] = max(abs(indi_wave(:)));
    [idy idx] = ind2sub(size(indi_wave),ind);
    if gt(idy+DIL_RAD,DIM(2))
        yMax = DIM(2);
    else
        yMax = idy+DIL_RAD;
    end
    if gt(idx+DIL_RAD,DIM(1))
        xMax = DIM(1);
    else
        xMax = idx+DIL_RAD;
    end
    if lt(idy-DIL_RAD,1)
        yMin = 1;
    else
        yMin = idy-DIL_RAD;
    end
    if lt(idx-DIL_RAD,1)
        xMin = 1;
    else
        xMin = idx-DIL_RAD;
    end
    wave_mask(yMin:yMax,xMin:xMax,wave_t) = 0;
    wave_t = wave_t + 1;
end
sub_wave = all_wave(:,:,1:100:end).*wave_mask;
threshold = max(abs(sub_wave(:)));
all_wave(all_wave>threshold)=threshold;
all_wave(all_wave<-threshold)=-threshold;

DIM(3) = 2*Z; %Need an equal set of noisy frames without the stimulus

%Scale wave
new_wave   = all_wave ./ std(all_wave(:)); %RMS Amplitude = standard deviation
long_wave  = zeros(DIM);
long_wave(:,:,DIM(3)/2+1:end) = new_wave;

clearvars -except long_wave SNR DIM BETA OUTPUT_DIR NOISE_SCALE NOISE_FILENAME CLOBBER_NOISE

if ~exist(['./',NOISE_FILENAME,'.mat']) || CLOBBER_NOISE
    %Generate noise
    disp('masterInput: Creating noise...');
    all_noise = spatialPattern(DIM, BETA);

    %Scale noise
    new_noise  = (all_noise ./ std(all_noise(:))) - mean(all_noise(:));
    disp('masterInput: Saving noise output...')
    save(['./',NOISE_FILENAME,'.mat'],'new_noise','-v7.3');
else
    disp('masterInput: Loading noise...');
    load(['./',NOISE_FILENAME,'.mat']);
end

disp('masterInput: Combining noise with simulation output...')
for i_snr = 1:length(SNR)
    NEW_OUTPUT_DIR = [OUTPUT_DIR,'_',num2str(SNR(i_snr)*100)];
    if ne(exist(NEW_OUTPUT_DIR),7)
       mkdir(NEW_OUTPUT_DIR);
    end

    %Scale input for imwrite
    new_input    = (SNR(i_snr)*long_wave) + new_noise;

    scale        = (max([abs(max(new_input(:))) abs(min(new_input(:)))]) * 2);
    assert(ne(scale,0));

    scaled_input = new_input ./ scale;
    scaled_input = scaled_input .* 255; 
    scaled_input = scaled_input + 128;
    scaled_input = uint8(floor(scaled_input));

    disp('masterInput: Writing files...');
    for i = 1:DIM(3)
        frame_str = sprintf('%05d',i);
        imwrite(abs(scaled_input(:,:,i)),[NEW_OUTPUT_DIR,'/input_',frame_str,'.jpg']);
    end
end

%clearvars -except new_input
disp('masterInput: Done.');
