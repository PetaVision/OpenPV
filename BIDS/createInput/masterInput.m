clear all;
close all;

DIM = [512, 512, 0];  % [X, Y, T=0]
BETA = -1;

DROP_WAVE = 1;             %If 1, continous sine wave, otherwise, discrete drops

%Discrete drop params
NUM_DROPS = 4;
DROP_STRENGTH = 5;         

%Sine wave params
WAVE_FREQUENCY = 300;    %Hz
WAVE_STRENGTH = 5;
TS_PER_PERIOD = 16;

%Source mask params
DROP_RADIUS = 5;
DROP_POS = [DIM(1)/4, DIM(2)/4];

MOVIE_NAME = '~/plot';
OUTPUT_DIR = '~/wave_stimulus';
NOISE_SCALE = .1; %1 - NOISE_SCALE = SNR (i.e. 80% is 0.2)



disp('MasterScript: Creating wave input...')

%Create input wave
createInput;

%Parse input wave to ones with stimulus
%all_wave = all_wave(:, :, 1:count);

%Remove orig drop from matrix
orig_drop = 1 - orig_drop;
all_wave = bsxfun(@times, orig_drop, all_wave);

[Y, X, Z] = size(all_wave);
DIM(3) = Z;

if ne(exist(OUTPUT_DIR),7)
   mkdir(OUTPUT_DIR);
end

disp('MasterScript: Creating noise...');
all_noise = spatialPattern(DIM, BETA);

%Find ranges for noise scaling
%range_wave = abs(max(all_wave(:))) + abs(min(all_wave(:)));
%range_noise = abs(max(all_noise(:))) + abs(min(all_noise(:)));
%min_scale = min(all_wave(:))/min(all_noise(:));
%max_scale = max(all_wave(:))/max(all_noise(:));
%avg_scale = mean([min_scale max_scale]);
%
%scaled_noise = all_noise .* (min_scale/NOISE_SCALE);

%master_input = all_wave + scaled_noise;

%Scale noise
std_noise = std(all_noise(:));
new_noise = all_noise ./ std_noise;
new_noise = new_noise .* NOISE_SCALE;

%Scale wave
range_wave = max(all_wave(:)) - min(all_wave(:));
new_wave = all_wave ./ range_wave;
new_input = new_wave + new_noise;

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
   if i < 10
      frame_str = ['00', num2str(i)];
   elseif i < 100
      frame_str = ['0', num2str(i)];
   else
      frame_str = num2str(i);
   end

   imwrite(abs(scaled_input(:, :, i)),[OUTPUT_DIR,'/input_',frame_str,'.jpg']);
   waitbar(i/DIM(3));
end
close(h);

disp('MasterScript: Done.');
