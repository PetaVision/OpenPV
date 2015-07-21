clear all

global ver w0  w0_rect  w  h  screen_xC  screen_yC  refresh_rate manual_timing exit_experiment exp

w = 512; h = 512;           % amoeba image width and height
ListenChar(2);              % prevents key stokes from displaying  
ver = version;              % MATLAB version
KbName('UnifyKeyNames');    % Switches internal naming scheme to MacOS-X naming scheme   
exit_experiment = false; 


%% random number generator
seed = sum(100*clock);
rand('twister', seed);      % reseed the random-number generator for each expt.
state = rand('twister');
data_file = ['..' filesep 'exp_data' filesep 'sosExp', num2str(round(seed))];


%% init window
[w0, w0_rect] = Screen('OpenWindow',0);
info = Screen('GetWindowInfo',w0);
screen_xC = w0_rect(3)/2;          % x midpoint
screen_yC = w0_rect(4)/2;          % y midpoint
Screen('TextSize', w0, 36);
refresh_rate = Screen('GetFlipInterval',w0);


%% init choice params
[instr_text, tmp] = ...
    sprintf('%6s\n', ...
    'Options:', ...
    'up arrow: target present', ...
    'down arrow: target absent', ...
    'right arrow: advance to next trial', ...
    'esc once to pause, twice to exit');
%         'left arrow: switch previous choice and discard current trial', ...

choice_text_rect = RectWidth( Screen( 'TextBounds', w0, instr_text ) );

%% init exp values

num_labels = 2;                 % amoeba
%num_labels = 4;                % image from file
exp = setExpValues(refresh_rate, num_labels);
exp.save_mode = false;
exp.amoeba = true;              % true: amoeba; false: image from file 
exp.save_mode = false;          % saves masks and generated images
exp.training = false;           % show score
exp.segment_total = 30;         % total number of segments (including target segments)
exp.mask_mode = 0;              % 0 mask with noise; 1 mask with segments 
exp.waitframes = 1;             % number of frames to wait before mask onset
exp.grayscale_flag = true;

%manual timing values
exp.t0 = zeros(exp.tot_trials,1);   
exp.t1 = zeros(exp.tot_trials,1);
exp.t2 = zeros(exp.tot_trials,1);
exp.t3 = zeros(exp.tot_trials,1);
exp.t4 = zeros(exp.tot_trials,1);
exp.t5 = zeros(exp.tot_trials,1);

if info.Beamposition == -1
    manual_timing = true;
else
    manual_timing = false;
end

duration_delta = ( exp.duration_max - exp.duration_min + ( exp.num_duration_vals <= 1 ) ) / ...
    ( exp.num_duration_vals - 1 + ( exp.num_duration_vals <= 1 ) );

exp.duration_vals = ...
    exp.duration_min : duration_delta : exp.duration_max;

exp.key_name = cell(exp.tot_trials, 1);
exp.obj = cell(exp.tot_trials, 2);

if ~exp.amoeba
   
    target_files = dir('../AnimalDB/Targets/*.jpg');
    control_files = dir('../AnimalDB/Distractors/*.jpg');

    [exp.shuffled_files, file_ndx] = Shuffle([target_files; control_files]);
    exp.target_flag = ones(exp.tot_trials,1);
    
elseif exp.amoeba
    exp.target_flag = logical(round(rand(exp.tot_trials,1)));
end

%set background color 
Screen('FillRect', w0, GrayIndex(w0));
Screen('Flip', w0);


%% start experiment

experiment();
   

