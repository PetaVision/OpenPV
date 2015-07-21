%% CONTINUOUS-TIME CORRELOGRAM
% using nccc.m and ccc.m provided by Il Park and Antonio Paiva.
close all
%clear all

% Make the global parameters available at the command-line for convenience.
global N  NX NY n_time_steps begin_step tot_steps
global spike_array num_target rate_array target_ndx vmem_array
global input_dir output_dir conn_dir output_path input_path
global patch_size write_step

if 0
st1 = [0 98 243 301];
st2 = [10 42 111 309];
tau = 40;
maxT = 350;
T = 310;
verbose = 0;

for tau = 5:5:100
    fprintf('tau = %d\n',tau);
    [Q, deltaT] = nccc(st1, st2, tau, maxT, T, verbose);
    plot(deltaT, Q,'-b');hold on
    pause
end
pause
end


input_dir = '/Users/manghel/Documents/workspace/soren/output/';
output_dir = '/Users/manghel/Documents/workspace/soren/output/';
conn_dir = '/Users/manghel/Documents/STDP-sim/conn_probes_8_8/';

read_spike_times = 1;

num_layers = 4;
n_time_steps = 100000; % the argument of -n; even when dt = 0.5 



begin_step = 0;       % where we start the analysis (used by readSparseSpikes)
end_step   = 2000;


% post neuron indices
ix_post = 17;
iy_post = 28;
nx_post = 64;
k_post = iy_post * nx_post + ix_post;

% pre neuron indices - we should loop over pre-syn
% neurons in the receptive fiel of the post_syn neuron

ix_pre = 2;
iy_pre = 5;
nx_pre = 16;
k_pre = iy_pre * nx_pre + ix_pre;

% Note: k_post and k_pre are 0 offset indices;
% in mlab we add 1 to these indices 

post_layer = 4; 

% Read relevant file names and scale parameters
[f_file, v_file, w_file, w_last, l_name, xScale, yScale] = stdp_globals( post_layer );

if read_spike_times
    [st_post, av_rate_post ] = ...
        stdp_readSpikes(f_file, k_post, begin_step, end_step);
    fprintf('ave_rate_post = %f\n',av_rate_post);
end

pre_layer = 3;

% Read parameters from previous layer
[f_file_pre, v_file_pre, w_file_pre, w_last_pre, ...
    xScale_pre, yScale_pre] = stdp_globals( pre_layer );

disp('spike time analysis')
begin_step = 0;  % where we start the analysis (used by readSparseSpikes)
end_step   = 100000;

if read_spike_times
    [st_pre, av_rate_pre] = ...
        stdp_readSpikes(f_file_pre, k_pre, begin_step, end_step);
    fprintf('ave_rate_pre = %f\n',av_rate_pre);
end


% shift spike times
T0 =     min([st_post(1) st_pre(1)]);
st_post = st_post - T0;
st_pre = st_pre - T0;

% params for estimating the continuous cross correlation
tau = 40; % in ms
T = max([st_post(end) st_pre(end)]);
maxT = 500;
verbose = 0;

%for tau = 5:5:100
    fprintf('tau = %d\n',tau);
    [Q, deltaT] = nccc(st_post, st_pre, tau, maxT, T, verbose);
    plot(deltaT, Q,'-b');hold on
    pause
%end


