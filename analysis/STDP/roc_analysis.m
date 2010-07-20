close all
%clear all

% Make the global parameters available at the command-line for convenience.
global NX NY n_time_steps 
global input_dir output_dir conn_dir image_dir output_path input_path


input_dir = '/Users/manghel/Documents/workspace/soren/output/';
%input_dir = '/Users/manghel/Documents/STDP-sim/soren15/';
output_dir = '/Users/manghel/Documents/workspace/soren/output/';
conn_dir = '/Users/manghel/Documents/STDP-sim/conn_probes_8_8/';

image_dir = '/Users/manghel/Documents/workspace/soren/';

num_layers = 5;
n_time_steps = 1200000; % the argument of -n; even when dt = 0.5 



NX=16;  % column size
NY=16;% 8;  % use xScale and yScale to get layer values
dT = 0.5;  % miliseconds
 
% makes ROC curves to measure detection performance

testTime = 50; % ms
restTime = 50; % ms
moveTime = 200000;   % ms  image moves
switchTime = 800000; % ms   image switches

test = zeros(1,20);  % records the number of spikes during test
rest = zeros(1,20);  % and rest periods



% average rates during testTime (image on)
for layer = 2:num_layers; % layer 1 here is layer 0 in PV
    
    if layer == 4
        
        [test_rate, rest_rate] = ...
            roc_layerAnalysis(layer, testTime,restTime, moveTime, switchTime);

        % plot distribution of rest and test firing spikes
        % for the chosen neuron
        figure('Name','Firing rate during 50ms test interval');
        bar(0:(length(test)-1), test, 'r')
        figure('Name','Firing rate during 50ms rest interval');
        bar(0:(length(rest)-1), rest, 'b')
        pause
    end

end



