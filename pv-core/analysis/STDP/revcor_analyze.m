%% Driver for performing reverse correlation analysis
% It computes conditional statistics of pre-synaptic neurons
% - we record the statistics conditioned on the post-synaptic neuron
% firing.
% - we record this statistics in a window of size W
% - The window can be causal (before the post-synaptic neuron fires)
% or acausal (after the post-synaptic neuron fires).
% NOTES: 
% 1) By passing any pair of layers we can analyze the conditional
% statistics betwen neurons in any two layers.
% 2) We drive the retina with random noise images and we turn off STDP
% learning
% 3) For each post-synaptic neuron we record the average activity in the 
% image. We do not need to consider only the receptive field of the neuron!
% By averaging over the noise, conditioned on the spiking time of the
% neuron, the image activity outside the receptive field that triggers the
% activity of the neuron will average to a uniform noise background. We
% only get some activity modulation for the image pixels in the receptive
% field of the neuron.

close all

% Make the global parameters available at the command-line for convenience.

global input_dir output_dir conn_dir n_time_steps dT

input_dir  = '/Users/manghel/Documents/workspace/earth/output/';
output_dir = '/Users/manghel/Documents/workspace/STDP-sim/movie10/rev_corr/';
conn_dir   = '/Users/manghel/Documents/workspace/STDP-sim/movie10/rev_corr/';


n_time_steps = 1000000;

    
NX            = 32;
NY            = 32;      % use xScale and yScale to get layer values
dT            = 0.5;     % miliseconds

comp_revCorr    = 0;
output_check    = 0;
analyze_revCorr = 1;


%% We compare the "spiking" activity in the image with the 
% saved image patch tif file.
if output_check
    
    begin_step   = 0;  % simulation steps
    end_step     = 100;

    [f_file, v_file, w_file, w_last, l_name, xScale, yScale] = ...
        stdp_globals( 1);

    stdp_outputCheck(f_file, begin_step, end_step);
    
    
end % output_check


layer1 = 4;  % post synaptic layer
layer2 = 1;  % pre synaptic layer

     
if (comp_revCorr)              
    disp('compute STDP statistics')
    % Read relevant file names and scale parameters for layer 1
    [f_file1, v_file1, w_file1, w_last1, l_name1, xScale1, yScale1] = ...
        stdp_globals( layer1 );
 
    % Read relevant file names and scale parameters for layer 1
    [f_file2, v_file2, w_file2, w_last2, l_name2, xScale2, yScale2] = ...
        stdp_globals( layer2 );

    for W=40:100:60
        
        fprintf('compute rev correlations for window size W = %d\n',W);
        
        [prePatchAverage, nPostSpikes, kPostIndices, timeWeightsSum] = ...
            stdp_revCorrelations(f_file1, f_file2, W);
        
        
    end % loop over W (pre-synaptic activity time window length)
    
end % comp_revCorr

    
if analyze_revCorr  
    
    % Read relevant file names and scale parameters for layer 1
    [f_file1, v_file1, w_file1, w_last1, l_name1, xScale1, yScale1] = ...
        stdp_globals( layer1 );
    NXpost = NX * xScale1;
    
    for W=40:100:60
        %% read indices of non-margin post-synaptic neurons
        % and their spiking numbers
        filename = [output_dir,'nPostSpikes_',num2str(W),'.dat'];
        fid = fopen(filename, 'r');
        data = load(filename); % N x 2 array
        kPostIndices = data(:,1);
        nPostSpikes = data(:,2);
        
        for nPost = 1:numel(kPostIndices)
            kPost = kPostIndices(nPost);
            fprintf('kPost = %d nPost = %d\n',kPost,nPostSpikes(nPost));
            
            kxPost=rem(kPost-1,NXpost);
            kyPost=(kPost-1-kxPost)/NXpost;
            filename = [output_dir,'STDP_pre_' num2str(W) '_' num2str(kPost) '.dat'];
            data = load(filename);
            imagesc(data,'CDataMapping','direct');
            title(['kPost ' num2str(kPost) ' (' num2str(kxPost) ' , ' num2str(kyPost) ' )'] );
            colorbar
            axis square
            axis off
            pause
            
        end % loop over post-synaptic neurons
        
    end % loop over W (window time length)
    
end % analyze_revcorr




    