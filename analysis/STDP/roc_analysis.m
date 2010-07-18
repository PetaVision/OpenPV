close all
%clear all

% Make the global parameters available at the command-line for convenience.
global NX NY n_time_steps 
global input_dir output_dir conn_dir output_path input_path


%input_dir = '/Users/manghel/Documents/workspace/STDP/output/';
input_dir = '/Users/manghel/Documents/workspace/soren/output/';
%input_dir = '/Users/manghel/Documents/STDP-sim/soren15/';
%input_dir = '/Users/manghel/Documents/STDP-sim/output-good-w1.5-ltd-1.2-phases/';
%input_dir = '/Users/manghel/Documents/STDP-sim/output-good-w1.5-ltd-1.2-vhbars/';
%input_dir = '/Users/manghel/Documents/STDP-sim/output-final_w3_bf40_bd10_dw04_bars/';
%output_dir = '/Users/manghel/Documents/STDP-sim/soren12/';
output_dir = '/Users/manghel/Documents/workspace/soren/output/';
conn_dir = '/Users/manghel/Documents/STDP-sim/conn_probes_8_8/';

num_layers = 5;
n_time_steps = 40000; % the argument of -n; even when dt = 0.5 
plot_rate = 0;

NX=16;  % column size
NY=16;% 8;  % use xScale and yScale to get layer values
dT = 0.5;  % miliseconds
burstFreq = 25; %50; % Hz  (50=20ms)
burstDuration = 9000000; % since patterns move %7.5; 

% makes ROC curves to measure detection performance

disp('read spikes and plot average activity (rate aray)')
begin_step = 0;  % where we start the analysis (used by readSparseSpikes)
end_step   = 0;

testTime = 50; % ms
restTime = 50; % ms

while end_step < n_time_steps
    
    % average rates during testTime (image on)
    for layer = 2:num_layers; % layer 1 here is layer 0 in PV
        test_array{layer} = [];
        % Read relevant file names and scale parameters
        [f_file, v_file, w_file, w_last, l_name, xScale, yScale] = stdp_globals( layer );
        [test_array{layer}, ave_rate] = stdp_readAverageActivity(f_file, begin_step, begin_step + 2*testTime);
        %disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate)]);

        % now plot rates
        NXscaled = NX * xScale;
        NYscaled = NY * yScale;

        if plot_rate
            figure('Name',['Average Rate for ',l_name, ' testTime']);
            recon2D = reshape( test_array{layer}, [NXscaled, NYscaled] );
            %     recon2D = rot90(recon2D);
            %     recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            imagesc( recon2D' );  % plots recon2D as an image
            colorbar
            axis square
            axis off
            %pause
        end

    end
    
    if begin_step == 0
        
        % load pre-synaptic weight patches and plot them  
        [f_file, v_file, w_file, w_last, l_name, xScale, yScale] = ...
            stdp_globals( 4 );
        NXscaled = NX * xScale;
        NYscaled = NY * yScale;
        
        nxMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
        nyMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
           
        % find neuron with largest firing rate
        % that is not a boundary neuron
        [sortR, sortI] = sort(test_array{4},'descend');
        
        
        for i=1:length(sortI)
            k = sortI(i);  % linear index
            I = mod(k-1,NXscaled);
            J = (k-1-I) / NXscaled;
            % check if not boundary neuron
                if J >= nyMar && J <= (NYscaled-nyMar-1) ...
                        && I >= nxMar && I <= (NXscaled-nxMar-1)
                    maxRate = test_array{4}(sortI(i));
                    maxI = sortI(i);
                    fprintf('max rate (I= %d, J = %d)\n',I,J);
                    break
                end
        end
        
        %[maxRate,maxI] = max(test_array{4});
        fprintf('%d test = %f ', maxI, maxRate);
               
        
        
        for i=1:length(w_last)
            [PATCH, patch_size, NXP, NYP] = ...
                roc_readPatch(w_last{i}, I, J, NXscaled, NYscaled);
            figure('Name',[l_name,' patch ',num2str(i)]);
            PATCH = reshape(PATCH,[NXP,NYP]);
            imagesc(PATCH,'CDataMapping','direct');
            colorbar
            axis square
            axis off
        end
        pause
        
    else
        fprintf('%d test = %f ', maxI, test_array{4}(maxI));
    end
    
    begin_step = begin_step + 2*testTime;

    % average rates during restTime (image off)
    for layer = 2:num_layers; % layer 1 here is layer 0 in PV
        rest_array{layer} = [];
        % Read relevant file names and scale parameters
        [f_file, v_file, w_file, w_last, l_name, xScale, yScale] = stdp_globals( layer );
        [rest_array{layer}, ave_rate] = stdp_readAverageActivity(f_file, begin_step, begin_step + 2*restTime);
        %disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate)]);

        % now plot rates
        NXscaled = NX * xScale;
        NYscaled = NY * yScale;

        if plot_rate
            figure('Name',['Average Rate for ',l_name,' restTime']);
            recon2D = reshape( rest_array{layer}, [NXscaled, NYscaled] );
            %     recon2D = rot90(recon2D);
            %     recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            imagesc( recon2D' );  % plots recon2D as an image
            colorbar
            axis square
            axis off
            %pause
        end
        
    end

    fprintf(' rest = %f\n', rest_array{4}(maxI));
    
    begin_step = begin_step + 2*restTime;
    end_step = end_step + testTime + restTime;
    
    %pause
    close all

end % while loop over end_step

