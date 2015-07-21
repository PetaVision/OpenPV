% analysis of adaptive wMax/VthRest models

% Make the global parameters available at the command-line for convenience.
global NX NY 
global input_dir output_dir conn_dir n_time_steps

if 1
    input_dir  = '/Users/manghel/Documents/workspace/soren/output/';
    output_dir = '/Users/manghel/Documents/workspace/soren/output/';
end

if 0
    input_dir  = '/Users/manghel/Documents/workspace/STDP-sim/HCmovie4/';
    output_dir = '/Users/manghel/Documents/workspace/STDP-sim/HCmovie4/';
end

if 0
    input_dir  = '/Users/manghel/Documents/workspace/STDP-sim/SC12/';
    output_dir = '/Users/manghel/Documents/workspace/STDP-sim/SC12/';
end

if 0
    input_dir  = '/Users/manghel/Documents/workspace/STDP-sim/wMax15/';
    output_dir = '/Users/manghel/Documents/workspace/STDP-sim/wMax15/';
end

conn_dir = '/Users/manghel/Documents/workspace/STDP-sim/PCProbes/';



INHIB = 0;
if INHIB
    num_layers = 7; % image + 2 retinas + S1 + S1Inh + C1 + C1Inh
else
    num_layers = 4; % image + 2 retinas + S1 + C1 or 
                    % image + 2 retinas + S1 + S1Inh (INHIB is misleading)
end
n_time_steps = 300000;

begin_step =  0;   % where we start the analysis (used by readSparseSpikes)
end_step   =  5000;

NX         = 64; %16; %32;
NY         = 64; %16; %32;   % use xScale and yScale to get layer values
averageR   = 15;   %10;

debug                  = 0;
plot_wMax              = 0;  % evolution of wMax distribution, R, Vth 
comp_scores            = 0;
plot_rate              = 0;
     read_R_evolution  = 0;
     read_spikes       = 0;  % inside plot_rate 
                             % Choose begin and end_step correctly to read
                             % spikes!!
     plot_raster       = 0;
     read_rate_array   = 1;  % computes average firing rate
                             % you don't ned to read_spikes first
                             % NOTE: if [begin_step, end_step] is not 
                             % within the [0 n_time_steps] interval, this 
                             % routine will read a
                             % truncated sequence - only the part within
                             % this interval, or nothing at all, and return
                             % an empty rate array.
         write_rate    = 0;  % writes computed rate to a file
             threshold_rate = 1; % threshold rate and write to file as binary
         plot_spikes   = 0;  % for rates above a chosen threshold
              
plot_weights_histogram = 0; % evolution of weights histogram
            HIST_STEP  = 3; % plot hist every HIST_STEP
plot_weights_field     = 1;
            FIELD_STEP = 3; % plot weights field every FIELD_STEP
plot_rate_field        = 0;
read_probe             = 0; 
    read_probe_data    = 1;
    advanceWmax        = 1;
    n_files            = 3; % number of probe files
    Cprobes            = 0;

%% K means analysis
comp_weights_Kmeans     = 0; % add Documents/MATLAB/Kmeans to path
comp_weights_KmeansAD   = 0; % add Documents/MATLAB/KmeansMA to path
comp_conc_weights_Kmeans= 0; % add Documents/MATLAB/Kmeans to path
numCenters              = 32;% param for comp_weights_Kmeans,
                             % comp_conc_weights_Kmeans, and
                             % comp_score_evolution
                             
                             
read_features          = 0;
    read_feature_files = 0;
    read_weights_files = 1;
    read_wMax_file     = 1;
    print_features     = 0;
        print_indexes  = 1;
        print_weights  = 1;
    comp_overlap       = 0;
    plot_overlap_hist  = 0;
    plot_features_weights = 0;
    comp_features_size = 1;
    
% layer 1 is the image layer
for layer = 1:num_layers; % layer 1 here is layer 0 in PV
    % Read relevant file names and scale parameters
    
    if INHIB
        [f_file, v_file, r_file, w_file, w_last, ...
            l_name, xScale, yScale] = wmax_globalsSC( layer );
    else
        [f_file, v_file, r_file, w_file, w_last, ...
            l_name, xScale, yScale] = wmax_globals( layer );
    end
    
    if comp_weights_Kmeans == 1 & layer >= 4
        disp('compute K-means of the  weight fields: ')
        for i=1:numel(w_last)
            stdp_compWeightsKmeans(w_last{i},numCenters, xScale,yScale);
            pause
        end
        
    end
    
    if comp_weights_KmeansAD == 1 & layer == 4
        disp('compute K-means of the  weight fields: ')
        disp('(select the number of centers based on Anderson-Darling test)')
        for i=1:numel(w_last)
            stdp_compWeightsKmeansAD(w_last{i},xScale,yScale);
            pause
        end
        
    end
    
    if comp_conc_weights_Kmeans == 1 & layer == 4
        disp('compute K-means for concatenated weight fields: ');
        stdp_compConcWeightsKmeans(w_last,numCenters, xScale,yScale);
    end
    
    %% Analyze rate and spikes
    if plot_rate & layer >= 2
        
        NXscaled = NX * xScale; % L1 size
        NYscaled = NY * yScale;
        
        
        if read_R_evolution % reads the evolution of rate vs time!!
            disp('read R evolution')
            fname = strcat(l_name,'_R.pvp');
            R = 1000.0 * wmax_readFile(fname, l_name, xScale, yScale);
            size(R)
            avR = mean(R,1);
        
        
            %% reshape and plot R (rate) field
        
            figure('Name',['last R for ',l_name]);
            recon2D = reshape(R, [NXscaled, NYscaled] );
            %recon2D = rot90(recon2D);
            %recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            imagesc( recon2D' );  % plots recon2D as an image
            colorbar
            axis square
            axis off
        end
        

        if 0
            for n=1:size(V,2)
                plot(R(:,n),'-b');
                pause
            end
        end


        % Read spike events

        if read_spikes           
            [spike_array, ave_rate] = ...
                stdp_readSparseSpikes(f_file, begin_step, end_step);
            disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate)]);
            % spike_array is T x N array
            tot_steps = size( spike_array, 1 );
            tot_neurons = size( spike_array, 2);
            fprintf('tot_steps = %d tot_neurons = %d\n',...
                tot_steps,tot_neurons);
            
            avS = mean(spike_array,1);
            
            %figure('Name','Average R vs Average Rate');
            %plot(avS, avR,'ob');
            
        end
        
        if plot_raster
            disp('plot raster')
            N = NX*xScale * NY *yScale;
            if ~isempty(spike_array)
                plot_title = ['Raster for ' l_name];
                figure('Name',plot_title);
                axis([0 tot_steps 0 N]);
                hold on
                box on
                
                [spike_time, spike_id] = find(spike_array);
                lh = plot(spike_time, spike_id, '.g');
                %set(lh,'Color',my_gray);
                pause
            end
        end
        
        % look at the distribution of R conditioned on a 
        % fixed number of spikes in this time interval
        if 0
            sumS = sum(spike_array,1);
            avR = mean(R,1);
            for k=1:9
                nbins = 20;
                fprintf('num spikes = %d\n',k);
                ind = find(sumS == k);
                hist(avR(ind),nbins);
                pause
                
            end
        end
        
        if read_rate_array
            disp('read spikes and plot average activity (rate aray)')
            %begin_step = 0;  % where we start the analysis (used by readSparseSpikes)
            %end_step   = 4500;
            [rate_array{layer}, ave_rate] = stdp_readAverageActivity(f_file, begin_step, end_step);
            %disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate)]);
            
            %disp('plot_rate_reconstruction')
            %stdp_reconstruct(rate_array{layer}, NX*xScale, NY * yScale, ...
            %    ['Rate reconstruction for layer  ', int2str(layer)]);
            
            NXscaled = NX * xScale;
            NYscaled = NY * yScale;
            recon2D = reshape( rate_array{layer}, [NXscaled, NYscaled] );
            %recon2D = rot90(recon2D);
            %recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            % threshold rate to remove boundary effects
            ind = find(recon2D > 45);
            recon2D(ind) = 0.0;
            fprintf('min_rate = %f max_rate = %f\n',...
                min(recon2D(:)),max(recon2D(:)));
            %% remove boundary layer
            % the size of the boundary is layer dependent
            if layer < 4
                %rate2D = recon2D(2:end-1,2:end-1);
            elseif layer == 4
                %rate2D = recon2D(8:end-7,8:end-7);
            else
                %rate2D = recon2D(2:end-1,2:end-1);
            end
            
            % compute rate
            %disp(['ave_rate(',num2str(layer),') = ', num2str(mean(rate2D(:)))]);
            disp(['ave_rate(',num2str(layer),') = ', num2str(mean(recon2D(:)))]);
            figure('Name',['Average Rate for ',l_name]);
            imagesc( recon2D');  % plots recon2D as an image
            %imagesc( rate2D');  % plots recon2D as an image
            title(['\fontsize{12}{\bf Rate for layer ' l_name '}']);
            %title(['{\bf Rate for layer ' l_name '}']);
            colorbar
            axis square
            axis off
            
            
            % histogram plot
            figure('Name',['Rate Histogram for ',l_name]);
            %hist(rate_array{layer}, 40 );
  
            nbins = 40;
            %hist(rate2D(:), nbins );
            hist(recon2D(:), nbins );
            pause
                        
            
            if plot_spikes
                Threshold = input('Rate threshold: ');
                ind = find(rate_array{layer} > Threshold);
                disp('plot spike activity');
                figure('Name','Spike Activity');
                for j=1:length(ind)
                   hold off
                   [T, bla] = find(spike_array(:,ind(j)));
                   for k=1:length(T)
                      plot([T(k) T(k)], [0 1],'-r');hold on 
                   end
                   pause
                end
            end
            
            if write_rate
                
                if threshold_rate
                    TRate = zeros(NYscaled,NXscaled);
                    while 1
                        TRate(:) = 0;
                        T = input('threshold rate: ');
                        for j=1:NYscaled
                            for i=1:NXscaled
                                
                                    if(recon2D(j,i) > T)
                                        TRate(j,i) = 1;
                                    else
                                        TRate(j,i) = 0;
                                    end
                           
                            end
                        end
                         imagesc(TRate);
                         axis square
                         axis off
                         reply = input('another rate? Y/N [Y]: ','s');
                         if isempty(reply)
                             reply = 'Y'
                         end
                         if reply == 'n' | reply == 'N'
                             break
                         end
                    end
                end
                
                filename = [output_dir, f_file, '.rate'];
                fid=fopen(filename,'w');
                for j=1:NYscaled
                    for i=1:NXscaled 
                        if threshold_rate
                            if(recon2D(j,i) > T)
                                fprintf(fid,'1 ')
                            else
                                fprintf(fid,'0 ')
                            end
                        else
                            fprintf(fid,'%f ',recon2D(j,i))
                        end
                    end
                    fprintf(fid,'\n');
                end
                fclose(fid);
            end
            
        end % read_rate_array
        
    end % plot_rate
    
    %% Analysis of adaptive wMax thresholds
    %if plot_wMax & (layer == 4 | layer == 6) % INHIB
    if plot_wMax & (layer == 4 | layer == 5)    
        
        NXscaled = NX * xScale; % L1 size
        NYscaled = NY * yScale;
        
        % read aWmax file
        fname = [l_name '_aWmax_last.pvp'];
        aWmax = wmax_readLastFile(fname, l_name, xScale, yScale);
        fprintf('length(aWmax) = %d\n',length(aWmax));
        %% histogram plot
        figure('Name',['aWax Histogram for ',l_name]);
        hist(aWmax,50);
        
        
        %% reshape and plot aWmax field
        if 1
            figure('Name',['aWmax for ',l_name]);
            % transposed: see the remarks on reshape in wmax_readLastFile.m
            recon2D = reshape(aWmax, [NXscaled, NYscaled] )';
            %recon2D = rot90(recon2D);
            %recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            imagesc( recon2D );  % plots recon2D as an image
            title(['\fontsize{12}{\bfaWmax for layer ' l_name '}']);
            colorbar
            axis square
            axis off
            pause
        end

        %fname = 'Wmax4_last.pvp';
        fname = [l_name '_Wmax_last.pvp'];
        wMax = wmax_readLastFile(fname, l_name, xScale, yScale);
        fprintf('length(wMax) = %d\n',length(wMax));
        %% histogram plot
        figure('Name',['wMax Histogram for ',l_name]);
        hist(wMax,50);
        
        
        %% reshape and plot wMax field
        if 1
            figure('Name',['wMax for ',l_name]);
            % transposed: see the remarks on reshape in wmax_readLastFile.m
            recon2D = reshape(wMax, [NXscaled, NYscaled] )';
            %recon2D = rot90(recon2D);
            %recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            imagesc( recon2D );  % plots recon2D as an image
            title(['\fontsize{12}{\bfwMax for layer ' l_name '}']);
            colorbar
            axis square
            axis off
        end

        fname = [l_name '_R_last.pvp'];
        R = 1000.0 * wmax_readLastFile(fname,l_name,xScale, yScale);
        fprintf('length(wMax) = %d\n',length(wMax));
        
        %% histogram plot
        figure('Name',['R Histogram for ',l_name]);
        hist(R,50);
        
        %% reshape and plot R field
        if 1
            figure('Name',['R for ',l_name]);
            % transposed: see the remarks on reshape in wmax_readLastFile.m
            recon2D = reshape(R, [NXscaled, NYscaled] )';
            %figure('Name','Rate Array ');
            imagesc( recon2D );  % plots recon2D as an image
            title(['\fontsize{12}{\bfR for layer ' l_name '}']);
            colorbar
            axis square
            axis off
        end
        
        figure('Name','wMax and Rate');
        %plot(wMax, averageR - R, 'ob');
        plot(wMax, R, 'ob');
        xlabel('wMax');
        %ylabel('avRate - Rate');
        ylabel('Rate');
        title('Rate vs wMax');
        
        if 1
            filename = [output_dir, l_name '_R_wMax.ds'];
            fid=fopen(filename,'w');
            for j=1:length(R)
                fprintf(fid,'%f %f\n',R(j),wMax(j))
            end
            fclose(fid);
        end
        
        if plot_rate & read_spikes
            figure('Name','R vs spike rate')
            plot(avS, R, 'ob');
            xlabel('<S>');
            ylabel('Rate');
            title('Rate vs <S>');
        end
        
        fname = [l_name '_VthRest_last.pvp'];
        VthRest = wmax_readLastFile(fname,l_name,xScale, yScale);

        %% histogram plot
        figure('Name',['VthRest Histogram for ',l_name]);
        hist(VthRest,50);
        
        %% reshape and plot VthRest field
        if 1
            figure('Name',['VthRest for ',l_name]);
            % transposed: see the remarks on reshape in wmax_readLastFile.m
            recon2D = reshape(VthRest, [NXscaled, NYscaled] )';
            %figure('Name','Rate Array ');
            imagesc( recon2D );  % plots recon2D as an image
            title(['\fontsize{12}{\bfVthRest for layer ' l_name '}']);
            colorbar
            axis square
            axis off
        end
        
        figure('Name','VthRest and Rate');
        %plot(wMax, averageR - R, 'ob');
        plot(R, VthRest, 'ob');
        ylabel('V_{thRest}');
        %ylabel('avRate - Rate');
        xlabel('Rate');
        title('V_{thRest} vs Rate');

       % 3D plot
       figure('Name','R, W_{max}, and V_{thRest}');
       plot3( R, wMax, VthRest + 55,'ob');
       xlabel('R');
       ylabel('W_{max}');
       zlabel('V_{thRest}');
       title(['\fontsize{12}{\bfR, W_{max}, and V_{thRest} for layer ' l_name '}']);
       grid on
       axis square
       
    end % plot_wmax
    
    %% read weights and compute learning scores for each neuron in 
    % this layer
    if comp_scores == 1 & layer >= 4
        disp('compute the learning score for each neuron: ');
        [S, F] = wmax_compScores(w_last,xScale,yScale); 
        sym = {'or','ob','og','ok'};
        
        for f=1:numel(w_last)
            figure('Name','Scores and wMax');
            plot(wMax,S{f},sym{f});
            xlabel('wMax');
            ylabel('Score');
            title([l_name,' Score vs wMax']);
        end
        
        
        
    end
    
    %% use scores to color plot 
    
    %if plot_weights_histogram == 1 & (layer == 4 | layer == 6) % INHIB
    if plot_weights_histogram == 1 & (layer == 4 | layer == 4)    
        disp('plot weights histogram')
        for i=1:numel(w_file)
            fprintf('Weights Field for %s\n',w_file{i});
            %figure('Name', ['Weights Histogram for ' l_name ' layer']);
            W = stdp_plotWeightsHistogramOnly(w_file{i},l_name,xScale,yScale,HIST_STEP);% W is t
            %pause
        end
    end
    
    
    
    %if plot_weights_field == 1 & (layer == 4 | layer == 6) % INHIB
    if plot_weights_field == 1 & (layer == 4 | layer == 5)    
        Xtarg = [];
        Ytarg = [];
        
        disp('plot weights field')
        if isempty(Xtarg) 
            disp('No target image: ');
        end
        % a layer may be connected to multiple pre-synaptic layers and each
        % connection has its own weights
        for i=1:numel(w_file)
            fprintf('Weights Field for %s\n',w_file{i});
           stdp_plotWeightsField(w_file{i},xScale,yScale,Xtarg,Ytarg, FIELD_STEP);
           pause
        end
        
    end    
    
    if plot_rate_field == 1 & layer >= 4
        
        Xtarg = [];
        Ytarg = [];
        disp('plot rate field')
       
        fname = 'L1_R.pvp';
        fprintf('Rate Field for %s\n',fname);
        stdp_plotRateField(fname,xScale,yScale,Xtarg,Ytarg);
        pause
        
        
    end  
    
    
    if read_probe == 1 & layer == 4
        % defines probe locations
        if Cprobes
            probeLoc = [15 13; 16 14; 22 28; 25 14; 5 7]; % C probes
            probeName = 'C1';
        else
            probeLoc = [15 13; 16 14; 22 30; 22 32; 50 14]; % S probes
            probeName = 'S1';
        end
        
        for p = 1: size(probeLoc,1)
        kx=probeLoc(p,1); 
        ky=probeLoc(p,2); 
        begin_step = 1;
        end_step = 100000; %200000; %10000;
        if read_probe_data
                        

            for ind=1:n_files
                if ind == 1
                    simTime = []; G_E = []; G_I = []; G_IB = []; V = []; Vth=[];
                    Rate = []; Wmax = []; aWmax = []; VthRest = []; Act = [];
                    lastTime = 0;
                end
                fname = [probeName '_' num2str(kx) '_' num2str(ky) '.txt' num2str(ind)];
                fprintf('Read probe data from %s\n',fname);
                % these are N x 1 arrays                
                % this reads advanceWmax from the LIF probe
                if advanceWmax
                    [vmem_time, vmem_G_E, vmem_G_I, vmem_G_IB, vmem_V, vmem_Vth, ...
                        vmem_R, vmem_Wmax, vmem_aWmax, vmem_VthRest, vmem_a] = ...
                        ptprobe_readValues(fname, begin_step, end_step);
                else
                    [vmem_time, vmem_G_E, vmem_G_I, vmem_G_IB, vmem_V, vmem_Vth, ...
                        vmem_R, vmem_Wmax, vmem_VthRest, vmem_a] = ...
                        ptprobe_readV(fname, begin_step, end_step);
                    
                end
                
                vmem_time = lastTime + vmem_time;    
                if size(vmem_time,1) == 1
                   simTime = [simTime; vmem_time'];
                else
                    simTime = [simTime; vmem_time];
                end
                lastTime = simTime(end);
                
                if size(vmem_G_E,1) == 1
                    G_E = [G_E; vmem_G_E'];
                else
                    G_E = [G_E; vmem_G_E];
                end
                if size(vmem_G_I,1) == 1
                    G_I = [G_I; vmem_G_I'];
                else
                    G_I = [G_I; vmem_G_I];
                end
                if size(vmem_G_IB,1) == 1
                   G_IB = [G_IB; vmem_G_IB'];
                else
                    G_IB = [G_IB; vmem_G_IB];
                end
                if size(vmem_V,1) == 1
                    V = [V; vmem_V'];
                else
                    V = [V; vmem_V];
                end
                if size(vmem_Vth,1) == 1
                    Vth = [Vth; vmem_Vth'];
                else
                    Vth = [Vth; vmem_Vth];
                end
                if size(vmem_R,1) == 1
                   Rate = [Rate; vmem_R'];
                else
                   Rate = [Rate; vmem_R];
                end
                if size(vmem_Wmax,1) == 1
                    Wmax = [Wmax; vmem_Wmax'];
                else
                    Wmax = [Wmax; vmem_Wmax];
                end
                if advanceWmax
                    if size(vmem_aWmax,1) == 1
                        aWmax = [aWmax; vmem_aWmax'];
                    else
                        aWmax = [aWmax; vmem_aWmax];
                    end
                end
                if size(vmem_VthRest,1) == 1
                    VthRest = [VthRest; vmem_VthRest'];
                else
                    VthRest = [VthRest; vmem_VthRest];
                end
                if size(vmem_a,1) == 1
                    Act = [Act; vmem_a'];
                else
                    Act = [Act; vmem_a];
                end
            end
        end % read_probe_data
        
       figure('Name',['Probe kx= ' num2str(kx) ' ky= ' num2str(ky) ' V, Vth, VthRest']);
       plot(simTime/1000.0,V,'-b');hold on
       xlabel('time');
       ylabel('V,V_{th}, V_{thRest}')
       plot(simTime/1000.0, Vth,'-k');
       plot(simTime/1000.0, VthRest,'or');
       legend('V','V_{th}','V_{thRest}');
       hold off
       
       if 0
       figure('Name',['Probe kx= ' num2str(kx) ' ky= ' num2str(ky) ' (<R> - R) and Wmax vs Time']);
       plot(simTime/1000.0, averageR - Rate,'-r');hold on
       xlabel('time');
       ylabel('(<R> - R) and W_{max}')
       plot(simTime/1000.0, Wmax,'-g');
       hold off
       end
       
       if advanceWmax
           figure('Name',['Probe kx= ' num2str(kx) ' ky= ' num2str(ky) ' R, Wmax, and V_{thRest} vs Time']);
           plot(simTime/1000.0, Rate,'-b');hold on
           xlabel('time');
           ylabel('R, W_{max}, aW_{max}, (V_{thRest} + 55)')
           plot(simTime/1000.0, Wmax,'-g');
           plot(simTime/1000.0, VthRest + 55,'.r');
           plot(simTime/1000.0, aWmax,'.k');
           legend('R','W_{max}','V_{thRest}','aW_{max}');
           hold off
       else
           
           figure('Name',['Probe kx= ' num2str(kx) ' ky= ' num2str(ky) ' R, Wmax, and V_{thRest} vs Time']);
           plot(simTime/1000.0, Rate,'-b');hold on
           xlabel('time');
           ylabel('R, W_{max}, (V_{thRest} + 55)')
           plot(simTime/1000.0, Wmax,'-g');
           plot(simTime/1000.0, VthRest + 55,'.r');
           legend('R','W_{max}','V_{thRest}');
           hold off
           
       end
       
       % 3D plot
       figure('Name',['Probe kx= ' num2str(kx) ' ky= ' num2str(ky) ' R, W_{max}, and V_{thRest}']);
       plot3( Rate, Wmax, VthRest + 55,'-b');hold on
       xlabel('R');
       ylabel('W_{max}');
       zlabel('V_{thRest}');
       grid on
       axis square
       
       figure('Name',['Probe kx= ' num2str(kx) ' ky= ' num2str(ky) ' Wmax vs R']);
       plot(Rate,Wmax,'-r');
       ylabel('W_{max}');
       %xlabel('<R> - R')
       xlabel('R')

       
       figure('Name',['Probe kx= ' num2str(kx) ' ky= ' num2str(ky) ' VthRest vs R']);
       plot(Rate, VthRest, '-r');
       ylabel('V_{thRest}');
       %xlabel('<R> - R')
       xlabel('R')
            
       pause
        end % loop over probe locations (neurons)
    end % read_probe
    
    if read_features & layer == 4
        NXpost = NX * xScale; % L1 size
        NYpost = NY * yScale;
        nxpPost = 5; % post-synaptic neuron patch size (receptive field)
        nypPost = 5;

        if read_feature_files
            % read features for connection to RetinaOn layer
            % the last parameter is a debug parameter
            % the features are normalized
            [preIndexesOn, featuresOn] = wmax_readFeatures(xScale, yScale, ...
                'a1.pvp.rate','PCPon', 0);
            
            
            % read features for connection to RetinaOff layer
            [preIndexesOff, featuresOff] = wmax_readFeatures(xScale, yScale, ...
                'a2.pvp.rate','PCPoff', 0);

        end
        
        
        if read_weights_files
            % The weights are normalized!
            [Won_array, NXP, NYP]  = wmax_readLastWeightsField(w_last{1}, xScale, yScale);
            [Woff_array, NXP, NYP] = wmax_readLastWeightsField(w_last{2}, xScale, yScale);         
        end
        
        
        if read_wMax_file & (~plot_wMax)
            fname = 'S1_Wmax_last.pvp';
            wMax = wmax_readLastFile(fname, l_name, xScale, yScale);
        end
        
        % print features
        if print_features
            for kyPost=0:(NYpost-1)
                for kxPost=0:(NXpost-1)
                    kPost = kyPost * NXpost + kxPost + 1; % shift by 1
                    fprintf('\n\tkyPost = %d kxPost = %d kPost = %d:\n\n',...
                        kyPost+1, kxPost+1, kPost);
                    %featuresOn{kPost} 
                    %featuresOff{kPost}
                    %pause
                    if length(featuresOn{kPost}) & length(featuresOff{kPost})
                        % pre-synaptic indexes
                        Ion = reshape(preIndexesOn{kPost},[nxpPost nxpPost])';
                        Ioff = reshape(preIndexesOff{kPost},[nxpPost nxpPost])';
                        % pre-synaptic features
                        % normalize features
                        Fon = reshape(featuresOn{kPost},[nxpPost nxpPost])';
                        Foff = reshape(featuresOff{kPost},[nxpPost nxpPost])';
                        % weight-patches
                        Won = reshape(Won_array(kPost,:),[NXP NYP])';
                        Woff = reshape(Woff_array(kPost,:),[NXP NYP])';
                        
                        for jp=1:nypPost
                            if print_indexes
                                for ip=1:nxpPost
                                    if Ion(jp,ip) < 10
                                        fprintf(' %d ', Ion(jp,ip));
                                    else
                                        fprintf('%d ', Ion(jp,ip));
                                    end
                                end
                            end
                            if print_weights
                                fprintf('   ');
                                for ip=1:nxpPost
                                    fprintf('%3.2f ', Won(jp,ip));
                                end
                            end
                            fprintf('   ');
                            for ip=1:nxpPost
                               fprintf('%3.2f ', Fon(jp,ip));
                            end
                            fprintf('   ');
                            if print_indexes
                                for ip=1:nxpPost
                                    if Ioff(jp,ip) < 10
                                        fprintf(' %d ', Ioff(jp,ip));
                                    else
                                        fprintf('%d ', Ioff(jp,ip));
                                    end
                                end
                            end
                            if print_weights
                                fprintf('   ');
                                for ip=1:nxpPost
                                    fprintf('%3.2f ', Woff(jp,ip));
                                end
                            end
                            fprintf('   ');
                            for ip=1:nxpPost
                                fprintf('%3.2f ', Foff(jp,ip));
                            end
                            fprintf('\n');
                        end
                        
                        pause
                    end % non-empty features condition
                    
                end % kxPost loop
            end % kyPost loop
        end % print_features
        
        %% compute overlap
        if comp_overlap
            for kyPost=0:(NYpost-1)
                for kxPost=0:(NXpost-1)
                    kPost = kyPost * NXpost + kxPost + 1; % shift by 1
                    %fprintf('\n\tkyPost = %d kxPost = %d kPost = %d:\n\n',...
                    %    kyPost+1, kxPost+1, kPost);
                    if length(featuresOn{kPost}) & length(featuresOff{kPost})
                        OverlapOn(kPost)  = Won_array(kPost,:) * featuresOn{kPost}';
                        OverlapOff(kPost) = Woff_array(kPost,:) * featuresOff{kPost}';
                    else
                        OverlapOn(kPost) = -1;
                        OverlapOff(kPost) = -1;
                    end
                end
            end
        end
        
        %% plot overlap between features and weight patches
        if plot_overlap_hist
            figure('Name',['Features On Overlap for ',l_name]);
            %hist(rate_array{layer}, 40 );
            hist(OverlapOn(OverlapOn > 0), 40 );
            %pause
            
            figure('Name',['Features Off Overlap ',l_name]);
            %hist(rate_array{layer}, 40 );
            hist(OverlapOff(OverlapOff > 0), 40 );
            %pause
        end
        
        if plot_features_weights
            ind = find( (OverlapOn > 0.6) & (OverlapOff > 0.6));
            fprintf('%d neurons with On and Off overlaps > 0.8\n',length(ind));
            figure('name','Large Overlaping Weight Patches')
            for k = 1:length(ind)
                kPost = ind(k);
                kxPost = rem(kPost-1,NXpost);
                kyPost = (kPost-1 - kxPost)/ NXpost;
                Fon = reshape(featuresOn{kPost},[nxpPost nxpPost])';
                Foff = reshape(featuresOff{kPost},[nxpPost nxpPost])';
                % weight-patches
                Won = reshape(Won_array(kPost,:),[NXP NYP])';
                Woff = reshape(Woff_array(kPost,:),[NXP NYP])';
                subplot(2,2,1)
                imagesc(Won);
                subplot(2,2,2)
                imagesc(Fon);
                subplot(2,2,3)
                imagesc(Woff);
                subplot(2,2,4)
                imagesc(Foff);
                fprintf('kyPost = %d kxPost = %d overOn = %f overOff = %f wMax = %f f_size = %d\n',...
                    kyPost,kxPost,OverlapOn(kPost),OverlapOff(kPost),...
                    wMax(kPost),length(find(Fon)) + length(find(Foff)) );
                pause
            end
        end
        
        if comp_features_size
            figure('name','wMax, max(Fon),max(Foff)')
            first_point = 1;
            for kPost = 1: NXpost * NYpost
                if length(featuresOn{kPost}) & length(featuresOff{kPost})
                    featuresSize(kPost) = length(find(featuresOn{kPost})) ...
                        + length(find(featuresOff{kPost}));
                    x = wMax(kPost);
                    y = max(featuresOn{kPost});
                    z = max(featuresOff{kPost});
                    plot3(x,y,x,'ob');
                    if first_point
                        xlabel('wMax');
                        ylabel('max(Fon)');
                        zlabel('max(Foff)');
                        grid on
                        axis square
                        hold on
                        first_point = 0;
                    end
                else
                    featuresSize(kPost) = 0;
                end
                
            end
            %plot(wMax, featuresSize,'or');
            
        end
        
    end % read_features
    
end % loop over layers



