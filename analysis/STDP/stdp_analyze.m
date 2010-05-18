%%
close all
%clear all

% Make the global parameters available at the command-line for convenience.
global N  NX NY n_time_steps begin_step tot_steps
global spike_array num_target rate_array target_ndx vmem_array
global input_dir output_dir conn_dir output_path input_path
global patch_size write_step

input_dir = '/Users/manghel/Documents/workspace/STDP/output/';
%input_dir = '/Users/manghel/Documents/STDP-sim/output13/';
%input_dir = '/Users/manghel/Documents/STDP-sim/output-good-w1.5-ltd-1.2-phases/';
%input_dir = '/Users/manghel/Documents/STDP-sim/output-good-w1.5-ltd-1.2-vhbars/';
%input_dir = '/Users/manghel/Documents/STDP-sim/output-final_w3_bf40_bd10_dw04_bars/';
output_dir = '/Users/manghel/Documents/STDP-sim/stdp_stat_14/';
conn_dir = '/Users/manghel/Documents/STDP-sim/conn_probes_8_8/';

num_layers = 2;
n_time_steps = 100000; % the argument of -n; even when dt = 0.5 
patch_size = 16;  % nxp * nyp
write_step = 10000; % set in params.stdp


begin_step = 5000;  % where we start the analysis (used by readSparseSpikes)
end_step   = 10000;
stim_begin = 1;  % generally not true, but I read spikes
                      % starting from begin_step
stim_end = 1000000;
stim_length = stim_end - stim_begin + 1;
stim_begin = stim_begin - begin_step + 1;
stim_end = stim_end - begin_step + 1;
stim_steps = stim_begin : stim_end;
bin_size = 10;

NX=8;% 8;  % defined by column 
NY=8;% 8;  % use xScale and yScale to get layer values
dT = 0.5;  % miliseconds
burstFreq = 50; % Hz  (50=20ms)
burstDuration = 7.5; 
   
   
my_gray = [.666 .666 .666];

target_ndx = cell(num_layers,1);
bkgrnd_ndx = cell(num_layers,1);

parse_tiff = 0;
read_spikes = 1;    % read spikes if you want plot_raster!
                    % reading is slow for long runs
spike_movie            = 1;
plot_raster             = 0;
plot_spike_activity     = 0; 
plot_rate_array         = 0;
plot_weights_rate_evolution = 0;
plot_membrane_potential  = 0;
plot_weights_field       = 0;
plot_weights_projections = 0;
plot_weights_histogram  = 0;
plot_weights_corr       = 0; % read spikes first
plot_patch              = 0;
comp_weights_PCA        = 0;
comp_weights_Kmeans     = 0; % add Documents/MATLAB/Kmeans to path
numCenters              = 16; % param for comp_weights_Kmeans 

% compute STDP statistics
comp_preSTDP            = 0;
comp_postSTDP           = 0;
firing_flag             = 0; % flag for comp_preSTDP

% STDP analysis
analyze_STDP_clique     = 0; % to compute the clique size distribution
                             % call analyze_STDP
analyze_STDP            = 0;
comp_STDP_Kmeans        = 0; 
preSTDP_anal            = 0; % flag for analyze_STDP & comp_STDP_Kmeans
postSTDP_anal           = 0; % flag for analyze_STDP & comp_STDP_Kmeans
 
if preSTDP_anal & postSTDP_anal
    disp('wrong arguments');
    return
end


if read_spikes | plot_rate_array
    spike_array = cell(num_layers,1);
    rate_array = cell(num_layers,1);
    spike_array_bkgrnd = spike_array;
end

if parse_tiff    
   tiff_path = [input_dir 'images/img2D_0.00.tif'];
   %imshow(tiff_path);
   [targ, Xtarg, Ytarg] = stdp_parseTiff( tiff_path );
   disp('parse tiff -> done');
   %pause
else
    Xtarg = [];
    Ytarg = [];
end


for layer = 1:num_layers; % layer 1 here is layer 0 in PV

    % Read relevant file names and scale parameters
    [f_file, v_file, w_file, w_last, xScale, yScale] = stdp_globals( layer );

    % Read spike events
    
    
    if read_spikes & layer >= 2
        disp('read spikes')
        [spike_array{layer}, ave_rate] = ...
            stdp_readSparseSpikes(f_file, begin_step, end_step);
        disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate)]);
        tot_steps = size( spike_array{layer}, 1 );
        tot_neurons = size( spike_array{layer}, 2);
        fprintf('tot_steps = %d tot_neurons = %d\n',...
            tot_steps,tot_neurons);
        num_bins = fix( tot_steps / bin_size );
        excess_steps = tot_steps - num_bins * bin_size;
        stim_bin_length = fix( ( stim_end - stim_begin + 1 ) / bin_size );
        stim_bin_begin = fix( ( stim_begin - begin_step + 1 ) / bin_size );
        stim_bin_end = fix( ( stim_end - begin_step + 1 ) / bin_size );
        stim_bins = stim_bin_begin : stim_bin_end;     
    end
    
    
    if plot_rate_array
        disp('read spikes and plot average activity (rate aray)')
        [rate_array{layer}, ave_rate] = stdp_readAverageActivity(f_file, begin_step, end_step);
        disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate)]);
        disp('plot_rate_reconstruction')
        stdp_reconstruct(rate_array{layer}, NX*xScale, NY * yScale, ...
            ['Rate reconstruction for layer  ', int2str(layer)]);
    end
    
    if spike_movie & layer >= 2
        disp('simple movie')
        retinaPeriod = 1000./burstFreq;
        if gcf ~= 1
            figure(gcf+1);
        else
            figure(gcf);
        end
        
        colormap gray 
        numRows = burstDuration / dT + 2;
        sumA = zeros(NX*xScale,NY*yScale);
        sumB = zeros(NX*xScale,NY*yScale);
        
        for t=1:n_time_steps
                         
            burstStatusA = mod(t*dT, retinaPeriod);% from 0 to burstDuration
                                                 % in steps of dT
            burstStatusB = mod(t*dT + 0.5*retinaPeriod, retinaPeriod);% from 0 to burstDuration
                                                 % in steps of dT                                     
            % plot only when Retina fires
            if burstStatusA <= burstDuration
                if ~burstStatusA 
                    sumA(:) = 0;
                end
                
                fprintf('%f*\n',t*dT);
                A = reshape(spike_array{layer}(t,:),NX*xScale,NY*yScale);
                sumA = sumA+A';
                figA = 2*burstStatusA/dT + 1;
                %fprintf('figA = %d\n',figA);
                subplot(numRows,2,figA)
                imagesc(A',[0 1])
                axis equal
                set(gca,'xtick',[],'ytick',[])
                axis tight
                %title([' t= ', num2str(t*dT)]);
                
                if burstStatusA == burstDuration % plot integrated signal
                    figA = 2*burstStatusA/dT + 3;
                    %fprintf('figAA = %d\n',figA);
                    subplot(numRows,2, figA)
                    imagesc(sumA / burstDuration,[0 1])
                    axis equal
                    set(gca,'xtick',[],'ytick',[])
                    axis tight
                    %pause
                end
                
                pause(0.1)
                
            elseif burstStatusB <= burstDuration
                if ~burstStatusB
                    sumB(:) = 0;
                end
                fprintf('%f#\n',t*dT);
                B = reshape(spike_array{layer}(t,:),NX*xScale,NY*yScale);
                sumB=sumB+B';
                figB = 2*(burstStatusB/dT + 1);
                %fprintf('figB = %d\n',figB);
                subplot(numRows,2,figB)
                imagesc(B',[0 1])
                axis equal
                set(gca,'xtick',[],'ytick',[])
                axis tight
                %title([' t= ', num2str(t*dT)]);
                
                if burstStatusB == burstDuration % plot integrated signal
                    figB = 2*(burstStatusB/dT + 1) + 2;
                    %fprintf('figBB = %d\n',figB);
                    subplot(numRows,2, figB)
                    imagesc(sumB/ burstDuration,[0 1])
                    axis equal
                    set(gca,'xtick',[],'ytick',[])
                    axis tight
                    %pause
                end
                
                pause(0.1)
                
            else
                fprintf('%f\n',t*dT);
            end
            
        end
        disp('pause')
        pause
    end
    
     if plot_spike_activity & layer > 1
         
        disp('compute rate array and spike activity array')
        rate_array{layer} = 1000.0 * full(mean(spike_array{layer},1) ) ;
        % this is a 1xN array where N=NX*NY
        disp('plot_rate_reconstruction')
        stdp_reconstruct(rate_array{layer}, NX*xScale, NY * yScale, ...
            ['Rate reconstruction for layer  ', int2str(layer)]);
        pause
        if 0
            spikes_array{layer} = 1000 * full( mean(spike_array{layer},2) );
            % this is Tx1 array
            plot_title = ['Spiking activity for layer  ',int2str(layer)];
            figure('Name',plot_title);
            plot(spikes_array{layer},'ob')
            xlabel=('t');
            ylabel=('num spikes');
            pause

            % redo this computation
            tot_steps = size( spike_array{layer}, 1 );
            num_bins = fix( tot_steps / bin_size );
            plot_title = ['Moving window rate average for layer  ',int2str(layer)];
            figure('Name',plot_title);
            bin_size = 100;
            tot_steps = size( spike_array{layer}, 1 );
            num_bins = fix( tot_steps / bin_size );
            moving_rate_array{layer} = ...
                mean(reshape(spikes_array{layer}(1:bin_size*num_bins),...
                bin_size, num_bins), 1);
            % reshape returns a bin_size x num_bins array: it oputs the first
            % bin_size values in column 1, the next bin_size values in column 2,
            % and so on, while mean applied to this matrix returns a
            % 1 x num_bins array
            plot(moving_rate_array{layer},'or')
            pause
        end
        
        % plot spikes for selected indices
        % stdp_plotSpikes(spike_array{layer},[]);
        
        
     end
    
     
     
      if (comp_preSTDP | comp_postSTDP) & layer > 1

          disp('compute STDP statistics')

          % Read parameters from previous layer
          [f_file_pre, v_file_pre, w_file_pre, w_last_pre, ...
              xScale_pre, yScale_pre] = stdp_globals( layer -1 );
    
          
          for W=4:4:60
              
              fprintf('compute STDP statistics for window size W = %d\n',W);
              if comp_preSTDP      
                  [prePatchAverage, nPostSpikes, kPostIndices, timeWeightsSum] = ...
                      stdp_compPreStatistics(f_file, f_file_pre, W, firing_flag);
              else
                  [prePatchAverage, nPostSpikes, kPostIndices, timeWeightsSum] = ...
                      stdp_compPostStatistics(f_file, f_file_pre, W, firing_flag);
              end

            
          end % loop over W (pre-synaptic activity time window length)

    end % comp_STDP
    
    
    if analyze_STDP & layer > 1
        
        
        %% read pre-synaptic indices for post-synaptic neurons
        filename = [conn_dir,'PCP_Indexes.dat'];
        fid = fopen(filename, 'r');
        C = textscan(fid,'%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d');
        kPostIndices=C{1}; % Nx1 array of kPost indices
        NXpost = NX*xScale;
        NYpost = NY+yScale;
        
        
        dT = 0.5;

        patch_fig =0;
        hist_fig = 0;
        time_fig = 0;
        corr_fig = 0;
        bin_centers = 0;
        bin_edges =1;
        
        for W=4:4:60
            
        for nPost = 1:numel(kPostIndices)
            kPost = kPostIndices(nPost);
            kxPost=rem(kPost-1,NXpost) + 1;
            kyPost=(kPost-1-kxPost)/NXpost + 1;
            %% open STDP files
            if comp_preSTDP
                if firing_flag
                    filename = [output_dir,'STDP_pre_' ...
                        num2str(W) '_' num2str(kPost) '.dat'];
                else
                    filename = [output_dir,'STDP_pre_notfire_' ...
                        num2str(W) '_' num2str(kPost) '.dat'];
                end
            else
                filename = [output_dir,'STDP_post_' num2str(W) '_' num2str(kPost) '.dat'];
            end
            fid = fopen(filename, 'r');

            data = fscanf(fid,'%f',[18 inf]);
            data=data'; % T x (2 + patchsize) matrix where T is the number
            % of post-synaptic spikes; the first record is the
            % timePost and the second is the total average number of pre
            % spikes, the next patchsize values are the average number of pre spikes for
            % each pre-synaptic neuron
            if numel(data)
                
                %% plots a space time history of pre-synaptic activity
                % and computes the correlation matrix for pre-synaptic activity
                % in the receptive field
                %
                if 1
                    if ~time_fig
                        time_fig = gcf;
                    end
                    figure(time_fig);
                    colormap autumn
                    imagesc(data(:,3:18)) % time sequence of average patch activity
                    %axis equal           % conditioned on firing post-synaptic neurons
                    title('Average pre-synaptic activity')
                    axis off

                    R = corrcoef(data(:,3:18));
                    if ~corr_fig
                        corr_fig = gcf + 1;
                    end
                    for i=1:size(R,1),R(i,i) = 0.0,end
                    figure(corr_fig);
                    imagesc(R','CDataMapping','direct');
                    title('Correlation Matrix');
                    %pause
                end

                %% computes average firing rate of pre-synaptic neurons
                % in the receptive field
                if 0
                    recField = sum(data(:,3:18)) ./ size(data,1);
                    patch = reshape(recField,[4 4])';
                    if ~patch_fig
                        patch_fig = gcf + 1;
                    end
                    figure(patch_fig);
                    imagesc(patch,'CDataMapping','direct');
                    title(['kPost ' num2str(kPost) ' (' num2str(kxPost) ' , ' num2str(kyPost) ' )'] );
                    colorbar
                    axis square
                    axis off
                    %pause
                end
                  
                %% makes a histogram plot of total pre-synaptic activity
                if 0
                    if ~hist_fig
                        hist_fig = gcf + 1;
                    end
                    figure(hist_fig);
                    % using bin centers
                    if bin_centers
                        x=0.5:1:20.5; % bin centers!!
                        hist(data(:,2),x)
                        title(['kPost ' num2str(kPost)]);
                    end
                    %using bin edges
                    if bin_edges
                        x=0:1:20; % bin edges!!
                        n=histc(data(:,2),x)
                        bar(x+0.5,n);
                        title(['kPost ' num2str(kPost)]);
                    end
                    pause
                end

                %% computes average clique size - the mean of the hist
                % distribution above
                avClique(nPost) = mean(data(:,2))/(W*dT);
                fprintf('kPost = %d avClique = %f \n',kPost,avClique(nPost));
                  end % numel
                  
              end % loop over post-synaptic neurons
              
              nbins = 20;
              minX = min(avClique);
              maxX = max(avClique);
              fprintf('\n\n minClique = %f maxClique = %f \n\n',minX,maxX);
              x=minX:(maxX-minX)/nbins:maxX; % bin centers!!
              n = hist(avClique,x);
              bar(x,n,'b');
              title(['Window ' num2str(W) ' Average Clique Size']);
              if comp_preSTDP
                filename = [output_dir,'avCliqueHist_pre_' num2str(W) '.dat'];
              else
                 filename = [output_dir,'avCliqueHist_post_' num2str(W) '.dat'];
              end
              fid=fopen(filename,'w');
              for i=1:length(n)
                  %fprintf(fid,'%f %f\n',x(i),n(i)/timeWeightsSum);
                  fprintf(fid,'%f %f\n',x(i),n(i));
              end
              fclose(fid);
              %pause
        
        end % loop over W (window time length)
        
    end % analyze_STDP
    
    
    
    if analyze_STDP_clique & layer > 1
        figure('Name','Clique Size Histograms');
        sym={'-r','-b','-g','-k'};
        dT=0.5;
        if 0
            n=1;
            W = 2;
            if preSTDP_anal
                filename = [output_dir,'avCliqueHist_pre_' num2str(W*dT) '.dat'];
            else
                filename = [output_dir,'avCliqueHist_post_' num2str(W*dT) '.dat'];
            end
            h= load(filename); % N x 2 array
            [M,I]=max(h(:,2));
            hMax(n)=M;
            xMax(n)=h(I,1);
            wMax(n) = W*dT;
            %subplot(1,2,1);
            plot(h(:,1),h(:,2),sym{4});
            hold on
        else
            n=0;
        end
        
        for W=4:4:60
            n=n+1;
            if preSTDP_anal
                filename = [output_dir,'avCliqueHist_pre_' num2str(W*dT) '.dat'];
            else
                filename = [output_dir,'avCliqueHist_post_' num2str(W*dT) '.dat'];
            end
            h= load(filename); % N x 2 array
            fprintf('W= %d Spikes=%d \n',W, sum(h(:,2)) );
            [M,I]=max(h(:,2));
            hMax(n)=M;
            xMax(n)=h(I,1);
            wMax(n) = W*dT;
            %subplot(1,2,1);
            plot(h(:,1),h(:,2),sym{mod(n,4)+1});
            hold on
            pause
        end
        
        figure('Name','Max Clique vs Window Size');
        %subplot(1,2,2)
        plot(wMax,xMax,'ob');
        
        pause
    end % analyze_STDP_clique
    
    
    
     if comp_STDP_Kmeans & layer > 1

          disp('compute STDP K-means: ')
          time_fig = 0;
          cluster_fig = 0;


          % Read parameters from previous layer
          [f_file_pre, v_file_pre, w_file_pre, w_last_pre, ...
              xScale_pre, yScale_pre] = stdp_globals( layer -1 );
    
          dT = 0.5;
          

          for W=60:4:60
              
              fprintf('compute STDP K-means for W = %d\n',W);

              %% read pre-synaptic indices for post-synaptic neurons
              filename = [conn_dir,'PCP_Indexes.dat'];
              fid = fopen(filename, 'r');
              C = textscan(fid,'%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d');
              kPostIndices=C{1}; % Nx1 array of kPost indices
              NXpost = NX*xScale;
              NYpost = NY+yScale;
              for nPost = 1:numel(kPostIndices)
                  
                  kPost = kPostIndices(nPost);
                  kxPost=rem(kPost-1,NXpost) + 1;
                  kyPost=(kPost-1-kxPost)/NXpost + 1;     
                  fprintf('kPost = %d kxPost = %d kyPost = %d\n',...
                      kPost,kxPost, kyPost);
                  
                  weightsPatch = stdp_getPatchWeights(w_last, xScale, yScale,kxPost, kyPost);
                  
                  %% open STDP files
                  if preSTDP_anal
                      filename = [output_dir,'STDP_pre_' num2str(W) '_' num2str(kPost) '.dat'];
                  else
                      filename = [output_dir,'STDP_post_' num2str(W) '_' num2str(kPost) '.dat'];
                  end
                  fid = fopen(filename, 'r');

                  data = fscanf(fid,'%f',[18 inf]);
                  data=data'; % T x (2 + patchsize) matrix where T is the number
                  % of post-synaptic spikes; the first record is the
                  % timePost and the second is the total average number of pre
                  % spikes, the next patchsize values are the average number of pre spikes for
                  % each pre-synaptic neuron
                  if numel(data)
                      
                      
                      if 1
                          if ~time_fig
                              time_fig = gcf;
                          end
                          figure(time_fig);
                          colormap gray
                          imagesc(1-data(:,3:18)) % time sequence of average patch activity
                                                % conditioned on firing
                                                % post-synaptic neurons
                          title([num2str(kPost) ' (' num2str(kxPost) ' , ' num2str(kyPost) ' )'] );
                          axis off
                          %pause
                      end
                      
                      
                      % plot receptive field synaptic weights
                      figure(gcf+1);
                      colormap gray
                      imagesc(weightsPatch);
                      colorbar
                      title([num2str(kPost) ' (' num2str(kxPost) ' , ' num2str(kyPost) ') Syn Weights'] );
                      axis square
                      axis off
                      
                      
                      %% find K-means clusters
                      numK = 4;   % number of clusters
                      % data is n x dim matrix where dim = patch_size
                      [centers,mincenter,mindist,q2,quality] = kmeans(data(:,3:18),numK);
                      %size(centers) % numK x patch_size
                      %size(mincenter)% n x 1
                      %q2
                      %quality
                      
                      %% compute weights
                      figure(gcf+1);
                      plot(mincenter,'ob');
                      title('Min Centers');
                      for k=1:numK
                          clustW(k) = sum(find(mincenter == k));
                      end
                      clustW = clustW ./sum(clustW);
                                                    
                      %% computes average firing rate of pre-synaptic neurons
                      % in the receptive field
                      figure(gcf+1);
                      for k=1:numK
                          fprintf('cluster %d w = %f\n',k,clustW(k));
                          %patch = clustW(k) * reshape(centers(k,:),[4 4])'; 
                          patch = reshape(centers(k,:),[4 4])';
                          subplot(2,2,k);
                          colormap gray
                          imagesc(patch,'CDataMapping','direct');
                          title([num2str(kPost) ' (' num2str(kxPost) ' , ' ...
                              num2str(kyPost) ' ) ' num2str(k) ' w = ' num2str(clustW(k),2) ] );
                          %xlabel(['w = ' num2str(clustW(k))] );
                          colorbar
                          axis square
                          axis off
                          
                      end

                          % write cluter centers
                      if 0
                          if preSTDP_anal
                              filename = [output_dir,'Clusters_pre_kPost_' num2str(kPost) '_' num2str(W) '.dat'];
                          else
                              filename = [output_dir,'Clusters_post _kPost_' num2str(kPost) '_' num2str(W) '.dat'];
                          end
                          fid=fopen(filename,'w');
                          for k=1:numK
                              for i=1:numel(centers(k,:))
                                  fprintf(fid,'%f ',centers(k,i));
                              end
                              fprintf(fid,'\n');
                          end
                          fclose(fid);

                          %pause
                      end


                  end % numel
                  
                  pause
                  
              end % loop over post-synaptic neurons
              

 
              %pause
              
          end % loop over W (pre-synaptic activity time window length)

    end % comp_STDP_Kmeans
    
    
  
    
    
    
    %parse the object identification files
    % pv_objects
    if parse_tiff
        % TODO: support for multiple objects--multiple calls to parseTiff?
        target_ndx{layer} = targ;
        bkgrnd_ndx{layer} = find(ones(N,1));
        bkgrnd_ndx{layer}(target_ndx{layer}) = 0;
        bkgrnd_ndx{layer} = find(bkgrnd_ndx{layer});
    end
    
    
    % raster plot
    
    if plot_raster
        disp('plot_raster')
        N = NX*xScale * NY *yScale;
        if ~isempty(spike_array{layer})
            plot_title = ['Raster for layer = ',int2str(layer)];
            figure('Name',plot_title);
            axis([0 tot_steps 0 N]);
            hold on
            box on
            
            [spike_time, spike_id] = find((spike_array{layer}));
            lh = plot(spike_time, spike_id, '.g');
            %set(lh,'Color',my_gray);    

        end
        pause
    end
    
    
    if plot_weights_corr & layer > 1
        
        disp('compute rate array and spike activity array')
        rate_array{layer} = 1000.0 * full(mean(spike_array{layer},1) ) ;
        % this is a 1xN array where N=NX*NY
        disp('plot_rate_reconstruction')
        stdp_reconstruct(rate_array{layer}, NX*xScale, NY * yScale, ...
            ['Rate reconstruction for layer  ', int2str(layer)]);
        pause
        
        % NOTE: Read spikes first!
        % Plot a histogram of rate activity
        %ind = find(A > 0.0);
        nbins = 200;
        [n,xout] = hist(rate_array{layer},nbins);
        plot_title = ['Rate Histogram for layer ', int2str(layer)];
        figure('Name',plot_title);
        %plot(xout,n,'-g','LineWidth',3);
        bar(xout,n,'g');
        pause
        
        T = input('print rate threshold ');
        ind = find(rate_array{layer} > T)
        stdp_plotWeightsCorr(w_file, ind, NX*xScale, NY*yScale,...
            20, ['Weights correlations for layer  ', int2str(layer)]);
        pause
        
        
    end
    
    if plot_weights_rate_evolution
        disp('plot weights rate evolution');
        % this is a cell aray: for each neuron it returns a temporal
        % array for the sum of its synaptic weights, i.e. sumW{n} is 
        % a T x 1 array where T is the number of weights snapshots
        [sumW, T] = stdp_compAverageWeightsEvol(w_file);
        % pass T-1
        [W,R] = stdp_analyzeWeightsRate(sumW, T-1, spike_array{layer});
        
        
        
    end

    
    % plot maximum stimulated membrane potential
    
    %read membrane potentials
    if plot_membrane_potential
        disp('plot_membrane_potential')
        vmem_array = stdp_readV(v_file, target_ndx{layer});
        num_max = 7;
        if ~isempty(spike_array{layer})
            plot_title = ['Membrane potential for layer = ',int2str(layer)];
            figure('Name',plot_title);
            color_order = get(gca,'ColorOrder');
            if ~isempty(spike_array{layer})
                % extract the mean rate of target neurons
                rate_array2 = rate_array{layer}( target_ndx{layer} );
                % sort in decreasing rate order
                [sorted_rate, sorted_ndx] = sort(rate_array2, 2, 'descend');
                spike_array2 = spike_array{layer}( :, target_ndx{layer} );
                vmem_rate = zeros(num_max,1);
                for i_max = 1:num_max
                    vmem_color = color_order(i_max,:);
                    lh = plot(vmem_array(:,sorted_ndx(i_max)), '-k');
                    set(lh, 'Color', vmem_color);
                    set(lh, 'LineWidth', 2.0);
                    hold on;
                    [spike_times, spike_id, spike_vals] = ...
                        find(spike_array2(:,sorted_ndx(i_max)));
                    vmem_rate(i_max) = length(spike_times);
                    lh = line( [max(1,spike_times-1), max(1,spike_times-1)]', ...
                        [vmem_array(max(1,spike_times-1), sorted_ndx(i_max)), 0.5*spike_vals]' );
                    set(lh, 'Color', vmem_color);
                    set(lh, 'LineWidth', 2.0);
                    display( [ 'rank:', num2str(i_max), ' vmem_rate(', num2str(layer), ') = ', ...
                        num2str( 1000*vmem_rate(i_max) / tot_steps ) ] );
                end
            end
        end
        clear col_ndx row_ndx Vmem_ndx vmem_array spike_array2
        pause
    end %plot_membrane_potential

    % Analyze the weights field and its distribution
    % Analyze the weights evoltion
    % NOTE: The number of weights distribution recorded is
    % (n_time_steps/write_step) 
    if plot_weights_field == 1 & layer > 1
        disp('plot weights field')
        if isempty(Xtarg) 
            disp('No target image: ');
        end
        stdp_plotWeightsField(w_file,xScale,yScale,Xtarg,Ytarg);
        
    end
    
    % Analyze the weights field and the weight patch projections
    % on a number of directions (features)
    % NOTE: The number of weights distribution recorded is
    % (n_time_steps/write_step) 
    if plot_weights_projections == 1 & layer > 1
        disp('plot weights projections')
        if isempty(Xtarg) 
            disp('No target image: ');
        end
        stdp_plotWeightsProjections(w_file,xScale,yScale,Xtarg,Ytarg);
        
    end
    
    
    if plot_weights_histogram == 1 & layer > 1
        disp('plot weights histogram')
        TSTEP = 1;
        W = stdp_plotWeightsHistogramOnly(w_file,xScale,yScale,TSTEP);% W is t
        %pause
    end
    
    
    if plot_patch % define the symbols using xtarg and ytarg so that
                  % it adapts to what a patch sees!!!
        patch_indexes = [];
        NXP=3;
        NYP=3;
        a = (NXP-1)/2;
        b = (NYP-1)/2;
             
        d='y';
        while d == 'y'
        %for p=1:length(Xtarg)
            I = input('input I index: ');
            J = input('input J index: ');
            %I=Xtarg(p);
            %J=Ytarg(p);
            fprintf('patch evolution for I= %d J= %d \n',I,J);
            linear_patch_index = [];
            k=0; % linear patch index 
            for j=1:NYP
                Jpatch = J-b + (j-1); % global XY image index
                %fprintf('\t');
                for i=1:NXP
                    k=k+1;
                    Ipatch = I-a + (i-1); % global XY image index
                    linear_patch_index(k) = (Ipatch-1)*NY + Jpatch;
                                      % global linear index
                    %fprintf('%d ',   linear_patch_index(k));
                    %fprintf('(%d %d) ',Jpatch,Ipatch)
                end
                %fprintf('\n');
            end
            plot_symbol = ismember(linear_patch_index, targ)
            %pause
            sym={'o-g','o-r'};
            plot_title = ['Weights evolution for neuron (' num2str(I) ',' num2str(J) ')'];
            PATCH = stdp_plotPatch(w_file, I,J, NX, NY, plot_title );
            figure('Name', plot_title);
            for k=1:9,
                %plot(PATCH(:,k),sym{k}),hold on,
                plot(PATCH(:,k),sym{plot_symbol(k)+1}),hold on,
            end
            AVERAGE_PATCH = reshape(mean(PATCH,1),[NXP NYP]);
            figure('Name', ['Average weights for neuron (' num2str(I) ',' num2str(J) ')']);
            imagesc(AVERAGE_PATCH', 'CDataMapping','direct');
            % NOTE: It seems that I need to transpose PATCH_RATE
            colorbar
        
            d = input('more patches? (y/n): ','s');
            if isempty(d)
                d='y';
            end
            
        end
    end
    
    if comp_weights_PCA == 1 & layer > 1
        disp('compute PCA of the  weight fields: ')       
        stdp_compPCA(w_last,xScale,yScale);       
    end
    
    if comp_weights_Kmeans == 1 & layer > 1
        disp('compute K-means of the  weight fields: ')       
        stdp_compWeightsKmeans(w_last,numCenters, xScale,yScale);       
    end
  
    
end % loop over all layers

plot_rates = tot_steps > 9 ;
if plot_rates  
    plot_title = ['PSTH target pixels'];
    figure('Name',plot_title);
    hold on
    co = get(gca,'ColorOrder');
    lh = zeros(4,1);
    for layer = 1:num_layers
        lh(layer) = plot((1:num_bins)*bin_size, psth_target{layer,i_target}, '-r');
        set(lh(layer),'Color',co(layer,:));
        set(lh(layer),'LineWidth',2);
    end
    leg_h = legend(lh(1:num_layers), ...
        [ ...
        'Retina  '; ...
        'V1      '; ...
        'V1 Inh  ' ...
        ]);
end

%%
plot_movie = 0; %tot_steps > 9;
if plot_movie
    for layer = [1 3 5 6]
        spike_movie = pv_reconstruct_movie( spike_array, layer);
        movie2avi(spike_movie,['layer',num2str(layer),'.avi'], ...
            'compression', 'None');
        show_movie = 1;
        if show_movie
            scrsz = get(0,'ScreenSize');
            [h, w, p] = size(spike_movie(1).cdata);  % use 1st frame to get dimensions
            fh = figure('Position',[scrsz(3)/4 scrsz(4)/4 560 420]);
            axis off
            box off
            axis square;
            axis tight;
            movie(fh,spike_movie,[1, 256+[1:256]], 12,[0 0 560 420]);
            close(fh);
        end
    end % layer
    clear spike_movie
end % plot_movie



%%

plot_xcorr = 0 && tot_steps > 9;
if plot_xcorr
    xcorr_eigenvector = cell( num_layers, 2);
    for layer = 1:num_layers
        pv_globals(layer);
%         pack;

        % autocorrelation of psth
        plot_title = ['Auto Correlations for layer = ',int2str(layer)];
        figure('Name',plot_title);
        ave_target_tmp = full(ave_target{layer,i_target}(stim_steps));
        maxlag= fix(length(ave_target_tmp)/4);
        target_xcorr = ...
            xcorr( ave_target_tmp, maxlag, 'unbiased' );
        target_xcorr = ( target_xcorr - mean(ave_target_tmp)^2 ) / ...
            (mean(ave_target_tmp)+(mean(ave_target_tmp)==0))^2;
        lh_target = plot((-maxlag:maxlag), target_xcorr, '-r');
        set(lh_target, 'LineWidth', 2);
        hold on
        ave_clutter_tmp = full(ave_clutter{layer,1}(stim_steps));
        clutter_xcorr = ...
            xcorr( ave_clutter_tmp, maxlag, 'unbiased' );
        clutter_xcorr = ( clutter_xcorr - mean(ave_clutter_tmp)^2 ) / ...
            (mean(ave_clutter_tmp)+(mean(ave_clutter_tmp)==0))^2;
        lh_clutter = plot((-maxlag:maxlag), clutter_xcorr, '-b');
        set(lh_clutter, 'LineWidth', 2);
        ave_bkgrnd_tmp = full(ave_bkgrnd{layer,1}(stim_steps));
        bkgrnd_xcorr = ...
            xcorr( ave_bkgrnd_tmp, maxlag, 'unbiased' );
        bkgrnd_xcorr = ( bkgrnd_xcorr - mean(ave_bkgrnd_tmp)^2 ) / ...
            (mean(ave_bkgrnd_tmp)+(mean(ave_bkgrnd_tmp)==0))^2;
        lh_bkgrnd = plot((-maxlag:maxlag), bkgrnd_xcorr, '-k');
        set(lh_bkgrnd, 'Color', my_gray);
        set(lh_bkgrnd, 'LineWidth', 2);
        target2clutter_xcorr = ...
            xcorr( ave_target_tmp, ave_clutter_tmp, maxlag, 'unbiased' );
        target2clutter_xcorr = ...
            ( target2clutter_xcorr - mean(ave_target_tmp) * mean(ave_clutter_tmp) ) / ...
            ( (mean(ave_target_tmp)+(mean(ave_target_tmp)==0)) * ...
            (mean(ave_clutter_tmp)+(mean(ave_clutter_tmp)==0)) );
        lh_target2clutter = plot((-maxlag:maxlag), target2clutter_xcorr, '-g');
        set(lh_target2clutter, 'LineWidth', 2);
        axis tight
        
        %plot power spectrum
        plot_title = ['Power for layer = ',int2str(layer)];
        figure('Name',plot_title);
        freq = 1000*(0:2*maxlag)/(2*maxlag + 1);
        min_ndx = find(freq > 400, 1,'first');
        target_fft = fft(target_xcorr);
        lh_target = plot(freq(2:min_ndx),...
            abs(target_fft(2:min_ndx)), '-r');
        set(lh_target, 'LineWidth', 2);
        hold on
        clutter_fft = fft(clutter_xcorr);
        lh_clutter = plot(freq(2:min_ndx),...
            abs(clutter_fft(2:min_ndx)), '-b');
        set(lh_clutter, 'LineWidth', 2);
        bkgrnd_fft = fft(bkgrnd_xcorr);
        lh_bkgrnd = plot(freq(2:min_ndx),...
            abs(bkgrnd_fft(2:min_ndx)), '-k');
        set(lh_bkgrnd, 'LineWidth', 2);
        set(lh_bkgrnd, 'Color', my_gray);
        target2clutter_fft = fft(target2clutter_xcorr);
        lh_target2clutter = plot(freq(2:min_ndx),...
            abs(target2clutter_fft(2:min_ndx)), '-g');
        set(lh_target2clutter, 'LineWidth', 2);
        axis tight

        %plot power reconstruction
        num_rate_sig = -1.0;
        mean_rate = mean( rate_array{layer} );
        std_rate = std( rate_array{layer} );
        rate_mask = find( rate_array{layer} > ( mean_rate + num_rate_sig * std_rate ) );
        num_rate_mask = numel(rate_mask);
        disp( [ 'mean_rate(', num2str(layer), ') = ', ...
            num2str(mean_rate), ' +/- ', num2str(std_rate) ] );
        disp( ['num_rate_mask(', num2str(layer), ') = ', num2str(num_rate_mask), ' > ', ...
            num2str( mean_rate + num_rate_sig * std_rate ) ] );
        while num_rate_mask > 2^14
            num_rate_sig = num_rate_sig + 0.5;
            rate_mask = find( rate_array{layer} > ( mean_rate + num_rate_sig * std_rate ) );
            num_rate_mask = numel(rate_mask);
            disp( ['num_rate_mask(', num2str(layer), ') = ', num2str(num_rate_mask), ' > ', ...
                num2str( mean_rate + num_rate_sig * std_rate ) ] );
        end
%         if num_rate_mask < 64
%             break;
%         end
        spike_full = full( spike_array{layer}(:, rate_mask) );
        freq_vals = 1000*(0:tot_steps-1)/tot_steps;
        min_ndx = find(freq_vals >= 40, 1,'first');
        max_ndx = find(freq_vals <= 60, 1,'last');
        power_array = fft( spike_full );
        power_array = power_array .* conj( power_array );
        peak_power = max(power_array(min_ndx:max_ndx,:));
        peak_power = sparse( rate_mask, 1, peak_power, N, 1 );
        pv_reconstruct(full(peak_power),  ['Power reconstruction for layer = ', int2str(layer)]);
        clear ave_target_tmp target_xcorr
        clear ave_clutter_tmp clutter_xcorr
        clear ave_bkgrnd_tmp bkgrnd_xcorr
        clear target2clutter_xcorr
        clear target_fft clutter_fft
        clear power_array spike_full bkgrnd_fft target2clutter_fft

        %cross-correlation using spike_array
        plot_pairwise_xcorr = 0;
        if plot_pairwise_xcorr
            plot_title = ['Cross Correlations for layer = ',int2str(layer)];
            figure('Name',plot_title);
            maxlag = fix( stim_length / 4 );
            freq = 1000*(0:2*maxlag)/(2*maxlag + 1);
            min_ndx = find(freq_vals >= 20, 1,'first');
            max_ndx = find(freq_vals <= 60, 1,'last');
            disp(['computing pv_xcorr_ave(', num2str(layer),')']);
            xcorr_ave_target = pv_xcorr_ave( spike_array{layer}(stim_steps, target_ndx{layer, i_target}), maxlag );
            lh_target = plot((-maxlag:maxlag), xcorr_ave_target, '-r');
            set(lh_target, 'LineWidth', 2);
            hold on
            xcorr_ave_clutter = pv_xcorr_ave( spike_array{layer}(stim_steps, clutter_ndx{layer}), maxlag );
            lh_clutter = plot((-maxlag:maxlag), xcorr_ave_clutter, '-b');
            set(lh_clutter, 'LineWidth', 2);
            hold on
            xcorr_ave_target2clutter = ...
                pv_xcorr_ave( spike_array{layer}(stim_steps, target_ndx{layer, i_target}), maxlag, ...
                spike_array{layer}(stim_steps, clutter_ndx{layer}) );
            lh_target2clutter = plot((-maxlag:maxlag), xcorr_ave_target2clutter, '-g');
            set(lh_target2clutter, 'LineWidth', 2);
            hold on
        end

        %make eigen amoebas
        plot_eigen = layer == -1 || layer == -3 || layer == -5 || layer == -6;
        if plot_eigen
            calc_power_mask = 1;
            if calc_power_mask
                num_power_sig = -1.0;
                mean_power = mean( peak_power(rate_mask) );
                std_power = std( peak_power(rate_mask) );
                disp( [ 'mean_power(', num2str(layer), ') = ', ...
                    num2str(mean_power), ' +/- ', num2str(std_power) ] );
                power_mask = find( peak_power > ( mean_power + num_power_sig * std_power ) );
                num_power_mask = numel(power_mask);
                disp( ['num_power_mask(', num2str(layer), ') = ', num2str(num_power_mask), ' > ', ...
                    num2str( mean_power + num_power_sig * std_power ) ] );
                while num_power_mask > 2^12
                    num_power_sig = num_power_sig + 0.5;
                    power_mask = find( peak_power > ( mean_power + num_power_sig * std_power ) );
                    num_power_mask = numel(power_mask);
                    disp( ['num_power_mask(', num2str(layer), ') = ', num2str(num_power_mask), ' > ', ...
                        num2str( mean_power + num_power_sig * std_power ) ] );
                end
%                 if num_power_mask < 64
%                     break;
%                 end
            else
                power_mask = sort( [target_ndx{layer, 1}, clutter_ndx{layer,1}] );
                num_power_mask = numel(power_mask);
            end % calc_power_mask
            disp(['computing xcorr(', num2str(layer),')']);
            %             pack;
            maxlag = fix( stim_length / 4 );
            freq_vals = 1000*(0:2*maxlag)/(2*maxlag + 1);
            min_ndx = find(freq_vals >= 20, 1,'first');
            max_ndx = max(find(freq_vals <= 60, 1,'last'), min_ndx);
            sparse_corr = ....
                pv_xcorr( spike_array{layer}(stim_steps, power_mask), ...
                spike_array{layer}(stim_steps, power_mask), ...
                maxlag, min_ndx,  max_ndx);
            for sync_true = 1 : 1 + (maxlag > 1)

                % find eigen vectors
                disp(['computing eigenvectors(', num2str(layer),')']);
                options.issym = 1;
                num_eigen = 6;
                [eigen_vec, eigen_value, eigen_flag] = ...
                    eigs( (1/2)*(sparse_corr{sync_true} + sparse_corr{sync_true}'), num_eigen, 'lm', options);
                for i_vec = 1:num_eigen
                    disp(['eigenvalues(', num2str(layer), ',' ...
                        , num2str(i_vec),') = ', num2str(eigen_value(i_vec,i_vec))]);
                end
                [max_eigen, max_eigen_ndx] = max( diag( eigen_value ) );
                xcorr_eigenvector{layer, sync_true} = eigen_vec(:,max_eigen_ndx);
                for i_vec = 1:num_eigen
                    plot_title = ['Eigen Reconstruction(', num2str(layer), '): sync_true = ', ...
                        num2str(2 - sync_true), ', i_vec = ', num2str(i_vec)];
                    %                     if mean(eigen_vec(power_mask,i_vec)) < 0
                    %                         eigen_vec(power_mask,i_vec) = -eigen_vec(power_mask,i_vec);
                    %                     end
                    eigen_vec_tmp = ...
                        sparse( power_mask, 1, eigen_vec(:,i_vec), ...
                        N, 1, num_power_mask);
                    pv_reconstruct(eigen_vec_tmp, plot_title);
                end % i_vec
            end % for sync_flag = 0:1
        end % plot_eigen
    end % layer
end % if plot_xcorr

