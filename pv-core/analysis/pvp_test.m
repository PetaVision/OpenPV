
close all
clear all
setenv("GNUTERM", "x11");

				  % set paths, may not be applicable to all octave installations
local_pwd = pwd;
user_index1 = findstr(local_pwd, 'Users')';
user_index1 = user_index1(1);
if ~isempty(user_index1)
  user_name = local_pwd(user_index1+6:length(local_pwd));
  user_index2 = findstr(user_name, '/');
  if isempty(user_index2)
    user_index2 = length(user_name);
    end
    user_name = user_name(1:user_index2-1);
    matlab_dir = ['/Users/', user_name, '/Documents/MATLAB'];
    addpath(matlab_dir);
    end
    

				  % Make the following global parameters available to all functions for convenience.
  global N_image NROWS_image NCOLS_image
  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer
  global NO NK dK % for the current layer
  global n_time_steps begin_step end_step time_steps tot_steps
  global stim_begin_step stim_end_step stim_steps
  global bin_size dt
  global begin_time end_time stim_begin_time stim_end_time
  global num_targets
  global rate_array
  global output_path input_path
  global NUM_BIN_PARAMS 
  global NUM_WGT_PARAMS
  NUM_BIN_PARAMS = 20;
  NUM_WGT_PARAMS = 6;


  input_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/';
  output_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/output/';

  num_layers = 3;
  pvp_order = 1;

				  % begin step, end_step and stim_begin_step may be adusted by
				  % pvp_readSparseSpikes
  begin_time = 0.0;  % (msec) start analysis here, used to exclude start up artifacts
  end_time = inf;
  stim_begin_time = 100.0;  % times (msec) before this and after begin_time can be used to calculate background
  stim_end_time = 900.0;
  bin_size = 10.0;  % (msec) used for all rate calculations

  stim_begin_bin = fix( ( stim_begin_step - begin_step + 1 ) / bin_size );
  stim_end_bin = fix( ( stim_end_step - begin_step + 1 ) / bin_size );
  stim_bins = stim_begin_bin : stim_end_bin;
  num_stim_bins = length(stim_bins);


				  % initialize to size of image (if known), these should be overwritten by each layer
  NROWS_image=128;
  NCOLS_image=128;
  NROWS = NROWS_image;
  NCOLS = NCOLS_image;
  NFEATURES = 12;

  NO = NFEATURES; % number of orientations
  NK = 1; % number of curvatures
  dK = 0; % spacing between curvatures (1/radius)

  my_gray = [.666 .666 .666];
  num_targets = 1;
  fig_list = [];

  read_spikes = 1:num_layers;  % list of spiking layers whose spike train are to be analyzed

  plot_vmem = 1;
  plot_weights_field = 1;
  min_plot_steps = 20;  % time-dependent quantities only plotted if tot_steps exceeds this threshold
  plot_reconstruct = uimatlab;
  plot_raster = uimatlab;
  plot_reconstruct_target = 1;

  spike_array = cell(num_layers,1);
  ave_rate = zeros(num_layers,1);

  %% read input image
  bmp_path = [input_path 'hard4.bmp'];
  [target, clutter, fig_tmp] = pvp_parseBMP( bmp_path, 1 );
  fig_list = [fig_list; fig_tmp];
  disp('parse BMP -> done');

				  % images may be annotated to indicate target and clutter pixels
  ave_target = cell(num_layers,num_targets);
  ave_clutter = cell(num_layers,1);
  ave_bkgrnd = cell(num_layers,1);
  psth_target = cell(num_layers,num_targets);
  psth_clutter = cell(num_layers,1);
  psth_bkgrnd = cell(num_layers,1);
  target_ndx = cell(num_layers, num_targets);
  clutter_ndx = cell(num_layers, num_targets);
  bkgrnd_ndx = cell(num_layers, num_targets);
  num_target_neurons = zeros(num_layers, num_targets);
  num_clutter_neurons = zeros(num_layers, 1);
  num_bkgrnd_neurons = zeros(num_layers, 1);
  num_rows = ones(num_layers,1);
  num_cols = ones(num_layers,1);
  num_features = ones(num_layers,1);
  target_rate = cell(num_layers, num_targets);
  target_rate_ndx = cell(num_layers, num_targets);
  clutter_rate = cell(num_layers, 1);
  clutter_rate_ndx = cell(num_layers, 1);



  %% Analyze spiking activity layer by layer
  for layer = read_spikes;
    disp(['analyzing layer: ', num2str(layer)]);

				  % Read spike events
    disp('reading spikes');
    [spike_array{layer}, ave_rate(layer)] = pvp_readSparseSpikes(layer, pvp_order);
    disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate(layer))]);

    num_bins = fix( tot_steps / bin_size );
    excess_steps = tot_steps - num_bins * bin_size;
    num_rows(layer) = NROWS;
    num_cols(layer) = NCOLS;
    num_features(layer) = NFEATURES;

				  % parse object segmentation info input image
    use_max = 1;
    bkgrnd_ndx{layer} = find(ones(N,1));
    clutter_ndx{layer} = ...
	pvp_image2layer( spike_array{layer}, clutter, stim_steps, use_max, pvp_order);
    num_clutter_neurons(layer) = length( clutter_ndx{layer} );
    ave_clutter{layer} = ...
	full( 1000*sum(spike_array{layer}(:,clutter_ndx{layer}),2) / ...
	     ( num_clutter_neurons(layer) + ( num_clutter_neurons(layer)==0 ) ) );
    bkgrnd_ndx{layer}(clutter_ndx{layer}) = 0;

    for i_target = 1:num_targets
      target_ndx{layer, i_target} = ...
	  pvp_image2layer( spike_array{layer}, target{i_target}, stim_steps, use_max, pvp_order);
      num_target_neurons(layer, i_target) = length( target_ndx{layer, i_target} );
      ave_target{layer,i_target} = ...
	  full( 1000*sum(spike_array{layer}(:,target_ndx{layer, i_target}),2) / ...
	       ( num_target_neurons(layer, i_target) + ( num_target_neurons(layer, i_target)==0 ) ) );
      bkgrnd_ndx{layer}(target_ndx{layer, i_target}) = 0;
      end % i_target

    bkgrnd_ndx{layer} = find(bkgrnd_ndx{layer});
    num_bkgrnd_neurons(layer) = N - sum(num_target_neurons(layer,:)) - num_clutter_neurons(layer);
    ave_bkgrnd{layer} = ...
	full( 1000*sum(spike_array{layer}(:,bkgrnd_ndx{layer}),2) / ...
	     ( num_bkgrnd_neurons(layer) + (num_bkgrnd_neurons(layer) == 0) ) );

    disp(['ave_target(',num2str(layer),') = ', num2str( mean( ave_target{layer,i_target}(:) ) ), 'Hz']);
    disp(['ave_clutter(',num2str(layer),') = ', num2str( mean( ave_clutter{layer,1}(:) ) ), 'Hz']);
    disp(['ave_bkgrnd(',num2str(layer),') = ', num2str( mean( ave_bkgrnd{layer,1}(:) ) ), 'Hz']);

    plot_rates = tot_steps > min_plot_steps;

    if ( plot_rates == 1 )
      plot_title = ['PSTH: layer = ',int2str(layer)];
      figure('Name',plot_title);

      for i_target = 1 : num_targets
	psth_target{layer,i_target} = ...
	    mean( reshape( ave_target{layer,i_target}(1:bin_size*num_bins), ...
			  bin_size, num_bins  ), 1);
	lh = plot((1:num_bins)*bin_size, psth_target{layer,i_target}, '-r');
	set(lh, 'LineWidth', 2);
	hold on;
	end % i_target

      psth_clutter{layer,1} = ...
	  mean( reshape( ave_clutter{layer,1}(1:bin_size*num_bins), ...
			bin_size, num_bins  ), 1);
      lh = plot((1:num_bins)*bin_size, psth_clutter{layer,1}, '-b');
      set(lh, 'LineWidth', 2);
      psth_bkgrnd{layer,1} = ...
	  mean( reshape( ave_bkgrnd{layer,1}(1:bin_size*num_bins), ...
			bin_size, num_bins  ), 1);
      lh = plot((1:num_bins)*bin_size, psth_bkgrnd{layer,1}, '-k');
      set(lh, 'LineWidth', 2);
      set(lh, 'Color', my_gray);
      end % plot_rates


   plot_raster = ( (plot_raster) && (tot_steps > min_plot_steps) && ...
			(~isempty(spike_array{layer})) );
   if plot_raster
     plot_title = ['Raster: layer = ',int2str(layer)'];
     fig_tmp = figure('Name',plot_title);
     fig_list = [fig_list; fig_tmp];
     [spike_time, spike_id] = find((spike_array{layer}));
     lh = plot(spike_time, spike_id, '.k');
     set(lh,'Color',my_gray);
     
     for i_target=1:num_targets
       [spike_time, spike_id] = ...
	   find((spike_array{layer}(:,target_ndx{layer, i_target})));
       plot(spike_time, target_ndx{layer, i_target}(spike_id), '.r');
       end % i_target
       
    [spike_time, spike_id] = find((spike_array{layer}(:,clutter_ndx{layer})));
    plot(spike_time, clutter_ndx{layer}(spike_id), '.b');
             
    end  % plot_raster
       
  if (plot_reconstruct_target && tot_steps > min_plot_steps)
    rate_array{layer} = 1000 * full( mean(spike_array{layer}(stim_steps,:),1) );
    for i_target = 1:num_targets
      target_rate{layer, i_target} = rate_array{layer}(1,target_ndx{layer, i_target});
      target_rate_array_tmp = ...
          sparse(1, target_ndx{layer, i_target}, target_rate{layer, i_target}, 1 , N, num_target_neurons(layer, i_target) );
      fig_tmp = pvp_reconstruct(full(target_rate_array_tmp), ...
				['Target rate reconstruction; layer = ', ...
				 int2str(layer), ', target index = ', ...
				 int2str(i_target)]);     
      fig_list = [fig_list; fig_tmp];
      [target_rate{layer, i_target}, target_rate_ndx{layer, i_target}] = ...
          sort( target_rate{layer, i_target}, 2, 'descend');
      for i_rank = [ 1 , ceil(num_target_neurons(layer, i_target)/2), num_target_neurons(layer, i_target) ]
        tmp_rate = target_rate{layer, i_target}(i_rank);
        tmp_ndx = target_rate_ndx{layer, i_target}(i_rank);
        k = target_ndx{layer, i_target}(tmp_ndx);
        [kf, kcol, krow] = ind2sub([num_features(layer), num_cols(layer), num_rows(layer)], k);
        disp(['rank:',num2str(i_rank),...
              ', target_rate(',num2str(layer),', ', num2str(i_target), ')', ...
              num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
	      num2str([k-1, kcol-1, krow-1, kf-1]) ]);
      end % i_rank
    end % i_target
    clutter_rate{layer, 1} = rate_array{layer}(1,clutter_ndx{layer, 1});
    [clutter_rate{layer, 1}, clutter_rate_ndx{layer, 1}] = ...
        sort( clutter_rate{layer, 1}, 2, 'descend');
    clutter_rate_array_tmp = ...
        sparse(1, clutter_ndx{layer, 1}, clutter_rate{layer, 1}, 1 , N, num_clutter_neurons(layer, 1) );
    fig_tmp = pvp_reconstruct(full(clutter_rate_array_tmp), ...
			      ['Clutter rate reconstruction: layer = ', ...
			       int2str(layer)]);
    fig_list = [fig_list; fig_tmp];
    for i_rank = [ 1 , ceil(num_clutter_neurons(layer, 1)/2), num_clutter_neurons(layer, 1) ]
      tmp_rate = clutter_rate{layer, 1}(i_rank);
      tmp_ndx = clutter_rate_ndx{layer, 1}(i_rank);
      k = clutter_ndx{layer, 1}(tmp_ndx);
      [kf, kcol, krow] = ind2sub([num_features(layer), num_cols(layer), num_rows(layer)], k);
      disp(['rank:',num2str(i_rank),...
            ', clutter_rate(', num2str(layer),')', ...
            num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
	    num2str([k-1, kcol-1, krow-1, kf-1]) ]);
    end % i_rank
  end %  reconstruc target/clutter
  
				% plot reconstructed image
  plot_rate_reconstruction = ( plot_reconstruct && tot_steps > ...
			      min_plot_steps );
  if plot_rate_reconstruction
    pvp_reconstruct(rate_array{layer}, ...
		    ['Rate reconstruction: layer = ', int2str(layer)]);
    fig_list = [fig_list; fig_tmp];
  end

end % layer

  %% plot connections
  global N_CONNECTIONS
  global NXP NYP NFP
  N_CONNECTIONS = 3;
  plot_weights = 1;
if ( plot_weights == 1 )
  weights = cell(N_CONNECTIONS, 1);
  nxp = cell(N_CONNECTIONS, 1);
  nyp = cell(N_CONNECTIONS, 1);
  pvp_header = cell(N_CONNECTIONS, 1);
  for i_conn = 1 : N_CONNECTIONS
    [ weights{i_conn}, nxp{i_conn}, nyp{i_conn}, pvp_header{i_conn}, pvp_index ] = pvp_readWeights(i_conn);
    NK = 1;
    NO = floor( NFEATURES / NK );
    pvp_header_tmp = pvp_header{i_conn};
    num_patches = pvp_header_tmp(pvp_index.WGT_NUMPATCHES);
    for i_patch = 1 : num_patches
      NCOLS = nxp{i_conn}(i_patch);
      NROWS = nyp{i_conn}(i_patch);
      N = NROWS * NCOLS * NFEATURES;
      pvp_reconstruct(weights{i_conn}{i_patch}, ['Weight reconstruction: connection = ', int2str(i_conn)]);
    end % i_patch
  end % i_conn
end % plot_weights
