

clear all;
close all;
setenv("GNUTERM","X11")

%% machine/run_type environment
if ismac
elseif isunix
  run_type = {"deep"; "noPulvinar"}; %%; "noTopDown"}; %%
  run_type_colormap = colormap("default");
  output_path = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128";
  mkdir(output_path);
  output_dir_root = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X";
  %%output_dir_root = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X2";
  threshold_vals = {"1"; "2"}%%; "3"; "4"};
  %%threshold_vals = {""; "_noise_05"; "_noise_10"};%%; "_noise_20"};
endif %% isunix

%% default paths
nonSparse_list = ...
    {["a9_"], ["Error1_2"]};
length_nonSparse_list = size(nonSparse_list,1);
Sparse_bins_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
Sparse_vals_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
mean_nonSparse_RMS_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
std_nonSparse_RMS_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
nonSparse_RMS_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
nonSparse_norm_RMS_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
nonSparse_times_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
ErrorVsSparse_fig = zeros(length_nonSparse_list,1);
for i_nonSparse = 1 : length_nonSparse_list
  for i_run_type = 1 : length(run_type)
    for i_thresh = 1:length(threshold_vals)
      ErrorVsSparse_dir = ...
	  [output_dir_root, threshold_vals{i_thresh}, "_", run_type{i_run_type}, filesep, "nonSparse"]
      if ~exist(ErrorVsSparse_dir)
	error(["ErrorVsSparse_dir does not exist: ", ErrorVsSparse_dir]);
      endif
      ErrorVsSparse_str = ...
	  [ErrorVsSparse_dir, filesep, "ErrorVsSparse", "_", "Noiseless", nonSparse_list{i_nonSparse,2}, "_", "*", ".mat"];
      ErrorVsSparse_glob = glob(ErrorVsSparse_str);
      num_ErrorVsSparse_glob = length(ErrorVsSparse_glob);
      if num_ErrorVsSparse_glob <= 0
	warning(["no files to load in: ", ErrorVsSparse_str]);
	ErrorVsSparse_str = ...
	    [ErrorVsSparse_dir, filesep, "ErrorVsSparse", "_", nonSparse_list{i_nonSparse,2}, "_", "*", ".mat"];
	ErrorVsSparse_glob = glob(ErrorVsSparse_str);
	num_ErrorVsSparse_glob = length(ErrorVsSparse_glob);	
	if num_ErrorVsSparse_glob <= 0
	  warning(["no files to load in: ", ErrorVsSparse_str]);
	  break;
	endif
      endif
      load("-mat", ErrorVsSparse_glob{num_ErrorVsSparse_glob});
      Sparse_bins_array{i_thresh, i_run_type, i_nonSparse} = ...
	  Sparse_bins;
      Sparse_vals_array{i_thresh, i_run_type, i_nonSparse} = ...
	  Sparse_vals;
      mean_nonSparse_RMS_array{i_thresh, i_run_type, i_nonSparse} = ...
	  mean_nonSparse_RMS;
      std_nonSparse_RMS_array{i_thresh, i_run_type, i_nonSparse} = ...
	  std_nonSparse_RMS;
      nonSparse_RMS_array{i_thresh, i_run_type, i_nonSparse} = ...
	  nonSparse_RMS;
      nonSparse_norm_RMS_array{i_thresh, i_run_type, i_nonSparse} = ...
	  nonSparse_norm_RMS;
      nonSparse_times_array{i_thresh, i_run_type, i_nonSparse} = ...
	  nonSparse_times;
    endfor%% i_thresh
  endfor%% i_run_type

  ErrorVsSparse_fig(i_nonSparse) = figure;
  %%axis([0.95 1.0 0.1 0.5])
  hold on
  for i_run_type = 1 : length(run_type)
    run_type_color = ...
	run_type_colormap(floor(1+length(run_type_colormap)*((i_run_type-1)/...
							     (length(run_type))+(length(run_type)==1))),:);
    for i_thresh = 1:length(threshold_vals)
      run_type_color2 = ...
	  run_type_colormap(min(floor(length(run_type_colormap) * ...
					(i_thresh / length(threshold_vals))),...
				length(run_type_colormap)),:);
      Sparse_vals = squeeze(Sparse_vals_array{i_thresh, i_run_type, i_nonSparse});
      normalized_nonSparse_RMS = nonSparse_RMS_array{i_thresh, i_run_type, i_nonSparse} ./ ...
	  (nonSparse_norm_RMS_array{i_thresh, i_run_type, i_nonSparse} + ...
	   (nonSparse_norm_RMS_array{i_thresh, i_run_type, i_nonSparse} == 0));
      normalized_nonSparse_RMS = squeeze(normalized_nonSparse_RMS);
      normalized_nonSparse_RMS = normalized_nonSparse_RMS(1:length(Sparse_vals));
      ErrorVsSparse_hndl = ...
	  plot(Sparse_vals, ...
	       normalized_nonSparse_RMS, ...
	       "."); 
      set(ErrorVsSparse_hndl, "color", run_type_color);
      Sparse_bins = Sparse_bins_array{i_thresh, i_run_type, i_nonSparse};
      skip_Sparse_val = (Sparse_bins(end) - Sparse_bins(1)) / length(Sparse_bins);
      mean_nonSparse_RMS = mean_nonSparse_RMS_array{i_thresh, i_run_type, i_nonSparse};
      std_nonSparse_RMS = std_nonSparse_RMS_array{i_thresh, i_run_type, i_nonSparse};
      eh(i_run_type) = errorbar(Sparse_bins+skip_Sparse_val/2, mean_nonSparse_RMS, std_nonSparse_RMS);
      set(eh(i_run_type), "color", run_type_color);
      set(eh(i_run_type), "linewidth", 1.5);
    endfor%% i_thresh
  endfor%% i_run_type
  legend(eh, run_type)
  set(ErrorVsSparse_fig(i_nonSparse), "name", ...
      ["ErrorVsSparse_", nonSparse_list{i_nonSparse,2}]);
  saveas(ErrorVsSparse_fig(i_nonSparse), ...
	 [output_path, filesep, ...
	  "ErrorVsSparse_", nonSparse_list{i_nonSparse,2}, ".png"], "png");
endfor  %% i_nonSparse





