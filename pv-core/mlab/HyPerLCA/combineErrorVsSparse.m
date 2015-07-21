

clear all;
close all;
setenv("GNUTERM","X11")

ErrorVsSparse_fig = figure;
hold on

%% machine/run_type environment
if ismac
elseif isunix
%%  run_type = {"deep"; "V1"}; %%
  run_type = {"lateral"}; 
%% run_type = {"deep"; "noPulvinar"}; %%; "noTopDown"}; %%
%%  num_features = [128; 160];
  num_features = [128];
  run_type_colormap = colormap("default");
  %%output_path = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x"; %% 128";
  output_path = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_3x3_9x9x"; %%128";
  output_dir_name = [output_path, num2str(num_features(1))];
  for i_feature = 2 : length(num_features)
    output_dir_name = [output_dir_name, "_", num2str(num_features(i_feature))];
  endfor
  mkdir(output_dir_name);
%%  output_dir_root = "_lambda_05X";
  output_dir_root = "_lambda_001X";
%%  threshold_vals = {"1"; "2"; "3"; "4"};
  threshold_vals = {"10"; "25"; "50"; "100"};
  %%threshold_vals = {""; "_noise_05"; "_noise_10"};%%; "_noise_20"};
endif %% isunix

%% default paths
%%nonSparse_list = ...
%%    {["a5_"], ["Recon"]; ...
%%     ["a13_"], ["Recon2PlusReconInfra"]};
%%nonSparse_list = ...
%%    {["a9_"], ["Error1_2"]};
%%nonSparse_list = ...
%%    {["a11_"], ["ReconInfra"]};
%%nonSparse_list = ...
%%    {["a5_"], ["Recon"]};
%%nonSparse_list = ...
%%    {["a3_"], ["Error"];
%%     ["a6_"], ["Error2"];
%%     ["a10_"], ["Error1_2"]};
%%nonSparse_list = ...
%%    {["a3_"], ["Error"];
%%     ["a10_"], ["Error1_2"]};
%%nonSparse_list = ...
%%    {["a3_"], ["Error"];
%%     ["a6_"], ["Error2"]};
nonSparse_list = ...
    {["a5_"], ["Recon"];
     ["a13_"], ["Recon2PlusReconInfra"];
     ["a16_"], ["ReconMaxPoolingV2X05"]};
length_nonSparse_list = size(nonSparse_list,1);
Sparse_bins_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
Sparse_vals_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
mean_nonSparse_RMS_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
std_nonSparse_RMS_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
nonSparse_RMS_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
nonSparse_norm_RMS_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
nonSparse_times_array = cell( length(threshold_vals), length(run_type),length_nonSparse_list);
%%ErrorVsSparse_fig = zeros(length_nonSparse_list,1);
%%axis([0.95 1.0 0.1 0.5])
hold on
nonSparse_colormap = [ [1 0 0]; [0 0 1]; [0 1 0]];
for i_nonSparse = 1 : length_nonSparse_list
  if length_nonSparse_list > 3
    nonSparse_color = ...
	run_type_colormap(floor(1+length(run_type_colormap)*((i_nonSparse-1)/...
							   (length_nonSparse_list+(length_nonSparse_list==1)))),:);
  else
    nonSparse_color = nonSparse_colormap(i_nonSparse, :);
  endif
  for i_run_type = 1 : length(run_type)
    for i_thresh = 1:length(threshold_vals)
      ErrorVsSparse_dir = ...
	  [output_path, num2str(num_features(i_run_type)), output_dir_root, threshold_vals{i_thresh}, "_", run_type{i_run_type}, filesep, "ErrorVsSparse"]
      if ~exist(ErrorVsSparse_dir)
	ErrorVsSparse_dir = ...
	  [output_path, num2str(num_features(i_run_type)), output_dir_root, threshold_vals{i_thresh}, "_", run_type{i_run_type}, filesep, "nonSparse"]
	if ~exist(ErrorVsSparse_dir)
	  error(["ErrorVsSparse_dir does not exist: ", ErrorVsSparse_dir]);
	endif
      endif
      ErrorVsSparse_str = ...
	  [ErrorVsSparse_dir, filesep, "ErrorVsSparse", "_", nonSparse_list{i_nonSparse,1}, nonSparse_list{i_nonSparse,2}, "_", "*", ".mat"];
      ErrorVsSparse_glob = glob(ErrorVsSparse_str);
      num_ErrorVsSparse_glob = length(ErrorVsSparse_glob);
      if num_ErrorVsSparse_glob <= 0
	warning(["no files to load in: ", ErrorVsSparse_str]);
	ErrorVsSparse_str = ...
	    [ErrorVsSparse_dir, filesep, "ErrorVsSparse", "_", nonSparse_list{i_nonSparse,2}, "_", "*", ".mat"];
	ErrorVsSparse_glob = glob(ErrorVsSparse_str);
	num_ErrorVsSparse_glob = length(ErrorVsSparse_glob);
      endif
      if num_ErrorVsSparse_glob <= 0
	warning(["no files to load in: ", ErrorVsSparse_str]);
	ErrorVsSparse_str = ...
	    [ErrorVsSparse_dir, filesep, "ErrorVsSparse", "_", "Noiseless", nonSparse_list{i_nonSparse,1}, nonSparse_list{i_nonSparse,2}, "_", "*", ".mat"];
	ErrorVsSparse_glob = glob(ErrorVsSparse_str);
	num_ErrorVsSparse_glob = length(ErrorVsSparse_glob);	
	if num_ErrorVsSparse_glob <= 0
	  warning(["no files to load in: ", ErrorVsSparse_str]);
	  ErrorVsSparse_str = ...
	      [ErrorVsSparse_dir, filesep, "ErrorVsSparse", "_", "Noiseless", nonSparse_list{i_nonSparse,2}, "_", "*", ".mat"];
	  ErrorVsSparse_glob = glob(ErrorVsSparse_str);
	  num_ErrorVsSparse_glob = length(ErrorVsSparse_glob);	
	endif
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

  for i_run_type = 1 : length(run_type)
    run_type_color = ...
	run_type_colormap(floor(1+length(run_type_colormap)*((i_run_type-1)/...
							     (length(run_type)+(length(run_type)==1)))),:);
    for i_thresh = 1:length(threshold_vals)
      thresh_color = ...
	  run_type_colormap(min(floor(length(run_type_colormap) * ...
					(i_thresh / length(threshold_vals))),...
				length(run_type_colormap)),:);
      Sparse_vals = squeeze(Sparse_vals_array{i_thresh, i_run_type, i_nonSparse});
      normalized_nonSparse_RMS = nonSparse_RMS_array{i_thresh, i_run_type, i_nonSparse} ./ ...
	  (nonSparse_norm_RMS_array{i_thresh, i_run_type, i_nonSparse} + ...
	   (nonSparse_norm_RMS_array{i_thresh, i_run_type, i_nonSparse} == 0));
      normalized_nonSparse_RMS = squeeze(normalized_nonSparse_RMS);
      normalized_nonSparse_RMS = normalized_nonSparse_RMS(1:length(Sparse_vals));
      figure(ErrorVsSparse_fig);
      hold on
      ErrorVsSparse_hndl(i_run_type,i_thresh) = ...
	  plot(Sparse_vals, ...
	       normalized_nonSparse_RMS, ...
	       "."); 
      set(ErrorVsSparse_hndl(i_run_type,i_thresh), "color", nonSparse_color);
      Sparse_bins = Sparse_bins_array{i_thresh, i_run_type, i_nonSparse};
      skip_Sparse_val = (Sparse_bins(end) - Sparse_bins(1)) / length(Sparse_bins);
      mean_nonSparse_RMS = mean_nonSparse_RMS_array{i_thresh, i_run_type, i_nonSparse};
      std_nonSparse_RMS = std_nonSparse_RMS_array{i_thresh, i_run_type, i_nonSparse};
      eh(i_nonSparse, i_run_type,i_thresh) = errorbar(Sparse_bins+skip_Sparse_val/2, mean_nonSparse_RMS, std_nonSparse_RMS);
      set(eh(i_nonSparse, i_run_type,i_thresh), "color", nonSparse_color);
      set(eh(i_nonSparse, i_run_type,i_thresh), "linewidth", 1.5);
    endfor%% i_thresh
  endfor%% i_run_type
endfor  %% i_nonSparse
legend_hndl = legend(squeeze(eh(:,1,1)), {"S1";"C1";"MaxPool"})
set(legend_hndl, "fontsize", 32)
set(gca, "fontsize", 16);
x_hndl = xlabel(gca, "Sparsity");
set(x_hndl, "fontsize", 24);
y_hndl = ylabel(gca, "% Error");
set(y_hndl, "fontsize", 24);
ErrorVsSparse_name = "ErrorVsSparse"; 
for i_nonSparse = 1 : length_nonSparse_list 
  ErrorVsSparse_name = [ErrorVsSparse_name, "_", nonSparse_list{i_nonSparse,2}];
endfor
set(ErrorVsSparse_fig, "name", ...
    ErrorVsSparse_name);
saveas(ErrorVsSparse_fig, ...
       [output_dir_name, filesep, ...
	ErrorVsSparse_name, ".png"], "png");





