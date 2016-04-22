

clear all;
close all;
setenv("GNUTERM","X11")

percent_change_fig = figure;
hold on
percent_active_fig = figure;
hold on
change_vs_active_fig = figure;
hold on

%% machine/run_type environment
if ismac
elseif isunix
  run_type = {"lateral"}; %%{"deep"; "noPulvinar"}; %%; "noTopDown"}; %%
  run_type_colormap = colormap("default");
  %%output_path = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128";
  output_path = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_3x3_9x9x128";
  mkdir(output_path);
  output_dir_root = "_lambda_001X";
  threshold_vals = {"10"; "25"; "50"; "100"};
endif %% isunix

%% default paths
Sparse_list = ...
    {["a4_"], ["V1"]; ...
     ["a7_"], ["V2"]; ...
     ["a14_"], ["MaxPoolingV2X05"]};
length_Sparse_list = size(Sparse_list,1);
Sparse_times_array = cell( length(threshold_vals), length(run_type),length_Sparse_list);
percent_change_array = cell( length(threshold_vals), length(run_type),length_Sparse_list);
percent_active_array = cell( length(threshold_vals), length(run_type),length_Sparse_list);

Sparse_colormap = [ [1 0 0]; [0 0 1]; [0 1 0]];

for i_Sparse = 1 : length_Sparse_list
%%  Sparse_color = ...
%%      run_type_colormap(floor(1+length(run_type_colormap)*((i_Sparse-1)/...
%%							   (length_Sparse_list+(length_Sparse_list==1)))),:);
  Sparse_color = Sparse_colormap(i_Sparse,:);
  Sparse_pointtype = {"x"; "o"; "+"};
  for i_run_type = 1 : length(run_type)
    for i_thresh = 1:length(threshold_vals)
      percent_change_dir = ...
	  [output_path, output_dir_root, threshold_vals{i_thresh}, "_", run_type{i_run_type}, filesep, "Sparse"]
      if ~exist(percent_change_dir)
	error(["percent_change_dir does not exist: ", percent_change_dir]);
      endif

      percent_change_str = ...
	  [percent_change_dir, filesep, "Sparse_percent_change", "_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      percent_change_glob = glob(percent_change_str);
      num_percent_change_glob = length(percent_change_glob);
      load("-mat", percent_change_glob{num_percent_change_glob});
      percent_change_array{i_thresh, i_run_type, i_Sparse} = ...
	  Sparse_percent_change;

      percent_active_str = ...
	  [percent_change_dir, filesep, "Sparse_percent_active", "_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      percent_active_glob = glob(percent_active_str);
      num_percent_active_glob = length(percent_active_glob);
      load("-mat", percent_active_glob{num_percent_active_glob});
      percent_active_array{i_thresh, i_run_type, i_Sparse} = ...
	  Sparse_percent_active;

      Sparse_times_array{i_thresh, i_run_type, i_Sparse} = ...
	  Sparse_times;
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
      thresh_thickness =  1.5 + (length(threshold_vals)*(i_thresh / length(threshold_vals)) / 4);

      figure(percent_change_fig)
      percent_change_hndl(i_Sparse) = ...
	  plot(squeeze(Sparse_times_array{i_thresh, i_run_type, i_Sparse}), ...
	       squeeze(percent_change_array{i_thresh, i_run_type, i_Sparse}), ...
	       "-"); 
      set(percent_change_hndl(i_Sparse), "color", Sparse_color);
      set(percent_change_hndl(i_Sparse), "linewidth", thresh_thickness);


      figure(percent_active_fig)
      percent_active_hndl(i_Sparse) = ...
	  plot(squeeze(Sparse_times_array{i_thresh, i_run_type, i_Sparse}), ...
	       squeeze(percent_active_array{i_thresh, i_run_type, i_Sparse}), ...
	       "-"); 
      set(percent_active_hndl(i_Sparse), "color", Sparse_color);
      set(percent_active_hndl(i_Sparse), "linewidth", thresh_thickness);

      figure(change_vs_active_fig)
      change_vs_active_hndl(i_Sparse) = ...
	  plot(squeeze(1-percent_active_array{i_thresh, i_run_type, i_Sparse}), ...
	       squeeze(percent_change_array{i_thresh, i_run_type, i_Sparse}), ...
	       Sparse_pointtype{i_Sparse}); 
      set(change_vs_active_hndl(i_Sparse), "color", Sparse_color);
      set(change_vs_active_hndl(i_Sparse), "linewidth", thresh_thickness);

    endfor%% i_thresh
  endfor%% i_run_type


endfor  %% i_Sparse
legend_list = {"S1";"C1";"MaxPool"};

figure(percent_change_fig)
set(gca, "fontsize", 16);
x_hndl = xlabel(gca, "time (msec)");
set(x_hndl, "fontsize", 24);
y_hndl = ylabel(gca, "% change frame-to-frame");
set(y_hndl, "fontsize", 24);
legend_hndl = legend(percent_change_hndl, legend_list);
set(legend_hndl, "fontsize", 32)
legend_hndl = legend(percent_change_hndl, legend_list);
%%legend(percent_change_hndl, Sparse_list(:,2))
percent_change_name = "percent_change"; 
for i_Sparse = 1 : length_Sparse_list 
  percent_change_name = [percent_change_name, "_", Sparse_list{i_Sparse,2}];
endfor
set(percent_change_fig, "name", ...
    percent_change_name);
saveas(percent_change_fig, ...
       [output_path, filesep, ...
	percent_change_name, ".png"], "png");

figure(percent_active_fig)
set(gca, "fontsize", 16);
x_hndl = xlabel(gca, "time (msec)");
set(x_hndl, "fontsize", 24);
y_hndl = ylabel(gca, "% active");
set(y_hndl, "fontsize", 24);
legend_hndl = legend(percent_active_hndl, legend_list);
set(legend_hndl, "fontsize", 32)
%%legend(percent_active_hndl, Sparse_list(:,2))
percent_active_name = "percent_active"; 
for i_Sparse = 1 : length_Sparse_list 
  percent_active_name = [percent_active_name, "_", Sparse_list{i_Sparse,2}];
endfor
set(percent_active_fig, "name", ...
    percent_active_name);
saveas(percent_active_fig, ...
       [output_path, filesep, ...
	percent_active_name, ".png"], "png");



figure(change_vs_active_fig)
set(gca, "fontsize", 16);
x_hndl = xlabel(gca, "Sparsity");
set(x_hndl, "fontsize", 24);
y_hndl = ylabel(gca, "% change");
set(y_hndl, "fontsize", 24);
legend_hndl = legend(change_vs_active_hndl, legend_list, "location", "northwest");
set(legend_hndl, "fontsize", 32)
%%legend(change_vs_active_hndl, Sparse_list(:,2))
change_vs_active_name = "change_vs_active"; 
for i_Sparse = 1 : length_Sparse_list 
  change_vs_active_name = [change_vs_active_name, "_", Sparse_list{i_Sparse,2}];
endfor
set(change_vs_active_fig, "name", ...
    change_vs_active_name);
saveas(change_vs_active_fig, ...
       [output_path, filesep, ...
	change_vs_active_name, ".png"], "png");





