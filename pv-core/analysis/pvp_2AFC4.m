%% plot 2AFC results vs two independent parameters

clear all
setenv("GNUTERM", "x11");
amoeba_flag = 0;
bowtie_flag = 0;
rendered_flag = 1;

%% set paths to twoAFC.mat files
if amoeba_flag
  twoAFC_path = '/Users/gkenyon/workspace/kernel/input/256/amoeba/';
elseif bowtie_flag
  twoAFC_path = '/Users/gkenyon/workspace2/geisler/input/256/bowtie/';
elseif rendered_flag
  twoAFC_path = '/Users/gkenyon/workspace2/geisler/input/256/dog/';  
endif

%% define params0
param0_list = [];
if amoeba_flag %% denote # Fourier Components
  param0_list = [4]; %%[2 4 6 8];
elseif bowtie_flag  %% denote # Fourier Components (always 4)
  param0_list = [4]; %%[2 4 6 8];
elseif rendered_flag  %% denote target/distractor model #'s
  num_models = 6;
  param0_list_dog = 1:num_models;  %% row ndx
  param0_list_cat = 1:num_models;  %% col ndx
  param0_list2D =  ...
      repmat(param0_list_dog', 1, num_models) + ...
      repmat((param0_list_cat-1)*num_models, num_models, 1);
  rendered_diag_flag = 1;  %% until all pairs have been tested ...
  if rendered_diag_flag == 1
    param0_list = diag(param0_list2D);
  else
    param0_list = param0_list2D(:);
  endif
  skip_param0 = num_models;
endif
num_params0 = length(param0_list);

%% define params
%% typically kernel strength
if amoeba_flag
  param_list = [287, 300, 325, 400];
elseif bowtie_flag
  param_list = [50, 75, 125, 175];
elseif rendered_flag
  param_list = [2000];
endif
num_params = length(param_list);


twoAFC_dir_list = cell(num_params0, num_params);
for i_param = 1 : num_params
  for i_param0 = 1 : num_params0
    twoAFC_dir_tmp = twoAFC_path;
    if amoeba_flag
      twoAFC_dir_tmp = ...
	  [twoAFC_dir_tmp, ...
	   'test_target40K_W', ...
	   num2str(param_list(i_param)), ...
	   '_target'];
      twoAFC_dir_tmp = [twoAFC_dir_tmp, '_G1/'];
      twoAFC_dir_tmp = ...
	  [twoAFC_dir_tmp, ...
	   num2str(num_params0), 'fc/'];
    elseif bowtie_flag
      twoAFC_dir_tmp = ...
	  [twoAFC_dir_tmp, ...
	   'target_', ...
	   num2str(param_list(i_param))];
      twoAFC_dir_tmp = [twoAFC_dir_tmp, '/'];    
      twoAFC_dir_tmp = [twoAFC_dir_tmp, ...
		    num2str(num_params0), 'fc/'];
    elseif rendered_flag
      [target_id, distractor_id] = ...
	  ind2sub( [num_models, num_models], param0_list(i_param0) );
      twoAFC_dir_tmp = ...
	  [twoAFC_dir_tmp, ...
	   "rendered_DoG_test_", ...
	   num2str(target_id)];
      twoAFC_dir_tmp = [twoAFC_dir_tmp, '/'];    
      twoAFC_dir_tmp = ...
	  [twoAFC_dir_tmp, ...
	   "cat_", ...
	   num2str(distractor_id), "/"];      
    endif
    twoAFC_dir_list{i_param0, i_param} = ...
	twoAFC_dir_tmp;
  endfor %% i_param0
endfor %% i_param
max_expNum = 1;  % maximum number of independent experiments to be combined
twoAFC_ROC_array = cell(num_params0, num_params, max_expNum);
twoAFC_AUC_array = cell(num_params0, num_params, max_expNum);
twoAFC_errorbar_array = cell(num_params0, num_params, max_expNum);
expNum_list = [1];  % list for combining results from several experiments
len_expNum = length(expNum_list);
local_path = pwd;
for i_param = 1 : num_params
  disp(['param = ', num2str(param_list(i_param))]);
  for i_param0 = 1 : num_params0
    disp(['param0 = ', num2str(param0_list(i_param0))]);
    twoAFC_dir_tmp = ...
	twoAFC_dir_list{i_param0, i_param};
    for i_expNum = 1 : len_expNum
      expNum = expNum_list(i_expNum);
      twoAFC_filename = ...
	  [twoAFC_dir_tmp, ...
	   'twoAFC', num2str(expNum), '.mat'];
      disp(["twoAFC_filename = ", twoAFC_filename]);
      load(twoAFC_filename)
      if len_expNum == 1
	twoAFC_ROC_array{i_param0, i_param} = twoAFC_ROC;
	twoAFC_AUC_array{i_param0, i_param} = twoAFC_AUC;
	twoAFC_errorbar_array{i_param0, i_param} = twoAFC_errorbar;
      else
	twoAFC_ROC_array{i_param0, i_param, i_expNum} = twoAFC_ROC;
	twoAFC_AUC_array{i_param0, i_param, i_expNum} = twoAFC_AUC;
	twoAFC_errorbar_array{i_param0, i_param, i_expNum} = twoAFC_errorbar;
      endif
      num_layers = size(twoAFC, 2);
      tot_trials = size(twoAFC, 3);
      num_2AFC_tests = 1; %size(twoAFC, 4);
      
      for i_2AFC_test = 1 : 1%num_2AFC_tests
	disp(['i_2AFC_test = ', num2str(i_2AFC_test)]);
	for layer = 1 : num_layers
	  disp( ['twoAFC_correct(', num2str(layer), ...
				 ',', num2str(i_2AFC_test), ') = ', ...
		 num2str(twoAFC_correct(layer, i_2AFC_test)) ] );
	endfor
      endfor % i_2AFC_test

      
      for i_2AFC_test = 1 : 1%num_2AFC_tests
	disp(['i_2AFC_test = ', num2str(i_2AFC_test)]);
	for layer = 1 : num_layers
	  disp( ['twoAFC_AUC(', num2str(layer), ...
			     ',', num2str(i_2AFC_test), ') = ', ...
		 num2str(twoAFC_AUC_array{i_param0, i_param, i_expNum}(layer, i_2AFC_test)), ...
		 ' +/- ', ...
		 num2str(twoAFC_errorbar_array{i_param0, i_param, i_expNum}(layer, i_2AFC_test)) ] );
	endfor
      endfor % i_2AFC_test

      
    endfor % i_expNum
  endfor % i_param0
endfor % i_param

%% plot ROC curve tableau
fig_list = [];
subplot_index = 0;
num_subplots = num_layers-1;
nrows_subplot = 2;
ncols_subplot = ceil( (num_layers-1) / 2 );
twoAFC_ROC_name = ['2AFC ROC Model'];
fig_tmp = figure('Name', twoAFC_ROC_name);
fig_list = [fig_list; fig_tmp];
i_2AFC_test = 1;
i_expNum = 1;
for layer = 2:num_layers
  subplot_index = subplot_index + 1;
  subplot(nrows_subplot, ncols_subplot, subplot_index);
  axis([0 1 0 1]);
  axis "nolabel"
  axis "square"
  th = title(["iter = ", num2str(layer-2)]);
  hold on;
  for i_param = 1 : num_params
    for i_param0 = 1:num_params0
      twoAFC_ROC_tmp = ...
	  twoAFC_ROC_array{i_param0, i_param, i_expNum};
      lh = plot(twoAFC_ROC_tmp{layer, i_2AFC_test}(1,:), ...
		twoAFC_ROC_tmp{layer, i_2AFC_test}(2,:), ...
		num2str(i_param0+(i_param-1)*num_params0));  
      set( lh, 'LineWidth', 2 );
    endfor % i_param0
  endfor % i_params
  axis([0 1 0 1]);
  axis "nolabel"
  axis "square"
endfor

%% plot AUC vs number of iterations
twoAFC_AUC_name = ['2AFC AUC Model'];
fig_tmp = figure('Name', twoAFC_AUC_name);
fig_list = [fig_list; fig_tmp];
AUC_layers = 3:num_layers;
axis( [0.95*min(AUC_layers) 1.05*max(AUC_layers) 0.45 1.0]);
hold on;
for i_param = 1 : num_params
  for i_param0 = 1:num_params0
    twoAFC_AUC_tmp = twoAFC_AUC_array{i_param0, i_param, i_expNum};
    twoAFC_errorbar_tmp = twoAFC_errorbar_array{i_param0, i_param, i_expNum};
    eh = errorbar(AUC_layers, ...
		  twoAFC_AUC_tmp(AUC_layers, i_2AFC_test), ...
		  twoAFC_errorbar_tmp(AUC_layers, i_2AFC_test));
    set( eh, 'LineWidth', 2 );
    lh = plot(AUC_layers, ...
	      twoAFC_AUC_tmp(AUC_layers, i_2AFC_test), ...		  
	      num2str(i_param0+(i_param-1)*num_params0));  
    set( lh, 'LineWidth', 2 );
    line_color = get( lh, 'Color');
    set( eh, 'Color', line_color);
  endfor %% i_param0
endfor %% i_param


%% plot AUC vs param
if num_params > 1
  twoAFC_param_name = ['2AFC vs. Param'];
  fig_tmp = figure('Name', twoAFC_param_name);
  fig_list = [fig_list; fig_tmp];
  param_layer = num_layers;
  axis( [0.95*min(param_list) 1.05*max(param_list) 0.5 1.0]);
  hold on;
  for i_param0 = 1:num_params0
    twoAFC_AUC_tmp = zeros(num_params,1);
    twoAFC_errorbar_tmp = zeros(num_params,1);
    for i_param = 1 : num_params
      twoAFC_AUC_tmp(i_param) = ...
	  twoAFC_AUC_array{i_param0, i_param, i_expNum}(param_layer, i_2AFC_test);
      twoAFC_errorbar_tmp(i_param) = ...
	  twoAFC_errorbar_array{i_param0, i_param, i_expNum}(param_layer, i_2AFC_test);
    endfor %% i_param
    for i_param = 1 : num_params
      eh = errorbar(param_list, ...
		    twoAFC_AUC_tmp, ...
		    twoAFC_errorbar_tmp);
      set( eh, 'LineWidth', 2 );
      lh = plot(param_list, ...
		twoAFC_AUC_tmp, ...		  
		num2str(i_param0+(i_param-1)*num_params0));  
      set( lh, 'LineWidth', 2 );
      line_color = get( lh, 'Color');
      set( eh, 'Color', line_color);
    endfor %% i_param
  endfor %% i_param0
endif %% num_params > 1


%% plot AUC vs param0
if num_params0 > 1
  twoAFC_param_name = ['2AFC vs. Param0'];
  fig_tmp = figure('Name', twoAFC_param_name);
  fig_list = [fig_list; fig_tmp];
  param_layer = num_layers;
  axis( [0.95*min(param0_list) 1.05*max(param0_list) 0.5 1.0]);
  hold on;
  for i_param = 1:num_params
    twoAFC_AUC_tmp = zeros(num_params0,1);
    twoAFC_errorbar_tmp = zeros(num_params0,1);
    for i_param0 = 1 : num_params0
      twoAFC_AUC_tmp(i_param0) = ...
	  twoAFC_AUC_array{i_param0, i_param, i_expNum}(param_layer, i_2AFC_test);
      twoAFC_errorbar_tmp(i_param0) = ...
	  twoAFC_errorbar_array{i_param0, i_param, i_expNum}(param_layer, i_2AFC_test);
    endfor %% i_param0
    eh = errorbar(param0_list, ...
		  twoAFC_AUC_tmp, ...
		  twoAFC_errorbar_tmp);
    set( eh, 'LineWidth', 2 );
    lh = plot(param0_list, ...
	      twoAFC_AUC_tmp, ...		  
	      num2str(i_param0+(i_param-1)*num_params0));  
    set( lh, 'LineWidth', 2 );
    line_color = get( lh, 'Color');
    set( eh, 'Color', line_color);
  endfor %% i_param
endif %% num_params0 > 1



pvp_saveFigList( fig_list, twoAFC_path, 'png');
fig_list = [];

