clear all
setenv("GNUTERM", "x11");
amoeba_flag = 1;
bowtie_flag = ~amoeba_flag;
if amoeba_flag
  twoAFC_path = '/Users/gkenyon/workspace/kernel/input/256/amoeba/';
elseif bowtie_flag
  twoAFC_path = '/Users/gkenyon/workspace2/geisler/input/256/bowtie/';
endif
FC_list = [4]; %%[2 4 6 8];
len_FC = length(FC_list);
if amoeba_flag
  strength_list = [287, 300, 325, 400];
elseif bowtie_flag
  strength_list = [50, 75, 125, 175];
endif
num_strengths = length(strength_list);
twoAFC_dir_list = cell(num_strengths,1);
for i_strength = 1 : num_strengths
  if amoeba_flag
    twoAFC_dir_list{i_strength} = ...
	[twoAFC_path, ...
	 'test_target40K_W', ...
	 num2str(strength_list(i_strength)), ...
	 '_target'];
  elseif bowtie_flag
    twoAFC_dir_list{i_strength} = ...
	[twoAFC_path, ...
	 'target_', ...
	 num2str(strength_list(i_strength))];
  endif
endfor
max_expNum = 1;  % maximum number of independent experiments to be combined
twoAFC_ROC_array = cell(len_FC, num_strengths, max_expNum);
twoAFC_AUC_array = cell(len_FC, num_strengths, max_expNum);
twoAFC_errorbar_array = cell(len_FC, num_strengths, max_expNum);
expNum_list = [1];  % list for combining results from several experiments
len_expNum = length(expNum_list);
local_path = pwd;
TRAINING_FLAG = -1;
for i_strength = 1 : num_strengths
  disp(['strength = ', num2str(strength_list(i_strength))]);
  twoAFC_dir = ...
      twoAFC_dir_list{i_strength};
  if amoeba_flag
    twoAFC_dir = [twoAFC_dir, '_G1/'];
  elseif bowtie_flag
    twoAFC_dir = [twoAFC_dir, '/'];    
  endif
  
  for i_fc = 1 : len_FC
    num_FC = FC_list(i_fc);
    disp(['num_FC = ', num2str(num_FC)]);
    if amoeba_flag
      twoAFC_dir = [twoAFC_dir, ...
		    num2str(num_FC), 'fc/'];
    elseif bowtie_flag
      twoAFC_dir = [twoAFC_dir, ...
		    num2str(num_FC), 'fc/'];
    endif
    for i_expNum = 1 : len_expNum
      expNum = expNum_list(i_expNum);
      twoAFC_filename = ...
	  [twoAFC_dir, ...
	   'twoAFC', num2str(expNum), '.mat'];
      load(twoAFC_filename)
      if len_expNum == 1
	twoAFC_ROC_array{i_fc, i_strength} = twoAFC_ROC;
	twoAFC_AUC_array{i_fc, i_strength} = twoAFC_AUC;
	twoAFC_errorbar_array{i_fc, i_strength} = twoAFC_errorbar;
      else
	twoAFC_ROC_array{i_fc, i_strength, i_expNum} = twoAFC_ROC;
	twoAFC_AUC_array{i_fc, i_strength, i_expNum} = twoAFC_AUC;
	twoAFC_errorbar_array{i_fc, i_strength, i_expNum} = twoAFC_errorbar;
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
		 num2str(twoAFC_AUC_array{i_fc, i_strength, i_expNum}(layer, i_2AFC_test)), ...
		 ' +/- ', ...
		 num2str(twoAFC_errorbar_array{i_fc, i_strength, i_expNum}(layer, i_2AFC_test)) ] );
	endfor
      endfor % i_2AFC_test

      
    endfor % i_expNum
  endfor % i_fc
endfor % i_strength


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
  for i_strength = 1 : num_strengths
    for i_fc = 1:len_FC
      twoAFC_ROC_tmp = ...
	  twoAFC_ROC_array{i_fc, i_strength, i_expNum};
      lh = plot(twoAFC_ROC_tmp{layer, i_2AFC_test}(1,:), ...
		twoAFC_ROC_tmp{layer, i_2AFC_test}(2,:), ...
		num2str(i_fc+(i_strength-1)*len_FC));  
      set( lh, 'LineWidth', 2 );
    endfor % i_fc
  endfor % i_strengths
endfor

twoAFC_AUC_name = ['2AFC AUC Model'];
fig_tmp = figure('Name', twoAFC_AUC_name);
fig_list = [fig_list; fig_tmp];
AUC_layers = 3:num_layers;
axis( [0.95*min(AUC_layers) 1.05*max(AUC_layers) 0.45 1.0]);
hold on;
for i_strength = 1 : num_strengths
  for i_fc = 1:len_FC
    twoAFC_AUC_tmp = twoAFC_AUC_array{i_fc, i_strength, i_expNum};
    twoAFC_errorbar_tmp = twoAFC_errorbar_array{i_fc, i_strength, i_expNum};
    eh = errorbar(AUC_layers, ...
		  twoAFC_AUC_tmp(AUC_layers, i_2AFC_test), ...
		  twoAFC_errorbar_tmp(AUC_layers, i_2AFC_test));
    set( eh, 'LineWidth', 2 );
    lh = plot(AUC_layers, ...
	      twoAFC_AUC_tmp(AUC_layers, i_2AFC_test), ...		  
	      num2str(i_fc+(i_strength-1)*len_FC));  
    set( lh, 'LineWidth', 2 );
    line_color = get( lh, 'Color');
    set( eh, 'Color', line_color);
  endfor %% i_fc
endfor %% i_strength

if num_strengths > 1
  twoAFC_strength_name = ['2AFC Kernel Strength'];
  fig_tmp = figure('Name', twoAFC_strength_name);
  fig_list = [fig_list; fig_tmp];
  strength_layer = num_layers;
  axis( [0.95*min(strength_list) 1.05*max(strength_list) 0.5 1.0]);
  hold on;
  for i_fc = 1:len_FC
    twoAFC_AUC_tmp = zeros(num_strengths,1);
    twoAFC_errorbar_tmp = zeros(num_strengths,1);
    for i_strength = 1 : num_strengths
      twoAFC_AUC_tmp(i_strength) = ...
	  twoAFC_AUC_array{i_fc, i_strength, i_expNum}(strength_layer, i_2AFC_test);
      twoAFC_errorbar_tmp(i_strength) = ...
	  twoAFC_errorbar_array{i_fc, i_strength, i_expNum}(strength_layer, i_2AFC_test);
    endfor
    for i_strength = 1 : num_strengths
      eh = errorbar(strength_list, ...
		    twoAFC_AUC_tmp, ...
		    twoAFC_errorbar_tmp);
      set( eh, 'LineWidth', 2 );
      lh = plot(strength_list, ...
		twoAFC_AUC_tmp, ...		  
		num2str(i_fc+(i_strength-1)*len_FC));  
      set( lh, 'LineWidth', 2 );
      line_color = get( lh, 'Color');
      set( eh, 'Color', line_color);
    endfor %% i_strength
  endfor %% i_fc
endif %% num_strengths > 1



pvp_saveFigList( fig_list, twoAFC_path, 'png');
fig_list = [];

