clear all
setenv("GNUTERM", "x11");
%pvp_matlabPath;
%%twoAFC_path = '/Users/gkenyon/workspace2/geisler/input/256/';
twoAFC_path = '/Users/gkenyon/workspace/kernel/input/256/';
FC_list = [4]; %%[2 4 6 8];
len_FC = length(FC_list);
max_expNum = 1;  % maximum number of independent experiments to be combined
twoAFC_array = cell(len_FC, max_expNum);
twoAFC_ROC = cell(len_FC, max_expNum);
twoAFC_AUC = cell(len_FC, max_expNum);
twoAFC_errorbar = cell(len_FC, max_expNum);
expNum_list = [1];  % list for combining results from several experiments
len_expNum = length(expNum_list);
local_path = pwd;
TRAINING_FLAG = -1;
for i_fc = 1 : len_FC
  num_FC = FC_list(i_fc);
  disp(['num_FC = ', num2str(num_FC)]);
  %twoAFC_dir = [twoAFC_path, 'target_125'];
  twoAFC_dir = [twoAFC_path, 'test_target40K_W325_target']; 
  if abs(TRAINING_FLAG) == 1
    twoAFC_dir = [twoAFC_dir, '_G1'];
  elseif abs(TRAINING_FLAG) == 2
    twoAFC_dir = [twoAFC_dir, '_G2'];
  elseif abs(TRAINING_FLAG) == 3
    twoAFC_dir = [twoAFC_dir, '_G3'];
  elseif abs(TRAINING_FLAG) == 4
    twoAFC_dir = [twoAFC_dir, '_G4'];
  else
    twoAFC_dir = [twoAFC_dir, ''];    
  endif
  twoAFC_dir = [twoAFC_dir, ...
		'/', num2str(num_FC), 'fc'];
  chdir(twoAFC_dir)
  for i_expNum = 1 : len_expNum
    expNum = expNum_list(i_expNum);
    twoAFC_filename = ['twoAFC', num2str(expNum), '.mat'];
    load(twoAFC_filename)
    twoAFC_array{i_fc, i_expNum} = twoAFC;
    num_layers = size(twoAFC, 2);
    tot_trials = size(twoAFC, 3);
    num_2AFC_tests = 1; %size(twoAFC, 4);
    
    if ~exist('twoAFC_correct')
      twoAFC_correct = zeros(num_layers, num_2AFC_tests);      
      for i_2AFC_test = 1 : 1%num_2AFC_tests
	for layer = 1 : num_layers
	  twoAFC_correct(layer, i_2AFC_test) = ...
	      sum( squeeze( twoAFC(1,layer,:,i_2AFC_test) > ...
			   twoAFC(2,layer,:,i_2AFC_test) ) ) / ...
	      ( tot_trials + (tot_trials == 0) );
        endfor
      endfor % i_2AFC_test
    endif
    
    for i_2AFC_test = 1 : 1%num_2AFC_tests
      disp(['i_2AFC_test = ', num2str(i_2AFC_test)]);
      for layer = 1 : num_layers
	disp( ['twoAFC_correct(', num2str(layer), ...
			       ',', num2str(i_2AFC_test), ') = ', ...
	       num2str(twoAFC_correct(layer, i_2AFC_test)) ] );
      endfor
    endfor % i_2AFC_test

    
    twoAFC_ROC{i_fc, i_expNum} = cell(num_layers, num_2AFC_tests);
    twoAFC_AUC{i_fc, i_expNum} = zeros(num_layers, num_2AFC_tests);
    twoAFC_errorbar{i_fc, i_expNum} = zeros(num_layers, num_2AFC_tests);
    for i_2AFC_test = 1 : 1%num_2AFC_tests
      for layer = 1 : num_layers
	twoAFC_ROC{i_fc, i_expNum}{layer, i_2AFC_test} = ...
	    [[0, fliplr( twoAFC_cumsum{2, layer, i_2AFC_test} ), 1]', ...
	     [0, fliplr( twoAFC_cumsum{1, layer, i_2AFC_test} ), 1 ]'];
	twoAFC_AUC{i_fc, i_expNum}(layer, i_2AFC_test) = ...
	    trapz(twoAFC_ROC{i_fc, i_expNum}{layer, i_2AFC_test}(:,1), ...
		  twoAFC_ROC{i_fc, i_expNum}{layer, i_2AFC_test}(:,2));
	if ( twoAFC_correct(layer, i_2AFC_test) * tot_trials ) ~= 0
	  twoAFC_errorbar{i_fc, i_expNum}(layer, i_2AFC_test) = ...
	      sqrt( 1 - twoAFC_correct(layer, i_2AFC_test) ) / ...
	      sqrt( twoAFC_correct(layer, i_2AFC_test) * tot_trials );
	else
	  twoAFC_errorbar{i_fc, i_expNum}(layer, i_2AFC_test) = ...
	      0;
	endif
      endfor
    endfor % i_2AFC_test

    for i_2AFC_test = 1 : 1%num_2AFC_tests
      disp(['i_2AFC_test = ', num2str(i_2AFC_test)]);
      for layer = 1 : num_layers
	disp( ['twoAFC_AUC(', num2str(layer), ...
			   ',', num2str(i_2AFC_test), ') = ', ...
	       num2str(twoAFC_AUC{i_fc, i_expNum}(layer, i_2AFC_test)), ...
	       ' +/- ', ...
	       num2str(twoAFC_errorbar{i_fc, i_expNum}(layer, i_2AFC_test)) ] );
      endfor
    endfor % i_2AFC_test

    
  endfor % i_expNum
endfor % i_fc

save 'twoAFC_ROC.mat' twoAFC_ROC twoAFC_AUC twoAFC_errorbar
chdir(local_path)


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
  for i_fc = 1:len_FC
    lh = plot(twoAFC_ROC{i_fc, i_expNum}{layer, i_2AFC_test}(:,1), ...
	      twoAFC_ROC{i_fc, i_expNum}{layer, i_2AFC_test}(:,2), ...
	      num2str(i_fc));  
    set( lh, 'LineWidth', 2 );
  endfor % i_fc
endfor

for layer = 3:num_layers
  for i_fc = 1:len_FC
    twoAFC_ROC_filename = [twoAFC_path, 'ROC_', 'K', num2str(2*i_fc), ...
			   '_L', num2str(layer-3), '.txt'];
    twoAFC_ROC_tmp = twoAFC_ROC{i_fc,1}{layer,1};
    save('-ascii', twoAFC_ROC_filename, 'twoAFC_ROC_tmp');
  endfor
endfor

twoAFC_AUC_name = ['2AFC AUC Model'];
fig_tmp = figure('Name', twoAFC_AUC_name);
fig_list = [fig_list; fig_tmp];
AUC_layers = 3:num_layers;
axis( [min(AUC_layers) max(AUC_layers) 0.5 1.0]);
hold on;
for i_fc = 1:len_FC
  eh = errorbar(AUC_layers, ...
		twoAFC_AUC{i_fc, i_expNum}(AUC_layers, i_2AFC_test), ...
		twoAFC_errorbar{i_fc, i_expNum}(AUC_layers, i_2AFC_test));
  set( eh, 'LineWidth', 2 );
  lh = plot(AUC_layers, ...
	    twoAFC_AUC{i_fc, i_expNum}(AUC_layers, i_2AFC_test), ...		  
	    num2str(i_fc));  
  set( lh, 'LineWidth', 2 );
  line_color = get( lh, 'Color');
  set( eh, 'Color', line_color);
endfor % target_ID

for i_fc = 1:len_FC
  twoAFC_AUC_filename = [twoAFC_path, 'AUC_', 'K', num2str(2*i_fc), '.txt'];
  twoAFC_AUC_tmp = ...
      [(AUC_layers-3)', ...
       twoAFC_AUC{i_fc,1}(AUC_layers,1), ...
       twoAFC_errorbar{i_fc,1}(AUC_layers,1)];
  save('-ascii', twoAFC_AUC_filename, 'twoAFC_AUC_tmp');
endfor




pvp_saveFigList( fig_list, twoAFC_path, 'png');
fig_list = [];

