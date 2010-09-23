clear all
setenv("GNUTERM", "x11");
pvp_matlabPath;
twoAFC_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/256/';
FC_list = [4]; %[2 4 6 8];
len_FC = length(FC_list);
max_expNum = 10;  % maximum number of independent experiments to be combined
twoAFC_array = cell(len_FC, max_expNum);
expNum_list = [1];  % list for combining results from several experiments
len_expNum = length(expNum_list);
local_path = pwd;
TRAINING_FLAG = -1
for i_fc = 1 : len_FC
  num_FC = FC_list(i_fc);
  disp(['num_FC = ', num2str(num_FC)]);
				%twoAFC_dir = [twoAFC_path, 'amoeba10K_', num2str(num_FC), 'fc'];
  twoAFC_dir = [twoAFC_path, 'test_target40K_W287_target']; %, num2str(num_FC)];
  if abs(TRAINING_FLAG) == 1
    twoAFC_dir = [twoAFC_dir, '_G1'];
  elseif abs(TRAINING_FLAG) == 2
    twoAFC_dir = [twoAFC_dir, '_G2'];
  elseif abs(TRAINING_FLAG) == 3
    twoAFC_dir = [twoAFC_dir, '_G3'];
  elseif abs(TRAINING_FLAG) == 4
    twoAFC_dir = [twoAFC_dir, '_G4'];
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
    num_2AFC_tests = size(twoAFC, 4);
    if ~exist('twoAFC_correct')
      twoAFC_correct = zeros(num_layers, num_2AFC_tests);      
      for i_2AFC_test = 1 : num_2AFC_tests
	for layer = 1 : num_layers
	  twoAFC_correct(layer, i_2AFC_test) = ...
	      sum( squeeze( twoAFC(1,layer,:,i_2AFC_test) > ...
			   twoAFC(2,layer,:,i_2AFC_test) ) ) / ...
	      ( tot_trials + (tot_trials == 0) );
        endfor
      endfor % i_2AFC_test
    endif
    
    for i_2AFC_test = 1 : num_2AFC_tests
      disp(['i_2AFC_test = ', num2str(i_2AFC_test)]);
      for layer = 1 : num_layers
	disp( ['twoAFC_correct(', num2str(layer), ',', num2str(i_2AFC_test), ') = ', ...
	       num2str(twoAFC_correct(layer, i_2AFC_test)) ] );
      endfor
    endfor % i_2AFC_test
    
  endfor % i_expNum
endfor % i_fc
chdir(local_path)

%twoAFC = zeros(2, num_layers, num_trials, num_2AFC_tests);
