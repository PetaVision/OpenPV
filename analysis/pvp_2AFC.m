clear all
setenv("GNUTERM", "x11");
pvp_matlabPath;
twoAFC_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/';
FC_list = [4]; % [2 4 6 8];
len_FC = length(FC_list);
max_expNum = 10;  % maximum number of independent experiments to be combined
twoAFC_array = cell(len_FC, max_expNum);
expNum_list = [1];  % list for combining results from several experiments
len_expNum = length(expNum_list);
local_path = pwd;
TRAINING_FLAG = -3
for i_fc = 1 : len_FC
  num_FC = FC_list(i_fc);
  twoAFC_dir = [twoAFC_path, 'amoeba10K_', num2str(num_FC), 'fc'];
  if abs(TRAINING_FLAG) == 2
    twoAFC_dir = [twoAFC_dir, '_G2'];
  elseif abs(TRAINING_FLAG) == 3
    twoAFC_dir = [twoAFC_dir, '_G3'];
  endif
  chdir(twoAFC_dir)
  for i_expNum = 1 : len_expNum
    expNum = expNum_list(i_expNum);
    twoAFC_filename = ['twoAFC', num2str(expNum), '.mat.z'];
    load(twoAFC_filename)
    twoAFC_array{i_fc, i_expNum} = twoAFC;

    num_layers = size(twoAFC, 2);
    tot_trials = size(twoAFC, 3);
    disp(['num_FC = ', num2str(num_FC)]);
    for layer = 1 : num_layers
      disp(['layer = ', num2str(layer)]);
      twoAFC_correct(layer) = ...
	  sum( squeeze( twoAFC(1,layer,:) > twoAFC(2,layer,:) ) ) / ...
	  ( tot_trials + (tot_trials == 0) );
      disp( ['twoAFC_correct(', num2str(layer), ') = ', ...
	     num2str(twoAFC_correct(layer)) ] );
    endfor
    
  endfor % i_expNum
endfor % i_fc
chdir(local_path)

