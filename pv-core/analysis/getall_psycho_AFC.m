clear all;
close all;
setenv('GNUTERM', 'x11');

%% AFC_MODE indicates which AFC protocol used
%% if equals 2 then standard 2AFC protocol
%% subject must choose between 2 conditions
%% (i.e. yes/no, left/right, first/second)
%% if equal to 1, then single condition (all positive or negative)
global AFC_MODE
AFC_MODE = 2;

%% index of xFactor that corresponds to SOA, <= 0
%% means SOA not used as experimental factor
global SOA_INDEX
SOA_INDEX = -1;

remake_anal_struct = 1;  %% set to 1 to ignore existing anal_struct file

global ROOT_PATH
ROOT_PATH = "/Users/MATLAB/gitPatches/";
if ~exist('ROOT_PATH', 'DIR')
  error(['ROOT_PATH does not exist:', ROOT_PATH]);
end

global ANALYSIS_PATH
ANALYSIS_PATH = [ROOT_PATH, "analysis/"];
mkdir(ANALYSIS_PATH); %% does not clobber existing dir

AFC_anal_struct_filename = ...
    [ANALYSIS_PATH, 'AFC_anal_struct', '.mat']
read_exp_list = ~exist(AFC_anal_struct_filename, "file") || remake_anal_struct

if read_exp_list
  AFC_anal_struct = struct;
  AFC_anal_struct.AFC_mode = AFC_MODE;
  AFC_anal_struct.SOA_index = SOA_INDEX;
else
  load(AFC_anal_struct_filename);
  AFC_MODE = AFC_anal_struct.AFC_mode;
  SOA_INDEX = AFC_anal_struct.SOA_index;
end

%% (re)set paths
AFC_anal_struct.root_path = ROOT_PATH;
AFC_anal_struct.analysis_path = ANALYSIS_PATH;
AFC_anal_struct.results_path = [AFC_anal_struct.root_path, "results/"];
AFC_anal_struct.invalid_path = [AFC_anal_struct.results_path, "invalid/"];
if ~exist('AFC_anal_struct.results_path', 'DIR')
  mkdir(AFC_anal_struct.results_path);  %% experimental data must be in anal_struct
  if ~isfield(AFC_anal_struct, 'exp_list')
    error(['exp_list does not exist in anal_struct read from:', ...
	   AFC_anal_struct.anal_path]);
  end
end
mkdir(AFC_anal_struct.invalid_path); %% does not clobber existing dir

AFC_anal_struct.filename = ...
    AFC_anal_struct_filename;

%%flip_interval = 5; %% msec, stored in exp_struct, assume 200 Hz

%% get list of subject IDs, build data structs to hold exp data and ROC analysis
if read_exp_list
  AFC_dir_struct = dir([anal_struct.results_path, '*.mat']);
  AFC_num_IDs = size(twoAFC_dir_struct,1);
  AFC_anal_struct.num_IDs = AFC_num_IDs;  %% init to this value, fix later
  AFC_IDs = zeros(AFC_num_IDs, 1);
  AFC_exp_list = cell(AFC_num_IDs, 1);
  AFC_ROC_list = cell(AFC_num_IDs, 1);
  AFC_anal_struct.random_IDs = [];
  AFC_anal_struct.num_IDs = 0;
  AFC_anal_struct.ROC_list = [];
  for AFC_loop_ndx = 1 : AFC_num_IDs
    AFC_ID_filename = AFC_ID_struct(AFC_loop_ndx,1).name;
    AFC_ID_ndx = strfind( AFC_ID_filename, ".mat" );
    AFC_ID_str = AFC_ID_filename(1:AFC_ID_ndx);
    AFC_ID = str2num(twoAFC_ID_str);
    AFC_filename = [AFC_anal_struct.results_path, AFC_ID_filename];
    load(AFC_filename);
    AFC_exp_struct = exp_struct;
    AFC_exp_struct.AFC_ID = AFC_ID;
    if AFC_ID ~= AFC_exp_struct.seed  %% should be error instead?
      warning(['AFC_ID ~= AFC_exp_struct.seed; ', ...
	       'AFC_ID = ', ...
	       AFC_ID_str, ...
	       '; ', ...
	       'AFC_exp_struct.seed = ', ...
	       num2str(AFC_exp_struct.seed)]);
    end
    %% check exp_struct to see if official expiment
    if ~AFC_exp_struct.offical_flag
      continue;
    end
    [AFC_ROC_struct, AFC_exp_struct] = ...
	psycho_AFC(AFC_exp_struct, AFC_anal_struct);
    if AFC_ROC_struct.valid_flag == 1
      AFC_anal_struct.num_IDs = ...
	  AFC_anal_struct.num_IDs + 1;
      AFC_anal_struct.random_IDs = ...
	  [AFC_anal_struct.random_IDs, AFC_ID];
      AFC_anal_struct.ROC_list = ...
	  [AFC_anal_struct.ROC_list; AFC_ROC_struct]; 
      AFC_anal_struct.exp_list = ...
	  [AFC_anal_struct.exp_list; AFC_exp_struct]; 
    else
      [STATUS, MSG, MSGID] = movefile(SOS_filename, SOS_invalid_path);
    endif
  endfor

  num_valid_IDs = valid_ID-1;
  SOS_struct.num_valid_IDs = num_valid_IDs;
  SOS_fieldnames = fieldnames(SOS_struct);

  save('-mat', SOS_struct_filename, 'SOS_struct');
endif % read_exp_list

%% group trials with same SOA and target_ID
if SOS_struct.num_valid_IDs > 1

  fig_list = [];

  group_exp_struct = struct;
  group_exp_struct.SOS_ID = [];
  group_exp_struct.SOA_vals = SOS_struct.exp_list{1}.SOA_vals;
  group_exp_struct.target_ID_vals = SOS_struct.exp_list{1}.target_ID_vals;
  for valid_ID = 1 : SOS_struct.num_valid_IDs
    group_exp_struct.SOA_vals = ...
	unique([group_exp_struct.SOA_vals; ...
		SOS_struct.exp_list{valid_ID}.SOA_vals]);
    group_exp_struct.target_ID_vals = ...
	unique([group_exp_struct.target_ID_vals; ...
		SOS_struct.exp_list{valid_ID}.target_ID_vals]);
  endfor
  group_ROC_struct = struct;
  group_ROC_struct.max_target_flag = SOS_struct.ROC_list{1}.max_target_flag;
  group_ROC_struct.num_layers = length(group_exp_struct.SOA_vals);
  group_ROC_struct.num_target_IDs = length(group_exp_struct.target_ID_vals);
  group_ROC_struct.layer_ndx = 1:group_ROC_struct.num_layers;
  group_ROC_struct.target_ID_ndx = 1:group_ROC_struct.num_target_IDs;
  group_ROC_struct.num_trials = ...
      zeros(group_ROC_struct.num_layers, group_ROC_struct.num_target_IDs);
  group_ROC_struct.twoAFC_ROC = ...
      cell(group_ROC_struct.num_layers, group_ROC_struct.num_target_IDs);
  group_ROC_struct.twoAFC_ROC2 = ...
      cell(group_ROC_struct.num_layers, group_ROC_struct.num_target_IDs);
  group_ROC_struct.twoAFC_correct = ...
      zeros(group_ROC_struct.num_layers, group_ROC_struct.num_target_IDs);
  group_ROC_struct.twoAFC_correct2 = ...
      zeros(group_ROC_struct.num_layers, group_ROC_struct.num_target_IDs);
  group_ROC_struct.twoAFC_AUC = ...
      zeros(group_ROC_struct.num_layers, group_ROC_struct.num_target_IDs);
  group_ROC_struct.twoAFC_AUC2 = ...
      zeros(group_ROC_struct.num_layers, group_ROC_struct.num_target_IDs);
  group_ROC_struct.twoAFC_errorbars = ...
      zeros(group_ROC_struct.num_layers, group_ROC_struct.num_target_IDs);
  for valid_ID = 1 : SOS_struct.num_valid_IDs
    ROC_struct = SOS_struct.ROC_list{valid_ID};
    exp_struct = SOS_struct.exp_list{valid_ID};
    layer_ndx = 1:ROC_struct.num_layers;
    for layer = layer_ndx
      group_layer = ...
	  find(group_exp_struct.SOA_vals == ...
	       exp_struct.SOA_vals(layer));
      if isempty(group_layer)
	continue;
      endif
      target_ID_ndx = 1:ROC_struct.num_target_IDs;
      for target_ID = target_ID_ndx
	group_target_ID = ...
	    find(group_exp_struct.target_ID_vals == ...
		 exp_struct.target_ID_vals(target_ID));
	if isempty(group_target_ID)
	  continue;
	endif
	group_ROC_struct.num_trials(group_layer, group_target_ID) = 1 + ...
	    group_ROC_struct.num_trials(group_layer, group_target_ID);
	if isempty(group_ROC_struct.twoAFC_ROC{group_layer, group_target_ID})
	  group_ROC_struct.twoAFC_ROC{group_layer, group_target_ID} = ...
	      ROC_struct.twoAFC_ROC{layer, target_ID};
	  group_ROC_struct.twoAFC_ROC2{group_layer, group_target_ID} = ...
	      ROC_struct.twoAFC_ROC{layer, target_ID}.^2;
	else
	  group_ROC_struct.twoAFC_ROC{group_layer, group_target_ID} = ...
	      group_ROC_struct.twoAFC_ROC{group_layer, group_target_ID} + ...
	      ROC_struct.twoAFC_ROC{layer, target_ID};
	  group_ROC_struct.twoAFC_ROC2{group_layer, group_target_ID} = ...
	      group_ROC_struct.twoAFC_ROC2{group_layer, group_target_ID} + ...
	      ROC_struct.twoAFC_ROC{layer, target_ID}.^2;
	endif
	group_ROC_struct.twoAFC_correct(group_layer, group_target_ID) = ...
	    group_ROC_struct.twoAFC_correct(group_layer, group_target_ID) + ...
	    ROC_struct.twoAFC_correct(layer, target_ID);
	group_ROC_struct.twoAFC_correct2(group_layer, group_target_ID) = ...
	    group_ROC_struct.twoAFC_correct2(group_layer, group_target_ID) + ...
	    ROC_struct.twoAFC_correct(layer, target_ID).^2;
	group_ROC_struct.twoAFC_AUC(group_layer, group_target_ID) = ...
	    group_ROC_struct.twoAFC_AUC(group_layer, group_target_ID) + ...
	    ROC_struct.twoAFC_AUC(layer, target_ID);
	group_ROC_struct.twoAFC_AUC2(group_layer, group_target_ID) = ...
	    group_ROC_struct.twoAFC_AUC2(group_layer, group_target_ID) + ...
	    ROC_struct.twoAFC_AUC(layer, target_ID).^2;
      endfor % target_ID
    endfor % layer
  endfor % valid_ID

  for group_layer = group_ROC_struct.layer_ndx
    for group_target_ID = group_ROC_struct.target_ID_ndx
      group_ROC_struct.twoAFC_ROC{group_layer, group_target_ID} = ...
	  group_ROC_struct.twoAFC_ROC{group_layer, group_target_ID} / ...
	  group_ROC_struct.num_trials(group_layer, group_target_ID);
      group_ROC_struct.twoAFC_ROC2{group_layer, group_target_ID} = ...
	  group_ROC_struct.twoAFC_ROC2{group_layer, group_target_ID} / ...
	  group_ROC_struct.num_trials(group_layer, group_target_ID);
      group_ROC_struct.twoAFC_correct(group_layer, group_target_ID) = ...
	  group_ROC_struct.twoAFC_correct(group_layer, group_target_ID) / ...
	  group_ROC_struct.num_trials(group_layer, group_target_ID);
      group_ROC_struct.twoAFC_correct2(group_layer, group_target_ID) = ...
	  group_ROC_struct.twoAFC_correct2(group_layer, group_target_ID) / ...
	  group_ROC_struct.num_trials(group_layer, group_target_ID);
      group_ROC_struct.twoAFC_AUC(group_layer, group_target_ID) = ...
	  group_ROC_struct.twoAFC_AUC(group_layer, group_target_ID) / ...
	  group_ROC_struct.num_trials(group_layer, group_target_ID);
      group_ROC_struct.twoAFC_AUC2(group_layer, group_target_ID) = ...
	  group_ROC_struct.twoAFC_AUC2(group_layer, group_target_ID) / ...
	  group_ROC_struct.num_trials(group_layer, group_target_ID);
      group_ROC_struct.twoAFC_errorbars(group_layer, group_target_ID) = ...
	  group_ROC_struct.twoAFC_AUC2(group_layer, group_target_ID) - ...
	  group_ROC_struct.twoAFC_AUC(group_layer, group_target_ID).^2;
      group_ROC_struct.twoAFC_errorbars(group_layer, group_target_ID) = ...
	  sqrt(group_ROC_struct.twoAFC_errorbars(group_layer, group_target_ID));
    endfor
  endfor
  

  [fig_tmp] = pvp_plotROC(group_ROC_struct, group_exp_struct);
  fig_list = [fig_list; fig_tmp];

  [fig_tmp] = pvp_plotAUC(group_ROC_struct, group_exp_struct);
  fig_list = [fig_list; fig_tmp];
  
 % pvp_saveFigList( fig_list, SOS_analysis_path, 'png');
  pvp_saveFigList( fig_list, SOS_analysis_path, 'jpg');

  SOS_struct.group_ROC_struct = group_ROC_struct;
  SOS_struct.group_exp_struct = group_exp_struct;
  save('-mat', SOS_struct_filename, 'SOS_struct');

endif