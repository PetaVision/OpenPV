clear all;
close all;
setenv('GNUTERM', 'x11');

SOS_path = "~/MATLAB/vadasg/";
SOS_results_path = [SOS_path, "results/official/"];
SOS_invalid_path = [SOS_results_path, "invalid/"];

global SOS_analysis_path
SOS_analysis_path = [SOS_path, "analysis/"];
SOS_struct_filename = ...
    [SOS_analysis_path, 'SOS_struct', '.mat']
SOS_struct.SOS_struct_filename = SOS_struct_filename;

flip_interval = 5; %% msec, should be stored in exp_struct, assume 200 Hz

read_exp_list = ~exist(SOS_struct_filename, "file") || 1
if read_exp_list
  %% get list of subject IDs, build struct to hold exp data and ROC analysis
  SOS_ID_struct = dir([SOS_results_path, '*.mat']);
  num_SOS_IDs = size(SOS_ID_struct,1);
  SOS_struct = struct;
  SOS_struct.num_SOS_IDs = num_SOS_IDs;
  SOS_struct.SOS_IDs = zeros(SOS_struct.num_SOS_IDs, 1);
  SOS_struct.exp_list = cell(SOS_struct.num_SOS_IDs, 1);
  SOS_struct.ROC_list = cell(SOS_struct.num_SOS_IDs, 1);
  valid_ID = 1;
  for ID = 1 : SOS_struct.num_SOS_IDs
    SOS_ID_filename = SOS_ID_struct(ID,1).name;
    ID_ndx = strfind( SOS_ID_filename, ".mat" );
    SOS_ID = str2num(SOS_ID_filename(1:ID_ndx));
    SOS_ID_str = num2str(SOS_struct.SOS_IDs(ID));
    SOS_filename = [SOS_results_path, SOS_ID_filename];
    load(SOS_filename);
    exp_struct_tmp = exp;
    clear exp
    exp_struct_tmp.SOS_ID = SOS_ID;
    %% check exp_struct to see if valid experiment?
    [ROC_struct_tmp, exp_struct_tmp] = ...
	pvp_SOS_2AFC(exp_struct_tmp, SOS_path);
    SOA_check_flag = exp_struct_tmp.SOA_check_flag;
    if SOA_check_flag == 1
      SOS_struct.SOS_IDs(valid_ID) = SOS_ID;
      SOS_struct.ROC_list{valid_ID} = ROC_struct_tmp;
      SOS_struct.exp_list{valid_ID} = exp_struct_tmp;
      valid_ID = valid_ID + 1;
    else
      [STATUS, MSG, MSGID] = movefile(SOS_filename, SOS_invalid_path);
    endif
  endfor

  num_valid_IDs = valid_ID-1;
  SOS_struct.num_valid_IDs = num_valid_IDs;
  SOS_fieldnames = fieldnames(SOS_struct);

  save('-mat', SOS_struct_filename, 'SOS_struct');
else
  load(SOS_struct_filename);
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