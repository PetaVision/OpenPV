
function [ROC_struct, exp_struct] = pvp_SOS_2AFC(exp_struct, SOS_path)

  %% analyze Speed of Sight 2AFC data for single subject
  
  
  %% use filenames to determine SOAs
  SOS_data_path = [SOS_path, "data/"];
  SOS_left_path = [SOS_data_path, "left/"];		    
  SOS_left_struct = dir([SOS_left_path, '*.png']);
  SOS_num_left = size(SOS_left_struct,1);
  exp_struct.SOA_ndx = zeros(SOS_num_left,1);
  for i_left = 1 : SOS_num_left
    SOS_left_filename = SOS_left_struct(i_left,1).name;
    exp_struct.SOA_ndx(i_left) = str2num(SOS_left_filename(10));
  endfor
  exp_struct.SOA_ndx = exp_struct.SOA_ndx(exp_struct.ndx);

  %% get trials corresponding to each SOA ndx
  max_SOA_ndx = max(exp_struct.SOA_ndx);
  min_SOA_ndx = min(exp_struct.SOA_ndx);
  num_SOAs = max_SOA_ndx - min_SOA_ndx + 1;
  exp_struct.SOA_trials = cell(num_SOAs,1);
  for i_SOA = 1 : num_SOAs
    exp_struct.SOA_trials{i_SOA} = ...
	find(exp_struct.SOA_ndx == (min_SOA_ndx + i_SOA - 1));
  endfor

  %% get/check SOA values
  exp_struct.SOA_vals = zeros(num_SOAs,1);
  exp_struct.SOA_check_flag = 1;
  for i_SOA = 1 : num_SOAs
    SOA_tmp = 1000 * ...
	(exp_struct.StimulusOnsetTime(exp_struct.SOA_trials{i_SOA}, 2) - ...
	 exp_struct.StimulusOnsetTime(exp_struct.SOA_trials{i_SOA}, 1) );
    exp_struct.SOA_vals(i_SOA) = round(mean(SOA_tmp));
    SOA_std = std(SOA_tmp);
    if exp_struct.SOA_vals(i_SOA) > 0
      SOA_check = SOA_std / exp_struct.SOA_vals(i_SOA);
    else
      SOA_check = inf;
    endif
    if (SOA_check > 0.001)
      warning(['i_SOA = ', num2str(i_SOA), ', ', ...
	       'SOA = ', num2str(exp_struct.SOA_vals(i_SOA)), ', ', ...
	       'SOA_std = ', num2str(SOA_std), ', ', ...
	       'SOA_check = ', num2str(SOA_check)]);
      exp_struct.SOA_check_flag = 0;
      return;
    endif
  endfor

  %% use filenames to determine target_id (i.e. cat, dog)
  exp_struct.target_ID = zeros(SOS_num_left,1);
  for i_left = 1 : SOS_num_left
    SOS_left_filename = SOS_left_struct(i_left,1).name;
    exp_struct.target_ID(i_left) = str2num(SOS_left_filename(12));
  endfor
  exp_struct.target_ID = exp_struct.target_ID(exp_struct.ndx);

  target_ID_vals = unique(exp_struct.target_ID);
  num_target_IDs = length(target_ID_vals);
  exp_struct.target_ID_trials = cell(num_target_IDs,1);
  for i_target_ID = 1 : num_target_IDs
    exp_struct.target_ID_trials{i_target_ID} = ...
	find(exp_struct.target_ID == target_ID_vals(i_target_ID));
  endfor
  exp_struct.target_ID_vals = target_ID_vals;

  %% get target_flag for each trial
  %% max_target_flag typically >= 2 (might extend this analysis to NAFC tasks,
  %% with N > 2).  max_target_flag == 1 would indicate no distractor images,
  %% which might be used simply to plot confidence histograms (not tested)
  max_target_flag = max(exp_struct.target_flag);
  min_target_flag = min(exp_struct.target_flag);
  num_target_flags = max_target_flag - min_target_flag + 1;
  exp_struct.target_flag_trials = cell(num_target_flags,1);
  for i_target_flag = 1 : num_target_flags
    exp_struct.target_flag_trials{i_target_flag} = ...
	find(exp_struct.target_flag == (min_target_flag + i_target_flag - 1));
  endfor


  %% build twoAFC data structure which organizes trials by target_flag,
  %% SOA and target_ID
  num_trials = SOS_num_left / ( num_SOAs * num_target_flags * num_target_IDs);
  twoAFC_data = zeros(max_target_flag, num_SOAs, num_trials, num_target_IDs);

  exp_struct.twoAFC_trials = cell(num_target_flags, num_SOAs, num_target_IDs);
  for i_target_flag = 1 : num_target_flags
    for i_SOA = 1 : num_SOAs
      trials_tmp = ...
	  intersect(exp_struct.SOA_trials{i_SOA}, exp_struct.target_flag_trials{i_target_flag});
      for i_target_ID = 1 : num_target_IDs
	trials_tmp2 = ...
	    intersect(exp_struct.target_ID_trials{i_target_ID}, trials_tmp);
	exp_struct.twoAFC_trials{i_target_flag, i_SOA, i_target_ID} = ...
	    trials_tmp2;
	twoAFC_data(i_target_flag, i_SOA, :, i_target_ID) = ...
	    exp_struct.confidence(trials_tmp2) .* ( 1 - 2 * exp_struct.choice(trials_tmp2) );
      endfor
    endfor
  endfor

  global num_hist_activity_bins
  num_hist_activity_bins = 5;

  exp_struct.twoAFC_data = twoAFC_data;
  [fig_list, ROC_struct] = pvp_ROC(exp_struct);

  ROC_path = [SOS_path, "analysis/"];
  ROC_filename = ...
      [ROC_path, 'ROC_', num2str(exp_struct.SOS_ID), '.mat']
  ROC_struct.ROC_filename = ROC_filename;

  save('-mat', ROC_filename, 'ROC_struct');

  pvp_saveFigList( fig_list, ROC_path, 'png');
  fig_list = [];
