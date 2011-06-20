
function [ROC_struct, exp_struct] = ...
      psycho_AFC(exp_struct, AFC_struct)


  %% analyze Speed of Sight 2AFC data for single subject
  global MAX_XFACTOR_VALS
  MAX_XFACTOR_VALS = 10;

  [num_trials, num_xFactors] = size(exp_struct.trial_struct.xfactors);
  if exp_struct.total ~= num_trials
    warning(['exp_struct.total ~= num_trials; ', ...
	     'exp_struct.total = ', ...
	     num2str(exp_struct.total),
	     '; ', ...
	     'num_trials = ', ...
	     num2str(num_trials)]);
  end
  ROC_struct = struct;
  ROC_struct.num_xfactor_vals = zeros(1,num_xfactors);
  ROC_struct.xfactor_vals = cell(1,num_xfactors);
  ROC_struct.xfactor_trials = cell(1,num_xfactors);

  for i_xfactor = 1 : num_xfactors
    xfactor_vals = unique(exp_struct.trial_struct.xfactor(:,i_xfactor));
    num_xfactor_vals = length(xfactor_vals);
    if num_xfactor_vals > MAX_XFACTOR_VALS
      [hist_xfactor, xfactor_vals] = ...
	  hist(squeeze(exp_struct.trials.xfactor(:,i_xfactor)), MAX_XFACTOR_VALS);
    end
    ROC_struct.xfactor_vals{i_xfactor} = xfactor_vals;
    num_xfactor_vals = length(xfactor_vals);
    ROC_struct.num_xfactor_vals(i_xfactor) = num_xfactor_vals;
    ROC_struct.xfactor_trials{i_xfactor} = cell(1,num_xfactor_vals);
    xfactor_min = min(squeeze(exp_struct.trial_struct.xfactor(:,i_xfactor)));
    xfactor_max = max(squeeze(exp_struct.trial_struct.xfactor(:,i_xfactor)));
    for i_val = 1 : num_xfactor_vals
      if i_val > 1
	xfactor_left = ...
	    (xfactor_vals(i_val - 1) + xfactor_vals(i_val)) / 2;
      else
	xfactor_left = xfactor_min*(0.99);
      end
      if i_val < num_xfactor_vals
	xfactor_right = ...
	    (xfactor_vals(i_val) + xfactor_vals(i_val + 1))/2;
      else
	xfactor_right = xfactor_max;
      end
      ROC_struct.xfactor_trials{i_xfactor}{i_val} = ...
	  find( (exp_struct.trial_struct.xfactor(:,i_xfactor) ...
		 > xfactor_left) ...
	       & ...
	       (exp_struct.trial_struct.xfactor(:,i_xfactor) ...
		 <= xfactor_right) );
    end %% i_val
  end %% i_xfactor

  ROC_struct.target_trials = cell(1,AFC_struct.AFC_mode);
  for i_AFC = 1 : AFC_struct.AFC_mode
    ROC_struct.target_trials{i_AFC} = ...
	find( exp_struct.trial_struct.target_index == i_AFC - 1);
  end


  %% get/check SOA values
  if AFC_struct.SOA_index > 0 &&  AFC_struct.SOA_index <= num_xfactors
    SOA_vals = 1000 * ...
	(exp_struct.StimulusOnsetTime(:, 2) - ...
	 exp_struct.StimulusOnsetTime(:, 1));
    num_SOAs = unique(SOA_vals);
    if num_SOAs ~= num_xfactor_vals(AFC_struct.SOA_index)
      warning(['num_xfactor_vals(AFC_struct.SOA_index) ~= num_SOAs; ', ...
	       'num_xfactor_vals(AFC_struct.SOA_index) = ', ...
	       num2str(num_xfactor_vals(AFC_struct.SOA_index)),
	       '; ', ...
	       'num_SOAs = ', ...
	       num2str(num_SOAs)]);    
    end
    for i_trial = 1 : num_trials
      SOA_check = ...
	  (SOA_vals(i_trial) - ...
	  exp_struct.trial_struct.xfactor(i_trial,AFC_struct.SOA_index)) ...
	  / ...
	  (SOA_vals(i_trial) + 0.001 ...
	  exp_struct.trial_struct.xfactor(i_trial,AFC_struct.SOA_index));
      if (SOA_check > 0.001)
	warning(['SOA_check > 0.001; ', ...
		 'SOA_vals(i_trial) = ', ...
		 num2str(SOA_vals(i_trial)),
		 '; ', ...
		 'xfactor(i_trial,AFC_struct.SOA_index) = ', ...
		 num2str(exp_struct.trial_struct.xfactor(i_trial,AFC_struct.SOA_index))]);    
      end
    endf %% i_trial
  end %% AFC_struct.SOA_index > 0



  %% build AFC data structure which organizes trials by target_flag
  %% and xfactor values
  num_trials_per_condition = ...
      num_trials / ( AFC_struct.AFC_mode * ...
		    sum(ROC_struct.num_xfactor_vals) );
  ROC_struct.AFC_data = ...
      zeros([AFC_struct.AFC_mode, ROC_struct.num_xfactor_vals]);
  ROC_struct.AFC_trials = ...
      cell([AFC_struct.AFC_mode, ROC_struct.num_xfactor_vals]);
  for i_AFC = 1 : AFC_struct.AFC_mode
    target_trials = ...
	ROC_struct.target_trials{i_AFC};
    for i_xfactor = 1 : num_xfactors
      num_xfactor_vals = ROC_struct.num_xfactor_vals(i_xfactor);
      for i_val = 1 : num_xfactor_vals
	ROC_struct.xfactor_vals{i_xfactor, i_val};
	xfactor_trials = ...
	    intersect(target_trials, ...
		      ROC_struct.xfactor_trials{i_xfactor, i_val});
      endfor  %% i_val
    end  %% i_xfactor
    ROC_struct.AFC_trials{
	     
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
