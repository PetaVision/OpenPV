function [fig_tmp] = pvp_plotAUC(ROC_struct, exp_struct)

  global SOS_analysis_path
  twoAFC_AUC_name = ['2AFC AUC'];
  if ~isempty(exp_struct.SOS_ID)
    twoAFC_AUC_name = [twoAFC_AUC_name, '(', num2str(exp_struct.SOS_ID), ')'];
  endif
  fig_tmp = figure('Name', twoAFC_AUC_name);
  %%axis "nolabel"
  axis([min(exp_struct.SOA_vals) max(exp_struct.SOA_vals) 0.5 1.0])
  hold on;
  for target_ID = ROC_struct.target_ID_ndx
    eh = errorbar(exp_struct.SOA_vals, ...
		  ROC_struct.twoAFC_AUC(:, target_ID), ...
		  ROC_struct.twoAFC_errorbars(:, target_ID));  
    set( eh, 'LineWidth', 2 );
    lh = plot(exp_struct.SOA_vals, ...
	      ROC_struct.twoAFC_AUC(:, target_ID), ...
	      num2str(target_ID));  
    set( lh, 'LineWidth', 2 );
    line_color = get( lh, 'Color');
    set( eh, 'Color', line_color);

    twoAFC_AUC_filename = ...
	[SOS_analysis_path, 'AUC_', ...
	 'K', num2str(exp_struct.target_ID_vals(target_ID))];
    if ~isempty(exp_struct.SOS_ID)
      twoAFC_AUC_filename = ...
	  [twoAFC_AUC_filename, '_', num2str(exp_struct.SOS_ID)];
    endif
    twoAFC_AUC_filename = ...
	[twoAFC_AUC_filename, '.txt'];
    twoAFC_AUC_tmp = ...
	[exp_struct.SOA_vals, ...
	 ROC_struct.twoAFC_AUC(:, target_ID), ...
	 ROC_struct.twoAFC_errorbars(:, target_ID)];
    save('-ascii', twoAFC_AUC_filename, 'twoAFC_AUC_tmp');
    
endfor % target_ID
