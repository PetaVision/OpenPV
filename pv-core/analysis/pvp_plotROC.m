function [fig_tmp] = pvp_plotROC(ROC_struct, exp_struct)

  global SOS_analysis_path
  len_layer_ndx = length(ROC_struct.layer_ndx);
  len_target_ID_ndx = length(ROC_struct.target_ID_ndx);
  subplot_index = 0;
  if len_layer_ndx > 3
    nrows_subplot = 2;
  else
    nrows_subplot = 1;
  endif
  ncols_subplot = ceil( (len_layer_ndx) / nrows_subplot );
  twoAFC_ROC_name = ['2AFC ROC'];
  if ~isempty(exp_struct.SOS_ID)
    twoAFC_ROC_name = [twoAFC_ROC_name, '(', num2str(exp_struct.SOS_ID), ')'];
  endif
  fig_tmp = figure('Name', twoAFC_ROC_name);
  for layer = ROC_struct.layer_ndx
    subplot_index = subplot_index + 1;
    subplot(nrows_subplot, ncols_subplot, subplot_index);
    axis "square";
    axis([0 1 0 1]);
    axis "nolabel"
    th = title(["SOA = ", num2str(exp_struct.SOA_vals(layer))]);
				%get(th)
				%set(th, "fontname", "helvecita");
    hold on;
    for target_ID = ROC_struct.target_ID_ndx
      lh = plot(ROC_struct.twoAFC_ROC{layer, target_ID}(:,1), ...
		ROC_struct.twoAFC_ROC{layer, target_ID}(:,2), ...
		num2str(target_ID));  
      set( lh, 'LineWidth', 2 );
      
      twoAFC_ROC_filename = ...
	  [SOS_analysis_path, 'ROC_', ...
	   'K', num2str(exp_struct.target_ID_vals(target_ID)), ...
	   '_SOA', num2str(exp_struct.SOA_vals(layer))];
      if ~isempty(exp_struct.SOS_ID)
	twoAFC_ROC_filename = ...
	    [twoAFC_ROC_name, '_', num2str(exp_struct.SOS_ID)];
      endif
      twoAFC_ROC_filename = ...
	  [twoAFC_ROC_filename, '.txt'];
      twoAFC_ROC_tmp = ROC_struct.twoAFC_ROC{layer, target_ID};
    save('-ascii', twoAFC_ROC_filename, 'twoAFC_ROC_tmp');
      
      
    endfor % target_ID
  endfor % layer
