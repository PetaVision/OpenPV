function pvp_saveFigList( fig_list, fig_path, fig_suffix);
num_figs = length(fig_list);
for i_fig = 1 : num_figs
  fig_hndl = fig_list(i_fig);
  fig_filename = get(fig_hndl, 'Name');
  fig_filename = [fig_path, fig_filename, '.', fig_suffix];
  fig_option = ['-d', fig_suffix];
  print(fig_hndl, fig_filename, fig_option);
endfor
