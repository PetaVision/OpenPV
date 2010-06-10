function pvp_saveFigList( fig_list, fig_path, fig_suffix)
num_figs = length(fig_list);
for i_fig = 2 : num_figs
  fig_hndl = fig_list(i_fig);
  figure(fig_hndl);
  axis normal
  fig_filename = get(fig_hndl, 'Name');
  fig_filename = [fig_path, fig_filename, '.', fig_suffix];
  fig_option = ['-d', fig_suffix];
  print(fig_hndl, fig_filename, fig_option);
end%%for
