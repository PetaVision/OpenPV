function pvp_saveFigList( fig_list, fig_path, fig_suffix)
%%keyboard
num_figs = length(fig_list);
for i_fig = 1 : num_figs
  fig_hndl = fig_list(i_fig);
  try
    figure(fig_hndl);
  catch
    continue;
  end
  axis normal
  fig_filename = get(fig_hndl, 'Name');
  fig_filename(fig_filename==" ")="";
  fig_filename(fig_filename=="(")="_";
  fig_filename(fig_filename==")")="_";
  fig_filename = [fig_path, fig_filename, '.', fig_suffix];
  if exist(fig_filename, 'file')
    delete(fig_filename);
  end%%if
  fig_option = ['-d', fig_suffix];
  %%saveas(fig_hndl, fig_filename, fig_suffix);
  print(fig_hndl, fig_filename, fig_option);
end%%for
