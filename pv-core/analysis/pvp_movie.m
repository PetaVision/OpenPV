function  [H_mov] = pvp_movie( spike_array, movie_title, H_mov )

  global spike_duration
  if ~exist('spike_duration', 'var') || isempty(spike_duration)
    spike_duration = 1;
  endif
  
  global first_frame
  if ~exist('first_frame', 'var') || isempty(first_frame)
    first_frame = 1;
  endif
  
  global last_frame
  if ~exist('last_frame', 'var') || isempty(last_frame)
    last_frame = size(spike_array,1);
  endif

  if ~exist('H_mov','var')          % tests if 'fh' is a variable in the workspace
    H_mov = figure;
  endif

  axis tight
  axis off
  box off
  axh = gca;
  set(axh,'nextplot','replacechildren');
  for i_frame = first_frame : last_frame
    plot_title = ['Spike Movie: layer = ', num2str(layer), ': frame = ', num2str(i_frame)];
        recon_array = spike_array{layer}( i_frame:(i_frame+spike_duration-1),: );
        recon_array = sum(recon_array,1);
        recon_array(recon_array > 1) = 1;
        fh = pvp_reconstruct(full(recon_array), plot_title, fh);
        figure(fh);
        fh = clf(fh);
    end % i_frame
end % i_layer
close(fh);












