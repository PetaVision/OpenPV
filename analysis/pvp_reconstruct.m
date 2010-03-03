
function [fh] = pvp_reconstruct( recon_array, plot_title, fh )
  
  global NK NO NROWS NCOLS
  
  if ~exist('DTH', 'var') || isempty(DTH)
    DTH = 180 / NO;
  endif
  
				%size(recon_array)
				%pause
  
				% plot reconstructed image
  if ~any(recon_array(:) ~= 0.0) % any(A) tests whether any of the elements
				% along various dimensions of an array is a 
				% nonzero number or is logical 1 (true). 
				% any ignores entries that are NaN (Not a Number).
    return;
  endif
  if ~exist('fh','var')          % tests if 'fh' is a variable in the workspace
				% returns 0 if 'fh' does not exists
    fh = figure;
  endif
  set(fh, 'Name', plot_title);
  ave_recon = mean(recon_array(recon_array ~= 0.0));
  if ave_recon < 0
    recon_array = -recon_array;
    ave_recon = -ave_recon;
  endif
  max_recon = max(recon_array(:));
  min_recon = min(recon_array(:));

				% if activity is sparse, plot all pixels, else only plot pixels > mean
  nz_recon = length(find(recon_array ~= 0.0));
  if nz_recon < length(recon_array) % / 2
    min_recon_val = min_recon;
  else
    min_recon_val = ave_recon;
  endif
  if min_recon_val == max_recon
    min_recon_val = 0;
  endif
  edge_len = sqrt(2)/2;
  if (NO==1)
    edge_len = sqrt(2)/2;
  endif
  theta_offset = 0;
  max_line_width = 2.5;
  axis([0 NCOLS+1 0 NROWS+1]);
  axis square;
  axis tight
  box on
  axis on
				%colorbar('East');
  colormap('gray');
				% cmap = colormap;
  hold on;
  if NO > 1 &&  max_recon > 0
    [recon_array, recon_ndx] = sort(recon_array(:));
    first_recon_ndx = find(recon_array > min_recon_val, 1, 'first');
    for recon_index = first_recon_ndx : length(recon_ndx)
      i_recon = recon_ndx(recon_index);
      recon_val = recon_array(recon_index);
      if recon_val <= min_recon_val
	continue;
      else
	recon_val = ( recon_val - min_recon_val ) / ( max_recon - min_recon_val + (max_recon == min_recon_val) );
      endif
      [i_k, i_theta, j_col, i_row] = ind2sub( [NK, NO, NCOLS, NROWS ], i_recon );
      clear i_k
      i_theta = i_theta - 1 + theta_offset;
      j_col = j_col;
      i_row = i_row;
      delta_col = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
      delta_row = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
      lh = line( [j_col - delta_col, j_col + delta_col], ...
		[(i_row - delta_row), (i_row + delta_row)] );
      line_width = 0.005 + max_line_width * recon_val;
      set( lh, 'LineWidth', line_width );
      line_color = 1 - recon_val;
      set( lh, 'Color', line_color*[1 1 1]);
    endfor
  elseif NO == 0
    recon2D = reshape( recon_array, [NCOLS, NROWS] )';
    recon2D = recon2D';
    imagesc( recon2D );  % plots recod2D as an image
  endif
  hold off;
				%pause	