
function [fh, recon_colormap] = ...
      pvp_reconstruct(recon_array, ...
		      plot_title, ...
		      fh, ...
		      size_recon, ...
		      plot_recon_flag, ...
		      pixels_per_cell)
  
  global NK NO NROWS NCOLS
  global ROTATE_FLAG
  global MIN_INTENSITY
  global FLAT_ARCH_FLAG
  global NUM2STR_FORMAT
  global OUTPUT_PATH
  global THETA_MAX
  
  if ~exist('NUM2STR_FORMAT') || isempty(NUM2STR_FORMAT)
    NUM2STR_FORMAT = '%03.3i';
  endif
  
  if ~exist('MIN_INTENSITY') || isempty(MIN_INTENSITY)
    MIN_INTENSITY = 0;
  endif
  
  if ~exist('THETA_MAX', 'var') || isempty(THETA_MAX)
   THETA_MAX = pi;
  endif

  if ~any(recon_array(:) ~= 0.0) % any(A) tests whether any of the elements
    return;
  endif

  if ~exist('size_recon', 'var') || isempty(size_recon) || nargin < 4
    size_recon = [NK NO NCOLS NROWS];
  endif
  NK = size_recon(1);
  NO  = size_recon(2);
  NCOLS  = size_recon(3);
  NROWS = size_recon(4);
  NFEATURES = NK * NO;
  
  if ~exist('DTH', 'var') || isempty(DTH)
    DTH = THETA_MAX / NO;
  endif

  if ~exist('plot_recon_flag', 'var') || isempty(plot_recon_flag) || nargin < 5
    plot_recon_flag = 1;
  endif

  if ~exist('pixels_per_cell', 'var') || isempty(pixels_per_cell) || nargin < 6
    pixels_per_cell = 1;
  endif

  flip_ij = 1;

				% disp(['size_recon = ', num2str(size_recon)]);

  max_recon_val = max(recon_array(:));
  min_recon_val = min(recon_array(:));
  recon_array = ...
      ( recon_array(:) - min_recon_val ) / ...
      ( max_recon_val - min_recon_val + (max_recon_val == min_recon_val) );
  zero_recon_val = ...
      ( 0.0 - min_recon_val ) / ...
      ( max_recon_val - min_recon_val + (max_recon_val == min_recon_val) );
  zero_recon_val = max(0, zero_recon_val);
  log_flag = 0;
  if log_flag
    recon_array = log(1 + recon_array);
    zero_recon_val = log(1 + zero_recon_val);
    max_recon_val = max(recon_array(:));
    min_recon_val = min(recon_array(:));
    recon_array = ...
	( recon_array(:) - min_recon_val ) / ...
	( max_recon_val - min_recon_val + (max_recon_val == min_recon_val) );
    zero_recon_val = ...
	( zero_recon_val - min_recon_val ) / ...
	( max_recon_val - min_recon_val + (max_recon_val == min_recon_val) );
    zero_recon_val = max(0, zero_recon_val);
  endif
  sigmoid_flag = 0;
  if sigmoid_flag
    red_flag = ...
	recon_array > (zero_recon_val + (1/4)*(1 - zero_recon_val) );
    blue_flag = ...
	recon_array < (zero_recon_val - (1/4)* zero_recon_val);
    green_flag = ~red_flag && ~blue_flag;
    recon_array(red_flag) = 1;
    recon_array(blue_flag) = 0;
    recon_array(green_flag) = zero_recon_val;
  endif
  disp(['min_recon_val = ', num2str(min_recon_val)]);
  disp(['max_recon_val = ', num2str(max_recon_val)]);

  edge_len = 3.5 * sqrt(2)/2;
  if (NO==1)
    edge_len = sqrt(2)/2;
  endif
  
  theta_offset = 0.5 * ROTATE_FLAG;
  max_line_width = 1.0;
  min_line_width = 1.0;
				% cmap = colormap;
  hold on;
  if ~FLAT_ARCH_FLAG && NO > 1  && plot_recon_flag
    if ~exist('fh','var') || isempty(fh) || nargin < 3
      fh = figure;
    endif
    set(fh, 'Name', plot_title);
    axis([0 NCOLS+1 0 NROWS+1]);
    if NROWS == NCOLS
      axis "square";
    endif
    axis "tight";
    axis "image";
    if flip_ij
      axis "ij"
    endif
    box "off"
    axis "off"

    [recon_array_tmp, recon_ndx] = sort(abs(recon_array(:)));
    recon_array = recon_array(recon_ndx);
    first_recon_ndx = 1;
    last_recon_ndx = length(recon_array);

    max_color_ndx = 128;
    recon_colormap = zeros(max_color_ndx,3);
    recon_color_ndx = (1:max_color_ndx)';
    zero_color_ndx = max(1, ceil(zero_recon_val * max_color_ndx));
    xmass_colormap = 0;
    if xmass_colormap
      red_norm = 1 - zero_recon_val;
      red_norm = red_norm + (red_norm == 0);
      blue_norm = zero_recon_val;
      blue_norm = blue_norm + (blue_norm == 0);
      red_ndx_norm = ...
	  max_color_ndx - zero_color_ndx + ...
	  (max_color_ndx == zero_color_ndx);
      red_recon_vals = ...
	  max( (recon_color_ndx - zero_color_ndx) / red_ndx_norm, 0 );
      blue_ndx_norm = ...
	  zero_color_ndx + (0 == zero_color_ndx);
      blue_recon_vals = ...
	  max((zero_color_ndx - recon_color_ndx) / blue_ndx_norm, 0 );
      green_recon_vals = 1 - red_recon_vals - blue_recon_vals;
    else
      green_recon_vals = ones(size(recon_color_ndx));
      red_ndx_norm = zero_color_ndx;
      red_recon_vals = recon_color_ndx;
      red_recon_vals ( red_recon_vals > red_ndx_norm ) = red_ndx_norm;
      red_recon_vals = red_recon_vals / red_ndx_norm;
      green_recon_vals( red_recon_vals < 1 ) = ...
	  red_recon_vals( red_recon_vals < 1 );
      blue_recon_vals = max_color_ndx - recon_color_ndx;
      blue_ndx_norm = max_color_ndx - zero_color_ndx;
      blue_ndx_norm = blue_ndx_norm + (blue_ndx_norm == 0);
      blue_recon_vals ( blue_recon_vals > blue_ndx_norm ) = blue_ndx_norm;
      blue_recon_vals = blue_recon_vals / blue_ndx_norm;
      green_recon_vals( blue_recon_vals < 1 ) = ...
	  blue_recon_vals( blue_recon_vals < 1 );
    endif
    
    recon_colormap(:,1) = red_recon_vals;
    recon_colormap(:,2) = green_recon_vals;
    recon_colormap(:,3) = blue_recon_vals;
    make_recon_image = 1;
    if make_recon_image
      colorbar('East');
      %%colormap('gray');
      colormap(recon_colormap);
      recon_image = ...
	  repmat(zero_color_ndx, pixels_per_cell * NROWS, pixels_per_cell * NCOLS);
      recon_values = recon_array;
      recon_values(recon_values < 0) = 0;
      recon_values(recon_values > 1) = 1;
      recon_color_index = ...
	  ceil(recon_values * max_color_ndx);
      recon_color_index(recon_color_index < 1) = 1;
      recon_color_index(recon_color_index > max_color_ndx) = max_color_ndx;      
      [i_k_values, i_theta_values, j_col_values, i_row_values] = ...
	  ind2sub( size_recon, recon_ndx );
      i_theta_values = i_theta_values - 1 + theta_offset;
      for edge_val = -edge_len : 1/pixels_per_cell : edge_len
	delta_row_values = ...
	    edge_val * ( sin(i_theta_values * DTH) );
	delta_col_values = ...
	    edge_val * ( cos(i_theta_values * DTH) ); 
	tmp_row_values = pixels_per_cell * ...
	    (i_row_values + delta_row_values);
	tmp_col_values = pixels_per_cell * ...
	    (j_col_values + delta_col_values);
	tmp_row_values = round(tmp_row_values);
	tmp_col_values = round(tmp_col_values);	
	tmp_row_values(tmp_row_values < 1) = 1;
	tmp_col_values(tmp_col_values < 1) = 1;
	tmp_row_values(tmp_row_values > pixels_per_cell * NROWS) = ...
	    pixels_per_cell * NROWS;
	tmp_col_values(tmp_col_values > pixels_per_cell * NCOLS) = ...
	    pixels_per_cell * NCOLS;
	for i_recon_ndx = 1 : length(recon_color_index)
	  pixel_row = tmp_row_values(i_recon_ndx);
	  pixel_col = tmp_col_values(i_recon_ndx);
	  pixel_val_old = recon_image(pixel_row, pixel_col);
	  pixel_val_new = recon_color_index(i_recon_ndx);
	  if (abs(pixel_val_new-zero_color_ndx) > abs(pixel_val_old-zero_color_ndx))
	    recon_image(pixel_row, pixel_col) = ...
		pixel_val_new;
	  endif
	endfor %% i_recon_ndx
      endfor %% edge_val
      line_width_values = ...
	  min_line_width + ...
	  (max_line_width - min_line_width) * recon_values;
      imagesc(1:1/pixels_per_cell:NROWS, 1:1/pixels_per_cell:NCOLS, recon_image);
      %%axis off;
      %%box off;
      colorbar;
    else
      for recon_index = first_recon_ndx : last_recon_ndx
	i_recon = recon_ndx(recon_index);
	recon_val = recon_array(recon_index);
	if recon_val <= min_recon_val
	  continue;
	endif
	[i_k, i_theta, j_col, i_row] = ind2sub( size_recon, i_recon );
	clear i_k
	i_theta = i_theta - 1 + theta_offset;
	j_col = j_col;
	i_row = i_row;
	delta_col = edge_len * ( cos(i_theta * DTH) ); %%* pi / 180 ) );
	delta_row = edge_len * ( sin(i_theta * DTH) ); %% * pi / 180 ) );
	lh = line( [j_col - delta_col, j_col + delta_col], ...
		  [(i_row - delta_row), (i_row + delta_row)] );
	%%line_color = 1 - recon_val;
	recon_color_index =  max(1, ceil(recon_val * max_color_ndx));
	recon_line_color = recon_colormap(recon_color_index,:);
	%%red_val = max( (recon_val - zero_recon_val) / red_norm, 0 );
	%%blue_val = max( (zero_recon_val - recon_val) / blue_norm, 0 );
	%%green_val = 1 - red_val - blue_val;
	%%line_color = [red_val, green_val, blue_val];
	set( lh, 'Color', recon_line_color );%%*(1-MIN_INTENSITY)*[1 1 1]);
	%%line_width = MIN_INTENSITY + (1-MIN_INTENSITY) * max_line_width * recon_val;
	line_width = ...
	    min_line_width + ...
	    (max_line_width - min_line_width) * (1-recon_line_color(2));
	set( lh, 'LineWidth', line_width );
      endfor %% recon_index
    endif %% make_recon_image
  elseif ~FLAT_ARCH_FLAG && NO == 1 && plot_recon_flag
    fh = zeros(1,NFEATURES);
    fh = figure;
    recon2D = reshape( recon_array(:), [NCOLS, NROWS] );
    set(fh, 'Name', plot_title);
    axis([0 NCOLS+1 0 NROWS+1]);
    if NROWS == NCOLS
      axis "square";
    endif
    axis "tight";
    axis "image";
    if flip_ij
      axis "ij"
    endif
    box "off"
    axis "off"
    imagesc( recon2D' );  % plots recod2D as an image
    colormap('gray');
  elseif FLAT_ARCH_FLAG
    NFEATURES = size_recon(1) * size_recon(2);
    recon3D = reshape( recon_array(:), [NFEATURES, NCOLS, NROWS] );
    recon3D = ...
	( recon3D - min_recon_val ) ./ ...
	( max_recon_val - min_recon_val + (max_recon_val == min_recon_val) );
    if plot_recon_flag
      if ( ~exist('fh','var') || isempty(fh) || nargin < 5 )
	fh = figure;
      else
	figure(fh);
      endif
      set(fh, 'Name', plot_title);
      axis([0 NCOLS+1 0 NROWS+1]);
      if NROWS == NCOLS
	axis "square";
      endif
      axis "tight";
      axis "image";
      if flip_ij
	axis "ij"
      endif
      box "off"
      axis "off"
      tmp = squeeze( max(recon3D,[],1) );
				%imagesc( gca, tmp' );  % plots recod2D as an image
      imagesc( tmp' );  % plots recod2D as an image
      colormap('gray');
    endif
%%    plot_title_tmp = ...
%%	[OUTPUT_PATH, plot_title, '.tiff'];
%%    recon3D = uint8(255*recon3D);
%%    imwrite( squeeze( max(recon3D,[],1) )', ...
%%	    plot_title_tmp);
    for i_feature = 1 : 0 % NFEATURES
      plot_title_tmp = ...
	  [OUTPUT_PATH, plot_title, '_', num2str(i_feature, NUM2STR_FORMAT), '.tiff'];
      imwrite( squeeze( recon3D(i_feature,:,:) )', ...
	      plot_title_tmp, 'tiff');
    endfor
  endif
 