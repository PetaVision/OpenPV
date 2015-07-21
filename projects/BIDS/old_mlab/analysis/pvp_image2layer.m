function [ layer_index, layer_index_max ] = ...
      pvp_image2layer(layer, ...
		      image_index, ...
		      timesteps, ...
		      use_max, ...
		      rate_array, ...
		      pvp_order )
  
				% Given row,col position in the image, return a list of the corresponding
				% rows, cols and feature indices in a subsequent layer 
				% use_max specifies only the indices of the cells with maximum firing rate over timesteps is returned, 
				% where the max is over all "features"
				% if N > N_image, then a patch of cells in the layer centered on each image pixel
				% is treated as a "flat" feature dimension, along with any explicit feature dimension
				% if use_max == 0, return all indices in patch
				% if pvp_order == 1, use [ NFEATURES, NCOLS, NROWS ]
				% if pvp_order == 0, use [ NROWS, NCOLS, NFEATURES ]

  global N_image NROWS_image NCOLS_image 
  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer

				%  global spike_array
				%  global LAYER
				# if nargin < 1 || isempty(layer)
				#   layer = LAYER;
				# end%if
				# if nargin < 2 || isempty(image_index)
				#   if ~isempty(rate_array)
				#     image_index = find(rate_array);
				#   else
				#     image_index = [];
				#   endif
				# endif
				# if nargin < 3 || isempty(timesteps)
				#   timesteps = size(spike_array, 1);
				# endif
				# if nargin < 4
				#   if ~isempty(rate_array)
				#     use_max = 1;
				#   else
				#     use_max = 0;
				#   endif
				# endif
				# if nargin < 5 || isempty(rate_array)
				#   rate_array = zeros(1,N);
				# endif

  row_scale = ceil( NROWS / NROWS_image );
  col_scale = ceil( NCOLS / NCOLS_image );

  num_image_index = length(image_index);

  [irow_image, jcol_image] = ind2sub( [NROWS_image NCOLS_image], image_index );

  irow_layer = ceil( irow_image * NROWS / NROWS_image ); 
  jcol_layer = ceil( jcol_image * NCOLS / NCOLS_image ); 

				% if row_scale, col_scale > 1, find all cells in patch around each image cell
  [ col_mesh , row_mesh ] = meshgrid( 0 : col_scale-1, 0 : row_scale-1 );
  row_mesh = row_mesh(:)';
  col_mesh = col_mesh(:)';
  num_mesh = length( row_mesh );
  irow_layer = repmat( irow_layer, 1, num_mesh );
  jcol_layer = repmat( jcol_layer, 1, num_mesh );
  row_mesh = repmat( row_mesh, num_image_index, 1 );
  col_mesh = repmat( col_mesh, num_image_index, 1 );
  irow_layer = irow_layer - row_mesh;
  jcol_layer = jcol_layer - col_mesh;
  irow_layer = repmat( irow_layer, [1, 1, NFEATURES] );
  jcol_layer = repmat( jcol_layer, [1, 1, NFEATURES] );

				% add feature index
  f_layer = zeros( 1, 1, NFEATURES );
  f_layer( 1, 1, : ) = 1 : NFEATURES;
  f_layer = repmat( f_layer, [ num_image_index, num_mesh, 1 ] );

  if pvp_order
    layer_index = sub2ind( [ NFEATURES, NCOLS, NROWS ], f_layer(:), jcol_layer(:), irow_layer(:) );
  else
    layer_index = sub2ind( [ NROWS, NCOLS, NFEATURES ], irow_layer(:), jcol_layer(:), f_layer(:) );
  endif
				% rate info not available except for present epoch
  if use_max && row_scale * col_scale * NFEATURES > 1
    rate_array = squeeze( rate_array( layer_index ) );
    rate_array = reshape( rate_array, [ num_image_index, row_scale * col_scale * NFEATURES ] );
    [ max_rate, max_index ] = max( rate_array, [], 2 );
    irow_layer = reshape( irow_layer, [ num_image_index, row_scale * col_scale * NFEATURES ] );   
    jcol_layer = reshape( jcol_layer, [ num_image_index, row_scale * col_scale * NFEATURES ] );   
    f_layer = reshape( f_layer, [ num_image_index, row_scale * col_scale * NFEATURES ] );
    max_ndx = sub2ind( [ num_image_index, row_scale * col_scale * ...
			NFEATURES ], (1:num_image_index)', max_index );
    irow_layer_max = squeeze( irow_layer( max_ndx ) );
    jcol_layer_max = squeeze( jcol_layer( max_ndx ) );
    f_layer_max = squeeze( f_layer( max_ndx ) );
    if pvp_order
      layer_index_max = sub2ind( [ NFEATURES, NCOLS, NROWS ], f_layer_max(:), jcol_layer_max(:), irow_layer_max(:) ); 
    else
      layer_index_max = sub2ind( [ NROWS, NCOLS, NFEATURES ], irow_layer_max(:), jcol_layer_max(:), f_layer_max(:) ); 
    endif
  else
    layer_index_max = layer_index;
  endif


				% find index of maximum activity associated with each image pixel when row_scale, col_scale > 1
				% if use_max && row_scale > 1 & col_scale > 1
				%    rate_array = ... 
				%       reshape( rate_array, [ row_scale, NROWS_image, col_scale, NCOLS_image, NFEATURES ] );
				%    rate_array = permute( rate_array, [ 3, 1, 4, 2, 5 ] );
				%    rate_array = reshape( rate_array, [1, 1, row_scale * col_scale * NFEATURES] );
				%    [ max_rate, max_index] = max( rate_array, 3 );
				%    [ max_row_index, max_col_index, max_feature_index ] = ...
				%       ind2sub( max_index, [ row_scale, col_scale, NFEATURES ] );
				%    base_row_index = repmat( ( 0 : row_scale : NROWS-row_scale )', 1, NCOLS_image );
				%    base_col_index = repmat( ( 0 : col_scale : NCOLS-col_scale ), 1, NROWS_image );
				%    max_row_index = max_row_index + base_row_index;
				%    max_col_index = max_col_index + base_col_index;
				% end

  