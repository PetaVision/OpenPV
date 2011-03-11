function [surf_hndl] = amoeba3D()

  %%setenv('GNUTERM', 'x11');
  setenv('GNUTERM', 'aqua');

  %% get dimensions of 3D amoeba
  box_size = [256, 256, 256];
  Rmax_scale = 4;
  Rmax_range = 4;
  Rmin_scale = 2;
  Rmax = box_size(1) / Rmax_scale;
  Rmin = Rmax / Rmax_range;
  amoeba_Rmax = Rmin + rand() * (Rmax - Rmin);
  amoeba_Rmin = ...
      ( amoeba_Rmax / Rmin_scale ) + ...
      rand() * (amoeba_Rmax - ( amoeba_Rmax / Rmin_scale ) );
  center_pos = max_size + rand(1,3) * ( box_size - 2*max_size );

  %% make random fourier outline, fit with nth-order polynomial
  N_poly = 8 * 2;  %should be multiple of 8
  %% N_poly: order of polynomial curve,
  %% sum_n C_n * z^n, 1 <= n <= N_poly
  coef_poly = zeros(3, N_poly);

  N_fourier = 5;
  %% N_Fourier: number of Fourier components (starting with 0) used to
  %% construct smoot random outline along each axis
  N_pts = N_poly / 2;
  %% N_pts: number of points at which to evalutate/fit polynomial
  %% divide by 2 because we fit both amplitude and 1st derivative

  symmetry_flag = zeros(1,3);

  %% i_plane: (row, col), (row, depth), (col,depth)
  %% dimensions order: (row, col, depth)
  amoeba_Xsection = zeros(N_pts, 3);
  fourier_arg = [0:(N_pts-1)] * (2*pi) / N_pts;
  fourier_arg2 = ...
      repmat([0:(N_fourier-1)]', [1, N_pts]) .* ...
      repmat( fourier_arg, [N_fourier, 1] );
  for i_plane = 1:3
    %%rand_phase = (2*pi) * rand(N_fourier, 1);
    %%rand_phase2 = repmat(rand_phase, [1, N_pts]);
    %%fourier_phase = exp( fourier_arg2 + rand_phase2 );
    fourier_coef = randn(N_fourier, 2);
    fourier_coef(1, :) = 0; %% add DC term separately
    fourier_coef(N_pts/2, :) = 0;
    if symmetry_flag(i_plane) == 1
      fourier_coef(:,1) = 0;
    endif
    cosine_coef2 = repmat( fourier_coef(:,1), [1, N_pts]);
    sine_coef2 = repmat( fourier_coef(:,2), [1, N_pts]);
    fourier_term = fourier_coef2 .* fourier_phase2;
    amoeba_Xsection(:,i_plane) = ...
	sum( fourier_term(2:N_pts), 1 );
  endfor

  %% match intercepts
  for i_plane = 2:3
    slope_rescale = ...
	(amoeba_Xsection(1, 1) - amoeba_Xsection(N_pts/2, 1) ) / ...
	(amoeba_Xsection(1, i_plane) - amoeba_Xsection(N_pts/2, i_plane) );
    offset_rescale = ...
	(amoeba_Xsection(N_pts/2, 1) * amoeba_Xsection(1, i_plane) - ...
	 amoeba_Xsection(1, 1) * amoeba_Xsection(N_pts/2, i_plane) ) / ...
	( amoeba_Xsection(1, i_plane) - amoeba_Xsection(N_pts/2, i_plane) );
  endfor  

  max_Xsection = max(amoeba_Xsection(:));
  min_Xsection = min(amoeba_Xsection(:));

  fourier_max = max(fourier_sum(:));
  fourier_min = min(fourier_sum(:));
  outer_diameter = ...
      ( rand(1) * ( 1 - amoeba_struct.target_outer_min ) + amoeba_struct.target_outer_min ) ...
      * amoeba_struct.target_outer_max * fix(amoeba_struct.image_rect_size/2);
  inner_diameter = ...
      ( rand(1) * ( 1 - amoeba_struct.target_inner_min ) + amoeba_struct.target_inner_min ) ...
      * amoeba_struct.target_inner_max * outer_diameter;
  r_phi = inner_diameter + ...
      ( fourier_sum - fourier_min ) * ...
      ( outer_diameter - inner_diameter ) / (fourier_max - fourier_min );


  target_outline = 
  
  x_vals = r_val * [1:box_size(1)] / box_size(1);
  y_vals = r_val * [1:box_size(2)] / box_size(2);
  x_vals = x_vals + x_center;
  y_vals = y_vals + y_center;
  [xmesh_vals, ymesh_vals] = meshgrid(x_vals,y_vals);
  zmesh_vals = z_center + ...
      real(sqrt( r2_val - (xmesh_vals - x_center).^2 - ...
		(ymesh_vals - y_center).^2 ));
  surf_hndl = mesh(xmesh_vals, ymesh_vals, zmesh_vals);
  axis([1 box_size(1) 1 box_size(2) 1 box_size(3)]);