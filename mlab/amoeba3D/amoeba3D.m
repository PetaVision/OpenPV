function [wire_hndl, ...
	  amoeba_x, ...
	  amoeba_y, ...
	  amoeba_z, ...
	  amoeba_Rnew, ...
	  amoeba_Rval, ...
	  fourier_arg] = amoeba3D()

  %%setenv('GNUTERM', 'x11');
  setenv('GNUTERM', 'aqua');

  %% get dimensions and position of 3D amoeba
  box_size = [256, 256, 256];
  Rmax_scale = 1/4; %% max radius as fraction of box size
  Rmax_range = 1/1; %% minimum value of max radius as fraction of max radius
  Ramp_max = 1/1; %% max amplitude of radial variation as fraction of Rmax
  Ramp_min = 1/1; %% min amplitude of radial variation as fraction of Rmax
  Rmax = box_size(1) * Rmax_scale;
  Rmin = Rmax * Rmax_range;
  amoeba_Rmax = Rmin + rand() * (Rmax - Rmin);  %% maximum radius of
  %% amoeba X-section
  amoeba_Ramp = Ramp_min + rand() * ( Ramp_max - Ramp_min);  %% radial
  %% variation of amoeba x-section as fraction of amoeba_Rmax
  amoeba_Rmin = amoeba_Rmax * amoeba_Ramp;
  amoeba_center = Rmax + rand(1,3) .* ( box_size - 2*Rmax );

  %% make random fourier outline, fit with nth-order polynomial
  N_poly = 16 * 2;  %should be multiple of 4 so that Npts/2 = N_poly/4
		   %is even (i.e. so that 0 and 180 degrees are both
		   %represented explicitly)
  %% N_poly: order of polynomial curve,
  %% sum_n C_n * z^n, 1 <= n <= N_poly
  coef_poly = zeros(3, N_poly);

  N_fourier = 5;
  %% N_Fourier: number of Fourier components (starting with 0) used to
  %% construct smoot random outline along each axis
  Npts = N_poly / 2;
  %% Npts: number of points at which to evalutate/fit polynomial
  %% divide by 2 because we fit both amplitude and 1st derivative

  symmetry_flag = zeros(1,3);
  symmetry_flag(1) = 0; %% only 1st dimension should ever be symmetric

  %% i_plane: (row, col), (row, depth), (col,depth)
  %% dimensions order: (row, col, depth)
  amoeba_Rval = zeros(Npts, 3);
  fourier_arg = [0:(Npts-1)] * (2*pi) / Npts;
  fourier_arg2 = ...
      repmat([0:(N_fourier-1)]', [1, Npts]) .* ...
      repmat( fourier_arg, [N_fourier, 1] );
  for i_plane = 1:3
    if symmetry_flag(i_plane) == 1
      rand_phase = repmat(i*pi/2, [N_fourier, 1]);
    else
      rand_phase = (2*pi) * rand(N_fourier, 1);
    endif
    rand_phase2 = repmat(rand_phase, [1, Npts]);
    fourier_phase2 = exp( i*fourier_arg2 + i*rand_phase2 );
    fourier_coef = randn(N_fourier, 1);
    fourier_coef(1) = 0; %% add DC term separately
    fourier_coef2 = repmat( fourier_coef, [1, Npts]);
    fourier_term = fourier_coef2 .* fourier_phase2;
    amoeba_Rval(:,i_plane) = sum( fourier_term, 1 );
  endfor

  %% scale Xsections to match specified min and max radii
  max_Rval = zeros(1,3);
  min_Rval = zeros(1,3);
  ave_Rval = zeros(1,3);
  %%amoeba_Rval(:,1) = real(amoeba_Rval(:,1));
  for i_plane = 1:3
    max_Rval(i_plane) = max(amoeba_Rval(:,i_plane));
    min_Rval(i_plane) = min(amoeba_Rval(:,i_plane));
    amoeba_Rval(:,i_plane) = ...
	amoeba_Rmin + ...
	(amoeba_Rval(:,i_plane) - min_Rval(i_plane)) * ...
	(amoeba_Rmax - amoeba_Rmin) / ...
	((max_Rval(i_plane) - min_Rval(i_plane)) + ...
	 ((max_Rval(i_plane) - min_Rval(i_plane))==0));
    ave_Rval(i_plane) = mean(amoeba_Rval(:,i_plane));
  endfor
  
  %% match intercepts
  %% Real part of cross-sections much match at intersections

  %% fix size of Xsection for i_plane == 1
  %% Xsections for i_plane == 2,3 are morphed to match Xsection for
  %% i_plane == 1,
  %% radial amplitude variations can be scaled and
  %% rotated by arbitrary magnitude and phase 
  %% average radius set to aritrary constant var_radius(1,3)
  %% giving 6 variables constrained by 6 equations (one for each intersection)

  Npts_quarter = 1 + Npts/4;
  Npts_half = 1 + Npts/2;
  Npts_3quarter = 1 + 3*Npts/4;
  var_radius = ones(1,3);
  var_A = ones(1,3);
  bar_B = ones(1,3);
  %% eq(1): xy intersect with xz
  %% real(amoeba_Rval(1,1)) = ...
  %%   real(var_radius(2) + var_A(1,2) * amoeba_Rval(Npts_quarter,2) + ...
  %%   i*var_B(1,2) * amoeba_Rval(Npts_quarter,2));
  %% eq(2)
  %% real(amoeba_Rval(Npts_half,1)) = ...
  %%   real(var_radius(2) + var_A(1,2) * amoeba_Rval(Npts_3quarter,2) + ...
  %%   i*var_B(1,2) * amoeba_Rval(Npts_3quarter,2));
  %% eq(3): xy intersect with yz
  %% real(amoeba_Rval(Npts_quarter,1)) = ...
  %%   real(var_radius(3) + var_A(1,3) * amoeba_Rval(1,3) + ...
  %%   i*var_B(1,3) * amoeba_Rval(1,3));
  %% eq(4)
  %% amoeba_Rval(Npts_3quarter,1) = ...
  %%   real(var_radius(3) + var_A(1,3) * amoeba_Rval(Npts_half,3) + ...
  %%   i*var_B(1,3) * amoeba_Rval(Npts_half,3));
  %% eq(5): xz intersect with yz
  %% 0 = real(var_radius(2) + var_A(1,2) * amoeba_Rval(1,2) + ...
  %%     i*var_B(1,2) * amoeba_Rval(1,2)) - ...
  %%     real(var_radius(3) + var_A(1,3) * amoeba_Rval(Npts_quarter,3) + ...
  %%     i*var_B(1,3) * amoeba_Rval(Npts_quarter,3));
  %% eq(6)
  %% 0 = real(var_radius(2) + var_A(1,2) * amoeba_Rval(Npts_half,2) + ...
  %%     i*var_B(1,2) * amoeba_Rval(Npts_half,2)) -
  %%     real(var_radius(3) + var_A(1,3) * amoeba_Rval(Npts_3quarter,3) + ...
  %%     i*var_B(1,3) * amoeba_Rval(Npts_3quarter,3));

  %% build matrix scale_mat and vector const_vec
  %%soln_vec = [ var_radius(2) var_A(2) var_B(2) var_radius(3) var_A(3) var_B(3) ];
  const_vec = real([amoeba_Rval(1,1) ...
		    amoeba_Rval(Npts_half,1) ...
		    amoeba_Rval(Npts_quarter,1) ...
		    amoeba_Rval(Npts_3quarter,1) ...
		    0 ...
		    0 ]);
  scale_mat = zeros(6);
  scale_mat(1,:) = [1 real(amoeba_Rval(Npts_quarter,2) - ave_Rval(2)) ...
		    -imag(amoeba_Rval(Npts_quarter,2) - ave_Rval(2)) 0 0 0];
  scale_mat(2,:) = [1 real(amoeba_Rval(Npts_3quarter,2) - ave_Rval(2)) ...
		    -imag(amoeba_Rval(Npts_3quarter,1) - ave_Rval(2)) 0 0 0];
  scale_mat(3,:) = [0 0 0 1 real(amoeba_Rval(1,3) - ave_Rval(3)) ...
		    -imag(amoeba_Rval(1,3) - ave_Rval(3))];
  scale_mat(4,:) = [0 0 0 1 real(amoeba_Rval(Npts_half,3) - ave_Rval(3)) ...
		    -imag(amoeba_Rval(Npts_half,3) - ave_Rval(3))];
  scale_mat(5,:) = [1 real(amoeba_Rval(1,2) - ave_Rval(2)) ...
		    -imag(amoeba_Rval(1,2) - ave_Rval(2)) ...
		    -1 -real(amoeba_Rval(Npts_quarter,3) - ave_Rval(3)) ...
		    +imag(amoeba_Rval(Npts_quarter,3) - ave_Rval(3))];
  scale_mat(6,:) = [1 real(amoeba_Rval(Npts_half,2) - ave_Rval(2)) ...
		    -imag(amoeba_Rval(Npts_half,2) - ave_Rval(2)) ...
		    -1 -real(amoeba_Rval(Npts_3quarter,3) - ave_Rval(3)) ...
		    +imag(amoeba_Rval(Npts_3quarter,3) - ave_Rval(3))];
  soln_vec = const_vec / scale_mat;
  var_radius(2) = real(soln_vec(1));
  var_A(2) = real(soln_vec(2));
  var_B(2) = real(soln_vec(3));
  var_radius(3) = real(soln_vec(4));
  var_A(3) = real(soln_vec(5));
  var_B(3) = real(soln_vec(6));

  amoeba_Rnew = zeros(size(amoeba_Rval));
  amoeba_Rnew(:,1) = amoeba_Rval(:,1);
  amoeba_Rnew(:,2) = var_radius(2) + ...
      var_A(2) * real(amoeba_Rval(:,2)) - ...
      var_B(2) * imag(amoeba_Rval(:,2));
  amoeba_Rnew(:,3) = var_radius(3) + ...
      var_A(3) * real(amoeba_Rval(:,3)) - ...
      var_B(3) * imag(amoeba_Rval(:,3));

  %% check
  diff_1 = amoeba_Rnew(1,1) - amoeba_Rnew(Npts_quarter,2);
  sum_1 = amoeba_Rnew(1,1) + amoeba_Rnew(Npts_quarter,2);
  delta_1 = diff_1 / (sum_1 + (sum_1==0));
  diff_2 = amoeba_Rnew(Npts_half,1) - amoeba_Rnew(Npts_3quarter,2);
  sum_2 = amoeba_Rnew(Npts_half,1) + amoeba_Rnew(Npts_3quarter,2);
  delta_2 = diff_2 / (sum_2 + (sum_2==0));
  diff_3 = amoeba_Rnew(Npts_quarter,1) - amoeba_Rnew(1,3);
  sum_3 = amoeba_Rnew(Npts_quarter,1) + amoeba_Rnew(1,3);
  delta_3 = diff_3 / (sum_3 + (sum_3==0));
  diff_4 = amoeba_Rnew(Npts_3quarter,1) - amoeba_Rnew(Npts_half,3);
  sum_4 = amoeba_Rnew(Npts_3quarter,1) + amoeba_Rnew(Npts_half,3);
  delta_4 = diff_4 / (sum_4 + (sum_4==0));
  diff_5 = amoeba_Rnew(Npts_quarter,3) - amoeba_Rnew(1,2);
  sum_5 = amoeba_Rnew(Npts_quarter,3) + amoeba_Rnew(1,2);
  delta_5 = diff_5 / (sum_5 + (sum_5==0));
  diff_6 = amoeba_Rnew(Npts_3quarter,3) - amoeba_Rnew(Npts_half,2);
  sum_6 = amoeba_Rnew(Npts_3quarter,3) + amoeba_Rnew(Npts_half,2);
  delta_6 = diff_6 / (sum_6 + (sum_6==0));
  disp(["diff = ", num2str([delta_1 delta_2 delta_3 delta_4 delta_5 delta_6])]);
 
  amoeba_x = zeros(Npts, 3);
  amoeba_y = zeros(Npts, 3);
  amoeba_z = zeros(Npts, 3);
  [amoeba_x(:,1) amoeba_y(:,1)] = ...
      pol2cart( fourier_arg', amoeba_Rnew(:,1) );
  [amoeba_z(:,2) amoeba_x(:,2)] = ...
      pol2cart( fourier_arg', amoeba_Rnew(:,2) );
  [amoeba_y(:,3) amoeba_z(:,3)] = ...
      pol2cart( fourier_arg', amoeba_Rnew(:,3) );

  figure;
  wire_hndl = plot3(amoeba_x(:,1:1:3), amoeba_y(:,1:1:3), amoeba_z(:,1:1:3));