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
  Rmax_scale = 1/4; %% max value of max amoeba radius as fraction of box size
  Rmax_range = 1/1; %% min value of max amoeba radius as fraction of max radius
  Rmax = box_size(1) * Rmax_scale;
  Rmin = Rmax * Rmax_range;
  amoeba_Rmax = Rmin + rand() * (Rmax - Rmin);  %% maximum radius of
  %% amoeba X-section
  disp(["amoeba_Rmax = ", num2str(amoeba_Rmax)]);
  Ramp_max = 3/4; %% max value of the min amoeba radius (in unis of amoeba_Rmax)
  Ramp_min = 3/4; %% min value of the min amoeba radius
  amoeba_Ramp = Ramp_min + rand() * ( Ramp_max - Ramp_min);  %% radial
  %% variation of amoeba x-section as fraction of amoeba_Rmax
  amoeba_Rmin = amoeba_Rmax * amoeba_Ramp;  %% minimum radius of amoeba X-section
  disp(["amoeba_Rmin = ", num2str(amoeba_Rmin)]);
  amoeba_center = Rmax + rand(1,3) .* ( box_size - 2*Rmax );

  N_fourier = 7;%%5;
  %% N_Fourier: number of Fourier components (starting with 0) used to
  %% construct smooth random outline along each axis
  Npts = 2*8*2;
  %% Npts: number of points at which to evalutate/fit polynomial
  %% must be even so that 0 and 180 degrees are explicitly represented

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

  %% scale realand imaginary parts of X-sections to match specified min and max radii
  max_real_Rval = zeros(1,3);
  min_real_Rval = zeros(1,3);
  ave_real_Rval = zeros(1,3);
  %%amoeba_Rval(:,1) = real(amoeba_Rval(:,1));
  real_Rval = real(amoeba_Rval);
  imag_Rval = imag(amoeba_Rval);
  for i_plane = 1:3
    max_real_Rval(i_plane) = max(real(amoeba_Rval(:,i_plane)));
    min_real_Rval(i_plane) = min(real(amoeba_Rval(:,i_plane)));
    real_Rval(:,i_plane) = ...
	amoeba_Rmin + ...
	(real_Rval(:,i_plane) - min_real_Rval(i_plane)) * ...
	(amoeba_Rmax - amoeba_Rmin) / ...
	((max_real_Rval(i_plane) - min_real_Rval(i_plane)) + ...
	 ((max_real_Rval(i_plane) - min_real_Rval(i_plane))==0));
    max_imag_Rval(i_plane) = max(imag(amoeba_Rval(:,i_plane)));
    min_imag_Rval(i_plane) = min(imag(amoeba_Rval(:,i_plane)));
    imag_Rval(:,i_plane) = ...
	amoeba_Rmin + ...
	(imag_Rval(:,i_plane) - min_imag_Rval(i_plane)) * ...
	(amoeba_Rmax - amoeba_Rmin) / ...
	((max_imag_Rval(i_plane) - min_imag_Rval(i_plane)) + ...
	 ((max_imag_Rval(i_plane) - min_imag_Rval(i_plane))==0));
  endfor
  amoeba_Rval = real_Rval + i * imag_Rval;
  ave_Rval = mean(amoeba_Rval,1);

  %% match intercepts
  %% Real part of cross-sections much match at intersections
  Npts_quarter = 1 + round(Npts/4);
  Npts_half = 1 + round(Npts/2);
  Npts_3quarter = 1 + round(3*Npts/4);
  amoeba_Rdiff = zeros(2,3);
  for axis_id = 1:3
    axis_id2 = axis_id + 2;
    axis_id2 = 1 + mod(axis_id2 - 1, 3);
    amoeba_Rdiff(1,axis_id) = ...
	amoeba_Rval(1,axis_id) - amoeba_Rval(Npts_quarter, axis_id2);
    amoeba_Rdiff(2,axis_id) = ...
	amoeba_Rval(Npts_half,axis_id) - amoeba_Rval(Npts_3quarter, axis_id2);
  endfor
  amoeba_Rdiff_original = amoeba_Rdiff;
  for axis_id = 1:3
    for j_arg = [ (Npts_3quarter + 1) : (Npts + Npts_quarter - 1)]
      i_arg = 1 + mod( j_arg - 1, Npts);
      amoeba_Rval(i_arg,axis_id) = ...
	  amoeba_Rval(i_arg,axis_id) - ...
	  amoeba_Rdiff(1,axis_id) * ...
	  min(j_arg - Npts_3quarter, Npts + Npts_quarter - j_arg) / ...
	  (Npts/4);
    endfor
    for i_arg = (Npts_quarter + 1) : (Npts_3quarter - 1)
      amoeba_Rval(i_arg,axis_id) = ...
	  amoeba_Rval(i_arg,axis_id) - ...
	  amoeba_Rdiff(2,axis_id) * ...
	  min(i_arg - Npts_quarter, Npts_3quarter - i_arg) / ...
	  (Npts/4);
    endfor
  endfor
  

  amoeba_Rnew = amoeba_Rval;

%%!!! deprecated: axis conventions are outdated
  linear_transform_flag = 0;
  if linear_transform_flag
    %% deprecated method to match intercepts using linear transformation
    %% fails most of the time because R must be positive
    var_radius = ones(1,3);
    var_A = ones(1,3);
    bar_B = ones(1,3);
    %% fix size of Xsection for i_plane == 1
    %% Xsections for i_plane == 2,3 are morphed to match Xsection for
    %% i_plane == 1,
    %% radial amplitude variations can be scaled and
    %% rotated by arbitrary magnitude and phase 
    %% average radius set to aritrary constant var_radius(1,3)
    %% giving 6 variables constrained by 6 equations (one for each intersection)

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
    const_vec = real([(amoeba_Rval(1,1) - ave_Rval(2)), ...
		      (amoeba_Rval(Npts_half,1) - ave_Rval(2)), ...
		      (amoeba_Rval(Npts_quarter,1) - ave_Rval(3)), ...
		      (amoeba_Rval(Npts_3quarter,1) - ave_Rval(3)), ...
		      (ave_Rval(3) - ave_Rval(2)), ...
		      (ave_Rval(3) - ave_Rval(2)) ]');
    scale_mat = zeros(6);
    scale_mat(1,:) = [1, ...
		      real(amoeba_Rval(Npts_quarter,2) - ave_Rval(2)), ...
		      -imag(amoeba_Rval(Npts_quarter,2) - ave_Rval(2)), ...
		      0, ...
		      0, ...
		      0];
    scale_mat(2,:) = [1, ...
		      real(amoeba_Rval(Npts_3quarter,2) - ave_Rval(2)), ...
		      -imag(amoeba_Rval(Npts_3quarter,1) - ave_Rval(2)), ...
		      0, ...
		      0, ...
		      0];
    scale_mat(3,:) = [0, ...
		      0, ...
		      0, ...
		      1, ...
		      real(amoeba_Rval(1,3) - ave_Rval(3)), ...
		      -imag(amoeba_Rval(1,3) - ave_Rval(3))];
    scale_mat(4,:) = [0, ...
		      0, ...
		      0, ...
		      1, ...
		      real(amoeba_Rval(Npts_half,3) - ave_Rval(3)), ...
		      -imag(amoeba_Rval(Npts_half,3) - ave_Rval(3))];
    scale_mat(5,:) = [1, ...
		      real(amoeba_Rval(1,2) - ave_Rval(2)), ...
		      -imag(amoeba_Rval(1,2) - ave_Rval(2)), ...
		      -1, ...
		      -real(amoeba_Rval(Npts_quarter,3) - ave_Rval(3)), ...
		      imag(amoeba_Rval(Npts_quarter,3) - ave_Rval(3))];
    scale_mat(6,:) = [1, ...
		      real(amoeba_Rval(Npts_half,2) - ave_Rval(2)), ...
		      -imag(amoeba_Rval(Npts_half,2) - ave_Rval(2)), ...
		      -1, ...
		      -real(amoeba_Rval(Npts_3quarter,3) - ave_Rval(3)), ...
		      imag(amoeba_Rval(Npts_3quarter,3) - ave_Rval(3))];
    soln_vec = scale_mat \ const_vec;
    var_radius(2) = real(soln_vec(1));
    var_A(2) = real(soln_vec(2));
    var_B(2) = real(soln_vec(3));
    var_radius(3) = real(soln_vec(4));
    var_A(3) = real(soln_vec(5));
    var_B(3) = real(soln_vec(6));

    amoeba_Rnew = zeros(size(amoeba_Rval));
    amoeba_Rnew(:,1) = amoeba_Rval(:,1);
    amoeba_Rnew(:,2) = var_radius(2) + real(ave_Rval(2)) + ...
	var_A(2) * real(amoeba_Rval(:,2) - ave_Rval(2)) - ...
	var_B(2) * imag(amoeba_Rval(:,2) - ave_Rval(2));
    amoeba_Rnew(:,3) = var_radius(3) + real(ave_Rval(3)) + ...
	var_A(3) * real(amoeba_Rval(:,3) - ave_Rval(3)) - ...
	var_B(3) * imag(amoeba_Rval(:,3) - ave_Rval(3));

  endif %% linear_transform_flag

  
  amoeba_Rnew = real(amoeba_Rnew);
  if any(amoeba_Rnew(:) < 0 )
    warning("any(amoeba_Rnew(:) < 0 )");
  endif

  %% check
  for axis_id = 1:3
    axis_id2 = axis_id + 2;
    axis_id2 = 1 + mod(axis_id2 - 1, 3);
    amoeba_Rdiff(1,axis_id) = ...
	amoeba_Rval(1,axis_id) - amoeba_Rval(Npts_quarter, axis_id2);
    amoeba_Rdiff(2,axis_id) = ...
	amoeba_Rval(Npts_half,axis_id) - amoeba_Rval(Npts_3quarter, axis_id2);
  endfor
  amoeba_Rdelta = ...
      abs(amoeba_Rdiff) / amoeba_Rmax;
  if any( amoeba_Rdelta > 0.5/amoeba_Rmax)
    warning(["any(delta_all > ", num2str(0.5/amoeba_Rmax)]);
  endif
  if any(abs(amoeba_Rdiff) > 0.5)
    warning("any(abs(amoeba_Rdiff) > 0.5)");
  endif

  
  amoeba_x = zeros(Npts, 3);
  amoeba_y = zeros(Npts, 3);
  amoeba_z = zeros(Npts, 3);
  [amoeba_x(:,1) amoeba_y(:,1)] = ...
      pol2cart( fourier_arg', amoeba_Rnew(:,1) );
  [amoeba_y(:,2) amoeba_z(:,2)] = ...
      pol2cart( fourier_arg', amoeba_Rnew(:,2) );
  [amoeba_z(:,3) amoeba_x(:,3)] = ...
      pol2cart( fourier_arg', amoeba_Rnew(:,3) );

  amoeba_x2 = [amoeba_x;amoeba_x(1,:)];
  amoeba_y2 = [amoeba_y;amoeba_y(1,:)];
  amoeba_z2 = [amoeba_z;amoeba_z(1,:)];

  wire_fig_hndl = figure;
  set(gca, "ColorOrder", zeros(8,3));
  wire_hndl = plot3(amoeba_x2(:,1:1:3), amoeba_y2(:,1:1:3), amoeba_z2(:,1:1:3));
  wire_filename = "wire_3D_amoeba.png";
  wire_option = "-dpng";
  print(wire_fig_hndl, wire_filename, wire_option);

  return;
  
  %% find polynomial of the form
  %% sum( coef_poly(i_pow, i_axis) * amoeba_xyz(i_pt, i_axis) ^ i_pow ) = ...
  %% const_poly(i_axis) == 1
  
  Npoly = Npts - 2; %%(3*Npts - 6)/3;
  %% each intersection appears twice
  %% remove 0 and 180 degree intersecton pts along each axis (same pts
  %% appear elsewhere)
  amoeba_xyz = zeros(Npoly, 3);
  amoeba_xyz(:,1) = ...
      [amoeba_x(2:(Npts_half-2),1), amoeba_x(Npts_half:Npts,1)];
  amoeba_xyz(:,2) = ...
      [amoeba_x(2:(Npts_half-2),2), amoeba_x(Npts_half:Npts,2)];
  amoeba_xyz(:,3) = ...
      [amoeba_x(2:(Npts_half-2),3), amoeba_x(Npts_half:Npts,3)];
  const_poly = ones(3*Npoly, 1);
  mat_poly2D = zeros(3*Npoly, 3*Npoly);
  for i_pt = 1 : 3*Npoly
    xyz_pt = 1 + mod(i_pt-1, Npoly);
    i_plane = ceil(i_pt / Npoly);
    for i_pow = 1 : Npoly
	  mat_poly2D(i_pt, i_pow + Npoly * (i_plane-1)) = ...
	      amoeba_xyz(xyz_pt, i_plane).^i_pow;
    endfor
  endfor
  coef_poly = mat_poly2D \ const_poly;
  coef_poly = reshape(coef_poly, [Npoly, 3]);

  %% draw polynomial surface
  amoeba_3D_fig = figure;
  num_ndx = 2*ceil(amoeba_Rmax);
  x_ndx = 1:num_ndx;
  y_ndx = 1:num_ndx;
  [x_sub, y_sub] = meshgrid(x_ndx, y_ndx);
  y_max = amoeba_y2(1,2);
  y_min = amoeba_y2(Npts_half, 2);
  y_delta = (y_max - y_min) / num_ndx;
  surf(amoeba_x2(:), amoeba_y2(:), amoeba_z2(:));

