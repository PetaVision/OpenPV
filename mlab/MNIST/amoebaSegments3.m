function [amoeba_image_x, amoeba_image_y, amoeba_struct] = ...
      amoebaSegments3(amoeba_struct, distractor_flag)

  if nargin < 2 || ~exist("distractor_flag") || isempty(distractor_flag)
    distractor_flag = 0;
  endif

  num_phi = amoeba_struct.num_phi;
  %% determine if amoeba is open (linear) or (semi-)closed
  amoeba_struct.closed_flag = rand() < amoeba_struct.closed_prob;
  if amoeba_struct.closed_flag 
    amoeba_struct.delta_phi = amoeba_struct.range_phi / num_phi;
  else
    amoeba_struct.delta_phi = 1;
  endif
  delta_segment = num_phi / amoeba_struct.num_segments;

%% make gaps of variabl size between segments
  gap_offest = delta_segment * rand(1);
  list_segments = [1; round(gap_offest:delta_segment:num_phi)'];
  list_segments = [list_segments, circshift(list_segments,-1)];
  list_segments(end, 2) = num_phi;
  delta_gap = ...
      [0; round( rand(amoeba_struct.num_segments, 1) * (amoeba_struct.max_gap - amoeba_struct.min_gap) + amoeba_struct.min_gap )];
  list_segments(:,1) = list_segments(:,1) + delta_gap;
  list_segments = min( list_segments, num_phi );
  list_segments = max( list_segments, 1 );

  %% make random fourier curve
  fourier_arg = (0:(num_phi-1)) * amoeba_struct.delta_phi;
  fourier_arg2 = ...
      repmat((0:(amoeba_struct.num_fourier-1))',[1, num_phi]) .* ...
      repmat( fourier_arg, [amoeba_struct.num_fourier, 1] );
				% exponentially decaying ampitudes produce amoebas that are too regular...
				% amoeba_struct.fourier_ratio = exp(-(0:(amoeba_struct.num_fourier-1))/amoeba_struct.num_fourier)';
  fourier_ratio = ones(amoeba_struct.num_fourier, 1);
  fourier_coef = amoeba_struct.fourier_ratio .* randn(amoeba_struct.num_fourier, 1);
  fourier_coef = fourier_coef .* 2.^(amoeba_struct.fourier_amp);
  fourier_coef2 = repmat( amoeba_struct.fourier_ratio .* fourier_coef, [1, num_phi]);
  fourier_phase = (pi) * rand(amoeba_struct.num_fourier, 1);
  %% could use pi/2 since amlitudes are normally distributed about zero
  fourier_phase2 = repmat(fourier_phase, [1, num_phi]);
  fourier_term = fourier_coef2 .* cos( amoeba_struct.fourier_arg2 + fourier_phase2);
  fourier_sum = sum(fourier_term, 1);
  fourier_max = max(fourier_sum(:));
  fourier_min = min(fourier_sum(:));
  fourier_ave = mean(fourier_sum(:));

  %% set size randomly within specified range
  outer_diameter = ...
      ( rand(1) * ( 1 - amoeba_struct.target_outer_min ) + amoeba_struct.target_outer_min ) ...
      * amoeba_struct.target_outer_max * fix(amoeba_struct.image_rect_size/2);
  inner_diameter = ...
      ( rand(1) * ( 1 - amoeba_struct.target_inner_min ) + amoeba_struct.target_inner_min ) ...
      * amoeba_struct.target_inner_max * outer_diameter;
  r_phi = inner_diameter + ...
      ( fourier_sum - fourier_min ) * ...
      ( outer_diameter - inner_diameter ) / (fourier_max - fourier_min );
  amoeba_struct.outer_diameter = outer_diameter;
  amoeba_struct.inner_diameter = inner_diameter;
  if ~amoeba_struct.closed_flag
    open_length = outer_diameter;
    open_amplitude = outer_diameter - inner_diameter;
    y_x = r_phi - inner_diameter;
    y_x = y_x - mean(y_x);
    num_x = num_phi;
    delta_x = open_length / num_x;
  endif

  seg_list = 1 : amoeba_struct.num_segments + 1;
  amoeba_image_x = cell(amoeba_struct.num_segments + 1,1);
  amoeba_image_y = cell(amoeba_struct.num_segments + 1,1);

				% extract x,y pairs for each segment
  if amoeba_struct.closed_flag

    %% make a (semi-)closed amoeba
    for i_seg = seg_list
      [amoeba_segment_x, amoeba_segment_y] = ...
	  pol2cart(amoeba_struct.fourier_arg( list_segments(i_seg,1):list_segments(i_seg,2) ), ...
		   r_phi( list_segments(i_seg,1):list_segments(i_seg,2) ) );
      amoeba_image_x{i_seg} = amoeba_segment_x;
      amoeba_image_y{i_seg} = amoeba_segment_y;
    endfor  
  
  else  %% ~amoeba_struct.closed_flag

    %% make a linear amoeba
    %% start with horzontal curve
    for i_seg = seg_list
      amoeba_segment_x = list_segments(i_seg,1):list_segments(i_seg,2);
      amoeba_segment_y = y_x( list_segments(i_seg,1):list_segments(i_seg,2) );
      amoeba_segment_x = amoeba_segment_x * delta_x;
      amoeba_image_x{i_seg} = amoeba_segment_x;
      amoeba_image_y{i_seg} = amoeba_segment_y;
    endfor
    %% now rotate by arbitrary angle about midpoint
    rand_theta = 2*pi*rand();
    ave_x = open_length / 2;
    ave_y = mean(y_x);
    i_seg_ndx = 0;
    for i_seg = seg_list
      i_seg_ndx = i_seg_ndx + 1;
      x_old = amoeba_image_x{i_seg};
      y_old = amoeba_image_y{i_seg};
      x_old = x_old - ave_x;
      y_old = y_old - ave_y;
      x_new = cos(rand_theta) * x_old + sin(rand_theta) * y_old;
      y_new = cos(rand_theta) * y_old - sin(rand_theta) * x_old;
      %%x_new = x_new + ave_x;
      %%y_new = y_new + ave_y;
      amoeba_image_x{i_seg} = x_new;
      amoeba_image_y{i_seg} = y_new;
    endfor  %%  i_seg
  endif %% amoeba_struct.closed_flag


				% if distractor_flag == 1, then rotate segments
  normalize_density_flag = 0;
  if distractor_flag == 1

    if normalize_density_flag && amoeba_struct.closed_flag
      amoeba_center_x = 0;
      amoeba_center_y = 0;
      amoeba_center_x2 = 0;
      amoeba_center_y2 = 0;
      amoeba_num_x = 0;
      amoeba_num_y = 0;
      for i_seg = 1 : amoeba_struct.num_segments + 1
        amoeba_center_x = amoeba_center_x + sum(amoeba_image_x{i_seg}(:));
        amoeba_center_y = amoeba_center_y + sum(amoeba_image_y{i_seg}(:));
        amoeba_center_x2 = amoeba_center_x2 + sum(amoeba_image_x{i_seg}(:).^2);
        amoeba_center_y2 = amoeba_center_y2 + sum(amoeba_image_y{i_seg}(:).^2);
	amoeba_num_x = amoeba_num_x + length(amoeba_image_x{i_seg}(:));
	amoeba_num_y = amoeba_num_y + length(amoeba_image_y{i_seg}(:));
      endfor
      amoeba_center_x = amoeba_center_x / amoeba_num_x;
      amoeba_center_y = amoeba_center_y / amoeba_num_y;
      amoeba_center_x2 = amoeba_center_x2 / amoeba_num_x;
      amoeba_center_y2 = amoeba_center_y2 / amoeba_num_y;
      amoeba_var_x2 = amoeba_center_x2 - amoeba_center_x * amoeba_center_x;
      amoeba_var_y2 = amoeba_center_y2 - amoeba_center_y * amoeba_center_y;      
    endif
    
    tot_segs = 0;
    while( tot_segs < amoeba_struct.num_segments )
      poisson_num =  fix( -log( rand(1) ) * amoeba_struct.segments_per_distractor * amoeba_struct.num_segments );
      if poisson_num < 1
        poisson_num = 1;
      endif
      poisson_num = min(3,poisson_num);
				%disp(['poisson num: ', num2str(poisson_num), '  spd:' ...
				%     num2str(amoeba_struct.segments_per_distractor), '  ns: ' ,num2str(amoeba_struct.num_segments)]);
      tot_segs = tot_segs + poisson_num;
      if tot_segs > amoeba_struct.num_segments + 1
        poisson_num = tot_segs - amoeba_struct.num_segments - 1;
        tot_segs = amoeba_struct.num_segments + 1;
      endif
      ave_x = 0;
      ave_y = 0;
      for i_seg = tot_segs - poisson_num + 1 : tot_segs
        ave_x = ave_x + mean(amoeba_image_x{i_seg}(:));
        ave_y = ave_y + mean(amoeba_image_y{i_seg}(:));
      endfor
      ave_x = ave_x / poisson_num;
      ave_y = ave_y / poisson_num;
				%rand_theta = (1+rand(1)) *  pi/2; % Vadas/Shawn version
      rand_theta = ( pi / 8 ) + rand(1) * ( 7* pi / 4 );  % Gar/Ben version
      for i_seg = tot_segs - poisson_num + 1 : tot_segs
        x_old = amoeba_image_x{i_seg};
        y_old = amoeba_image_y{i_seg};
        x_old = x_old - ave_x;
        y_old = y_old - ave_y;
        x_new = cos(rand_theta) * x_old + sin(rand_theta) * y_old;
        y_new = cos(rand_theta) * y_old - sin(rand_theta) * x_old;
        x_new = x_new + ave_x;
        y_new = y_new + ave_y;
        amoeba_image_x{i_seg} = x_new;
        amoeba_image_y{i_seg} = y_new;
      endfor
    endwhile

    if normalize_density_flag && amoeba_struct.closed_flag
      clutter_center_x = 0;
      clutter_center_y = 0;
      clutter_center_x2 = 0;
      clutter_center_y2 = 0;
      clutter_num_x = 0;
      clutter_num_y = 0;
      for i_seg = 1 : amoeba_struct.num_segments + 1
        clutter_center_x = clutter_center_x + sum(amoeba_image_x{i_seg}(:));
        clutter_center_y = clutter_center_y + sum(amoeba_image_y{i_seg}(:));
        clutter_center_x2 = clutter_center_x2 + sum(amoeba_image_x{i_seg}(:).^2);
        clutter_center_y2 = clutter_center_y2 + sum(amoeba_image_y{i_seg}(:).^2);
	clutter_num_x = clutter_num_x + length(amoeba_image_x{i_seg}(:));
	clutter_num_y = clutter_num_y + length(amoeba_image_y{i_seg}(:));
      endfor
      clutter_center_x = clutter_center_x / clutter_num_x;
      clutter_center_y = clutter_center_y / clutter_num_y;
      clutter_center_x2 = clutter_center_x2 / clutter_num_x;
      clutter_center_y2 = clutter_center_y2 / clutter_num_y;
      clutter_var_x2 = clutter_center_x2 - clutter_center_x * clutter_center_x;
      clutter_var_y2 = clutter_center_y2 - clutter_center_y * clutter_center_y;          

      expansion_x = sqrt( amoeba_var_x2 /  clutter_var_x2 );
      expansion_y = sqrt( amoeba_var_y2 /  clutter_var_y2 );
      for i_seg = 1 : amoeba_struct.num_segments + 1
	amoeba_image_x{i_seg}(:) = ...
	    expansion_x * ( amoeba_image_x{i_seg}(:) - clutter_center_x ) + clutter_center_x;
 	amoeba_image_y{i_seg}(:) = ...
	    expansion_y * ( amoeba_image_y{i_seg}(:) - clutter_center_y ) + clutter_center_y;
      endfor

    endif % normalize_density_flag

  endif % distractor_flag


  
  %% add thickness
  min_thickness = 1;
  resize_factor = 0.75;
  for i_segment = seg_list
    
    x_thin = amoeba_image_x{i_segment}(:);
    y_thin = amoeba_image_y{i_segment}(:);
    num_pix = length(x_thin(:));
    x_thick = amoeba_struct.min_resize + ...
	rand()*(amoeba_struct.max_resize - amoeba_struct.min_resize);
    x_thick = round(resize_factor * min_thickness*x_thick/2);
    y_thick = amoeba_struct.min_resize + ...
	rand()*(amoeba_struct.max_resize - amoeba_struct.min_resize);
    y_thick = round(resize_factor * min_thickness*y_thick/2);
    amoeba_segment_x = [];
    amoeba_segment_y = [];
    for x_pad = -x_thick : x_thick
      x_fat = repmat(x_pad, [num_pix,1]) + x_thin;
      for y_pad = -y_thick : y_thick
	y_fat = repmat(y_pad, [num_pix,1]) + y_thin;
	amoeba_segment_x = [amoeba_segment_x; x_fat];
	amoeba_segment_y = [amoeba_segment_y; y_fat];      
      endfor
    endfor
    amoeba_image_x{i_segment} = amoeba_segment_x;
    amoeba_image_y{i_segment} = amoeba_segment_y;
    
  endfor  %% i_seg
    

  [amoeba_image_x, amoeba_image_y, amoeba_struct] = ...
      offsetAmoebaSegments(amoeba_struct, amoeba_image_x, amoeba_image_y);
  
