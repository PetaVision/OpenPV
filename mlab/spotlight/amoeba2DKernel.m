function [amoeba_info] = ...
      amoeba2DKernel(amoeba_struct)


  num_phi = amoeba_struct.num_phi;
  delta_phi = 2 * pi / num_phi;

  %% make random RF curve
  RF_arg = (0:(num_phi-1)) * delta_phi;
  RF_arg2 = ...
      repmat((0:(amoeba_struct.num_RF-1))',[1, num_phi]) .* ...
      repmat( RF_arg, [amoeba_struct.num_RF, 1] );
				% exponentially decaying ampitudes produce amoebas that are too regular...
				% amoeba_struct.RF_ratio = exp(-(0:(amoeba_struct.num_RF-1))/amoeba_struct.num_RF)';
  amoeba_struct.RF_ratio = ones(amoeba_struct.num_RF, 1);
  RF_coef = amoeba_struct.RF_ratio .* randn(amoeba_struct.num_RF, 1);
  RF_coef = RF_coef .* 2.^(amoeba_struct.RF_amp);
  RF_coef2 = repmat( amoeba_struct.RF_ratio .* RF_coef, [1, num_phi]);
  RF_phase = (2*pi) * rand(amoeba_struct.num_RF, 1);
  %% could use pi/2 since amlitudes are normally distributed about zero
  RF_phase2 = repmat(RF_phase, [1, num_phi]);
  RF_term = RF_coef2 .* cos( RF_arg2 + RF_phase2);
  RF_sum = sum(RF_term, 1);
  RF_max = max(RF_sum(:));
  RF_min = min(RF_sum(:));
  RF_ave = mean(RF_sum(:));

  %% set size randomly within specified range
  outer_diameter = ...
      ( rand(1) * ( 1 - amoeba_struct.target_outer_min ) + amoeba_struct.target_outer_min ) ...
      * amoeba_struct.target_outer_max * fix(min(amoeba_struct.image_rect_size)/2);
  inner_diameter = ...
      ( rand(1) * ( 1 - amoeba_struct.target_inner_min ) + amoeba_struct.target_inner_min ) ...
      * amoeba_struct.target_inner_max * outer_diameter;
  r_phi = inner_diameter + ...
      ( RF_sum - RF_min ) * ...
      ( outer_diameter - inner_diameter ) / (RF_max - RF_min );
  amoeba_struct.outer_diameter = outer_diameter;
  amoeba_struct.inner_diameter = inner_diameter;
 
    %% make a (semi-)closed amoeba
  [amoeba_outline_x, amoeba_outline_y] = ...
      pol2cart(RF_arg, r_phi);
  
  [amoeba_outline_x, amoeba_outline_y, amoeba_struct] = ...
      offsetAmoebaSegments2(amoeba_struct, amoeba_outline_x, amoeba_outline_y);

  
  %% fill
  amoeba_center_col = mean(amoeba_outline_x(:));
  amoeba_center_row = mean(amoeba_outline_y(:));
  amoeba2D_mask = zeros(amoeba_struct.image_rect_size);
  outline_ndx = sub2ind(amoeba_struct.image_rect_size, fix(amoeba_outline_y), fix(amoeba_outline_x));
  amoeba2D_mask(outline_ndx) = 1;
  [amoeba2D_mask, amoeba2D_ndx] = bwfill(amoeba2D_mask, amoeba_center_col, amoeba_center_row, 4);
  amoeba_info = struct;
  amoeba_info.amoeba_struct = amoeba_struct;
  amoeba_info.amoeba2D_mask = amoeba2D_mask;
  amoeba_info.amoeba2D_ndx = amoeba2D_ndx;
  amoeba_info.amoeba_outline_x = amoeba_outline_x;
  amoeba_info.amoeba_outline_y = amoeba_outline_y;



