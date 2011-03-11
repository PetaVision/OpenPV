
function [amoeba_image] = amoeba2D3(amoeba_struct, j_trial)

  global mean_amoeba_xy std_amoeba_xy mean_distractor_xy std_distractor_xy
  %%global j_trial num_trials

  target_type = amoeba_struct.target_type;
  target_id = amoeba_struct.target_id;
  num_fourier = amoeba_struct.num_fourier;
  
  num_objects = amoeba_struct.num_targets + amoeba_struct.num_distractors;
  amoeba_image = cell( 2*num_objects, 2 );
  amoeba_struct.delta_phi = amoeba_struct.range_phi / amoeba_struct.num_phi;
  amoeba_struct.fourier_arg = (0:(amoeba_struct.num_phi-1)) * amoeba_struct.delta_phi;
  amoeba_struct.fourier_arg2 = ...
      repmat((0:(amoeba_struct.num_fourier-1))',[1, amoeba_struct.num_phi]) .* ...
      repmat( amoeba_struct.fourier_arg, [amoeba_struct.num_fourier, 1] );
				% exponentially decaying ampitudes produce amoebas that are too regular...
				% amoeba_struct.fourier_ratio = exp(-(0:(amoeba_struct.num_fourier-1))/amoeba_struct.num_fourier)';
  amoeba_struct.fourier_ratio = ones(amoeba_struct.num_fourier, 1);
  amoeba_struct.delta_segment = amoeba_struct.num_phi / amoeba_struct.num_segments;

%make targets & distractors
for i_amoeba = 1 : amoeba_struct.num_targets
  if target_type == 0
    [amoeba_image_x, amoeba_image_y, amoeba_struct] = ...
	amoebaSegments3(amoeba_struct, 0);
  elseif target_type == 1
    [amoeba_image_x, amoeba_image_y, amoeba_struct] = ...
	MNISTSegments(amoeba_struct, 1, target_id, target_type);
  endif
  
    amoeba_image{i_amoeba, 1} = amoeba_image_x;
    amoeba_image{i_amoeba, 2} = amoeba_image_y;

    %% gather 1st and 2nd order stats
    mean_amoeba_x = 0;
    mean_amoeba_y = 0;
    std_amoeba_x = 0;
    std_amoeba_y = 0;
    num_amoeba_xy = 0;
    for segment = 1:size(amoeba_image_x,1)
      mean_amoeba_x = mean_amoeba_x + sum(amoeba_image_x{segment});
      mean_amoeba_y = mean_amoeba_y + sum(amoeba_image_y{segment});
      std_amoeba_x = std_amoeba_x + sum(amoeba_image_x{segment}.^2);
      std_amoeba_y = std_amoeba_y + sum(amoeba_image_y{segment}.^2);
      num_amoeba_xy = num_amoeba_xy + length(amoeba_image_x{segment});
    endfor %% segment
    trial_ndx = (j_trial-1) * amoeba_struct.num_targets + i_amoeba;
    mean_amoeba_xy(trial_ndx, 1) = mean_amoeba_x / num_amoeba_xy;
    mean_amoeba_xy(trial_ndx, 2) = mean_amoeba_y / num_amoeba_xy;
    std_amoeba_xy(trial_ndx, 1) = ...
	sqrt( (std_amoeba_x / num_amoeba_xy) - (mean_amoeba_x / num_amoeba_xy).^2 );
    std_amoeba_xy(trial_ndx, 1) = ...
	sqrt( (std_amoeba_y / num_amoeba_xy) - (mean_amoeba_y / num_amoeba_xy).^2 );

endfor %% i_amoeba

%make distractors
for i_amoeba = amoeba_struct.num_targets + 1 : 2*num_objects
    [amoeba_image_x, amoeba_image_y, amoeba_struct] = ...
	amoebaSegments3(amoeba_struct, 1);
    amoeba_image{i_amoeba, 1} = amoeba_image_x;
    amoeba_image{i_amoeba, 2} = amoeba_image_y;

    mean_distractor_x = 0;
    mean_distractor_y = 0;
    std_distractor_x = 0;
    std_distractor_y = 0;
    num_distractor_xy = 0;
    for segment = 1:size(amoeba_image_x,1)
      mean_distractor_x = mean_distractor_x + sum(amoeba_image_x{segment});
      mean_distractor_y = mean_distractor_y + sum(amoeba_image_y{segment});
      std_distractor_x = std_distractor_x + sum(amoeba_image_x{segment}.^2);
      std_distractor_y = std_distractor_y + sum(amoeba_image_y{segment}.^2);
      num_distractor_xy = num_distractor_xy + length(amoeba_image_x{segment});
    endfor %% segment
    trial_ndx = ...
	(j_trial-1) * ( 2 * num_objects - amoeba_struct.num_targets ) ...
	+ i_amoeba - amoeba_struct.num_targets;
    mean_distractor_xy(trial_ndx, 1) = mean_distractor_x / num_distractor_xy;
    mean_distractor_xy(trial_ndx, 2) = mean_distractor_y / num_distractor_xy;
    std_distractor_xy(trial_ndx, 1) = ...
	sqrt( (std_distractor_x / num_distractor_xy) - (mean_distractor_x / num_distractor_xy).^2 );
    std_distractor_xy(trial_ndx, 1) = ...
	sqrt( (std_distractor_y / num_distractor_xy) - (mean_distractor_y / num_distractor_xy).^2 );


endfor %% i_amoeba
