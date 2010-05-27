function [amoeba_image_x, amoeba_image_y] = amoebaSegments2(amoeba_struct, distractor_flag)

if nargin == 1
    distractor_flag = 0;
end
gap_offest = amoeba_struct.delta_segment * rand(1);
list_segments = [1; round(gap_offest:amoeba_struct.delta_segment:amoeba_struct.num_phi)'];
list_segments = [list_segments, circshift(list_segments,-1)];
list_segments(end, 2) = amoeba_struct.num_phi;
delta_gap = ...
    [0; round( rand(amoeba_struct.num_segments, 1) * (amoeba_struct.max_gap - amoeba_struct.min_gap) + amoeba_struct.min_gap )];
list_segments(:,1) = list_segments(:,1) + delta_gap;
list_segments = min( list_segments, amoeba_struct.num_phi );
list_segments = max( list_segments, 1 );
fourier_coef = amoeba_struct.fourier_ratio .* randn(amoeba_struct.num_fourier, 1);
fourier_coef2 = repmat( amoeba_struct.fourier_ratio .* fourier_coef, [1, amoeba_struct.num_phi]);
fourier_phase = (pi/2) * rand(amoeba_struct.num_fourier, 1);
fourier_phase2 = repmat(fourier_phase, [1, amoeba_struct.num_phi]);
fourier_term = fourier_coef2 .* cos( amoeba_struct.fourier_arg2 + fourier_phase2);
fourier_sum = sum(fourier_term, 1);
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

seg_list = 1 : amoeba_struct.num_segments + 1;
amoeba_image_x = cell(amoeba_struct.num_segments + 1,1);
amoeba_image_y = cell(amoeba_struct.num_segments + 1,1);

% extract x,y pairs for each segment
i_seg_ndx = 0;
for i_seg = seg_list
    i_seg_ndx = i_seg_ndx + 1;
    [amoeba_segment_x, amoeba_segment_y] = pol2cart( ...
        amoeba_struct.fourier_arg( list_segments(i_seg,1):list_segments(i_seg,2) ), ...
        r_phi( list_segments(i_seg,1):list_segments(i_seg,2) ) );
    amoeba_image_x{i_seg_ndx} = amoeba_segment_x;
    amoeba_image_y{i_seg_ndx} = amoeba_segment_y;
end    


% if distractor_flag == 1, then rotate segments
if distractor_flag == 1
    tot_segs = 0;
    while( tot_segs < amoeba_struct.num_segments )
        poisson_num =  fix( -log( rand(1) ) * amoeba_struct.segments_per_distractor * amoeba_struct.num_segments );
        if poisson_num < 1
            poisson_num = 1;
        end
	poisson_num = min(3,poisson_num);
	%disp(['poisson num: ', num2str(poisson_num), '  spd:' ...
	%     num2str(amoeba_struct.segments_per_distractor), '  ns: ' ,num2str(amoeba_struct.num_segments)]);
        tot_segs = tot_segs + poisson_num;
        if tot_segs > amoeba_struct.num_segments + 1
            poisson_num = tot_segs - amoeba_struct.num_segments - 1;
            tot_segs = amoeba_struct.num_segments + 1;
        end
        ave_x = 0;
        ave_y = 0;
        for i_seg = tot_segs - poisson_num + 1 : tot_segs
            ave_x = ave_x + mean(amoeba_image_x{i_seg}(:));
            ave_y = ave_y + mean(amoeba_image_y{i_seg}(:));
        end
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
        end
    end
end

    
% fix boundary conditions (use mirror BCs)
%  added conditional
%  || distractor_flag == 1
% to eliminate difference in global density of clutter vs amoebas,
% especiaolly at corners...
offset_x = 2 * ( rand(1) - 0.5 ) * ( fix(amoeba_struct.image_rect_size/2) - ...
    ( distractor_flag == 0 || distractor_flag == 1 ) * outer_diameter );
offset_y = 2 * ( rand(1) - 0.5 ) * ( fix(amoeba_struct.image_rect_size/2) - ...
    ( distractor_flag == 0 || distractor_flag == 1 ) * outer_diameter );
i_seg_ndx = 0;
for i_seg = seg_list
    i_seg_ndx = i_seg_ndx + 1;
    amoeba_segment_x = amoeba_image_x{i_seg_ndx};
    amoeba_segment_y = amoeba_image_y{i_seg_ndx};
    amoeba_segment_x = ...
        amoeba_segment_x + fix(amoeba_struct.image_rect_size/2) + offset_x;
    amoeba_segment_x = ...
        amoeba_segment_x .* ((amoeba_segment_x >= 1) & (amoeba_segment_x <= amoeba_struct.image_rect_size)) + ...
        (2 * amoeba_struct.image_rect_size - amoeba_segment_x ) .* (amoeba_segment_x > amoeba_struct.image_rect_size) + ...
        (1 - amoeba_segment_x) .* (amoeba_segment_x < 1);
    amoeba_segment_y = ...
        amoeba_segment_y + fix(amoeba_struct.image_rect_size/2) + offset_y;
    amoeba_segment_y = ...
        amoeba_segment_y .* ((amoeba_segment_y >= 1) & (amoeba_segment_y <= amoeba_struct.image_rect_size)) + ...
        (2 * amoeba_struct.image_rect_size - amoeba_segment_y ) .* (amoeba_segment_y > amoeba_struct.image_rect_size) + ...
        (1 - amoeba_segment_y) .* (amoeba_segment_y < 1);
    amoeba_image_x{i_seg_ndx} = amoeba_segment_x;
    amoeba_image_y{i_seg_ndx} = amoeba_segment_y;
end
