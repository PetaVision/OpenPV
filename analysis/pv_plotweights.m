function [weight3D] = pv_plotweights(vmem_array)
% plot "weights" (typically after turning on just one neuron)
global NK NO NX NY DTH 

figure;
weight3D = reshape(vmem_array(2,:), [NK, NO, NX, NY]);
i_x0 = fix(NX/2) + 1;
i_y0 = fix(NY/2) + 1;
%weight3D(i_k0, i_theta0 + 1, i_x0, i_y0) = 0;%-70.0;
%weight3D = weight3D - min_weight;
ave_weight = mean(weight3D(:));
weight3D = weight3D - ave_weight;
min_weight = min(weight3D(:));
max_weight = max(weight3D(:));
if abs(min_weight) > abs(max_weight)
    weight3D = -weight3D;
elseif min_weight == max_weight
    return
end
ave_weight = mean(weight3D(:));
% min_weight = min(weight3D(:));
max_weight = max(weight3D(:));
edge_len = sqrt(2)/2;
max_line_width = 3;
axis([0 NX -NY 0]);
axis square;
if ~isOctave
	box ON
end
hold on;
if NO > 1
    i_theta0 = 0.5;
    % i_k0 = 0 + 1;
    %k0 = i_k + (i_theta0)*NK + (i_x0-1) * NO * NK + (i_y0-1) * NX * NO * NK;
    delta_x = edge_len * ( cos(i_theta0 * DTH * pi / 180 ) );
    delta_y = edge_len * ( sin(i_theta0 * DTH * pi / 180 ) );
    lh = line( [i_x0 - delta_x, i_x0 + delta_x]', ...
        [0 - (i_y0 - delta_y), 0 - (i_y0 + delta_y)]' );
    line_width = max_line_width;
    set( lh, 'LineWidth', line_width );
    set( lh, 'Color', [1 0 0]);
    for i_k = 1:NK
        for i_theta = 0:NO-1
            delta_x = edge_len * ( cos((i_theta+0.5) * DTH * pi / 180 ) );
            delta_y = edge_len * ( sin((i_theta+0.5) * DTH * pi / 180 ) );
            for i_x = 1:NX
                for i_y = 1:NY
                    if weight3D(i_k,i_theta+1,i_x,i_y) < ave_weight
                        continue;
                    end
                    lh = line( [i_x - delta_x, i_x + delta_x]', ...
                        [0 - (i_y - delta_y), 0 - (i_y + delta_y)]' );
                    line_width = 0.05 + ...
                        max_line_width * (weight3D(i_k,i_theta+1,i_x,i_y) - ave_weight) / ...
                        (max_weight - ave_weight);
                    set( lh, 'LineWidth', line_width );
                    line_color = (weight3D(i_k,i_theta+1,i_x,i_y) - ave_weight) / ...
                        (max_weight - ave_weight);
                    set( lh, 'Color', (1-line_color) * [1 1 1]);
                end
            end
        end
    end
else
    axis([0 NX 0 NY]);
    imagesc( flipud(squeeze( weight3D(1,1,:,:) ) ) ); colormap('gray');
end
