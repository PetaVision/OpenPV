function [] = pv_recon3d( spike_array )

global N NO NX NY NK DTH n_time_steps begin_step rate_array

ave_rate = 1000 * sum(spike_array(:)) / ( N * n_time_steps );

real_steps = n_time_steps - begin_step + 1

%if (real_steps > 20) real_steps = 20 % don't try more than 20 timsteps

% plot reconstructed image
figure
max_rate = max(rate_array(:));
edge_len = sqrt(2)/2;
max_line_width = 3;
axis([-1 NX -1 NY]);
axis square;

if (~isOctave)
    box ON
end

hold on;

for time=1:real_steps
    rate3D = reshape(spike_array(time,:), [NK, NO, NX, NY]);
    for i_nk = 1:NK
        for i_theta = 0:NO-1
            delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
            delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
            for i_x = 1:NX
                for i_y = 1:NY
                    if rate3D(i_nk, i_theta+1,i_x,i_y) < 1.0
                        continue;
                    end
                    lh = line( [i_x - delta_x, i_x + delta_x]', ...
                        [NX-(i_y-delta_y),NX-(i_y+delta_y)]', [time,time]' );
                    line_width = 1.5;
                    set( lh, 'LineWidth', line_width );
                    line_color = 1.0;
                    set( lh, 'Color', line_color*[0 0 1]);
                end
            end
        end
    end
end %time

