clear all
n_time_steps = 1;
NX = 48;
NY = 48;
NO = 9;
N = NX * NY * NO;
DTH  = 20;%22.5;
output_path = 'C:\cygwin\home\user\myzucker\output\';

events_filename = 'events.bits';
events_filename = [output_path, events_filename];
fid = fopen(events_filename, 'r', 'native');
event_mask = fread(fid, [N/8, n_time_steps], 'uchar');
fclose(fid);

spike_array = sparse(n_time_steps, N, 0);
for i_bit = 1:8
    event_ndx = find(bitget(event_mask,i_bit));
    event_ndx = ( event_ndx - 1 ) * 8 + i_bit;
    [spike_time, spike_id] = ind2sub([n_time_steps, N], event_ndx);
    spike_array(spike_time, spike_id) = 1;
end

ave_rate = 1000 * sum(spike_array(:)) / ( N * n_time_steps );

figure;
[spike_time, spike_id, spike_val] = find(spike_array);
plot(spike_time, spike_id, '.k');

figure;
rate_array = 1000 * sum(spike_array,1) / ( n_time_steps );
max_rate = max(rate_array(:));
edge_len = sqrt(2)/2;
max_line_width = 2;
axis([-1 NX -1 NY]);
box ON
hold on;
rate3D = reshape(rate_array, [NO, NX, NY]);
for i_theta = 0:NO-1
    delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
    delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
    for i_x = 1:NX
        for i_y = 1:NY
            if rate3D(i_theta+1,i_x,i_y) < ave_rate
                continue;
            end
            lh = line( [i_x - delta_x, i_x + delta_x]', ...
                [i_y - delta_y, i_y + delta_y]' );
            line_width = 0.05 + ...
                max_line_width * rate3D(i_theta+1,i_x,i_y) / max_rate;
            set( lh, 'LineWidth', line_width );
            line_color = 1 - rate3D(i_theta+1,i_x,i_y) / max_rate;
            set( lh, 'Color', line_color*[1 1 1]);
        end
    end
end

