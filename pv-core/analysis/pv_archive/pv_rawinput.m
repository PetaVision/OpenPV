function [] = pv_rawinput();

    global N NX NY DTH n_time_steps begin_step input_path

    NO=1; % we know the input file has no orientation
    N=NX*NY;

    % plot raw input image
    input_filename = 'input.bin';
    input_filename = [input_path, input_filename];
    if exist(input_filename,'file')
        fid = fopen(input_filename, 'r', 'native');
        input_array = fread(fid, N, 'float');
        fclose(fid);
    else
	disp(['File not found.'];
	return;
    end

    figure;
    axis([-1 NX -1 NY]);
    axis square;
    %box ON % Octave doesn't like box
    hold on;
    ave_input = sum( input_array(:) ) / ( N );
    disp(['ave_input = ', num2str(ave_input)]);
    max_input = max(input_array(:));
    min_input = min(input_array(:));
    edge_len = sqrt(2)/2;
    max_line_width = 3;
    input3D = reshape(input_array, [NO, NX, NY]);
    for i_theta = 0:NO-1
        delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
        delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
        for i_x = 1:NX
            for i_y = 1:NY
                if input3D(i_theta+1,i_x,i_y) <= ave_input
                    continue;
                end
                %plot(i_x, i_y, '.k');
                lh = line( [i_x - delta_x, i_x + delta_x]', ...
                    [i_y - delta_y, i_y + delta_y]' );
                line_width = 0.05 + ...
                    max_line_width * ...
                    ( input3D(i_theta+1,i_x,i_y) - min_input ) / ...
                    ( max_input - min_input );
                set( lh, 'LineWidth', line_width );
                line_color = 1 - ...
                    ( input3D(i_theta+1,i_x,i_y) - min_input ) / ...
                    ( max_input - min_input );
                set( lh, 'Color', line_color*[1 1 1]);
            end
        end
    end
end
