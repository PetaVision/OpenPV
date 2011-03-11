% raw file contains 8 * 384 * 288 32-bit float values between 0.0 and 1.0
% (data order is orientations * columns * rows).  endian issues: this was
% generated on a mac g4.

clear all
NX = 48;
NY = 48;
NO = 18;
N = NX * NY * NO;
DTH  = 180.0/NO;%22.5;
input_path = 'C:\cygwin\home\gkenyon\PetaVision\input\';

image_filename = 'roadrunner_18_tile.raw';
image_filename = [input_path, image_filename];
fid = fopen(image_filename, 'r', 'l');
image_array = fread(fid, N, 'float32');
fclose(fid);

% plot raw input image
figure;
axis([-1 NX -1 NY]);
axis square
box ON
hold on;
ave_image = mean( image_array(:) );
disp(['ave_input = ', num2str(ave_image)]);
max_image = max(image_array(:));
edge_len = sqrt(2)/2;
max_line_width = 3;
image3D = reshape(image_array, [NO, NX, NY]);
for i_theta = 0:NO-1
    delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
    delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
    for i_x = 1:NX
        for i_y = 1:NY
            if image3D(i_theta+1,i_x,i_y) < ave_image
                continue;
            end
            %plot(i_x, i_y, '.k');
            lh = line( [i_x - delta_x, i_x + delta_x]', ...
                [i_y - delta_y, i_y + delta_y]' );
            line_width = 0.05 + ...
                max_line_width * image3D(i_theta+1,i_x,i_y) / max_image;
            set( lh, 'LineWidth', line_width );
            line_color = 1 - image3D(i_theta+1,i_x,i_y) / max_image;
            set( lh, 'Color', line_color*[1 1 1]);
        end
    end
end