
function [fh] = pv_reconstruct( recon_array, plot_title, fh )

global NK NO NX NY DTH

%size(recon_array)
%pause

% plot reconstructed image
if ~any(recon_array(:) ~= 0.0) % any(A) tests whether any of the elements
                               % along various dimensions of an array is a 
                               % nonzero number or is logical 1 (true). 
                               % any ignores entries that are NaN (Not a Number).
    return;
end
if ~exist('fh','var')          % tests if 'fh' is a variable in the workspace
                               % returns 0 if 'fh' does not exists
    fh = figure('Name',plot_title);
else
    set(fh, 'Name', plot_title);
end
ave_recon = mean(recon_array(recon_array ~= 0.0));
if ave_recon < 0
    recon_array = -recon_array;
    ave_recon = -ave_recon;
end
max_recon = max(recon_array(recon_array ~= 0.0));
min_recon = min(recon_array(recon_array ~= 0.0));
nz_recon = length(find(recon_array ~= 0.0));
if nz_recon < length(recon_array) / 2
    min_recon_val = min_recon;
else
    min_recon_val = ave_recon;
end
if min_recon_val == max_recon
    min_recon_val = 0;
end
edge_len = sqrt(2)/2.5;
if (NO==1)
    edge_len = 1/sqrt(2)/2;
end
max_line_width = 2.5;
axis([1 NX 1 NY]);
axis square;
axis tight
box off
axis off
if (~isOctave)
    box ON
end
colorbar;
cmap = colormap;
hold on;
if NO > 1 &&  max_recon > 0
    recon_ndx = find(recon_array ~= 0.0);
    for recon_index = 1 : length(recon_ndx)
        i_recon = recon_ndx(recon_index);
        recon_val = recon_array(i_recon);
        if recon_val <= min_recon_val
            continue;
        else
            recon_val = ( recon_val - min_recon_val ) / ( max_recon - min_recon_val + (max_recon == min_recon_val) );
        end
        [i_k, i_theta, i_x, i_y] = ind2sub( [NK, NO, NX, NY ], i_recon );
        clear i_k
        i_theta = i_theta - 0.5;
        delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
        delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
        lh = line( [i_x - delta_x, i_x + delta_x]', ...
            [NY - (i_y - delta_y), NY-(i_y + delta_y)]' );
        line_width = 0.05 + max_line_width * recon_val;
        set( lh, 'LineWidth', line_width );
%         line_color = 1 - recon_val;
        line_color = recon_val;
%         set( lh, 'Color', line_color*[1 1 1]);
        set( lh, 'Color', cmap(floor(line_color*63)+1,:,:,:));         
    end
elseif max_recon > 0
    recon2D = reshape( recon_array, [NX, NY] );
%     recon2D = rot90(recon2D);
%     recon2D = 1 - recon2D;
    imagesc( recon2D );  % plots recod2D as an image
%     colormap(gca, gray);
end
hold off;
pause