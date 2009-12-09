
function [fh] = stdp_reconstruct( recon_array, NX, NY, plot_title, fh )


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

axis([1 NX 1 NY]);
axis square;
axis tight
box off
axis off
colorbar;
cmap = colormap;
hold on;

recon2D = reshape( recon_array, [NX, NY] );
%     recon2D = rot90(recon2D);
%     recon2D = 1 - recon2D;
imagesc( recon2D' );  % plots recod2D as an image
%     colormap(gca, gray);

hold off;
