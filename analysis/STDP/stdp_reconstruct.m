
function [fh] = stdp_reconstruct( recon_array, NX, NY, plot_title )

fprintf('reconstruct average rate: NX = %d NY = %d\n',NX,NY)
%size(recon_array)
%pause

% plot reconstructed image
if ~any(recon_array(:) ~= 0.0) % any(A) tests whether any of the elements
                               % along various dimensions of an array is a 
                               % nonzero number or is logical 1 (true). 
                               % any ignores entries that are NaN (Not a Number).
    
   fprintf('recon_array has only zero entries: return\n')                            
   return;
end

fh = figure('Name',plot_title);
ave_recon = mean(recon_array(recon_array ~= 0.0));
fprintf('ave_recon = %f ',ave_recon);
if ave_recon < 0
    recon_array = -recon_array;
    ave_recon = -ave_recon;
end
max_recon = max(recon_array(recon_array ~= 0.0));
min_recon = min(recon_array(recon_array ~= 0.0));
fprintf('min_recon = %f max_recon = %f\n',min_recon,max_recon);

% nz_recon = length(find(recon_array ~= 0.0));
% if nz_recon < length(recon_array) / 2
%     min_recon_val = min_recon;
% else
%     min_recon_val = ave_recon;
% end
% if min_recon_val == max_recon
%     min_recon_val = 0;
% end

axis([1 NX 1 NY]);
axis square;
axis tight
box off
axis off
colorbar;
cmap = colormap;
% hold on;

recon2D = reshape( recon_array, [NX, NY] );
%     recon2D = rot90(recon2D);
%     recon2D = 1 - recon2D;
figure('Name','Rate Array ');
imagesc( recon2D' );  % plots recod2D as an image
colorbar
axis square
axis off
%     colormap(gca, gray);
%hold off;
