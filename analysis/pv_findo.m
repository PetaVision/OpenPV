function [ indices ] = pv_findo( spikes, xy, timesteps, ny, nx, no, nk )

% Given X,Y position, return a list of the single maximally stimulated,
% orientation-selective neurons at each X,Y position.

rates = sum(spikes,1); % total spikes over all time
resh = reshape(rates, [nk*no, ny*nx]);

indices=[];
use_all_orientations_curvatures=0;
if use_all_orientations_curvatures
    for i=1:size(xy,1)
        pixels = (xy-1)*no*nk+1; % calc the indicies
        indices = [indices, pixels(i) + 0:(no*nk-1)];
    end
else % pull out the max orientation

    % Find which orientation is maximal for each position
    [y,max_xy]=max(resh,[],1);
    indices = (xy'-1)*no*nk + max_xy( xy' ); % calc the indicies
end

end % end function
