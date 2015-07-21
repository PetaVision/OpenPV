function [sparse_corr] = pv_xcorr_ave( pre_spike_train, maxlag, post_spike_train )

if nargin == 2
    post_spike_train = pre_spike_train;
end
[num_steps, num_pre] = size(pre_spike_train);
[num_steps, num_post] = size(post_spike_train);
sparse_corr = zeros( 2*maxlag + 1, 1 );
for i_lag = -maxlag : maxlag
    if i_lag < 0
        xcorr_steps = 1:(num_steps-abs(i_lag));
    else
        xcorr_steps = (1+abs(i_lag)):num_steps;
    end
    mean_pre = mean( pre_spike_train(xcorr_steps,:), 1 );
    mean_post = mean( post_spike_train(xcorr_steps,:), 1 );
    post_shift_train = circshift( post_spike_train, [i_lag, 0] );
    for i_shift = 1 : num_post
        post_shift_train2 = repmat( post_shift_train(:, i_shift), 1, num_pre );
        mean_corr2 = mean_pre * mean_post(i_shift);
        sparse_corr(i_lag + maxlag + 1) = sparse_corr(i_lag + maxlag + 1) + ...
            sum( ( mean( pre_spike_train(xcorr_steps,:) .* post_shift_train2(xcorr_steps,:), 1 ) - ...
            mean_corr2 ) ./ ( mean_corr2 + (mean_corr2 == 0) ), 2 );
    end
%             if mod(i_lag, 10) == 0
%                 disp(['i_lag = ', num2str(i_lag)]);
%             end
end
sparse_corr = sparse_corr / (num_pre * num_post);
