function [sparse_corr] = pv_xcorr( pre_spike_train, post_spike_train, ...
    maxlag, min_ndx,  max_ndx  )

% if nargin == 2
%     post_spike_train = pre_spike_train;
% end
[num_steps, num_pre] = size(pre_spike_train);
[num_steps, num_post] = size(post_spike_train);
sparse_corr = cell(2,1);
sparse_corr{1,1} = zeros( num_pre, num_post );
sparse_corr{2,1} = zeros( num_pre, num_post );
sparse_corr_lags = zeros( 2 * maxlag + 1, num_pre );
for i_post = 1 : num_post
    for i_lag = -maxlag : maxlag
        if i_lag < 0
            xcorr_steps = 1:(num_steps-abs(i_lag));
        else
            xcorr_steps = (1+abs(i_lag)):num_steps;
        end
        mean_pre = mean( pre_spike_train(xcorr_steps,:), 1 );
        mean_post = mean( post_spike_train(xcorr_steps,:), 1 );
        mean_corr2 = mean_pre * mean_post(i_post);
        post_shift_train = circshift( post_spike_train, [i_lag, 0] );
        post_shift_train2 = repmat( post_shift_train(:, i_post), 1, num_pre );
        sparse_corr_lags(i_lag + maxlag + 1, : ) = ...
            mean( ( pre_spike_train(xcorr_steps,:) .* post_shift_train2(xcorr_steps,:) ), 1 ) - mean_corr2;
    end
    sparse_corr{1}(:, i_post ) = sparse_corr_lags(maxlag + 1, :);
    sparse_corr_lags = real( fft( sparse_corr_lags ) );
    sparse_corr{2}(:, i_post ) = mean( sparse_corr_lags( min_ndx:max_ndx,:) );
    if mod(i_post, 100) == 0
        disp(['i_post = ', num2str(i_post)]);
    end
end
