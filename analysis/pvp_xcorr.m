function [xcorr_array, xcorr_std, xcorr_dist, xcorr_mean, xcorr_lags, xcorr_figs] = ...
    pvp_xcorr( pre_spike_train, ...
    post_spike_train, ...
    max_lag, ...
    pre_ndx, size_pre, post_ndx, size_post, ...
    is_auto, plot_interval )
% multiply xcorr_arry by ( 1000 / dt )^2 to convert to joint firing rate
% with units Hz^2
global dt
[num_steps, num_pre] = size(pre_spike_train);
if nargin < 2 || isempty(post_spike_train)
    post_spike_train = pre_spike_train;
    if isempty( is_auto )
        is_auto = 1;
    end%%if
end%%if
[num_steps, num_post] = size(post_spike_train);
if nargin < 3 || isempty(max_lag)
    max_lag = num_steps / 2;
end%%if
if nargin < 4 || isempty( pre_ndx )
    pre_ndx = ( 1 : num_pre );
end%%if
if nargin < 5 || isempty( size_pre )
    size_pre = [ num_pre, 1, 1 ];
end%%if
if nargin < 6 || isempty( post_ndx )
    post_ndx = ( 1 : num_post );
end%%if
if nargin < 7 || isempty( size_post )
    size_post = [ num_post, 1, 1 ];
end%%if
if nargin < 8 && isempty(is_auto)
    is_auto = 0;
end%%if
if nargin < 9 || isempty(plot_interval)
    plot_interval = num_pre * num_post;
end%%if

xcorr_array = zeros( num_pre, num_post, max_lag + 1 );
xcorr_std = zeros( num_pre, num_post );
xcorr_dist = zeros( num_pre, num_post );
xcorr_mean = zeros( num_pre, num_post );
xcorr_lags = -max_lag : max_lag;
xcorr_figs = zeros( floor( num_pre * num_post / plot_interval ), 1 );
[pre_row_index, pre_col_index, pre_f_index] = ...
    ind2sub( size_pre, pre_ndx );
[post_row_index, post_col_index, post_f_index]  = ...
    ind2sub( size_post, post_ndx );

i_plot = 0;
j_plot = 1;
for i_post = 1 : num_post
    post_row_tmp = repmat( post_row_index(i_post), [1, num_pre] );
    post_col_tmp = repmat( post_col_index(i_post), [1, num_pre] );
    xcorr_dist( :, i_post ) = ...
        sqrt( ( post_row_tmp - pre_row_index ).^2 + ...
        ( post_col_tmp - pre_col_index ).^2 );
    sum_pre = sum( pre_spike_train, 1 );
    sum_post = sum( post_spike_train(:,i_post), 1 );
    num_pre_steps = size( pre_spike_train, 1 );
    num_post_steps = size( post_spike_train, 1 );
    mean_corr2 = sum_pre * sum_post / (num_pre_steps * num_post_steps);
    xcorr_mean( :, i_post ) = mean_corr2;
    xcorr_std( :, i_post ) = ...
        sqrt( mean_corr2 ) .* sqrt( (1./(sum_pre+(sum_pre~=0))) + (1/(sum_post+(sum_post~=0))) );
    for i_lag = 0 : max_lag  % lag == pre - post
        if i_lag < 0
            if num_pre_steps >= num_post_steps
                xcorr_post_steps = 1:(num_post_steps-abs(i_lag));
            else
                xcorr_post_steps = 1:(num_pre_steps-abs(i_lag));
            end%%if
        else
            if num_pre_steps >= num_post_steps
                xcorr_post_steps = (1+abs(i_lag)):num_post_steps;
            else
                xcorr_post_steps = (1+abs(i_lag)):num_pre_steps;
            end%%if
        end%%if
        post_shift_train = circshift( post_spike_train, [i_lag, 0] );
        post_shift_train2 = repmat( post_shift_train(:, i_post), 1, num_pre );
        xcorr_array( :, i_post, i_lag + 1 ) = ...
            mean( ( pre_spike_train(xcorr_post_steps,:) .* post_shift_train2(xcorr_post_steps,:) ), 1 ) - mean_corr2;
    end%%for % i_lag
end%%for % i_post


for i_plot = plot_interval : plot_interval : num_pre * num_post
    [i_pre, i_post] = ind2sub( [num_pre, num_post], i_plot);
    disp(['i_pre = ', num2str(i_pre), ': i_post = ', num2str(i_post)]);
    plot_title = [' xcorr: i_pre = ',int2str(i_pre), ': i_post = ', num2str(i_post)];
    fig_tmp = figure('Name',plot_title);
    hold on
    xcorr_figs = [xcorr_figs; fig_tmp];
    xcorr_tmp = zeros( 2 * max_lag + 1, 1 );
    xcorr_tmp(max_lag + 1 : end)  = xcorr_array( i_pre, i_post, : );
    xcorr_tmp(1 : max_lag) =  ...
        flipdim( xcorr_array( i_post, i_pre, 2 : end ), 3 );
    plot( (-max_lag : max_lag)*dt, xcorr_tmp, '-k');
    xcorr_std_tmp = xcorr_std( i_pre, i_post );
    lh = line( [-max_lag, max_lag]*dt, ...
        [ xcorr_std_tmp xcorr_std_tmp ] );
    lh = line( [-max_lag, max_lag]*dt, ...
        [ -xcorr_std_tmp -xcorr_std_tmp ] );
end%%for % i_plot