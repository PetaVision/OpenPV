function [mass_xcorr, ...
	  mass_autocorr, ...
	  mass_xcorr_mean, ...
	  mass_xcorr_std, ...
	  mass_xcorr_lags, ...
	  xcorr_array, ...
	  xcorr_dist, ...
	  min_freq_ndx, ...
	  max_freq_ndx ] = ...
      pvp_xcorr2( pre_spike_train, ...
		 post_spike_train, ...
		 max_lag, ...
		 pre_ndx, ...
		 size_pre, ...
		 post_ndx, ...
		 size_post, ...
		 is_auto, ...
		 min_freq, ...
		 max_freq, ...
		 xcorr_flag )
% multiply xcorr_arry by ( 1000 / DELTA_T )^2 to convert to joint firing rate
% with units Hz^2
  global DELTA_T
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
  if nargin < 9 || isempty(min_freq)
    min_freq = 0;
  end%%if
  if nargin < 10 || isempty(max_freq)
    max_freq = 1000/DELTA_T;
  end%%if
  if nargin < 11 || isempty(xcorr_flag)
    xcorr_flag = 1;
  end%%if

  xcorr_array = zeros( num_pre, num_post, 2 );
  mass_xcorr = zeros( 2 * max_lag + 1, 1 );
  mass_xcorr_mean = 0;
  mass_xcorr_std = 0;
  mass_xcorr_lags = -max_lag : max_lag;
  mass_autocorr = zeros( 2 * max_lag + 1, 1 );
  mass_autocorr_mean = 0;
  mass_autocorr_std = 0;
  xcorr_dist = zeros( num_pre, num_post );

  [pre_row_index, pre_col_index, pre_f_index] = ...
      ind2sub( size_pre, pre_ndx );
  [post_row_index, post_col_index, post_f_index]  = ...
      ind2sub( size_post, post_ndx );
  freq_vals = 1000*(1/DELTA_T)*(0:2*max_lag)/(1+2*max_lag);
  min_freq_ndx = find(freq_vals >= min_freq, 1,'first');
  max_freq_ndx = find(freq_vals <= max_freq, 1,'last');
  disp([ 'min_freq_ndx = ', num2str(min_freq_ndx) ]);
  disp([ 'max_freq_ndx = ', num2str(max_freq_ndx) ]);
%  i_plot = 0;
%  xcorr_figs = [];
%% for each i_post, compute xcorr for all i_pre in parallel
for i_post = 1 : num_post
    disp_interval = fix(num_post / 10);
    if mod(i_post, disp_interval) == 1
        disp(['pvp_xcorr2: i_post = ', num2str(i_post)]);
    end
    post_row_tmp = repmat( post_row_index(i_post), [num_pre, 1] );
    post_col_tmp = repmat( post_col_index(i_post), [num_pre, 1] );
    xcorr_dist( :, i_post ) = ...
        sqrt( ( ( post_row_tmp - reshape(pre_row_index, size(post_row_tmp)) ).^2 ) + ...
        ( ( post_col_tmp - reshape(pre_col_index, size(post_col_tmp)) ).^2 ) );
    sum_pre = sum( pre_spike_train, 1 );
    sum_post = sum( post_spike_train(:,i_post), 1 );
    num_pre_steps = size( pre_spike_train, 1 );
    num_post_steps = size( post_spike_train, 1 );
    xcorr_mean_tmp = sum_pre * sum_post / (num_pre_steps * num_post_steps);
    xcorr_std_tmp = ...
        sqrt( xcorr_mean_tmp ) .* sqrt( (1./(sum_pre+(sum_pre~=0))) + (1/(sum_post+(sum_post~=0))) );
    mass_xcorr_mean = mass_xcorr_mean + ...
        sum( xcorr_mean_tmp, 1 );
    mass_xcorr_std = mass_xcorr_std + ...
        sum( xcorr_std_tmp .^ 2 );
    fft_xcorr_array = zeros( 2 * max_lag + 1, num_pre);
    for i_lag = -max_lag : max_lag % lag == pre - post
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
        xcorr_array_tmp = ...
            mean( ( pre_spike_train(xcorr_post_steps,:) .* post_shift_train2(xcorr_post_steps,:) ), 1 ) - xcorr_mean_tmp;
        fft_xcorr_array( i_lag + max_lag + 1, :) = full(xcorr_array_tmp);
        mass_xcorr( i_lag + max_lag + 1 ) = mass_xcorr( i_lag + max_lag + 1 ) + ...
            sum( xcorr_array_tmp, 2 );
        if is_auto %% 
            mass_autocorr( i_lag + max_lag + 1 ) = mass_autocorr( i_lag + max_lag + 1 ) + ...
               xcorr_array_tmp(i_post);
        end%%if
    end%%for % i_lag
    if is_auto %% subtract autocorr at zero lag
        mass_xcorr( max_lag + 1 ) = mass_xcorr( max_lag + 1 ) - ...
            xcorr_array_tmp(i_post);
        mass_autocorr( max_lag + 1 ) = mass_autocorr( max_lag + 1 ) - ...
            xcorr_array_tmp(i_post);
    end%%if  %% is_auto

%    i_plot = i_plot + num_post;

%    if mod( i_plot, plot_interval ) == -1
%        i_pre = fix( num_pre * rand(1) ) + 1;
%        disp(['i_pre = ', num2str(i_pre), ', i_post = ', num2str(i_post)]);
%        plot_title = ['xcorr(', num2str(i_pre), ',', num2str(i_post), ')'];
%        fig_tmp = figure;
%        set(fig_tmp, 'Name', plot_title);
%        hold on
%        xcorr_figs = [xcorr_figs; fig_tmp];
%        xcorr_tmp = zeros( 2 * max_lag + 1, 1 );
%        xcorr_tmp(max_lag + 1 : 2*max_lag + 1)  = ...
%            fft_xcorr_array(max_lag + 1 : 2*max_lag + 1, i_pre  );
%        xcorr_tmp(1 : max_lag) =  ...
%            flipdim( fft_xcorr_array( max_lag + 2 : 2*max_lag + 1, i_pre ), 1 );
%        plot( (-max_lag : max_lag)*DELTA_T, xcorr_tmp, '-k');
%        xcorr_std_tmp = xcorr_std_tmp( i_pre );
%        lh = line( [-max_lag, max_lag]*DELTA_T, ...
%            [ xcorr_std_tmp xcorr_std_tmp ] );
%        lh = line( [-max_lag, max_lag]*DELTA_T, ...
%            [ -xcorr_std_tmp -xcorr_std_tmp ] );
%    end%%if % i_plot
    
    if xcorr_flag == 1
        fft_xcorr_array = ...
            real( fft( fft_xcorr_array, [], 1 ) );
        xcorr_array(:, i_post, 1) = ...
            squeeze( max( fft_xcorr_array( min_freq_ndx : max_freq_ndx, : ) ) );
        xcorr_array(:, i_post, 2) = ...
            squeeze( mean( fft_xcorr_array( min_freq_ndx : max_freq_ndx, : ), 1 ) );
    end%%if  %xcorr_flag

end%%for % i_post

mass_xcorr = mass_xcorr / ( num_pre * num_post );
%mass_xcorr_flip = ...
%    mass_xcorr( max_lag + 2 : 2 * max_lag + 1, 1 );
%mass_xcorr_flip = ...
%    flipdim( mass_xcorr_flip, 1 );
%mass_xcorr(1 : max_lag, 1) =  ...
%    mass_xcorr_flip;

mass_xcorr_mean = mass_xcorr_mean / ( num_pre * num_post );
mass_xcorr_std = sqrt( mass_xcorr_std / ( num_pre * num_post ) );

%fft_mass_xcorr = ...
%    real( fft( mass_xcorr - mass_xcorr_mean, [], 1 ) );

mass_autocorr = mass_autocorr / ( num_post );
%mass_autocorr_flip = ...
%    mass_autocorr( max_lag + 2 : 2 * max_lag + 1, 1 );
%mass_autocorr_flip = ...
%    flipdim( mass_autocorr_flip, 1 );
%mass_autocorr(1 : max_lag, 1) =  ...
%    mass_autocorr_flip;

