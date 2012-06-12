function [mass_xcorr, ...
	  mass_autocorr, ...
	  mass_xcorr_mean, ...
	  mass_xcorr_std, ...
	  xcorr_array, ...
	  xcorr_dist ] = ...
      pvp_xcorr2(pre_spike_train, ...
		 post_spike_train, ...
		 xcorr_struct, ...
		 pre_ndx, ...
		 size_pre, ...
		 post_ndx, ...
		 size_post, ...
		 is_auto, ...
		 xcorr_flag )
				% multiply xcorr_arry by ( 1000 / DELTA_T )^2 to convert to joint firing rate
				% with units Hz^2
  global DELTA_T
  [num_steps, num_pre] = size(pre_spike_train);
  if nargin < 2 || isempty(post_spike_train)
    post_spike_train = pre_spike_train;
    if isempty( is_auto )
      is_auto = 1;
    endif
  endif
  [num_steps, num_post] = size(post_spike_train);
  if nargin < 3 || isempty(xcorr_struct)
    max_lag = num_steps / 2;
    min_freq = 0;
    max_freq = 1000/DELTA_T;
    freq_vals = 1000*(1/DELTA_T)*(0:2*max_lag)/(1+2*max_lag);
    min_freq_ndx = find(freq_vals >= min_freq, 1,'first');
    max_freq_ndx = find(freq_vals <= max_freq, 1,'last');
    disp([ 'min_freq_ndx = ', num2str(min_freq_ndx) ]);
    disp([ 'max_freq_ndx = ', num2str(max_freq_ndx) ]);
  else
    max_lag = xcorr_struct.max_lag;
    min_freq_ndx = xcorr_struct.min_freq_ndx;
    max_freq_ndx = xcorr_struct.max_freq_ndx;
  endif
  if nargin < 4 || isempty( pre_ndx )
    pre_ndx = ( 1 : num_pre );
  endif
  if nargin < 5 || isempty( size_pre )
    size_pre = [ num_pre, 1, 1 ];
  endif
  if nargin < 6 || isempty( post_ndx )
    post_ndx = ( 1 : num_post );
  endif
  if nargin < 7 || isempty( size_post )
    size_post = [ num_post, 1, 1 ];
  endif
  if nargin < 8 && isempty(is_auto)
    is_auto = 0;
  endif
  if nargin < 9 || isempty(xcorr_flag)
    xcorr_flag = 1;
  endif

DEBUG_FLAG = 0;
if DEBUG_FLAG
  disp(["size(pre_spike_train) = ", num2str(size(pre_spike_train))]);
  disp(["size(post_spike_train) = ", num2str(size(post_spike_train))]);
  disp(["xcorr_struct = ", num2str(cell2mat(struct2cell(xcorr_struct)))]);
  disp(["size(pre_ndx) = ", num2str(size(pre_ndx))]);
  disp(["size_pre = ", num2str(size_pre)]);
  disp(["size(post_ndx) = ", num2str(size(post_ndx))]);
  disp(["size_post = ", num2str(size_post)]);
  disp(["is_auto = ", num2str(is_auto)]);
  disp(["xcorr_flag = ", num2str(xcorr_flag)]);
endif


  if xcorr_flag
    xcorr_array = zeros( num_pre, num_post, 2 );
    xcorr_dist = zeros( num_pre, num_post );
  endif
  mass_xcorr = zeros( 2 * max_lag + 1, 1 );
  mass_xcorr_mean = 0;
  mass_xcorr_std = 0;
  mass_xcorr_lags = -max_lag : max_lag;
  mass_autocorr = zeros( 2 * max_lag + 1, 1 );
  mass_autocorr_mean = 0;
  mass_autocorr_std = 0;

  [pre_row_index, pre_col_index, pre_f_index] = ...
      ind2sub( size_pre, pre_ndx );
  [post_row_index, post_col_index, post_f_index]  = ...
      ind2sub( size_post, post_ndx );
				%  i_plot = 0;
				%  xcorr_figs = [];
  %% for each i_post, compute xcorr for all i_pre 
  num_pre_steps = size( pre_spike_train, 1 );
  num_post_steps = size( post_spike_train, 1 );
  for i_post = 1 : num_post
    disp_interval = 1; %fix(num_post / 10);
    if mod(i_post, disp_interval) == 1
      disp(['pvp_xcorr2: i_post = ', num2str(i_post)]);
    endif
    if xcorr_flag
      post_row_tmp = repmat( post_row_index(i_post), [num_pre, 1] );
      post_col_tmp = repmat( post_col_index(i_post), [num_pre, 1] );
      xcorr_dist( :, i_post ) = ...
          sqrt( ( ( post_row_tmp - reshape(pre_row_index, size(post_row_tmp)) ).^2 ) + ...
	       ( ( post_col_tmp - reshape(pre_col_index, size(post_col_tmp)) ).^2 ) );
    endif
    sum_pre = sum( pre_spike_train, 1 );
    sum_post = sum( post_spike_train(:,i_post), 1 );
    xcorr_mean_tmp = sum_pre * sum_post / (num_pre_steps * num_post_steps);
    xcorr_std_tmp = ...
        sqrt( xcorr_mean_tmp ) .* ...
	sqrt( (1./(sum_pre+(sum_pre==0))) + (1/(sum_post+(sum_post==0))) );
    mass_xcorr_mean = mass_xcorr_mean + ...
        sum( xcorr_mean_tmp, 2 );
    mass_xcorr_std = mass_xcorr_std + ...
        sum( xcorr_std_tmp .^ 2 );
    
    if xcorr_flag == 1
      fft_xcorr_array = zeros( 2 * max_lag + 1, num_pre);
    endif
    for i_lag = -max_lag : max_lag % lag == pre - post
      if i_lag < 0
        if num_pre_steps >= num_post_steps
          xcorr_post_steps = 1:(num_post_steps-abs(i_lag));
        else
          xcorr_post_steps = 1:(num_pre_steps-abs(i_lag));
        endif
      else
        if num_pre_steps >= num_post_steps
          xcorr_post_steps = (1+abs(i_lag)):num_post_steps;
        else
          xcorr_post_steps = (1+abs(i_lag)):num_pre_steps;
        endif
      endif
      post_shift_train = circshift( post_spike_train, [i_lag, 0] );
      post_shift_train2 = repmat( post_shift_train(:, i_post), 1, num_pre );
      xcorr_array_tmp = ...
          mean( ( pre_spike_train(xcorr_post_steps,:) .* ...
		 post_shift_train2(xcorr_post_steps,:) ), 1 ) - xcorr_mean_tmp;
      if xcorr_flag == 1
        fft_xcorr_array( i_lag + max_lag + 1, :) = full(xcorr_array_tmp);
      endif
      mass_xcorr( i_lag + max_lag + 1 ) = ...
	  mass_xcorr( i_lag + max_lag + 1 ) + ...
          sum( xcorr_array_tmp, 2 );
      if is_auto  
	mass_autocorr( i_lag + max_lag + 1 ) = ...
	    mass_autocorr( i_lag + max_lag + 1 ) + ...
	    xcorr_array_tmp(1,i_post);
	if i_lag == 0
          mass_xcorr( i_lag + max_lag + 1 ) = ...
	      mass_xcorr( i_lag + max_lag + 1 ) - ...
	      xcorr_array_tmp(1,i_post);
	  mass_autocorr( i_lag + max_lag + 1 ) = ...
	      mass_autocorr( i_lag + max_lag + 1 ) - ...
	      xcorr_array_tmp(1,i_post);
	endif %% i_lag == 0
      endif %% is_auto
    endfor % i_lag
    if is_auto %% subtract autocorr at zero lag
      mass_xcorr( max_lag + 1 ) = ...
	  0.5 * mass_xcorr( max_lag + 2 ) + ...
          0.5 * mass_xcorr( max_lag );
      mass_autocorr( max_lag + 1 ) = ...
	  0.5 * mass_autocorr( max_lag + 2 ) + ...
          0.5 * mass_autocorr( max_lag ); 
    endif  %% is_auto
    
    if xcorr_flag == 1
      fft_xcorr_array = ...
          real( fft( fft_xcorr_array, [], 1 ) );
      xcorr_array(:, i_post, 1) = ...
          squeeze( max( fft_xcorr_array( min_freq_ndx : max_freq_ndx, : ) ) );
      xcorr_array(:, i_post, 2) = ...
          squeeze( mean( fft_xcorr_array( min_freq_ndx : max_freq_ndx, : ), 1 ) );
    endif  %xcorr_flag

  endfor % i_post

  if is_auto
    mass_xcorr_norm = num_pre * (num_pre - 1)  %% num_pre == num_post
  else
    mass_xcorr_norm = num_pre * num_post;
  endif
  mass_xcorr = mass_xcorr / mass_xcorr_norm;

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

  if is_auto
    mass_autocorr = mass_autocorr / ( num_post );  %% num_pre == num_post
  endif
				%mass_autocorr_flip = ...
				%    mass_autocorr( max_lag + 2 : 2 * max_lag + 1, 1 );
				%mass_autocorr_flip = ...
				%    flipdim( mass_autocorr_flip, 1 );
				%mass_autocorr(1 : max_lag, 1) =  ...
				%    mass_autocorr_flip;

