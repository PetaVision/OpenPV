function [power_array, ...
	  mass_autocorr] = ...
      pvp_autocorr(layer, ...
		   layer_ndx, ...
		   epoch_struct, ...
		   xcorr_struct, ...
		   spike_array )
				% multiply autocorr_arry by ( 1000 / DELTA_T )^2 to convert to joint firing rate
				% with units Hz^2
				% intermediate size of mass_autocorr = [ 2*max_lag+1, length(layer_ndx) ]
  global DELTA_T
  [num_steps, num_neurons] = size(spike_array);
  max_lag = xcorr_struct.max_lag;
  min_freq = xcorr_struct.min_freq;
  max_freq = xcorr_struct.max_freq;
  min_freq_ndx = xcorr_struct.min_freq_ndx;
  max_freq_ndx = xcorr_struct.max_freq_ndx;
  if nargin < 5 || ...
	isempty( layer ) || ...
	isempty( layer_ndx ) || ...
	isempty( epoch_struct ) ||...
	isempty( xcorr_struct ) || ...
	isempty( spike_array )
    error("nargin < 5 in pvp_autocorr");
  endif
				#  if nargin < 4 || isempty( layer_ndx )
				#    layer_ndx = ( 1 : num_neurons );
				#  end%%if
				# if nargin < 5 || isempty(min_freq)
				#    min_freq = 0;
				#  end%%if
				#  if nargin < 6 || isempty(max_freq)
				#    max_freq = 1000/DELTA_T;
				#  end%%if

  num_corr = length(layer_ndx(:));
  power_array = zeros( num_corr, 2 );
  mass_autocorr_mean = 0;
  mass_autocorr_std = 0;
  autocorr_lags = -max_lag : max_lag;
  max_num_corr = floor( (2^22) / ( 2*max_lag+1 ) );
  num_blocks = ceil( num_corr / max_num_corr );
  for i_block = 1 : num_blocks
    start_corr = (i_block - 1) * max_num_corr + 1;
    stop_corr = i_block * max_num_corr;
    stop_corr = min( stop_corr, num_corr);
    corr_ndx = start_corr : stop_corr;
    len_corr = length(corr_ndx);
    mass_autocorr = zeros( 2*max_lag+1, len_corr );

    sum_spikes = sum( spike_array(:, layer_ndx(corr_ndx)), 1 );
    autocorr_mean = sum_spikes / num_steps;
    autocorr_std = sqrt(sum_spikes) / num_steps;
    autocorr_mean = autocorr_mean .^ 2;
    autocorr_std = autocorr_std .^ 2;
    mass_autocorr_mean = mass_autocorr_mean + sum( autocorr_mean );
    mass_autocorr_std = mass_autocorr_std + sum( autocorr_std.^2 );
    for i_lag = 1 : max_lag % lag == pre - post
      autocorr_steps = (1+abs(i_lag)):num_steps;
      shift_train = ...
	  circshift( spike_array(:, layer_ndx(corr_ndx)), [i_lag, 0] );
      mass_autocorr( i_lag + max_lag + 1, :) = ...
          mean( ( spike_array(autocorr_steps, layer_ndx(corr_ndx)) .* ...
		 shift_train(autocorr_steps,:) ), 1 ) - ...
	  autocorr_mean;
    endfor % i_lag
    mass_autocorr(1 : max_lag, :) =  ...
	flipdim( mass_autocorr( max_lag + 2 : 2 * max_lag + 1, : ), 1);
    mass_autocorr = ...
        real( fft( mass_autocorr, [], 1 ) );
    power_array(layer_ndx(corr_ndx), 1) = ...
	squeeze( max( mass_autocorr( min_freq_ndx : max_freq_ndx, : ) ) );
    power_array(layer_ndx(corr_ndx), 2) = ...
	squeeze( mean( mass_autocorr( min_freq_ndx : max_freq_ndx, : ), 1 ) );
    mass_autocorr = mean(mass_autocorr, 2);
  endfor % i_block
  mass_autocorr_mean = mass_autocorr_mean / num_corr;
  mass_autocorr_std = sqrt(mass_autocorr_std) / num_corr;
