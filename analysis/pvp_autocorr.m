function [power_array, ...
	  mass_autocorr] = ...
      pvp_autocorr( max_lag, ...
		   layer_ndx, ...
		   min_freq, ...
		   max_freq )
% multiply autocorr_arry by ( 1000 / DELTA_T )^2 to convert to joint firing rate
% with units Hz^2
% requires making a copy of SPIKE_ARRAY{LAYER}
% intermediate size of mass_autocorr = [ 2*max_lag+1, length(layer_ndx) ]
  global DELTA_T
  global SPIKE_ARRAY
  global LAYER
  [num_steps, num_neurons] = size(SPIKE_ARRAY{LAYER});
  if nargin < 1 || isempty(max_lag)
    max_lag = num_steps / 2;
  end%%if
  if nargin < 2 || isempty( layer_ndx )
    layer_ndx = ( 1 : num_neurons );
  end%%if
 if nargin < 3 || isempty(min_freq)
    min_freq = 0;
  end%%if
  if nargin < 4 || isempty(max_freq)
    max_freq = 1000/DELTA_T;
  end%%if

  num_corr = length(layer_ndx(:));
  power_array = zeros( num_corr, 2 );
  mass_autocorr = zeros( 2*max_lag+1, num_corr );
  autocorr_mean = 0;
  autocorr_std = 0;
  autocorr_lags = -max_lag : max_lag;

  sum_spikes = sum( SPIKE_ARRAY{LAYER}(:, layer_ndx), 1 );
  autocorr_mean = sum_spikes / num_steps;
  autocorr_std = sqrt(sum_spikes) / num_steps;
  autocorr_mean = autocorr_mean .^ 2;
  autocorr_std = autocorr_std .^ 2;
  mass_autocorr_mean =  sum( sum_spikes(:) ) / num_corr;
  mass_autocorr_std = sqrt( sum( sum_spikes(:) ) ) / num_corr;
  mass_autocorr_mean = mass_autocorr_mean .^ 2;
  mass_autocorr_std = mass_autocorr_std .^ 2;
  for i_lag = 1 : max_lag % lag == pre - post
    autocorr_steps = (1+abs(i_lag)):num_steps;
    shift_train = circshift( SPIKE_ARRAY{LAYER}(:, layer_ndx), [i_lag, 0] );
    mass_autocorr( i_lag + max_lag + 1, :) = ...
        mean( ( SPIKE_ARRAY{LAYER}(autocorr_steps, layer_ndx) .* ...
	       shift_train(autocorr_steps,:) ), 1 ) - ...
	autocorr_mean;
  end%%for % i_lag
  clear shift_train
  mass_autocorr(1 : max_lag, :) =  ...
      flipdim( mass_autocorr( max_lag + 2 : 2 * max_lag + 1, : ), 1);
  mass_autocorr = ...
            real( fft( mass_autocorr, [], 1 ) );
  freq_vals = 1000*(1/DELTA_T)*(0:2*max_lag)/(1+2*max_lag);
  min_freq_ndx = find(freq_vals >= min_freq, 1,'first');
  max_freq_ndx = find(freq_vals <= max_freq, 1,'last');
  disp([ 'min_freq_ndx = ', num2str(min_freq_ndx) ]);
  disp([ 'max_freq_ndx = ', num2str(max_freq_ndx) ]);
  power_array(:, 1) = ...
      squeeze( max( mass_autocorr( min_freq_ndx : max_freq_ndx, : ) ) );
  power_array(:, 2) = ...
      squeeze( mean( mass_autocorr( min_freq_ndx : max_freq_ndx, : ), 1 ) );
  mass_autocorr = mean(mass_autocorr, 2);
