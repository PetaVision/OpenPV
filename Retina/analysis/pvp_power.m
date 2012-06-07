
function [power_array, ...
	  mass_power] = ...
      pvp_power(layer, ...
                layer_ndx, ...
		epoch_stuct, ...
		xcorr_struct, ...
		spike_array )
% computes power of each spike train in SPIKE_ARRY{LAYER}
% averages over segments of size
% returns mean and peak power between min and max freq
  global DELTA_T
  # global spike_array
  # global layer
  [num_steps, num_neurons] = size(spike_array);
  max_lag = xcorr_struct.max_lag;
  min_freq = xcorr_struct.min_freq;
  max_freq = xcorr_struct.max_freq;
  min_freq_ndx = xcorr_struct.min_freq_ndx;
  max_freq_ndx = xcorr_struct.max_freq_ndx;
  power_win_size = xcorr_struct.power_win_size;
  if nargin < 5 || ...
	isempty( layer ) || ...
	isempty( layer_ndx ) || ...
	isempty( epoch_struct ) ||...
	isempty( xcorr_struct ) || ...
	isempty( spike_array )
    error("nargin < 5 in pvp_autocorr");
  endif
  # if nargin < 1 || isempty(power_win_size)
  #   power_win_size = num_steps;
  # end%%if
  # if nargin < 2 || isempty( layer_ndx )
  #   layer_ndx = ( 1 : num_neurons );
  # end%%if
  # if nargin < 3 || isempty(min_freq)
  #   min_freq = 0;
  # end%%if
  # if nargin < 4 || isempty(max_freq)
  #   max_freq = 1000/DELTA_T;
  # end%%if

  num_corr = length(layer_ndx(:));
  power_array = zeros( num_corr, 2 );

  power_inc = ceil( power_win_size );
  power_w_type = 3; % rectangle
  num_windows = fix( length( num_steps ) / power_win_size );
  mass_power = zeros( power_win_size, num_corr );
  for i_trial = 1 : num_windows
    power_start = ( i_trial - 1 ) * power_inc;
    power_stop = power_start + power_win_size - 1;
    mass_power = mass_power + ...
        fft( ( spike_array( power_start : power_stop, layer_ndx ) ), [], 1 );
  end%%for
  mass_power = mass_power / num_windows;
  mass_power = mass_power .* conj( mass_power );
  power_array{layer, 1} = max(mass_power(min_ndx:max_ndx,:));
  power_array{layer, 2} = mean(mass_power(min_ndx:max_ndx,:));
  mass_power = mean( mass_power, 2 );
