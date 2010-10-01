
function [power_array, ...
	  mass_power] = ...
      pvp_power( power_win_size, ...
		layer_ndx, ...
		min_freq, ...
		max_freq )
% computes power of each spike train in SPIKE_ARRY{LAYER}
% averages over segments of size
% returns mean and peak power between min and max freq
  global DELTA_T
  global SPIKE_ARRAY
  global LAYER
  [num_steps, num_neurons] = size(SPIKE_ARRAY);
  if nargin < 1 || isempty(power_win_size)
    power_win_size = num_steps;
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

  power_inc = ceil( power_win_size );
  power_w_type = 3; % rectangle
  num_windows = fix( length( num_steps ) / power_win_size );
  freq_vals = 1000*(1/DELTA_T)*(0:power_win_size-1)/power_win_size;
  min_ndx = find(freq_vals >= min_freq, 1,'first');
  max_ndx = find(freq_vals <= max_freq, 1,'last');
  disp([ 'min_freq_ndx = ', num2str(min_freq_ndx) ]);
  disp([ 'max_freq_ndx = ', num2str(max_freq_ndx) ]);
  mass_power = zeros( power_win_size, num_corr );
  for i_trial = 1 : num_windows
    power_start = ( i_trial - 1 ) * power_inc;
    power_stop = power_start + power_win_size - 1;
    mass_power = mass_power + ...
        fft( ( SPIKE_ARRAY( power_start : power_stop, layer_ndx ) ), [], 1 );
  end%%for
  mass_power = mass_power / num_windows;
  mass_power = mass_power .* conj( mass_power );
  power_array{LAYER, 1} = max(mass_power(min_ndx:max_ndx,:));
  power_array{LAYER, 2} = mean(mass_power(min_ndx:max_ndx,:));
  mass_power = mean( mass_power, 2 );
