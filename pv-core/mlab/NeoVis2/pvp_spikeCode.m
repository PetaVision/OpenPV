function [status_info] = ...
      pvp_spikeCode(image_pathname, ...
		    base_rate, max_rate, refractory_period, ...
		    gray_intensity, max_intensity, ...
		    integration_period, ...
		    rand_state)

  %% Generates a gamma-like distribution of "spike" events for each pixel location,
  %% with the number of events encoding each pixel intensity 
  %% being quasi-proportional to the image intensity at each pixel, subject to the specified refractory period.
  %% Event generation is based on a Poisson-distribution with a hard refractory period.
  %% Intensities equal to the specified gray value yield events at the specified baseline rate.
  %% intensities below zero are truncated at zero, intensities above the specified
  %% maximum intensity are not truncated at the maximum intensity however and produce events at the
  %% rate determined by the stochastic process with the specified refractory period.  
  %% The maximum rate is used only to set the slope of a linear encoding.
  %% Gray pixels are guaranteed to produce events at the specified base rate if possible.

  %%keyboard;
  num_argin = 0;
  num_argin = num_argin + 1;
  if nargin < num_argin
    image_pathname = "~/Pictures/lena/lena.png";
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    base_rate = 50; %% gray level activity
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    max_rate = 100;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    refractory_period = 0.001;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    gray_intensity = 128;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    max_intensity = 255;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    integration_period = 0.030;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    rand_state = [];
  endif

  setenv('GNUTERM', 'x11');

  if (exist("rand_state", "var") && ~isempty(rand_state))
    rand("state", rand_state);
  else
    rand_state = rand("state");
  endif

  input_image = imread(image_pathname);
  image_size = size(input_image);
  if isrgb(input_image)
    gray_image = rgb2gray(input_image);
  elseif ndims(input_image) == 2
    gray_image = input_image;
  elseif ndims(input_image) == 3
    gray_image = squeeze(mean(input_image,3));
  else
    gray_image = squeeze(input_image(:,:,1));
  endif

  plot_gray_image = 0;
  if plot_gray_image
    figure;
    imagesc(gray_image);
    colormap(gray);
    axis off;
    axis image;
  endif

  status_info = struct;
  if integration_period == 0
    status_info.eventCount_array = double(gray_image);
    return;
  endif
    

  %%keyboard;

  %% calculate renormalized_base_rate
  %% Time between spikes is refractory_period plus exponentially-distributed random variable
  %% Call renormalized_base_rate the mean value of exp.-distributed variable.
  %% refractory_period + 1/renormalized_base_rate = 1/base_rate
  %% renormalized_base_rate = 1 / ( (1/base_rate) - refractory_period )

  renormalized_base_rate = base_rate / ( 1 - base_rate * refractory_period );
  %% rand_max = exp( -base_rate * refractory_period / ( 1 - base_rate * refractory_period ) );
  renormalized_max_rate = max_rate / ( 1 - base_rate * refractory_period );

  %%keyboard;
  if (max_intensity > gray_intensity)
    gray_array = double(gray_image);
    rate_array = ...
	renormalized_base_rate + ...
	(gray_array - gray_intensity) * ...
	(renormalized_max_rate - renormalized_base_rate) / (max_intensity - gray_intensity);
  else
    rate_array = repmat(renormalized_rate, image_size(1:2));
  endif

  num_isi = ceil(3 * max_rate * integration_period);
  min_rate = 1 / num_isi;
  rate_array(rate_array < min_rate) = min_rate;
  %%rate_array(rate_array > max_rate) = max_rate;
  tau_array = repmat(1./rate_array, [ 1, 1, num_isi ] );
  
  %%keyboard;
  isi_arg = rand([image_size(1:2), num_isi]);
  isi_arg(isi_arg == 0) = exp(-1);
  isi_array = -(tau_array) .* log(isi_arg) + refractory_period;
  
  %% isi_arg = rand_max * rand([image_size(1:2), num_isi]);
  %% isi_arg(isi_arg == 0) = exp(-1);
  %% isi_array = -(tau_array) .* log(isi_arg);
  eventTime_3D = zeros([image_size(1:2), num_isi+1]);
  eventTime_3D(:,:,2:num_isi+1) = cumsum( isi_array, 3);
  eventTime_3D(eventTime_3D > integration_period) = -1;
  [max_eventTime, eventCount_array] = ...
      max(eventTime_3D - repmat(integration_period, size(eventTime_3D)), [], 3);
  eventCount_array = eventCount_array - 1;  %% count >= 1
  
  plot_eventCount = 0;
  if plot_eventCount
    figure;
    eventCount_image = uint8(eventCount_array);
    imagesc(eventCount_image);
    colormap(gray);
  endif

  hist_eventCount = 0;
  if hist_eventCount
    figure;
    [event_hist, event_bins] = hist(eventCount_array(:), 10);
    bar(event_bins, event_hist);
  endif

  status_info.eventCount_array = eventCount_array;
  status_info.rand_state = rand_state;



endfunction %% pvp_spikeCode