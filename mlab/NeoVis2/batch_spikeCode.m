function [status_info] = ...
      batch_spikeCode(input_dir, output_dir, ...
		      base_rate, max_rate, refractory_period, ...
		      gray_intensity, max_intensity, ...
		      integration_period, num_procs)

  num_argin = 0;
  num_argin = num_argin + 1;
  if nargin < num_argin
    input_dir = "~/Pictures/amoeba/256/4/t/";
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    output_dir = "~/Pictures/spikeCode/amoeba/4/";
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
    integration_period = 0.100;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    num_procs = 1;
  endif

  image_type = ".png";
  input_path = ...
      [input_dir, '*', image_type];
  input_pathnames = glob(input_path);
  num_images = size(input_pathnames,1);
  disp(['num_images = ', num2str(num_images)]);

  mean_hist = zeros(1, num_images);
  std_hist = zeros(1, num_images);
  std_mean_hist = zeros(1, num_images);

  


  if num_procs > 1
    [status_info] = ...
	parcellfun(num_procs, @pvp_spikeCode, ...
		   input_pathnames, ...
		   num2cell(repmat(base_rate, num_images, 1)), ...
		   num2cell(repmat(max_rate, num_images, 1)), ...
		   num2cell(repmat(refractory_period, num_images, 1)), ...
		   num2cell(repmat(gray_intensity, num_images, 1)), ...
		   num2cell(repmat(max_intensity, num_images, 1)), ...
		   num2cell(repmat(integration_period, num_images, 1)), ...
		   "UniformOutput", false);
  else
    [status_info] = ...
	cellfun(@pvp_spikeCode, ...
		input_pathnames, ...
		num2cell(repmat(base_rate, num_images, 1)), ...
		num2cell(repmat(max_rate, num_images, 1)), ...
		num2cell(repmat(refractory_period, num_images, 1)), ...
		num2cell(repmat(gray_intensity, num_images, 1)), ...
		num2cell(repmat(max_intensity, num_images, 1)), ...
		num2cell(repmat(integration_period, num_images, 1)), ...
		"UniformOutput", false);
  endif
