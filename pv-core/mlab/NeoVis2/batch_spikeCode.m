function [status_info] = ...
      batch_spikeCode(input_dir, output_dir, ...
		      base_rate, max_rate, refractory_period, ...
		      gray_intensity, max_intensity, ...
		      integration_period, ...
		      max_images, num_procs)
  more off;
  num_argin = 0;
  num_argin = num_argin + 1;
  if nargin < num_argin
    input_dir = "~/Pictures/AnimalDB/Targets/"; %% "~/Pictures/MNIST/6/"; %% "~/Pictures/amoeba/256/4/t/";
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    output_dir = "~/Pictures/spikeCode/AnimalDB/Targets/0msec/";
  endif
  mkdir(output_dir);
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
    integration_period = 0; %% 0.025; %% 0.05; %% 0.1; %% 0.2; %% 0; %% 
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    max_images = 0;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin
    num_procs = 4;
  endif

  %%keyboard;
  setenv('GNUTERM', 'x11');
  image_type = ".jpg";
  
  %% path to generic image processing routines
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);
  
  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);

  input_path = ...
      [input_dir, '*', image_type];
  input_pathnames = glob(input_path);
  num_images = size(input_pathnames,1);
  if max_images > 0 && num_images > max_images
    num_images = max_images;
    input_pathnames = input_pathnames(1:num_images,1);
  endif
  disp(['num_images = ', num2str(num_images)]);

  mean_hist = zeros(1, num_images);
  std_hist = zeros(1, num_images);
  std_mean_hist = zeros(1, num_images);

  %%keyboard;
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

  %%keyboard;
  max_count = 0;
  min_count = 10000;
  for i_image = 1 : num_images
    eventCount_array = status_info{i_image}.eventCount_array;
    max_count = max(max(eventCount_array(:)), max_count);
    min_count = min(min(eventCount_array(:)), min_count);
  endfor
  disp(["max_count = ", num2str(max_count)]);
  disp(["min_count = ", num2str(min_count)]);

  mkdir(output_dir);
  for i_image = 1 : num_images
    input_filename = strFolderFromPath(input_pathnames{i_image});
    input_rootname = strRemoveExtension(input_filename);
    spikeCode_filename = [output_dir, input_rootname, ".png"];
    eventCount_array = status_info{i_image}.eventCount_array;
    disp(["size(eventCount_array) = ", num2str(size(eventCount_array))]);
    image_tmp = uint8(floor(255*(eventCount_array - min_count) / ...
		      (max_count - min_count + ((max_count - min_count)==0))));
    disp(["max(image) = ", num2str(max(image_tmp(:)))]);
    disp(["min(image) = ", num2str(min(image_tmp(:)))]);
    imwrite(image_tmp, spikeCode_filename, "png");
  endfor