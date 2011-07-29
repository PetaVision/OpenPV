function [] = commands3(image_size, target_type, num_trials)

  if nargin < 1 || ~exist("image_size") || isempty(image_size)
    image_size = [256 256]; %
  endif
  if nargin < 2 || ~exist("target_type") || isempty(target_type)
    target_type = 0; %% radial frequency pattern + clutter
    %% target_type = 1; %% MNIST digits (resized) + clutter
  endif
  if nargin < 3 || ~exist("num_trials") || isempty(num_trials)
    num_trials = 1; %
  endif

  global image_dim
  image_dim = image_size;
  addpath("../MNIST", "-begin");

  disp(["num_trials = ", num2str(num_trials)]);

  global plot_amoeba2D fh_amoeba2D
  plot_amoeba2D = 0;
  setenv('GNUTERM', 'x11');
  if plot_amoeba2D
    fh_amoeba2D = figure;
  endif
  
  global machine_path  %% assume we're in the /workspace-*/PetaVision/mlab/MNIST directory
  machine_path = '../../../../MATLAB/figures/amoeba/'; %%'/Users/gkenyon/MATLAB/captcha/';
  if ~exist( 'machine_path', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', machine_path); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif

  global amoeba_file_path
  amoeba_file_path =  [ machine_path, num2str(image_dim(1)), '_png/']
  if ~exist( 'amoeba_file_path', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', amoeba_file_path ); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif

  global image_file_path image_file_name

  if target_type == 0
  %% set number of fourier components
  %%fourC = [2 4 6 8];
    fourC = [4];
    min_target_ndx = 1;
    max_target_ndx = length(fourC);
    target_list = fourC;
  elseif target_type == 1
    %% set range of MNIST targets
    target_list = [6];
    min_target_ndx = 1;
    max_target_ndx = length(target_list);
  endif

  global nz_image
  nz_image = zeros(3, num_trials);
  nz_image_cell = cell(length(target_list), 1);

  global mean_amoeba_xy std_amoeba_xy mean_distractor_xy std_distractor_xy
  std_xy_hist = cell(length(target_list), 2);

  for target_ndx = min_target_ndx : max_target_ndx
%%    nfour = fourC(target_list(target_ndx));
    %%disp(["target_ndx = ", num2str(target_ndx)]);
    %%disp(["target_list(target_ndx) = ", num2str(target_list(target_ndx))]);
    image_file_path = [ amoeba_file_path, num2str(target_list(target_ndx)), '/']
    if ~exist( 'image_file_path', 'dir')
      [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', image_file_path ); 
      if SUCCESS ~= 1
        error(MESSAGEID, MESSAGE);
      endif
    endif
    
    for j_trial = 1:num_trials
      getAmoebaStats4(j_trial, target_list(target_ndx), target_type, 4, num_trials);
      if mod(j_trial, fix(num_trials/20)) == 0
        disp(num2str(j_trial));
      endif
    endfor
    
    nz_image_cell{target_ndx} = nz_image;
    figure
    [nz_image_hist, nz_image_bins] = ...
        hist( nz_image_cell{target_ndx}(1,:) );
    bh = bar( nz_image_bins, nz_image_hist, 0.8);
    set(bh, 'EdgeColor', [1 0 0]);
    set(bh, 'FaceColor', [1 0 0]);
    hold on;
    nz_image_hist = ...
        hist( nz_image_cell{target_ndx}(2,:), nz_image_bins );
    bh = bar( nz_image_bins, nz_image_hist, 0.6);
    set(bh, 'EdgeColor', [0 0 1]);
    set(bh, 'FaceColor', [0 0 1]);
    
    figure
    std_amoeba = ...
	sqrt( std_amoeba_xy(:,1).^2 + std_amoeba_xy(:,2).^2 );
    [std_amoeba_hist, std_amoeba_bins] = ...
        hist( std_amoeba );
    std_amoeba_hist = std_amoeba_hist / sum(std_amoeba_hist(:));
    bh = bar( std_amoeba_bins, std_amoeba_hist, 0.8);
    set(bh, 'EdgeColor', [1 0 0]);
    set(bh, 'FaceColor', [1 0 0]);
    hold on;
    std_distractor = ...
	sqrt( std_distractor_xy(:,1).^2 + std_distractor_xy(:,2).^2 );
    std_distractor_hist = ...
        hist( std_distractor, std_amoeba_bins );
    std_distractor_hist = std_distractor_hist / sum(std_distractor_hist(:));
    bh = bar( std_amoeba_bins, std_distractor_hist, 0.6);
    set(bh, 'EdgeColor', [0 0 1]);
    set(bh, 'FaceColor', [0 0 1]);

    std_xy_hist{target_list(target_ndx), 1} = std_amoeba_hist; 
    std_xy_hist{target_list(target_ndx), 2} = std_distractor_hist; 

    
  endfor 

  amoeba_hist_filename = ...
      [image_file_path, 'amoeba_hist', '.mat']
  save('-mat', amoeba_hist_filename, 'nz_image_cell');

  for i = 1:size(nz_image_cell,1)
    mean_nz_tmp = mean(nz_image_cell{i}(1,:));
    std_nz_tmp = std(nz_image_cell{i}(1,:));
    mean_nz_tmp2 = mean(nz_image_cell{i}(2,:));
    std_nz_tmp2 = std(nz_image_cell{i}(2,:));
    disp( ['mean_nz(', num2str(i), ',:) =', num2str(mean_nz_tmp), ...
	   ' +/- ', num2str(std_nz_tmp), ', ', num2str(mean_nz_tmp2), ...
	   ' +/- ', num2str(std_nz_tmp2)] );
  endfor
  
  std_xy_hist_filename = ...
      [image_file_path, 'std_xy_hist', '.mat']
  save('-mat', std_xy_hist_filename, 'std_xy_hist');

  if exist("std_xy_hist")
  for target_ndx = min_target_ndx:max_target_ndx
    if ~exist("std_xy_hist{target_ndx,1}") continue; endif
    if ~exist("std_xy_hist{target_ndx,2}") continue; endif
    mean_std_amoeba_tmp = mean(std_xy_hist{target_ndx,1});
    mean_std_distractor_tmp = mean(std_xy_hist{target_ndx,2});
    std_xy_tmp = std_xy_hist{target_ndx,1};
    if std_xy_tmp ~= 0
      std_std_amoeba_tmp = std(std_xy_tmp);
    else
      std_std_amoeba_tmp = 0;
    endif
    std_xy_tmp = std_xy_hist{target_ndx,2};
    if std_xy_tmp ~= 0
      std_std_distractor_tmp = std(std_xy_tmp);
    else
      std_std_distractor_tmp = 0;
    endif
    disp( ['mean_std_amoeba(', num2str(target_list(target_ndx)), ') =', ...
	   num2str(mean_std_amoeba_tmp), ...
	   ' +/- ', ...
	   num2str(std_std_amoeba_tmp) ] );
    disp( ['mean_std_distractor(', num2str(target_list(target_ndx)), ') =', ...
	   num2str(mean_std_distractor_tmp), ...
	   ' +/- ', ...
	   num2str(std_std_distractor_tmp) ] );
  endfor
  endif
  
				%Screen('CloseAll');

				%if ( uioctave )
				%endif

