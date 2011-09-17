function [] = commands2(image_size)

  addpath("/Users/gkenyon/workspace-indigo/PetaVision/mlab/imgProc/");
  setenv('GNUTERM', 'x11');

  if nargin < 1
    image_size =  [256 256]; %[128 128]; %
  endif

  global image_dim
  image_dim = image_size;

				% number of targets/fourier component
  numT = 10000; %%10000;
  global trial num_trials
  num_trials = numT;
  global plot_amoeba2D fh_amoeba2D
  plot_amoeba2D = 0;
  setenv('GNUTERM', 'x11');
  if plot_amoeba2D
    fh_amoeba2D = figure;
  endif

  %% sets number of fourier components
  fourC = [2 4 6 8];
  %%fourC = [4];
  global nz_image
  nz_image = zeros(3, numT);
  nz_image_cell = cell(length(fourC), 1);

  global mean_amoeba_xy std_amoeba_xy mean_distractor_xy std_distractor_xy
  std_xy_hist = cell(length(fourC), 2);

  global machine_path
  %%machine_path = '/nh/home/gkenyon/Documents/MATLAB/amoeba_ltd/';
  machine_path = '/Users/gkenyon/Pictures/amoeba/';
  if ~exist( 'machine_path', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', machine_path); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif

  global amoeba_file_path
  amoeba_file_path =  [ machine_path, num2str(image_dim(1)), "/"] %%, '_png/']
  if ~exist( 'amoeba_file_path', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', amoeba_file_path ); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif

  global image_file_path image_file_name

  for i_FC = 1:length(fourC)
    nfour = fourC(i_FC);
    nz_image = zeros(2,0);
    image_file_path = [ amoeba_file_path, num2str(nfour), '/']
    if ~exist( 'image_file_path', 'dir')
      [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', image_file_path ); 
      if SUCCESS ~= 1
        error(MESSAGEID, MESSAGE);
      endif
    endif
    
    for j = 1:numT
      getAmoebaStats3(j, nfour);
      if mod(j, fix(numT/20)) == 0
        disp(num2str(j));
      endif
    endfor
    
    nz_image_cell{i_FC} = nz_image;
    figure
    [nz_image_hist, nz_image_bins] = ...
        hist( nz_image_cell{i_FC}(1,:) );
    bh = bar( nz_image_bins, nz_image_hist, 0.8);
    set(bh, 'EdgeColor', [1 0 0]);
    set(bh, 'FaceColor', [1 0 0]);
    hold on;
    nz_image_hist = ...
        hist( nz_image_cell{i_FC}(2,:), nz_image_bins );
    bh = bar( nz_image_bins, nz_image_hist, 0.6);
    set(bh, 'EdgeColor', [0 0 1]);
    set(bh, 'FaceColor', [0 0 1]);
    
    figure
    std_amoeba = sqrt( std_amoeba_xy(:,1).^2 + std_amoeba_xy(:,2).^2 );
    [std_amoeba_hist, std_amoeba_bins] = ...
        hist( std_amoeba );
    std_amoeba_hist = std_amoeba_hist / sum(std_amoeba_hist(:));
    bh = bar( std_amoeba_bins, std_amoeba_hist, 0.8);
    set(bh, 'EdgeColor', [1 0 0]);
    set(bh, 'FaceColor', [1 0 0]);
    hold on;
    std_distractor = sqrt( std_distractor_xy(:,1).^2 + std_distractor_xy(:,2).^2 );
    std_distractor_hist = ...
        hist( std_distractor, std_amoeba_bins );
    std_distractor_hist = std_distractor_hist / sum(std_distractor_hist(:));
    bh = bar( std_amoeba_bins, std_distractor_hist, 0.6);
    set(bh, 'EdgeColor', [0 0 1]);
    set(bh, 'FaceColor', [0 0 1]);

    std_xy_hist{i_FC, 1} = std_amoeba_hist; 
    std_xy_hist{i_FC, 2} = std_distractor_hist; 

    
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

  num_FC = size(std_xy_hist,1);
  for i_FC = 1:num_FC
    mean_std_amoeba_tmp = mean(std_xy_hist{i_FC,1});
    mean_std_distractor_tmp = mean(std_xy_hist{i_FC,2});
    std_std_amoeba_tmp = std(std_xy_hist{i_FC,1});
    std_std_distractor_tmp = std(std_xy_hist{i_FC,2});
    disp( ['mean_std_amoeba(', num2str(i_FC), ') =', ...
	   num2str(mean_std_amoeba_tmp), ...
	   ' +/- ', ...
	   num2str(std_std_amoeba_tmp) ] );
    disp( ['mean_std_distractor(', num2str(i_FC), ') =', ...
	   num2str(mean_std_distractor_tmp), ...
	   ' +/- ', ...
	   num2str(std_std_distractor_tmp) ] );
  endfor
