function [nz_image_cell] = commands2(image_size)

  if nargin < 1
    image_size = [256 256];
  end

  global image_dim
  image_dim = image_size;

%  addpath('/Applications/Psychtoolbox/');

% number of targets/fourier component
  numT = 1000;
%  screen_color = [];
				%screen_rect = [0 0 256 256];
%  screen_rect = [0 0 128 128];
%[w0, window_rect]  = Screen('OpenWindow', 0, screen_color, screen_rect);
%Screen('FillRect', w0, GrayIndex(w0));

  global plot_amoeba2D fh_amoeba2D
plot_amoeba2D = 0;
  setenv("GNUTERM", "x11");
if plot_amoeba2D
  fh_amoeba2D = figure;
end

%fourC = [2 4 6 8];
				fourC = [4];
  global nz_image 
  nz_image = zeros(3, numT);
  nz_image_cell = cell(length(fourC), 1);

    global amoeba_file_path
    amoeba_file_path =  [ num2str(image_dim(1)), '_png/']

    global image_file_path image_file_name
    
for i = 1:length(fourC)
  nfour = fourC(i);
  nz_image = zeros(2,0);
  image_file_path = [ amoeba_file_path, num2str(nfour), '/']
  for j = 1:numT
    getAmoebaStats3(j, nfour);
    disp(num2str(j));
  end
  nz_image_cell{i} = nz_image;
  figure
  [nz_image_hist, nz_image_bins] = ...
      hist( nz_image_cell{1}(1,:) );
  bh = bar( nz_image_bins, nz_image_hist, 0.8);
  set(bh, 'EdgeColor', [1 0 0]);
  set(bh, 'FaceColor', [1 0 0]);
  hold on;
  nz_image_hist = ...
      hist( nz_image_cell{1}(2,:), nz_image_bins );
  bh = bar( nz_image_bins, nz_image_hist, 0.6);
  set(bh, 'EdgeColor', [0 0 1]);
  set(bh, 'FaceColor', [0 0 1]);
    
    amoeba_hist_filename = ...
	[image_file_path, 'amoeba_hist', '.mat.z']
    save('-z', '-mat', amoeba_hist_filename, 'nz_image');
end
    
				%Screen('CloseAll');

%if ( uioctave )
%endif

