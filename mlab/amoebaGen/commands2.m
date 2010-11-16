function [nz_image_cell] = commands2(image_size)

if nargin < 1
    image_size = [128 128]; [256 256];
end

global image_dim
image_dim = image_size;

%  addpath('/Applications/Psychtoolbox/');

% number of targets/fourier component
numT = 10000;
%  screen_color = [];
%screen_rect = [0 0 256 256];
%  screen_rect = [0 0 128 128];
%[w0, window_rect]  = Screen('OpenWindow', 0, screen_color, screen_rect);
%Screen('FillRect', w0, GrayIndex(w0));

global plot_amoeba2D fh_amoeba2D
plot_amoeba2D = 0;
setenv('GNUTERM', 'x11');
if plot_amoeba2D
    fh_amoeba2D = figure;
end

%% sets number of fourier components
%fourC = [2 4 6 8];
fourC = [4];
global nz_image
nz_image = zeros(3, numT);
nz_image_cell = cell(length(fourC), 1);

global machine_path
machine_path = '/nh/home/gkenyon/Documents/MATLAB/amoeba_ltd/';
%machine_path = '/Users/gkenyon/Documents/MATLAB/amoeba_ltd/';
if ~exist( 'machine_path', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', machine_path); 
    if SUCCESS ~= 1
        error(MESSAGEID, MESSAGE);
    end%%if
end%%if

global amoeba_file_path
amoeba_file_path =  [ machine_path, num2str(image_dim(1)), '_png/']
if ~exist( 'amoeba_file_path', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', amoeba_file_path ); 
    if SUCCESS ~= 1
        error(MESSAGEID, MESSAGE);
    end%%if
end%%if

global image_file_path image_file_name

for i = 1:length(fourC)
    nfour = fourC(i);
    nz_image = zeros(2,0);
    image_file_path = [ amoeba_file_path, num2str(nfour), '/']
    if ~exist( 'image_file_path', 'dir')
       [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', image_file_path ); 
       if SUCCESS ~= 1
           error(MESSAGEID, MESSAGE);
       end%%if
    end%%if
    
    for j = 1:numT
        getAmoebaStats3(j, nfour);
        disp(num2str(j));
    end
    nz_image_cell{i} = nz_image;
    figure
    [nz_image_hist, nz_image_bins] = ...
        hist( nz_image_cell{i}(1,:) );
    bh = bar( nz_image_bins, nz_image_hist, 0.8);
    set(bh, 'EdgeColor', [1 0 0]);
    set(bh, 'FaceColor', [1 0 0]);
    hold on;
    nz_image_hist = ...
        hist( nz_image_cell{i}(2,:), nz_image_bins );
    bh = bar( nz_image_bins, nz_image_hist, 0.6);
    set(bh, 'EdgeColor', [0 0 1]);
    set(bh, 'FaceColor', [0 0 1]);
    
    amoeba_hist_filename = ...
        [image_file_path, 'amoeba_hist', '.mat']
    save('-mat', amoeba_hist_filename, 'nz_image');
end


for i = 1:size(nz_image_cell,1)
    mean_nz_tmp = mean(nz_image_cell{i}(1,:));
    std_nz_tmp = std(nz_image_cell{i}(1,:));
    mean_nz_tmp2 = mean(nz_image_cell{i}(2,:));
    std_nz_tmp2 = std(nz_image_cell{i}(2,:));
    disp( ['mean_nz(', num2str(i), ',:) =', num2str(mean_nz_tmp), ...
	  ' +/- ', num2str(std_nz_tmp), ', ', num2str(mean_nz_tmp2), ...
	  ' +/- ', num2str(std_nz_tmp2)] );
end
    
%Screen('CloseAll');

%if ( uioctave )
%endif

