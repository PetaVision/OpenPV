function plotAmoebaMask(amoeba_image, amoeba_struct, trial)
%global w0
global image_dim
global amoeba_file_path
global image_file_path image_file_name

trial = trial-1;
s = size(amoeba_image); % 3,2
num_segments = length(amoeba_image{1,1});

x0 = [];
y0 = [];
for i_amoeba = 1:s(1)
  num_segments = size(amoeba_image{i_amoeba,1});
    for j = 1:num_segments
        
        x = (amoeba_image{i_amoeba,1}{j});
        y = (amoeba_image{i_amoeba,2}{j});
        
        xM = mean(amoeba_image{i_amoeba,1}{j});
        yM = mean(amoeba_image{i_amoeba,2}{j});
        
        xD = x - xM;
        yD = y - yM;
        theta = (45+rand*90) * pi/180;
        rX = (xD*cos(theta) + yD*(-sin(theta))) + xM;
        rY = (xD*sin(theta) + yD*cos(theta)) + yM;
        %draw(rX,rY,0);
        x0 = [x0; rX(:)];
        y0 = [y0; rY(:)];
        
        
    endfor
endfor

image = zeros(image_dim);
x_pixel = round(x0);
y_pixel = round(y0);
x_pixel(x_pixel < 1) = 1;
y_pixel(y_pixel < 1) = 1;
x_pixel(x_pixel > image_dim(1)) = image_dim(1);
y_pixel(y_pixel > image_dim(2)) = image_dim(2);
image( sub2ind( image_dim, x_pixel(:), y_pixel(:) ) ) = 255;

if trial < 10
    suffix_zeros = '000';
elseif trial < 100
    suffix_zeros = '00';
elseif trial < 1000
    suffix_zeros = '0';
else
    suffix_zeros = '';
end

dname = 'm/';
fname = 'm_';
image_file_dir = ...
    [image_file_path, dname];
if ~exist(image_file_dir, 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', image_file_dir );
    if SUCCESS ~= 1
        error(MESSAGEID, MESSAGE);
    end%%if
end%%if
image_filename = ...
    [image_file_dir, fname, suffix_zeros, num2str(trial)];
savefile2(image_filename, image);

global plot_amoeba2D fh_amoeba2D
if plot_amoeba2D
    figure(fh_amoeba2D);
    imagesc(image)
end
