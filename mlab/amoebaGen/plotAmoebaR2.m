function plotAmoebaR2(A,t,nfour)
%global w0
global image_dim
global amoeba_file_path
global image_file_path image_file_name

t = t-1;
s = size(A); % 3,2
numA = length(A{1,1});

x0 = [];
y0 = [];
for i = 1:s(1)
    for j = 1:numA
        
        x = (A{i,1}{j});
        y = (A{i,2}{j});
        
        xM = mean(A{i,1}{j});
        yM = mean(A{i,2}{j});
        
        xD = x - xM;
        yD = y - yM;
        theta = (45+rand*90) * pi/180;
        rX = (xD*cos(theta) + yD*(-sin(theta))) + xM;
        rY = (xD*sin(theta) + yD*cos(theta)) + yM;
        %draw(rX,rY,0);
        x0 = [x0; rX(:)];
        y0 = [y0; rY(:)];
        
        
    end
end

image = zeros(image_dim);
x_pixel = round(x0);
y_pixel = round(y0);
x_pixel(x_pixel < 1) = 1;
y_pixel(y_pixel < 1) = 1;
x_pixel(x_pixel > image_dim(1)) = image_dim(1);
y_pixel(y_pixel > image_dim(2)) = image_dim(2);
image( sub2ind( image_dim, x_pixel(:), y_pixel(:) ) ) = 255;

if t < 10
    suffix_zeros = '000';
elseif t < 100
    suffix_zeros = '00';
elseif t < 1000
    suffix_zeros = '0';
else
    suffix_zeros = '';
end

%savefile(['256_png/',num2str(nfour),'/m/m_',suffix_zeros, num2str(t)], 128, 128);
% savefile(['128_png/',num2str(nfour),'/m/m_',suffix_zeros, num2str(t)], 64, 64);

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
    [image_file_dir, fname, suffix_zeros, num2str(t)];
% image_filename = ...
%     [image_file_path, fname, suffix_zeros, num2str(t)];
savefile2(image_filename, image);

global plot_amoeba2D fh_amoeba2D
if plot_amoeba2D
    figure(fh_amoeba2D);
    imagesc(image)
end
