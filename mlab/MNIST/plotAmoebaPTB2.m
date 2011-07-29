function plotAmoebaPTB2(amoeba_image, ...
			amoeba_struct, ...
			trial, ...
			amoeba_flag, ...
			aflag)
% global w0
global image_dim
global nz_image
global amoeba_file_path
global image_file_path image_file_name

if nargin < 5 || isempty(aflag)
    aflag = 0;
end

trial = trial-1;
s = size(amoeba_image); % 6,2
num_segments = length(amoeba_image{1,1});

num_objects = amoeba_struct.num_targets + amoeba_struct.num_distractors;

if amoeba_flag
  if ~aflag
    start = 1;
    fin = num_objects; %%s(1)-3;
    dname = 't/';
    fname = 'tar_';
    fflag = 'a';
  elseif aflag %% amoeba only
    start = 1;
    fin = amoeba_struct.num_targets;
    dname = 'a/';
    fname = 'tar_';
    fflag = 'a';
  endif
elseif ~amoeba_flag
    start = num_objects + 1; %%s(1)-2;
    fin = 2*num_objects; %%s(1);
    dname = 'd/';
    fname = 'tar_';
    fflag = 'n';
endif


% start == fin == 1 to plot amoeba only, fname = 'a/tar_';

x = [];
y = [];
for i_amoeba = start:fin
  num_segments = size(amoeba_image{i_amoeba,1},1);
    for segment = 1:num_segments
        x = [x; amoeba_image{i_amoeba,1}{segment}(:)];
        y = [y; amoeba_image{i_amoeba,2}{segment}(:)];
    end
end

image = zeros(image_dim);
x_pixel = round(x);
y_pixel = round(y);
x_pixel(x_pixel < 1) = 1; %% should use mirror BCs
y_pixel(y_pixel < 1) = 1;
x_pixel(x_pixel > image_dim(1)) = image_dim(1);
y_pixel(y_pixel > image_dim(2)) = image_dim(2);
image( sub2ind( image_dim, x_pixel(:), y_pixel(:) ) ) = 255;

if amoeba_flag && ~aflag
    nz_image(1,trial+1) = sum(image(:));
elseif ~amoeba_flag
    nz_image(2,trial+1) = sum(image(:));
elseif amoeba_flag && aflag
    nz_image(3,trial+1) = sum(image(:));
end


if trial < 10
    suffix_zeros = '000';
elseif trial < 100
    suffix_zeros = '00';
elseif trial < 1000
    suffix_zeros = '0';
else
    suffix_zeros = '';
end


%savefile(['256_png/',num2str(nfour),'/',fname,suffix_zeros, num2str(t), '_', fflag], 128, 128);
image_file_dir = ...
    [image_file_path, dname];
if ~exist(image_file_dir, 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', image_file_dir );
    if SUCCESS ~= 1
        error(MESSAGEID, MESSAGE);
    end%%if
end%%if
image_file_name = ...
    [image_file_dir, fname, suffix_zeros, num2str(trial), '_', fflag];
savefile2(image_file_name, image);

global plot_amoeba2D fh_amoeba2D
if plot_amoeba2D
    figure(fh_amoeba2D);
    imagesc(image);
    drawnow
end
