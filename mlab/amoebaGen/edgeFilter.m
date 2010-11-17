
close all
clear all
setenv('GNUTERM', 'x11');
image_dir = ...
      '~/workspace/kernel/input/256/animalDB/'; 
%%    '~/eclipse-workspace/kernel/input/segmented_images/'; 
image_dir = ...
    [image_dir, 'targets/']; %'distractors/']; %'annotated_animals/']; %'annotated_distractors/'];  %
original_dir = ...
    [ image_dir, 'original/' ];
image_path = ...
    [image_dir, 'original/', '*.jpg'];
sigma_canny = 0.5;% 1.0;%
if sigma_canny == 1
  canny_dir = ...
      [ image_dir, 'canny_sigma_1_0/' ];
elseif sigma_canny == 0.5
  canny_dir = ...
      [ image_dir, 'canny_sigma_0_5/' ];
endif
if ~exist( 'canny_dir', 'dir')
  [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', canny_dir); 
  if SUCCESS ~= 1
    error(MESSAGEID, MESSAGE);
  endif
endif

DoG_flag = 1;
if DoG_flag
  amp_center = 1;
  sigma_center = sigma_canny;
  amp_surround = 1;
  sigma_surround = 2 * sigma_canny;
  DoG_dir = ...
      [ image_dir, 'DoG/' ];
  if ~exist( 'DoG_dir', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', DoG_dir); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif
endif

[image_struct] = dir(image_path);
num_images = size(image_struct,1);
disp(['num_images = ', num2str(num_images)]);
for i_image = 1 : num_images
  image_name = image_struct(i_image).name;
  base_name = image_name(1:strfind(image_name, '.jpg')-1);
  image_name = [original_dir, image_name];
  [image_color, image_map, image_alpha] = ...
      imread(image_name);
  image_gray = col2gray(image_color);
  
  [image_canny, image_orient] = ...
      canny(image_gray, sigma_canny);
  canny_filename = ...
      [canny_dir, base_name];
  savefile2(canny_filename, image_canny);

  if DoG_flag == 1
    [image_DoG] = ...
	DoG(image_gray, ...
	    amp_center, ...
	    sigma_center, ...
	    amp_surround, ...
	    sigma_surround);
    DoG_filename = ...
	[DoG_dir, base_name];
    savefile2(DoG_filename, image_DoG);
  endif

  
  if mod( i_image, ceil(num_images/10) ) == 1
    disp(['i_image = ', num2str(i_image)]);
    fh = figure('name', 'Original');
    imagesc(image_gray); colormap(gray);
    fh = figure('name', 'canny');
    imagesc(image_canny); colormap(gray);
    if DoG_flag == 1
      fh = figure('name', 'DoG');
      imagesc(image_DoG); colormap(gray);
    endif

  endif
  
endfor



