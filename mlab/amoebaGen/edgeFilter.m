 
clear all
setenv('GNUTERM', 'x11');
image_dir = ...
      '~/eclipse-workspace/kernel/input/256/animalDB/'; 
%%    '~/eclipse-workspace/kernel/input/segmented_images/'; 
image_dir = ...
    [image_dir, 'targets/']; %'distractors/']; %'annotated_animals/']; %'annotated_distractors/'];  %
original_dir = ...
    [ image_dir, 'original/' ];
image_path = ...
    [image_dir, 'original/', '*.jpg'];
canny_dir = ...
    [ image_dir, 'canny/' ];
if ~exist( 'canny_dir', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', canny_dir); 
    if SUCCESS ~= 1
        error(MESSAGEID, MESSAGE);
    end%%if
end%%if

[image_struct] = dir(image_path);
num_images = size(image_struct,1);
disp(['num_images = ', num2str(num_images)]);
sigma_canny = 1;
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
  if mod( i_image, ceil(num_images/10) ) == 1
    disp(['i_image = ', num2str(i_image)]);
    figure;
    imagesc(image_canny); colormap(gray);
    figure;
    imagesc(image_gray); colormap(gray);
  endif
  
endfor



