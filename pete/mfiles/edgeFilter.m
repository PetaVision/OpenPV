function edgeFilter(image_dir, output_dir, in_extension, out_extension)
% edgeFilter(image_dir, output_dir, in_extension, out_extension)
% performs batch Canny edge detection.
%
%
% image_dir is a directory containing the input images
% output_dir is the directory the output images will be written to.
% Per the matlab mkdir command, if output_dir does not exist, it
% will be created, along with any necessary parent directories.
%
% in_extension is the file extension of the images in image_dir.
% out_extension is the file extension of the images to be written to
% output_dir.
%
% Example:
%    Suppose the directory inputfiles contains the files 'apple.jpg',
%    'banana.jpg' and 'chile.jpg', and there is a directory 'output',
%    but no subdirectory 'output/canny'.
%    Then the command
%        edgeFilter('inputfiles','output/canny/outfiles','jpg','png')
% would create the directory 'output/canny/outfiles/', and put files
% 'apple.png', 'banana.png', and 'chile.png' in that directory.
%
% To do:  if any files in output directory would be overwritten, 
% warn the user and provide options to overwrite that file, overwrite
% all files, abort.
%

%clear all
setenv('GNUTERM', 'x11');
padding = 10;
%image_dir = ...
%      '/Users/pschultz/Workspace/NMC/CNS/AnimalDB/'; 
%    '~/eclipse-workspace/kernel/input/segmented_images/'; 
% image_dir = ...
%     [image_dir, 'Distractors/']; %'Targets/']; %'annotated_animals/']; %'annotated_distractors/'];  %
% original_dir = ...
%     [ image_dir, 'original/' ];
% image_path = ...
%     [image_dir, 'original/', '*.jpg'];
% canny_dir = ...
%     [ image_dir, 'canny/' ];
if ~exist( 'output_dir', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', output_dir); 
    if SUCCESS ~= 1
        error(MESSAGEID, MESSAGE);
    end%%if
end%%if

[image_struct] = dir([image_dir '/*.' in_extension]);
num_images = size(image_struct,1);
disp(['num_images = ', num2str(num_images)]);
sigma_canny = 1;
for i_image = 1 : num_images
  image_name = image_struct(i_image).name;
  base_name = image_name(1:strfind(image_name, '.jpg')-1);
  image_path = [image_dir, '/', image_name];
  [image_color, image_map, image_alpha] = imread(image_path);
  image_gray = col2gray(image_color);
  
  % pad image with mirror boundary conditions, and crop after canny edge
  % detection, so that the boundary isn't called an edge.
  [m, n] = size(image_gray);
  image_padded = image_gray(:,[padding+1:-1:2 1:n n-1:-1:n-padding]);
  image_padded = image_padded([padding+1:-1:2 1:m m-1:-1:m-padding],:);
  % edge detection  
  [canny_padded, image_orient] = canny(image_padded, sigma_canny);
  % now crop back to the original size
  image_canny = canny_padded(padding+1:padding+m, padding+1:padding+n);
  
  canny_filename = [output_dir, '/', base_name '.' out_extension];
  image_uint8 = uint8(image_canny);
  imwrite(image_uint8, canny_filename);
    % savefile2(canny_filename, image_canny);
  if mod( i_image, ceil(num_images/10) ) == 0
    figure;
    imagesc(image_gray); colormap(gray);
    figure;
    imagesc(image_canny); colormap(gray);
    drawnow;
    disp(['Image ', num2str(i_image) ' of ' num2str(num_images)]);
  end%%if
  
end%%for



