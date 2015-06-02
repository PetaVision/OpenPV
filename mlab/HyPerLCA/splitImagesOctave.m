function splitImagesOctave(batch_name, cnt)

%
%   __   __         ___     __     ___       __                   __   ___  __  
%  /__` |__) |    |  |     /  ` | |__   /\  |__)    |  |\/|  /\  / _` |__  /__` 
%  .__/ |    |___ |  |     \__, | |    /~~\ |  \    |  |  | /~~\ \__> |___ .__/ 
%                                                                              
% Modified extractImagesOctave that generates three images from each color channel with a corresponding
% image list                              
%
% 1) Download CIFAR dataset from http://www.cs.toronto.edu/~kriz/cifar.html
%	Make sure to download the Matlab version
%	Unzip the .tar file
%
% 2) Modify this file so that:
%	mkdir('path/to/where/you/want/your/images')
%	output_dir('match/the/first/path/to/where/you/want/your/images/')   Make sure to end with a '/'
%
% 3) Extract the CIFAR dataset:
%	Navigate to the location of your unzipped cifar.mat files
%	Open octave 
%
%	> addpath('~/path/to/PetaVision/mlab/HyPerLCA')
%	> splitImagesOctave(batch_name, cnt)
%
%	 batch_name:	mat file given as string, e.g 'data_batch_1.mat'
% 	 cnt:		most significant digit in unique file number
%		        accounts for having individual .mat files
%		        cnt = 3 will result in numbers 3xxxx
%
%	*NOTE: you can also navigate to ~/path/to/PetaVision/mlab/HyPerLCA
%	       and avoid needing to use addpath('')
%
% 	You'll need to do this for each batch.mat file or you can make a 
% 	wee lil script to loop over all the files yourself.
%
% 4) Go forth and prosper	

 
 
 load(batch_name)
  % variables in batch_name: batch_label = testing batch 1 of 1 (if using test_batch)
  %                          data      size(data) = 10000 x 3072
  %                          labels    size(labels) = 10000 x 1; random numbers 0 - 9
 
% Set the names of the dimensions you want to separate from the dataset  
  dimensions = ['R';'G';'B'];

  output = '/Users/bbroompeltz/Documents/workspace/dataset/CIFAR/';  % should end with / 
  mkdir([output]);
  if ~isempty(strfind(batch_name, 'data_batch_'))
    base_batch_name_start = strfind(batch_name, 'data_batch_');
  elseif ~isempty(strfind(batch_name, 'test_batch'))
    base_batch_name_start = strfind(batch_name, 'test_batch');
  else
    error('batch_name does not contain expected substrings ''data'' or ''test''');
  endif
  base_batch_name_end = strfind(batch_name, '.mat')-1;
  base_batch_name = batch_name(base_batch_name_start:base_batch_name_end);
  output_dir = [output,base_batch_name];
  mkdir(output_dir);

% create different sub-directories for each dimension 
  i = 1
  for i=1:size(dimensions,1)
    color_dim{i} = [output,base_batch_name,'/',dimensions(i)];
    mkdir(color_dim(i));
  endfor
 
% get dimension of data extracted from .mat file
  [xl,yl] = size(data); 		% xl = number of images, yl = size of image 

% initialize image matrix to CIFAR image dimensions
  xdim = 32;
  ydim = 32;
  coldim = size(dimensions,1);			% hard-coded since we are splitting the img into r g b
  im = zeros(xdim,ydim,coldim);

% get min and max label used to create subfolders for each label
  mi = min(labels);
  ma = max(labels);

% create subfolders for all label categories in each dimension sub-directory
  for j=1:size(dimensions,1)
  for i=mi:ma
      dir_flag = mkdir([color_dim{j},'/',int2str(i)]);
      if(dir_flag != 1)
        error(["Unable to create sub-folders"])
      endif
  endfor
  endfor

% create randorder file for PV. Order within .mat file is random already
% appends new lines to the end of the file

  for j=1:size(dimensions,1)
  randorder_pathname{j} = [color_dim{j},'/',base_batch_name,'_randorder_',dimensions(j),'.txt'];
  fid{j} = fopen(randorder_pathname{j}, 'w');
     if fid{j} <=0 
       error(["fid = ", num2str(fid), ": randorder_pathname = ", randorder_pathname]);
      endif
  endfor;

%  
% loop through all elements in .mat file
%
  for i=1 : xl

% get image in 3 1D vectors
% first third of vector has red component, second third green, and last third blue
%% from here on everything is kluged to match an RGB image, and would need to be 
% modified to be generalized to any dimensional 

    red = data(i,1:yl/3);
    green = data(i,yl/3+1:2*yl/3);
    blue = data(i,2*yl/3+1:yl);
 
% collect rgb values out of 1 d vectors to create 3 images
    for j=1:xdim           
      idx = (j-1)*ydim+1;
      im_r(j,:,1) = red(idx:idx+ydim-1);
      im_g(j,:,1) = green(idx:idx+ydim-1);
      im_b(j,:,1) = blue(idx:idx+ydim-1);
    endfor


 % create unique file number (%05d makes it a 5 digit long number with padding zeros)
    num = sprintf("%05d",i+cnt*10000);

% RED IMAGES    
	CIFAR_name_r = strcat(color_dim{1},"/", int2str(labels(i)),'/CIFAR_R_',num,'.png');

% save filename and path to randorder file
	fprintf(fid{1},"%s\n",CIFAR_name_r);

% convert to uint8 and write image
        im1_r = uint8(im_r);
        imwrite(im1_r, CIFAR_name_r);

% GREEN IMAGES    
	CIFAR_name_g = strcat(color_dim{2},"/", int2str(labels(i)),'/CIFAR_G_',num,'.png');

% save filename and path to randorder file
	fprintf(fid{2},"%s\n",CIFAR_name_g);

% convert to uint8 and write image
        im1_g = uint8(im_g);
        imwrite(im1_g, CIFAR_name_g);

% BLUE IMAGES    
	CIFAR_name_b = strcat(color_dim{3},"/", int2str(labels(i)),'/CIFAR_B_',num,'.png');

% save filename and path to randorder file
	fprintf(fid{3},"%s\n",CIFAR_name_b);

% convert to uint8 and write image
        im1_b = uint8(im_b);
        imwrite(im1_b, CIFAR_name_b);

  endfor

for k=1:size(dimensions,1)
  fclose(fid{k});
endfor

keyboard;
endfunction
