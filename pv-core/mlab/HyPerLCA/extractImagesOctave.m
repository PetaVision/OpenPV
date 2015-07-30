function extractImagesOctave(batch_name, cnt)

%
%   ___     ___  __        __  ___     __     ___       __                   __   ___  __  
%  |__  \_/  |  |__)  /\  /  `  |     /  ` | |__   /\  |__)    |  |\/|  /\  / _` |__  /__` 
%  |___ / \  |  |  \ /~~\ \__,  |     \__, | |    /~~\ |  \    |  |  | /~~\ \__> |___ .__/ 
%                                                                                          
%
% 1) Download CIFAR dataset from http://www.cs.toronto.edu/~kriz/cifar.html
%	Download the Matlab version    
%          $ wget "http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz"
%       Unzip the tar file
%          $ tar -zxvf cifar-10-matlab.tar.gz
%
% 3) Extract the CIFAR dataset:
%	Navigate to the location of your unzipped cifar.mat files
%	Open octave 
%
%	> addpath('~/path/to/master/pv-core/mlab/HyPerLCA')
%	> extractImagesOctave(batch_name, cnt)
%
%	 batch_name:	mat file given as string, e.g 'data_batch_1.mat'
% 	 cnt:		most significant digit in unique file number
%		        accounts for having individual .mat files
%		        cnt = 3 will result in numbers 3xxxx
%
%	*NOTE: you can also navigate to ~/master/pv-core/mlab/HyPerLCA
%	       and avoid needing to use addpath('')
%
% 	You'll need to do this for each batch.mat file or you can make a 
% 	wee lil script to loop over all the files yourself.
%
% 4) Go forth and prosper	

 pwd = pwd(); 

 load(batch_name)
  if ~isempty(strfind(batch_name, 'data_batch_'))
    base_batch_name_start = strfind(batch_name, 'data_batch_');
  elseif ~isempty(strfind(batch_name, 'test_batch'))
    base_batch_name_start = strfind(batch_name, 'test_batch');
  else
    error('batch_name does not contain expected substrings ''data'' or ''test''');
  endif
  base_batch_name_end = strfind(batch_name, '.mat')-1;
  base_batch_name = batch_name(base_batch_name_start:base_batch_name_end);
  output_dir = [base_batch_name];
  mkdir(output_dir);
				% get dimension of data extracted from .mat file
				% xl = number of images, yl = size of image
  [xl,yl] = size(data);	
				% initialize image matrix to CIFAR image dimensions
  xdim = 32;
  ydim = 32;
  coldim = 3;
  im = zeros(xdim,ydim,coldim);
				% get min and max label used to create subfolders for each label
  mi = min(labels);
  ma = max(labels);
				% create subfolders
  for i=mi:ma
    mkdir([output_dir,'/',int2str(i)]);
  endfor
  
				% create randorder file for PV. Order within .mat file is random already
				% appends new lines to the end of the file
  randorder_pathname = [output_dir,'/',base_batch_name,'_randorder.txt'];
  fid = fopen(randorder_pathname, 'w');
  if fid <=0 
    error(["fid = ", num2str(fid), ": randorder_pathname = ", randorder_pathname]);
  endif

 progress = sprintf('Finished %s',base_batch_name)
 disp(progress) 
				% loop through all elements in .mat file
  for i=1 : xl
				% get image in 3 1D vectors
				% first third of vector has red component, second third green, and last third blue
    red = data(i,1:yl/3);
    green = data(i,yl/3+1:2*yl/3);
    blue = data(i,2*yl/3+1:yl);
				% collect rgb values out of 1 d vectors to create images
    for j=1:xdim
      idx = (j-1)*ydim+1;
      im(j,:,1) = red(idx:idx+ydim-1);
      im(j,:,2) = green(idx:idx+ydim-1);
      im(j,:,3) = blue(idx:idx+ydim-1);
    endfor
				% create unique file number (%05d makes it a 5 digit long number with padding zeros)
    num = sprintf("%05d",i+cnt*10000);
    CIFAR_name = strcat(output_dir,"/", int2str(labels(i)),'/CIFAR_',num,'.png');
				% save filename and path to randorder file
    fprintf(fid,"%s/%s\n",pwd,CIFAR_name);
				% convert to uint8 and write image
    im1 = uint8(im);
    imwrite(im1, CIFAR_name);
  endfor
  fclose(fid);		
copyfile(randorder_pathname,pwd);
endfunction
