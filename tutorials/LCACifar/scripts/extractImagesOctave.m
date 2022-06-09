function extractImagesOctave(batch_name, cnt, append_mixed=true)
% batch_name: mat file given as string, e.g 'data_batch_1.mat'
% cnt:        most significant digit in unique file number. Accounts for having 
%             individual .mat files. cnt = 3 will result in numbers 3xxxx

 load(batch_name)
  if ~isempty(strfind(batch_name, 'data_batch_'))
    base_batch_name_start = strfind(batch_name, 'data_batch_');
  elseif ~isempty(strfind(batch_name, 'test_batch'))
    base_batch_name_start = strfind(batch_name, 'test_batch');
  else
    error('batch_name does not contain expected substrings ''data'' or ''test''');
  end%if
  base_batch_name_end = strfind(batch_name, '.mat')-1;
  base_batch_name = batch_name(base_batch_name_start:base_batch_name_end);
  parent_path = cd(cd(fileparts(batch_name)));
  output_dir = [parent_path, '/', base_batch_name];
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
  end%for
  
                                % create randorder file for PV. Order within .mat file is random already
                                % appends new lines to the end of the file
  randorder_pathname = [output_dir,'/',base_batch_name,'_randorder.txt'];
  mixed_file_pathname = [parent_path, '/mixed_cifar.txt'];
  
  batch_file = fopen(randorder_pathname, 'w');
  if batch_file <=0 
    error(["batch_file = ", num2str(batch_file), ": randorder_pathname = ", randorder_pathname]);
  end%if
  
  mixed_file = fopen(mixed_file_pathname, 'a');
  if mixed_file <=0 
    error(["mixed_file = ", num2str(mixed_file), ": randorder_pathname = ", randorder_pathname]);
  end%if

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
    end%for
                                % create unique file number (%05d makes it a 5 digit long number with padding zeros)
    num = sprintf("%05d",i+cnt*10000);
    CIFAR_name = strcat(output_dir,"/", int2str(labels(i)),'/CIFAR_',num,'.png');
                                % save filename and path to randorder file
    fprintf(batch_file,"%s\n",CIFAR_name);
    if append_mixed
        fprintf(mixed_file,"%s\n",CIFAR_name);
    end%if
                                % convert to uint8 and write image
    im1 = uint8(im);
    imwrite(im1, CIFAR_name);
  end%for
  fclose(batch_file);
  fclose(mixed_file);
  copyfile(randorder_pathname, parent_path);
  printf("Finished %s\n",base_batch_name);
endfunction
