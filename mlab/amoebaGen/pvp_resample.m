

clear all
setenv('GNUTERM', 'x11');
image_dir = ...
    '~/eclipse-workspace/kernel/input/256/animalDB/';
%%    '~/eclipse-workspace/kernel/input/segmented_images/'; 
image_dir = ...
    [image_dir, 'distractors/']; %'targets/']; %'annotated_distractors/'];  % 'annotated_animals/']; % 
canny_dir = ...
    [ image_dir, 'canny/' ];
canny_files = ...
    [canny_dir, '*.png'];

filenames_dir = ...
    '/Users/gkenyon/eclipse-workspace/kernel/input/256/noanimal/';
file_type = '.png';

				% [gradient or] = canny(im, sigma)
[image_struct] = dir(canny_files);
N_images = size(image_struct,1);
disp(['N_images = ', num2str(N_images)]);

N_train = floor( 0.8 * N_images );  % N_images - 1
disp(['N_train = ', num2str(N_train)]);
N_test = N_images - N_train;
disp(['N_test = ', num2str(N_test)]);
N_samples = floor( N_images / (N_images - N_train ) );
disp(['N_samples = ', num2str(N_samples)]);


global NUM2STR_FORMAT
NUM2STR_FORMAT = '%04.4i';
filenames_train = cell(N_samples, N_train);
filenames_test = cell(N_samples, N_test);
rand_state = rand('state'); % should save...
for i_sample = 1 : N_samples
  image_rank = randperm(N_images);
  train_ndx = image_rank(1:N_train);
  test_ndx = image_rank(N_train+1:N_images);
  for i_train = 1 : N_train
    i_image = train_ndx(i_train);
    filename_tmp = image_struct(i_image).name;
    filenames_train{i_sample, i_train} = filename_tmp;
  endfor
  for i_test = 1 : N_test
    i_image = test_ndx(i_test);
    filename_tmp = image_struct(i_image).name;
    filenames_test{i_sample, i_test} = filename_tmp;
  endfor
endfor

for i_sample = 1 : N_samples
  filenames_path = [filenames_dir, num2str(i_sample, NUM2STR_FORMAT), '/'];
  if ~exist( 'filenames_path', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', filenames_path ); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif
  filenames_path = [filenames_path, 'train/'];
  if ~exist( 'filenames_path', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', filenames_path ); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif
  filenames_tmp = [filenames_path, 'train_filenames.txt'];
  fid = fopen(filenames_tmp, 'w', 'native');
  for i_train = 1 : N_train
    filename_str = ...
	filenames_train{i_sample, i_train};
    filename_str = ...
	[canny_dir, filename_str];
    fprintf(fid, "%s\n", filename_str);
  endfor
  fclose(fid);
	    
  filenames_path = [filenames_dir, num2str(i_sample, NUM2STR_FORMAT), '/'];
  filenames_path = [filenames_path, 'test/'];
  if ~exist( 'filenames_path', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', filenames_path ); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif
  filenames_tmp = [filenames_path, 'test_filenames.txt'];
  fid = fopen(filenames_tmp, 'w', 'native');
  for i_test = 1 : N_test
    filename_str = ...
	filenames_test{i_sample, i_test};
    filename_str = ...
	[canny_dir, filename_str];
    fprintf(fid, '%s\n', filename_str);
  endfor
  fclose(fid);
endfor %% i_sample

