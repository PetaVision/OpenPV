function pvp_resample(input_dir, output_dir, filetype, trainingfraction)
% pvp_resample(input_dir, output_dir, filetype, trainingfraction)
% takes a catalog of images and randomly sorts them into several bins
% 
% input_dir is a string giving the directory containing the input image files.
%     The slash at the end of the directory is optional.
%
% output_dir is a string specifying the directory the lists of images will be
% written to.  The routine creates the following files within output_dir:
%    0001/test/test_filenames.txt, 0001/train/train_filenames.txt,
%    0002/test/test_filenames.txt, 0002/train/train_filenames.txt,
%    etc.
% If necessary directories do not exist, they will be created.
% See trainingfraction for how many directories are created
%
% filetype is the file extension to look for in the input directory.
% Do not include the period separating the file extension.
%
% trainingfraction is a number between zero and one that determines the
% percentage of images that will be put into the train directories.
% The remainder are sent to the test directories.
% The number of directories 0001, 0002, etc. created is approximately
% 1/(1-trainingfraction)

if strcmp(input_dir(1:2),'~/')
    input_dir = [getenv('HOME') input_dir(2:end)];
end%%if
if strcmp(output_dir(1:2),'~/')
    output_dir = [getenv('HOME') output_dir(2:end)];
end%%if
if ~exist( output_dir, 'dir')
  [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', output_dir ); 
  if SUCCESS ~= 1
    error(MESSAGEID, MESSAGE);
  end%%if
end%%if

setenv('GNUTERM', 'x11');
[image_struct] = dir([input_dir '/*.' filetype]);
N_images = size(image_struct,1);
disp(['N_images = ', num2str(N_images)]);

N_train = floor( trainingfraction * N_images );  % N_images - 1
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
save([output_dir '/rand_state.mat'],'rand_state');
for i_sample = 1 : N_samples
  image_rank = randperm(N_images);
  train_ndx = image_rank(1:N_train);
  test_ndx = image_rank(N_train+1:N_images);
  for i_train = 1 : N_train
    i_image = train_ndx(i_train);
    filename_tmp = image_struct(i_image).name;
    filenames_train{i_sample, i_train} = filename_tmp;
  end%%for i_train
  for i_test = 1 : N_test
    i_image = test_ndx(i_test);
    filename_tmp = image_struct(i_image).name;
    filenames_test{i_sample, i_test} = filename_tmp;
  end%%for i_test
end%%for i_sample

for i_sample = 1 : N_samples
  output_path = [output_dir, '/', num2str(i_sample, NUM2STR_FORMAT), '/'];
  if ~exist( output_path, 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', output_path ); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    end%%if
  end%%if
  train_path = [output_path, 'train/'];
  if ~exist( train_path, 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', train_path ); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    end%%if
  end%%if
  fileoffilespath = [train_path, 'train_filenames.txt'];
  fid = fopen(fileoffilespath, 'w', 'native');
  for i_train = 1 : N_train
    filename_str = [input_dir, '/', filenames_train{i_sample, i_train}];
    fprintf(fid, '%s\n', filename_str);
  end%%for i_train
  fclose(fid);
	    
  output_path = [output_dir, '/', num2str(i_sample, NUM2STR_FORMAT), '/'];
  test_path = [output_path, 'test/'];
  if ~exist( test_path, 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', test_path ); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    end%%if
  end%%if
  fileoffilespath = [test_path, 'test_filenames.txt'];
  fid = fopen(fileoffilespath, 'w', 'native');
  for i_test = 1 : N_test
    filename_str = [input_dir, '/', filenames_test{i_sample, i_test}];
    fprintf(fid, '%s\n', filename_str);
  end%%for i_test
  fclose(fid);
end%%for i_sample

