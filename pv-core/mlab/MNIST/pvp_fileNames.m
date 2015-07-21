function filenames_cell = pvp_fileNames(N_ofeach, target_list, ...
					min_image_id, random_order)

  global NUM2STR_FORMAT
  NUM2STR_FORMAT = '%04.4i';
  file_path = ...
      '../../MATLAB/captcha/256_png/';
%      '/nh/home/gkenyon/Documents/MATLAB/amoeba/128_png/'
%      '../PetaVision/mlab/amoebaGen/256/';
  file_root = 'tar_'; % 
  target_flag = '_a'; % '_n'; %
  file_dir = 't'; %'a'; % 'd'; % 
  file_type = '.png';
  if nargin < 4 || isempty(random_order)
    random_order = 0; 
  endif
  if nargin < 3 || isempty(min_image_id)
    min_image_id = 0; %%2500; 
  endif
  if nargin < 2 || isempty(FC_list)
    target_list = [6]; %%[ 2 4 6 8 ];
  endif
  num_targets = length(target_list);
  if nargin < 1 || isempty(N_ofeach)
    N_ofeach = 10000; %2500;
  endif
  N = num_targets * N_ofeach;
  image_id = [];
  for i_target = 1 : num_targets
    image_id = [ image_id; repmat(target_list(i_target), N_ofeach, 1) ];
  endfor
  if random_order
    image_id = randperm(N) - 1;
  else
    image_id = 1:N;
    image_id = image_id - 1;
  endif
  filenames_cell = cell(N,1);
  fid = fopen('fileNames.txt', 'w', 'native');
  for i = 1 : N
    target_str = ...
	num2str( target_list( ceil( (1+image_id(i)) / N_ofeach ) ) );
    image_id_tmp = mod( image_id(i), N_ofeach );
    image_id_tmp = image_id_tmp + min_image_id;
    id_str = ...
	num2str( image_id_tmp, NUM2STR_FORMAT );
    filename_str = ...
	[ file_path, ...
	 target_str, ...
	 '/',  file_dir, '/', ...
	 file_root, ...
	 id_str, ...
	 target_flag, ...
	 file_type ];
    filenames_cell{i} = filename_str;
    fprintf(fid, '%s\n', filename_str);
  endfor
  fclose(fid);