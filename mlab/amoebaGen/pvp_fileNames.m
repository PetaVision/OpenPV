function filenames_cell = pvp_fileNames(N_ofeach, FC_list, ...
					min_image_id, random_order)

  global NUM2STR_FORMAT
  NUM2STR_FORMAT = '%04.4i';
  file_path = ...
      '../../MATLAB/figures/amoeba/256_png/';
%      '/nh/home/gkenyon/Documents/MATLAB/amoeba/128_png/'
%      '../PetaVision/mlab/amoebaGen/256/';
  file_root = 'tar_'; % 
  target_flag = '_a'; % '_n'; %
  file_dir = 'a'; %'a'; % 'd'; % 
  file_type = '.png';
  if nargin < 4 || isempty(random_order)
    random_order = 1; 
  endif
  if nargin < 3 || isempty(min_image_id)
    min_image_id = 1000; %% 2500; 
  endif
  if nargin < 2 || isempty(FC_list)
    FC_list = [ 2 4 6 8 ]; %%[4]; %% 
  endif
  num_FCs = length(FC_list);
  if nargin < 1 || isempty(N_ofeach)
    N_ofeach = 9000;
  endif
  N = num_FCs * N_ofeach;
  image_id = [];
  for i_FC = 1 : num_FCs
    image_id = [ image_id; repmat(FC_list(i_FC), N_ofeach, 1) ];
  endfor
  if random_order
    image_id = randperm(N) - 1;
  else
    image_id = 1:N;
    image_id = image_id - 1;
  endif
  filenames_cell = cell(N,1);
  fid = fopen([file_path, "amoeba", num2str(num_FCs), "x", num2str(N_ofeach), "_fileNames.txt"], 'w', 'native');
  for i = 1 : N
    FC_str = ...
	num2str( FC_list( ceil( (1+image_id(i)) / N_ofeach ) ) );
    image_id_tmp = mod( image_id(i), N_ofeach );
    image_id_tmp = image_id_tmp + min_image_id;
    id_str = ...
	num2str( image_id_tmp, NUM2STR_FORMAT );
    filename_str = ...
	[ file_path, ...
	 FC_str, ...
	 '/',  file_dir, '/', ...
	 file_root, ...
	 id_str, ...
	 target_flag, ...
	 file_type ];
    filenames_cell{i} = filename_str;
    fprintf(fid, "%s\n", filename_str);
  endfor
  fclose(fid);