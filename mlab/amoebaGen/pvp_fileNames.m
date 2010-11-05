function filenames_cell = pvp_fileNames(N_ofeach, FC_list)

  global NUM2STR_FORMAT
  NUM2STR_FORMAT = '%04.4i';
  file_path = ...
      '/nh/home/gkenyon/Documents/MATLAB/amoeba/128_png/'
%      '../../../../Documents/MATLAB/amoeba/128_png/';
%      '../PetaVision/mlab/amoebaGen/256/';
  file_root = 'tar_'; % 
  target_flag = '_a'; % '_n'; %
  file_dir = 't'; %'a'; % 'd'; % 
  file_type = '.png';
  if nargin < 2 || isempty(FC_list)
    FC_list = [ 2 4 6 8 ];
  end%%if
  num_FCs = length(FC_list);
  if nargin < 1 || isempty(N_ofeach)
    N_ofeach = 10000; %2500;
  end%%if
  N = num_FCs * N_ofeach;
  image_id = [];
  for i_FC = 1 : num_FCs
    image_id = [ image_id; repmat(FC_list(i_FC), N_ofeach, 1) ];
  end%%for
  image_id = randperm(N) - 1;
  filenames_cell = cell(N,1);
  fid = fopen('fileNames.txt', 'w', 'native');
  for i = 1 : N
    FC_str = ...
	num2str( FC_list( ceil( (1+image_id(i)) / N_ofeach ) ) );
    id_str = ...
	num2str( mod( image_id(i), N_ofeach ), NUM2STR_FORMAT );
    filename_str = ...
	[ file_path, ...
	 FC_str, ...
	 '/',  file_dir, '/', ...
	 file_root, ...
	 id_str, ...
	 target_flag, ...
	 file_type ];
    filenames_cell{i} = filename_str;
    fprintf(fid, '%s\n', filename_str);
  end%%for
  fclose(fid);