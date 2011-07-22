function filenames_cell = pvp_makeFileOfFilenames()

  image_dir_base = ...
          "~/MATLAB/figures/amoeba/64_png/4/";       
%%    "~/Pictures/Textured_Dog_Cat_Spherical_T1/";       
  image_dir = ...
      [ image_dir_base ];
%%      [ image_dir_base, "Cat/"]; 
  image_subdir_name = "t"; %% "DoG";
  image_subdir = [image_dir, image_subdir_name, "/"];
  image_path = ...
      [image_subdir, "*.png"];
  [image_struct] = dir(image_path);
  num_images = size(image_struct,1);
  disp(['num_images = ', num2str(num_images)]);
  shuffle_images = 1;
  if shuffle_images == 1
    [tmp, image_ndx] = sort(rand(num_images,1));
  else
    image_ndx = 1:num_images
  endif
  image_skip = 10;
  filenames_cell = cell(num_images,2);
  num_names = zeros(2,1);
  for i_image = 1 : num_images
    image_name = image_struct(image_ndx(i_image)).name;
    image_full_name = [image_subdir, image_name];
    base_name = image_name(1:strfind(image_name, ".png")-1);
    if mod( i_image, image_skip ) == 0
      num_names(1) = num_names(1) + 1;
      filenames_cell{num_names(1),1} = image_full_name;
    else
      num_names(2) = num_names(2) + 1;
      filenames_cell{num_names(2),2} = image_full_name;
    endif
  endfor
  file_type = {"test";"train"};
  for i_mode = 1 : 2
    if num_names(i_mode) > 0
      file_of_filenames = [image_dir, image_subdir_name, "_", ...
				   file_type{i_mode}, "Files.txt"];
      fid = fopen(file_of_filenames, 'w', 'native');
    else
      continue;
    endif
    for i_image = 1 : num_names(i_mode)
      filename_str = filenames_cell{i_image, i_mode};
      fprintf(fid, "%s\n", filename_str);
    endfor
    fclose(fid);
  endfor