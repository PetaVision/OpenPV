function filenames_cell = pvp_animalFilenames()

  image_dir = ...
      "/Users/gkenyon/workspace/kernel/input/256/AnimalDB/Distractors/"; 
  original_dir = ...
      [ image_dir, "original/" ];
  image_path = ...
      [image_dir, "original/", "*.jpg"];
  gray_dir = ...
      [ image_dir, "gray/" ];
  if ~exist( 'gray_dir', 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', gray_dir); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif
  [image_struct] = dir(image_path);
  num_images = size(image_struct,1);
  disp(['num_images = ', num2str(num_images)]);
  filenames_cell = cell(num_images,1);
  for i_image = 1 : num_images
    image_name = image_struct(i_image).name;
    image_full_name = [original_dir, image_name];
    base_name = image_name(1:strfind(image_name, ".jpg")-1);
    [image_color, image_map, image_alpha] = ...
	imread(image_full_name);
    image_gray = col2gray(image_color);
    gray_filename = [gray_dir, base_name];
    filename_str = [gray_filename, ".png"];
    filenames_cell{i_image} = filename_str;
    savefile2(gray_filename, image_gray);
  endfor
  file_of_filenames = [image_dir, "noanimalFileNames.txt"];
  fid = fopen(file_of_filenames, 'w', 'native');
  for i_image = 1 : num_images
    filename_str = filenames_cell{i_image};
    fprintf(fid, "%s\n", filename_str);
  endfor
  fclose(fid);