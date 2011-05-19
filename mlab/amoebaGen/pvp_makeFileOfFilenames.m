function filenames_cell = pvp_makeFileOfFilenames()

  image_dir_base = ...
    "~/Pictures/Textured_Dog_Cat_Spherical_T1/";       
  image_dir = ...
      [ image_dir_base, "Dog/"]; 
  image_subdir_name = "DoG";
  image_subdir = [image_dir, image_subdir_name, "/"];
  image_path = ...
      [image_subdir, "*.png"];
  [image_struct] = dir(image_path);
  num_images = size(image_struct,1);
  disp(['num_images = ', num2str(num_images)]);
  filenames_cell = cell(num_images,1);
  for i_image = 1 : num_images
    image_name = image_struct(i_image).name;
    image_full_name = [image_subdir, image_name];
    base_name = image_name(1:strfind(image_name, ".png")-1);
    filenames_cell{i_image} = image_full_name;
  endfor
  file_of_filenames = [image_dir, image_subdir_name, "_fileOfFilenames.txt"];
  fid = fopen(file_of_filenames, 'w', 'native');
  for i_image = 1 : num_images
    filename_str = filenames_cell{i_image};
    fprintf(fid, "%s\n", filename_str);
  endfor
  fclose(fid);