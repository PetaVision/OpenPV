clear all

num_images = 20;
machine_path = "~/Pictures/amoeba2D/";
mkdir(machine_path);

max_RF_list = [4]; %%[2 4 6 8];
num_RFs = length(max_RF_list);
image_dim = [256 256];
num_procs = 1;

amoeba_struct = struct;

amoeba_struct.rand_state = {rand('state')};
amoeba_struct.image_rect_size = image_dim;

amoeba_struct.num_targets = 1;
amoeba_struct.target_outer_max = 0.5;%max/min outer radius of target annulus, units of image rect
amoeba_struct.target_outer_min = 0.5; %% value in Geisler paper
amoeba_struct.target_inner_max = 0.5;%max/min inner radius in units of outer radius
amoeba_struct.target_inner_min = 0.5; %% value in Geisler paper
				% set amp of largest fourier component factor of 2 larger to make more distinct amoebas
				% amoeba_struct.fourier_amp(amoeba_struct.num_fourier,1) = 1;
amoeba_struct.num_phi = 1024;
amoeba_struct.foreground_amp = 0.25;

fInv_struct = struct;
fInv_struct.image_rect_size = image_dim;
fInv_struct.background_amp = 0.1;

for i_RF =  1 : num_RFs

  amoeba_struct.num_RF = max_RF_list(i_RF);
  amoeba_struct.RF_amp = zeros(amoeba_struct.num_RF, 1);

  amoeba_struct_array = cell(num_images,1);
  amoeba_struct_array(1:num_images,1) = amoeba_struct;
  
  if num_procs >= 1
    [foreground_array] = cellfun( @amoeba2DKernel, amoeba_struct_array, "UniformOutput", false);
  endif

  fInv_struct_array = cell(num_images,1);
  fInv_struct_array(1:num_images,1) = fInv_struct;

  if num_procs >= 1
    [background_array] = cellfun( @fInvKernel, fInv_struct_array, "UniformOutput", false);
  endif

  if num_procs >= 1
    [spotlight_array] = cellfun( @spotlightKernel, background_array, foreground_array, "UniformOutput", false);
  endif

  output_path = [machine_path, num2str(max_RF_list(i_RF)), "FC", filesep];
  mkdir(output_path);

  target_path = [output_path, "target", filesep];
  mkdir(target_path);
  for i_image = 1 : num_images
    spotlight_image = uint8(spotlight_array{i_image});
    imwrite(spotlight_image, [target_path, "amoeba2D_", num2str(2*i_image-1, "%4.4i"), ".png"], "png");
  endfor

  if num_procs >= 1
    [distractor_array] = cellfun( @fInvKernel, fInv_struct_array, "UniformOutput", false);
  endif

  distractor_path = [output_path, "fInv", filesep];
  mkdir(distractor_path);
  for i_image = 1 : num_images
    fInv_image = uint8(distractor_array{i_image}.background_image);
    imwrite(fInv_image, [distractor_path, "amoeba2D_", num2str(2*i_image, "%4.4i"), ".png"], "png");
  endfor

endfor
