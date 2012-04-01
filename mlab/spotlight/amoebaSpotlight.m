

num_images_per_max_RF = 1;
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

amoeba_struct.spotlight_amp = 0.5;
amoeba_struct.background_amp = 0.1;

for i_RF =  1 : num_RFs

  amoeba_struct.num_RF = max_RF_list(i_RF);
  amoeba_struct.RF_amp = zeros(amoeba_struc.num_RF, 1);

  amoeba_struct_array = cell(num_images_per_max_RF,1);
  amoeba_struct_array(1:num_images_per_max_RF,1) = amoeba_struct;
  
  if num_procs >= 1
    [amoeba_info] = cellfun( @amoeba2DKernel, amoeba_struct_array);
  endif

endfor
