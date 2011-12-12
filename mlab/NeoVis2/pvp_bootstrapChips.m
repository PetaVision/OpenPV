function [num_target_chips, num_distractor_chips] = ...
      pvp_bootstrapChips(frame_pathname, hit_list, miss_list);

  global target_bootstrap_dir
  global distractor_bootstrap_dir

  num_target_chips = 0;

  %% path to generic image processing routines
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);
  
  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);
    
  num_miss_BB = length(miss_list);
  num_distractor_chips = 0;
  if num_miss_BB > 0
    ave_miss_confidence = 0;
    std_miss_confidence = 0;
    for i_miss_BB = 1 : num_miss_BB
      ave_miss_confidence = ave_miss_confidence + miss_list{i_miss_BB}.Confidence;
      std_miss_confidence = std_miss_confidence + miss_list{i_miss_BB}.Confidence.^2;
    endfor
    ave_miss_confidence = ave_miss_confidence / num_miss_BB;
    std_miss_confidence = sqrt((std_miss_confidence / num_miss_BB) - (ave_miss_confidence.^2));
    for i_miss_BB = 1 : num_miss_BB
      if miss_list{i_miss_BB}.Confidence < ave_miss_confidence
	continue;
      endif
      x_BB_min2 = miss_list{i_miss_BB}.patch_X1;
      x_BB_max2 = miss_list{i_miss_BB}.patch_X2;
      y_BB_min2 = miss_list{i_miss_BB}.patch_Y1;
      y_BB_max2 = miss_list{i_miss_BB}.patch_Y3;
      pvp_image = imread(frame_pathname);
      miss_chip = pvp_image(y_BB_min2:y_BB_max2, x_BB_min2:x_BB_max2);
      miss_chip_imagename = strFolderFromPath(frame_pathname);
      miss_chip_rootname = strRemoveExtension(miss_chip_imagename);
      miss_chip_title = [miss_chip_rootname, "_", num2str(i_miss_BB, "%3.3d"), ".png"];
      miss_chip_pathname = [distractor_bootstrap_dir, miss_chip_title];
      imwrite(miss_chip, miss_chip_pathname);
      num_distractor_chips = num_distractor_chips + 1;
    endfor  %% i_BB
  endif  %% num_miss_BB > 0
  

