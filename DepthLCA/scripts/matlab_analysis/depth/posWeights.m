clear all; close all;

workspace_path = "/home/slundquist/workspace";
output_dir = "/nh/compneuro/Data/Depth/LCA/restart"; 

addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
addpath([workspace_path, filesep, "/PetaVision/mlab/HyPerLCA"]);

filesep = "/";
weights_list = ...
   {
      %["BinocularV1S1ToLeftDepthError_W"]; ...
      ["BinocularV1S2ToPosError_W"]; ...
      %["BinocularV1S1ToRightDepthError_W"]; ...
      %["BinocularV1S2ToRightDepthError_W"]; ...
   };
pre_list = ...
   {
      %["BinocularV1S1_A"]; ...
      ["BinocularV1S2_A"]; ...
   };

last_checkpoint_ndx = 10000;
checkpoint_path = [output_dir, filesep, "Checkpoints", filesep,  "Checkpoint", num2str(last_checkpoint_ndx, '%i')]; %% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
       {...
        %["a36_"], ["BinocularV1S1"]; ...
        ["a37_"], ["BinocularV1S2"]; ...
        };

    load_Sparse_flag = 0;
    plot_Sparse_flag = 1;
    fraction_Sparse_frames_read = 1;
    min_Sparse_skip = 1;
    fraction_Sparse_progress = 10;
    num_procs = nproc;

    [Sparse_hdr, ...
          Sparse_hist_rank_array, ...
          Sparse_times_array, ...
          Sparse_percent_active_array, ...
          Sparse_percent_change_array, ...
          Sparse_std_array] = ...
          analyzeSparsePVP(Sparse_list, ...
                output_dir, ...
                load_Sparse_flag, ...
                plot_Sparse_flag, ...
                fraction_Sparse_frames_read, ...
                min_Sparse_skip, ...
                fraction_Sparse_progress, ...
                num_procs);

num_weights_list = size(weights_list, 1);
for i_weights = 1:num_weights_list
   i_pre = i_weights;
   pre_file = [checkpoint_path, filesep, pre_list{i_pre}, '.pvp'];
   weights_file = [checkpoint_path, filesep, weights_list{i_weights}, '.pvp'];
   pre_fid = fopen(pre_file);
   pre_hdr{i_pre} = readpvpheader(pre_fid);
   fclose(pre_fid);
   [weights_struct, weights_hdr_tmp] = ...
   readpvpfile(weights_file);
   i_arbor = 1;
   i_frame = 1;
   if weights_hdr_tmp.nfp == 1
      weight = squeeze(weights_struct{i_frame}.values{i_arbor});
   else
      weight = weights_struct{i_frame}.values{i_arbor};
   endif
   pre_hist_rank = Sparse_hist_rank_array{i_weights};
   weight_time = squeeze(weights_struct{i_frame}.time);
   [nx, ny, nf, np] = size(weight);
   maxWeight = max(weight(:));
   minWeight = min(weight(:));
   if abs(maxWeight) >= abs(minWeight)
      scaleVal = abs(maxWeight);
   else
      scaleVal = abs(minWeight);
   endif
   %load("weight.mat");
   num_patches = np;
   num_patches_rows = floor(sqrt(np));
   num_patches_cols = ceil(num_patches / num_patches_rows);
   weights_fig = figure;
   set(weights_fig, "name", ["Weights_", weights_list{i_weights}, "_", num2str(weight_time)]);
   for i_patch = 1:np
      i = pre_hist_rank(i_patch);
      plotme = weight(:, :, :, i);
      [x, y, z] = meshgrid(1:nx, 1:ny, 1:nf);
      scatter3(x(:), y(:), z(:), 10, plotme(:));
      caxis([-scaleVal, scaleVal]);
      colorbar;
      pause;
   end
end
