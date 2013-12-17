clear all; close all;

workspace_path = "/home/slundquist/workspace";
output_dir = "/nh/compneuro/Data/Depth/LCA/dataset01"; 
filesep = "/";
weights_list = ...
   {
      %["BinocularV1S1ToLeftDepthError_W"]; ...
      ["BinocularV1S2ToLeftDepthError_W"]; ...
      %["BinocularV1S1ToRightDepthError_W"]; ...
      %["BinocularV1S2ToRightDepthError_W"]; ...
   };
pre_list = ...
   {
      %["BinocularV1S1_A"]; ...
      ["BinocularV1S2_A"]; ...
   };

addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
last_checkpoint_ndx = 380000;
checkpoint_path = [output_dir, filesep, "Checkpoints", filesep,  "Checkpoint", num2str(last_checkpoint_ndx, '%i')]; %% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Sparse_list = ...
   {
   %["a34_"], ["BinocularV1S1"];...
   ["a35_"], ["BinocularV1S2"];...
   };
num_Sparse_list = size(Sparse_list,1);
Sparse_hdr = cell(num_Sparse_list,1);
Sparse_hist_rank = cell(num_Sparse_list,1);
Sparse_dir = [output_dir, filesep, "Sparse"];
mkdir(Sparse_dir);
for i_Sparse = 1 : num_Sparse_list
 Sparse_file = [output_dir, filesep, Sparse_list{i_Sparse,1}, Sparse_list{i_Sparse,2}, ".pvp"]
 if ~exist(Sparse_file, "file")
   error(["file does not exist: ", Sparse_file]);
 endif
 Sparse_fid = fopen(Sparse_file);
 Sparse_hdr{i_Sparse} = readpvpheader(Sparse_fid);
 fclose(Sparse_fid);
 tot_Sparse_frames = Sparse_hdr{i_Sparse}.nbands;
 num_Sparse = tot_Sparse_frames;
 progress_step = ceil(tot_Sparse_frames / 10);
 [Sparse_struct, Sparse_hdr_tmp] = ...
readpvpfile(Sparse_file, progress_step, tot_Sparse_frames, tot_Sparse_frames-num_Sparse+1);
 nx_Sparse = Sparse_hdr{i_Sparse}.nx;
 ny_Sparse = Sparse_hdr{i_Sparse}.ny;
 nf_Sparse = Sparse_hdr{i_Sparse}.nf;
 n_Sparse = nx_Sparse * ny_Sparse * nf_Sparse;
 num_frames = size(Sparse_struct,1);
 Sparse_hist = zeros(nf_Sparse+1,1);
 Sparse_hist_edges = [0:1:nf_Sparse]+0.5;
 Sparse_current = zeros(n_Sparse,1);
 Sparse_abs_change = zeros(num_frames,1);
 Sparse_percent_change = zeros(num_frames,1);
 Sparse_current_active = 0;
 Sparse_tot_active = zeros(num_frames,1);
 Sparse_times = zeros(num_frames,1);
 for i_frame = 1 : 1 : num_frames
   Sparse_times(i_frame) = squeeze(Sparse_struct{i_frame}.time);
   Sparse_active_ndx = squeeze(Sparse_struct{i_frame}.values);
   Sparse_previous = Sparse_current;
   Sparse_current = full(sparse(Sparse_active_ndx+1,1,1,n_Sparse,1,n_Sparse));
   Sparse_abs_change(i_frame) = sum(Sparse_current(:) ~= Sparse_previous(:));
   Sparse_previous_active = Sparse_current_active;
   Sparse_current_active = nnz(Sparse_current(:));
   Sparse_tot_active(i_frame) = Sparse_current_active;
   Sparse_max_active = max(Sparse_current_active, Sparse_previous_active);
   Sparse_percent_change(i_frame) = ...
  Sparse_abs_change(i_frame) / (Sparse_max_active + (Sparse_max_active==0));
   Sparse_active_kf = mod(Sparse_active_ndx, nf_Sparse) + 1;
   if Sparse_max_active > 0
Sparse_hist_frame = histc(Sparse_active_kf, Sparse_hist_edges);
   else
Sparse_hist_frame = zeros(nf_Sparse+1,1);
   endif
   Sparse_hist = Sparse_hist + Sparse_hist_frame;
 endfor %% i_frame
 Sparse_hist = Sparse_hist(1:nf_Sparse);
 Sparse_hist = Sparse_hist / (num_frames * nx_Sparse * ny_Sparse); %% (sum(Sparse_hist(:)) + (nnz(Sparse_hist)==0));
 [Sparse_hist_sorted, Sparse_hist_rank{i_Sparse}] = sort(Sparse_hist, 1, "descend");
endfor

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
   pre_hist_rank = Sparse_hist_rank{i_weights};
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
