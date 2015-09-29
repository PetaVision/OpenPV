
clear all;
close all;
%setenv("GNUTERM","X11")

workspace_path = "/home/slundquist/workspace";
output_dir = "/nh/compneuro/Data/Depth/NIPS/finetuned/validate/aws_icapatch_LCA_fine"
%output_dir = "/nh/compneuro/Data/Depth/LCA/arbortest/"; 

addpath([workspace_path, filesep, "/OpenPV/pv-core/mlab/util"]);
addpath([workspace_path, filesep, "/OpenPV/pv-core/mlab/HyPerLCA"]);
checkpoint_path = [output_dir, filesep, "Last/"];
max_history = 100000000;
numarbors = 1;

max_patches = 512;

analyze_Sparse_flag = true;
if analyze_Sparse_flag
    Sparse_list = ...
       {["a12_"], ["V1_LCA"]; ...
        };

    load_Sparse_flag = 0;
    plot_flag = 1;

    fraction_Sparse_frames_read = 1;
    min_Sparse_skip = 1;
    fraction_Sparse_progress = 1;
    num_procs = 8;
    num_epochs = 4;
    Sparse_frames_list = [];

  [Sparse_hdr, ...
   Sparse_hist_rank_array, ...
   Sparse_times_array, ...
   Sparse_percent_active_array, ...
   Sparse_percent_change_array, ...
   Sparse_std_array, ...
   Sparse_struct_array] = ...
      analyzeSparseEpochsPVP2(Sparse_list, ...
			     output_dir, ...
			     load_Sparse_flag, ...
			     plot_flag, ...
			     fraction_Sparse_frames_read, ...
			     min_Sparse_skip, ...
			     fraction_Sparse_progress, ...
			     Sparse_frames_list, ...
			     num_procs, ...
			     num_epochs);
endif

plot_flag = 0;
%%keyboard;
plot_weights = true;
if plot_weights

   weights_list = ...
       { ...
        ["LCA_V1ToLeftRecon_W"]; ...
        ["LCA_V1ToRightRecon_W"]; ...
        };
   pre_list = ...
       { ...
        ["V1_A"]; ...
        ["V1_A"]; ...
        };
   sparse_ndx = ...
        [   ...
        1;  ...
        1;  ...
        ];
   checkpoints_list = {checkpoint_path};
   num_checkpoints = size(checkpoints_list,1);
   checkpoint_weights_movie = true;
   no_clobber = false;
   weights_movie_dir = [output_dir, filesep, "weights_movie"]


  num_weights_list = size(weights_list,1);
  weights_hdr = cell(num_weights_list,1);
  pre_hdr = cell(num_weights_list,1);
  if checkpoint_weights_movie
    weights_movie_dir = [output_dir, filesep, "weights_movie"]
    [status, msg, msgid] = mkdir(weights_movie_dir);
    if status ~= 1
      warning(["mkdir(", weights_movie_dir, ")", " msg = ", msg]);
    endif 
  endif
  weights_dir = [output_dir, filesep, "weights"]
  [status, msg, msgid] = mkdir(weights_dir);
  if status ~= 1
    warning(["mkdir(", weights_dir, ")", " msg = ", msg]);
  endif 
  for i_weights = 1 : num_weights_list
    max_weight_time = 0;
    max_checkpoint = 0;
    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights_file = [checkpoint_dir, filesep, weights_list{i_weights,1}, ".pvp"];
      if ~exist(weights_file, "file")
	warning(["file does not exist: ", weights_file]);
	continue;
      endif
      weights_fid = fopen(weights_file);
      weights_hdr{i_weights} = readpvpheader(weights_fid);    
      fclose(weights_fid);

      weight_time = weights_hdr{i_weights}.time;
      if weight_time > max_weight_time
	max_weight_time = weight_time;
	max_checkpoint = i_checkpoint;
      endif
    endfor %% i_checkpoint

    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights_file = [checkpoint_dir, filesep, weights_list{i_weights,1}, ".pvp"];
      if ~exist(weights_file, "file")
	warning(["file does not exist: ", weights_file]);
	continue;
      endif
      weights_fid = fopen(weights_file);
      weights_hdr{i_weights} = readpvpheader(weights_fid);    
      fclose(weights_fid);
      weights_filedata = dir(weights_file);
      weights_framesize = ...
	  weights_hdr{i_weights}.recordsize*weights_hdr{i_weights}.numrecords+weights_hdr{i_weights}.headersize;
      tot_weights_frames = weights_filedata(1).bytes/weights_framesize;
      num_weights = 1;
      progress_step = ceil(tot_weights_frames / 10);
      [weights_struct, weights_hdr_tmp] = ...
	  readpvpfile(weights_file, progress_step, tot_weights_frames, tot_weights_frames-num_weights+1);
      i_frame = num_weights;
      i_arbor = 1;
      weight_vals = squeeze(weights_struct{i_frame}.values{i_arbor});
      weight_time = squeeze(weights_struct{i_frame}.time);
      weights_name =  [weights_list{i_weights,1}, "_", num2str(weight_time, "%08d")];
      if no_clobber && exist([weights_movie_dir, filesep, weights_name, ".png"]) && i_checkpoint ~= max_checkpoint
	continue;
      endif
      tmp_ndx = sparse_ndx(i_weights);
      if analyze_Sparse_flag
	tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
	tmp_rank = [];
      endif
      if analyze_Sparse_flag && ~isempty(tmp_rank)
	pre_hist_rank = tmp_rank;
      else
	pre_hist_rank = (1:weights_hdr{i_weights}.nf);
      endif
      %% make tableau of all patches
      %%keyboard;
      i_patch = 1;
      num_weights_dims = ndims(weight_vals);
      num_patches = size(weight_vals, num_weights_dims);
      num_patches = min(num_patches, max_patches);
      num_patches_rows = floor(sqrt(num_patches));
      num_patches_cols = ceil(num_patches / num_patches_rows);
      num_weights_colors = 1;
      if num_weights_dims == 4
	num_weights_colors = size(weight_vals,3);
      endif
      if plot_flag && i_checkpoint == max_checkpoint
	weights_fig = figure;
	set(weights_fig, "name", weights_name);
      endif
      weight_patch_array = [];
      for j_patch = 1  : num_patches
	i_patch = pre_hist_rank(j_patch);
	if plot_flag && i_checkpoint == max_checkpoint
	  subplot(num_patches_rows, num_patches_cols, j_patch); 
	endif
	if num_weights_colors == 1
	  patch_tmp = squeeze(weight_vals(:,:,i_patch));
	else
	  patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
	endif
	patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
	min_patch = min(patch_tmp2(:));
	max_patch = max(patch_tmp2(:));
	patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
	patch_tmp2 = uint8(permute(patch_tmp2, [2,1,3])); %% uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
	if plot_flag && i_checkpoint == max_checkpoint
	  imagesc(patch_tmp2); 
	  if num_weights_colors == 1
	    colormap(gray);
	  endif
	  box off
	  axis off
	  axis image
	  %if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
	  %  [~, max_label] = max(squeeze(labelWeights_vals(:,i_patch)));
	  %  text(size(weight_vals,1)/2, -size(weight_vals,2)/6, num2str(max_label-1), "color", [1 0 0]);
	  %endif %% ~empty(labelWeights_vals)
	  %%drawnow;
	endif %% plot_flag
	if isempty(weight_patch_array)
	  weight_patch_array = ...
	      zeros(num_patches_rows*size(patch_tmp2,1), num_patches_cols*size(patch_tmp2,2), size(patch_tmp2,3));
	endif
	col_ndx = 1 + mod(j_patch-1, num_patches_cols);
	row_ndx = 1 + floor((j_patch-1) / num_patches_cols);
	weight_patch_array(((row_ndx-1)*size(patch_tmp2,1)+1):row_ndx*size(patch_tmp2,1), ...
			   ((col_ndx-1)*size(patch_tmp2,2)+1):col_ndx*size(patch_tmp2,2),:) = ...
	    patch_tmp2;
      endfor  %% j_patch
      if plot_flag && i_checkpoint == max_checkpoint
	  saveas(weights_fig, [weights_dir, filesep, weights_name, ".png"], "png");
      endif
      imwrite(uint8(weight_patch_array), [weights_movie_dir, filesep, weights_name, ".png"], "png");
      %% make histogram of all weights
      if plot_flag && i_checkpoint == max_checkpoint
	weights_hist_fig = figure;
	[weights_hist, weights_hist_bins] = hist(weight_vals(:), 100);
	bar(weights_hist_bins, log(weights_hist+1));
	set(weights_hist_fig, "name", ...
	    ["Hist_",  weights_list{i_weights,1}, "_", num2str(weight_time, "%08d")]);
	saveas(weights_hist_fig, ...
	       [weights_dir, filesep, "weights_hist_", num2str(weight_time, "%08d")], "png");
      endif
    endfor %% i_checkpoint
  endfor %% i_weights
endif  %% plot_weights

