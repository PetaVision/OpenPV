
clear all;
close all;
%setenv("GNUTERM","X11")

workspace_path = "/home/ec2-user/workspace";
output_dir = "/home/ec2-user/mountData/benchmark/train/aws_rcorr_LCA_saved/"; 
%

addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
addpath([workspace_path, filesep, "/PetaVision/mlab/HyPerLCA"]);

function I = disp_map(I)

map = [0 0 0 114; 0 0 1 185; 1 0 0 114; 1 0 1 174; ...
       0 1 0 114; 0 1 1 185; 1 1 0 114; 1 1 1 0];

bins  = map(1:end-1,4);
cbins = cumsum(bins);
bins  = bins./cbins(end);
cbins = cbins(1:end-1) ./ cbins(end);
ind   = min(sum(repmat(I(:)', [6 1]) > repmat(cbins(:), [1 numel(I)])),6) + 1;
bins  = 1 ./ bins;
cbins = [0; cbins];

I = (I-cbins(ind)) .* bins(ind);
I = min(max(map(ind,1:3) .* repmat(1-I, [1 3]) + map(ind+1,1:3) .* repmat(I, [1 3]),0),1);
end

function I = disp_to_color (D,max_disp)
% computes color representation of disparity map
% code adapted from Oliver Woodford's sc.m
% max_disp optionally specifies the scaling factor

D = double(D);

if nargin==1
  max_disp = max(D(:));
else
  max_disp = max(max_disp,1);
end

I = disp_map(min(D(:)/max_disp,1));
I = reshape(I, [size(D,1) size(D,2) 3]);
end




plot_flag = true;

last_checkpoint_ndx = 90;
checkpoint_path = output_dir; %% 
%checkpoint_path = [output_dir, filesep, "Last"]; %% 
max_history = 196000;
numarbors = 1;

%%keyboard;
plot_StatsProbe_vs_time = 0;
if plot_StatsProbe_vs_time
  first_StatsProbe_line = 1; %%max([(last_StatsProbe_line - StatsProbe_plot_lines), 1]);
  StatsProbe_plot_lines = 30000;
%%  StatsProbe_list = ...
%%      {["Error"],["_Stats.txt"]; ...
%%       ["V1"],["_Stats.txt"]};
  StatsProbe_list = ...
       {["BinocularV1S1"],["_Stats.txt"]; ...
       ["BinocularV1S2"],["_Stats.txt"]; ...
       ["BinocularV1S3"],["_Stats.txt"]; ...
       ["LeftError1"],["_Stats.txt"]; ...
       ["LeftError2"],["_Stats.txt"]; ...
       ["LeftError3"],["_Stats.txt"]; ...
       ["RightError1"],["_Stats.txt"]; ...
       ["RightError2"],["_Stats.txt"]; ...
       ["RightError3"],["_Stats.txt"]; ...
       ["LeftDepthError"],["_Stats.txt"]; ...
       ["RightDepthError"],["_Stats.txt"]; ...
       ["PosError"],["_Stats.txt"]; ...
       };
  StatsProbe_vs_time_dir = [output_dir, filesep, "StatsProbe_vs_time"];
  mkdir(StatsProbe_vs_time_dir);
  num_StatsProbe_list = size(StatsProbe_list,1);
  StatsProbe_sigma_flag = ones(1,num_StatsProbe_list);
  StatsProbe_sigma_flag([1]) = 0;
  StatsProbe_sigma_flag([2]) = 0;
  StatsProbe_sigma_flag([3]) = 0;
  StatsProbe_nnz_flag = ~StatsProbe_sigma_flag;
  for i_StatsProbe = 1 : num_StatsProbe_list
    StatsProbe_file = [output_dir, filesep, StatsProbe_list{i_StatsProbe,1}, StatsProbe_list{i_StatsProbe,2}]
    if ~exist(StatsProbe_file,"file")
      error(["StatsProbe_file does not exist: ", StatsProbe_file]);
    endif
    [status, wc_output] = system(["cat ",StatsProbe_file," | wc"], true, "sync");
    if status ~= 0
      error(["system call to compute num lines failed in file: ", StatsProbe_file, " with status: ", num2str(status)]);
    endif
    wc_array = strsplit(wc_output, " ", true);
    StatsProbe_num_lines = str2num(wc_array{1});
    StatsProbe_fid = fopen(StatsProbe_file, "r");
    StatsProbe_line = fgets(StatsProbe_fid);
    StatsProbe_time_vals = [];
    StatsProbe_sigma_vals = [];
    StatsProbe_nnz_vals = [];
    skip_StatsProbe_line = 1; %%2000 per time update
    last_StatsProbe_line = StatsProbe_plot_lines; %% StatsProbe_num_lines - 2;
    num_lines = floor((last_StatsProbe_line - first_StatsProbe_line)/ skip_StatsProbe_line);

    %StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    %StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    %StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);

    for i_line = 1:first_StatsProbe_line-1
      StatsProbe_line = fgets(StatsProbe_fid);
    endfor
    %% extract N
    StatsProbe_N_ndx1 = strfind(StatsProbe_line, "N==");
    StatsProbe_N_ndx2 = strfind(StatsProbe_line, "Total==");
    StatsProbe_N_str = StatsProbe_line(StatsProbe_N_ndx1+3:StatsProbe_N_ndx2-2);
    StatsProbe_N = str2num(StatsProbe_N_str);
    for i_line = 1:num_lines
      %Skip lines based on how many was skipped
      for s_line = 1:skip_StatsProbe_line
         StatsProbe_line = fgets(StatsProbe_fid);
      endfor
      %% extract time
      StatsProbe_time_ndx1 = strfind(StatsProbe_line, "t==");
      StatsProbe_time_ndx2 = strfind(StatsProbe_line, "N==");
      StatsProbe_time_str = StatsProbe_line(StatsProbe_time_ndx1+3:StatsProbe_time_ndx2-2);
      StatsProbe_time_vals(i_line) = str2num(StatsProbe_time_str);
      %% extract sigma
      StatsProbe_sigma_ndx1 = strfind(StatsProbe_line, "sigma==");
      StatsProbe_sigma_ndx2 = strfind(StatsProbe_line, "nnz==");
      StatsProbe_sigma_str = StatsProbe_line(StatsProbe_sigma_ndx1+7:StatsProbe_sigma_ndx2-2);
      StatsProbe_sigma_vals(i_line) = str2num(StatsProbe_sigma_str);
      %% extract nnz
      StatsProbe_nnz_ndx1 = strfind(StatsProbe_line, "nnz==");
      StatsProbe_nnz_ndx2 = length(StatsProbe_line); 
      StatsProbe_nnz_str = StatsProbe_line(StatsProbe_nnz_ndx1+5:StatsProbe_nnz_ndx2-1);
      StatsProbe_nnz_vals(i_line) = str2num(StatsProbe_nnz_str);
    endfor %%i_line
    fclose(StatsProbe_fid);
    StatsProbe_vs_time_fig(i_StatsProbe) = figure;
    if StatsProbe_nnz_flag(i_StatsProbe)
      StatsProbe_vs_time_hndl = plot(StatsProbe_time_vals, StatsProbe_nnz_vals/StatsProbe_N); axis tight;
      axis tight
      set(StatsProbe_vs_time_fig(i_StatsProbe), "name", [StatsProbe_list{i_StatsProbe,1}, " nnz"]);
      saveas(StatsProbe_vs_time_fig(i_StatsProbe), ...
	     [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
	      "_nnz_vs_time_", num2str(StatsProbe_time_vals(end), "%i")], "png");
    else
      StatsProbe_vs_time_hndl = plot(StatsProbe_time_vals, StatsProbe_sigma_vals); axis tight;
      axis tight
      set(StatsProbe_vs_time_fig(i_StatsProbe), "name", [StatsProbe_list{i_StatsProbe,1}, " sigma"]);
      saveas(StatsProbe_vs_time_fig(i_StatsProbe), ...
	     [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
	      "_sigma_vs_time_", num2str(StatsProbe_time_vals(end), "%i")], "png");
    endif %% 
    drawnow;
  endfor %% i_StatsProbe
endif  %% plot_StatsProbe_vs_time
%Prints weights from checkpoints

%%keyboard;
plot_weights = true;
if plot_weights

   weights_list = ...
       { ...
        %["LCA_V1ToLeftRecon_W"]; ...
        %["LCA_V1ToRightRecon_W"]; ...
        ["LCA_V1ToDepthGT_W"]; ...
        };
   pre_list = ...
       { ...
        %["LCA_V1_A"]; ...
        %["LCA_V1_A"]; ...
        ["LCA_V1_A"]; ...
        };
   sparse_ndx = ...
        [   ...
        %1;  ...
        %1;  ...
        1;  ...
        ];
   checkpoints_list = {checkpoint_path};
   num_checkpoints = size(checkpoints_list,1);
   checkpoint_weights_movie = true;
   no_clobber = false;
   max_patches = 999999;
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

      if(ndims(weight_vals) == 4)
         %Average x and y of pre
         pre_dim = weights_hdr_tmp.numPatches / weights_hdr_tmp.nf;
         [nxp, nyp, nfp, numPatches] = size(weight_vals);
         if(pre_dim ~= 1)
            %Extract nf from pre patch
            weight_vals = reshape(weight_vals, [nxp, nyp, nfp, weights_hdr_tmp.nf, pre_dim]);
            %Find mean of the new dimension
            weight_vals = mean(weight_vals, 5);
         end
         assert(ndims(weight_vals) == 4);
      end

      weight_time = squeeze(weights_struct{i_frame}.time);
      weights_name =  [weights_list{i_weights,1}, "_", num2str(weight_time, "%08d")];
      if no_clobber && exist([weights_movie_dir, filesep, weights_name, ".png"]) && i_checkpoint ~= max_checkpoint
	continue;
      endif
      tmp_ndx = sparse_ndx(i_weights);
      tmp_rank = [];
      pre_hist_rank = (1:weights_hdr{i_weights}.nf);

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
     %Average over x and y to get only bin values
     if(size(patch_tmp2, 3) == 1 || size(patch_tmp2, 3) == 3)
        imagesc(patch_tmp2); 
     else
        [nxp, nyp, nfp] = size(patch_tmp2);
        [drop, patch_tmp2] = max(patch_tmp2, [], 3);
        color_patch = disp_to_color(patch_tmp2, nfp);
        imagesc(color_patch, [0 1]);
     end
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
			   ((col_ndx-1)*size(patch_tmp2,2)+1):col_ndx*size(patch_tmp2,2),:) = patch_tmp2;
   endfor  %% j_patch

   if plot_flag && i_checkpoint == max_checkpoint
     saveas(weights_fig, [weights_dir, filesep, weights_name, ".png"], "png");
   endif

   weight_patch_array = disp_to_color(weight_patch_array);
   weight_patch_array = weight_patch_array * 255;

   imwrite((uint8)(weight_patch_array), [weights_movie_dir, filesep, weights_name, ".png"], "png");

    endfor %% i_checkpoint
  endfor %% i_weights
endif  %% plot_weights
