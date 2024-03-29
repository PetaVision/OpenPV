
clear all;
close all;
%setenv("GNUTERM","X11")

workspace_path = "~/workspaceGit/OpenPV/";
output_dir = "~/workspaceGit/OpenPV/demo/binocDemo/output/";
%output_dir = "/nh/compneuro/Data/Depth/LCA/arbortest/"; 

addpath([workspace_path, filesep, "/pv-core/mlab/util"]);
addpath([workspace_path, filesep, "/pv-core/mlab/HyPerLCA"]);
last_checkpoint_ndx = 10;
checkpoint_path = [output_dir, filesep, "Checkpoints", filesep,  "Checkpoint", num2str(last_checkpoint_ndx, '%i')]; %% 
max_history = 100000000;
numarbors = 1;

plot_flag = 1;
analyze_Sparse_flag = true;
if analyze_Sparse_flag
    Sparse_list = ...
       {[""], ["V1"]; ...
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
end%if

analyze_nonSparse_flag = true;
if analyze_nonSparse_flag
    nonSparse_list = ...
        {[""], ["LeftError"]; ...
         [""], ["RightError"]; ...
         };
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(10, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {...
         [""], ["LeftRescale"]; ...
         [""], ["RightRescale"]; ...
         }; ...
    nonSparse_norm_strength = [1 1];
    Sparse_std_ndx = [0 0];
    plot_flag = true;

  if ~exist("Sparse_std_ndx")
    Sparse_std_ndx = zeros(num_nonSparse_list,1);
  end%if
  if ~exist("nonSparse_norm_strength")
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
  end%if

  fraction_nonSparse_frames_read = 1;
  min_nonSparse_skip = 1;
  fraction_nonSparse_progress = 10;
  [nonSparse_times_array, ...
   nonSparse_RMS_array, ...
   nonSparse_norm_RMS_array, ...
   nonSparse_RMS_fig] = ...
      analyzeNonSparsePVP(nonSparse_list, ...
                       nonSparse_skip, ...
                       nonSparse_norm_list, ...
                       nonSparse_norm_strength, ...
                       Sparse_times_array, ...
                       Sparse_std_array, ...
                       Sparse_std_ndx, ...
                       output_dir, ...
                       plot_flag, ...
                       fraction_nonSparse_frames_read, ...
                       min_nonSparse_skip, ...
                       fraction_nonSparse_progress);

end%if %% analyze_nonSparse_flag

%%keyboard;
plot_flag = false;
plot_weights = true;
if plot_weights

   weights_list = ...
       { ...
        ["V1ToLeftError_W"]; ...
        ["V1ToRightError_W"]; ...
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
   max_patches = 512;
   weights_movie_dir = [output_dir, filesep, "weights_movie"]


  num_weights_list = size(weights_list,1);
  weights_hdr = cell(num_weights_list,1);
  pre_hdr = cell(num_weights_list,1);
  if checkpoint_weights_movie
    weights_movie_dir = [output_dir, filesep, "weights_movie"]
    [status, msg, msgid] = mkdir(weights_movie_dir);
    if status ~= 1
      warning(["mkdir(", weights_movie_dir, ")", " msg = ", msg]);
    end%if 
  end%if
  weights_dir = [output_dir, filesep, "weights"]
  [status, msg, msgid] = mkdir(weights_dir);
  if status ~= 1
    warning(["mkdir(", weights_dir, ")", " msg = ", msg]);
  end%if 
  for i_weights = 1 : num_weights_list
    max_weight_time = 0;
    max_checkpoint = 0;
    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights_file = [checkpoint_dir, filesep, weights_list{i_weights,1}, ".pvp"];
      if ~exist(weights_file, "file")
        warning(["file does not exist: ", weights_file]);
        continue;
      end%if
      weights_fid = fopen(weights_file);
      weights_hdr{i_weights} = readpvpheader(weights_fid);    
      fclose(weights_fid);

      weight_time = weights_hdr{i_weights}.time;
      if weight_time > max_weight_time
        max_weight_time = weight_time;
        max_checkpoint = i_checkpoint;
      end%if
    end%for %% i_checkpoint

    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights_file = [checkpoint_dir, filesep, weights_list{i_weights,1}, ".pvp"];
      if ~exist(weights_file, "file")
        warning(["file does not exist: ", weights_file]);
        continue;
      end%if
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
      end%if
      tmp_ndx = sparse_ndx(i_weights);
      if analyze_Sparse_flag
        tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
        tmp_rank = [];
      end%if
      if analyze_Sparse_flag && ~isempty(tmp_rank)
        pre_hist_rank = tmp_rank;
      else
        pre_hist_rank = (1:weights_hdr{i_weights}.nf);
      end%if

   %   if length(labelWeights_list) >= i_weights && ...
        %    ~isempty(labelWeights_list{i_weights}) && ...
        %    plot_flag && ...
        %    i_checkpoint == max_checkpoint
        %labelWeights_file = ...
        %    [checkpoint_dir, filesep, labelWeights_list{i_weights,1}, labelWeights_list{i_weights,2}, ".pvp"]
        %if ~exist(labelWeights_file, "file")
        %  warning(["file does not exist: ", labelWeights_file]);
        %  continue;
        %end%if
        %labelWeights_fid = fopen(labelWeights_file);
        %labelWeights_hdr{i_weights} = readpvpheader(labelWeights_fid);    
        %fclose(labelWeights_fid);
        %num_labelWeights = 1;
        %labelWeights_filedata = dir(labelWeights_file);
        %labelWeights_framesize = ...
        %    labelWeights_hdr{i_weights}.recordsize * ...
        %    labelWeights_hdr{i_weights}.numrecords+labelWeights_hdr{i_weights}.headersize;
        %tot_labelWeights_frames = labelWeights_filedata(1).bytes/labelWeights_framesize;
        %[labelWeights_struct, labelWeights_hdr_tmp] = ...
        %    readpvpfile(labelWeights_file, ...
        %                progress_step, ...
        %                tot_labelWeights_frames, ...
        %                tot_labelWeights_frames-num_labelWeights+1);
        %labelWeights_vals = squeeze(labelWeights_struct{i_frame}.values{i_arbor});
        %labelWeights_time = squeeze(labelWeights_struct{i_frame}.time);
   %   else
        %labelWeights_vals = [];
        %labelWeights_time = [];
   %   end%if

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
      end%if
      if plot_flag && i_checkpoint == max_checkpoint
        weights_fig = figure;
        set(weights_fig, "name", weights_name);
      end%if
      weight_patch_array = [];
      for j_patch = 1  : num_patches
        i_patch = pre_hist_rank(j_patch);
        if plot_flag && i_checkpoint == max_checkpoint
          subplot(num_patches_rows, num_patches_cols, j_patch); 
        end%if
        if num_weights_colors == 1
          patch_tmp = squeeze(weight_vals(:,:,i_patch));
        else
          patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
        end%if
        patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
        min_patch = min(patch_tmp2(:));
        max_patch = max(patch_tmp2(:));
        patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
        patch_tmp2 = uint8(permute(patch_tmp2, [2,1,3])); %% uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
        if plot_flag && i_checkpoint == max_checkpoint
          imagesc(patch_tmp2); 
          if num_weights_colors == 1
            colormap(gray);
          end%if
          box off
          axis off
          axis image
          %if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
          %  [~, max_label] = max(squeeze(labelWeights_vals(:,i_patch)));
          %  text(size(weight_vals,1)/2, -size(weight_vals,2)/6, num2str(max_label-1), "color", [1 0 0]);
          %end%if %% ~empty(labelWeights_vals)
          %%drawnow;
        end%if %% plot_flag
        if isempty(weight_patch_array)
          weight_patch_array = ...
              zeros(num_patches_rows*size(patch_tmp2,1), num_patches_cols*size(patch_tmp2,2), size(patch_tmp2,3));
        end%if
        col_ndx = 1 + mod(j_patch-1, num_patches_cols);
        row_ndx = 1 + floor((j_patch-1) / num_patches_cols);
        weight_patch_array(((row_ndx-1)*size(patch_tmp2,1)+1):row_ndx*size(patch_tmp2,1), ...
                           ((col_ndx-1)*size(patch_tmp2,2)+1):col_ndx*size(patch_tmp2,2),:) = ...
            patch_tmp2;
      end%for  %% j_patch
      if plot_flag && i_checkpoint == max_checkpoint
          saveas(weights_fig, [weights_dir, filesep, weights_name, ".png"], "png");
      end%if
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
      end%if

%      if ~isempty(labelWeights_vals) && ...
%            ~isempty(labelWeights_time) && ...
%            plot_flag && ...
%            i_checkpoint == max_checkpoint
%        %% plot label weights as matrix of column vectors
%        [~, maxnum] = max(labelWeights_vals,[],1);
%        [maxnum,maxind] = sort(maxnum);
%        label_weights_fig = figure;
%        imagesc(labelWeights_vals(:,maxind))
%        label_weights_str = ...
%            ["LabelWeights_", labelWeights_list{i_weights,1}, labelWeights_list{i_weights,2}, ...
%             "_", num2str(labelWeights_time, "%08d")];
%        %%title(label_weights_fig, label_weights_str);
%        figure(label_weights_fig, "name", label_weights_str); title(label_weights_str);
%        saveas(label_weights_fig, [weights_dir, filesep, label_weights_str, ".png"] , "png");
%
%        %% Plot the average movie weights for a label %%
%        labeledWeights_str = ...
%            ["labeledWeights_", ...
%             weights_list{i_weights,1}, "_", ...
%             num2str(weight_time, "%08d")];
%        labeledWeights_fig = figure("name", labeledWeights_str);
%        title(labeledWeights_str);
%        rows_labeledWeights = ceil(sqrt(size(labelWeights_vals,1)));
%        cols_labeledWeights = ceil(size(labelWeights_vals,1) / rows_labeledWeights);
%        for label = 0 : size(labelWeights_vals,1)-1 %% anything 0:0
%          subplot(rows_labeledWeights, cols_labeledWeights, label+1);
%          if num_weights_colors == 1
%            imagesc(squeeze(mean(weight_vals(:,:,maxind(maxnum==(label+1))),3))')
%          else
%            imagesc(permute(squeeze(mean(weight_vals(:,:,:,1+mod(maxind(maxnum==(label+1))-1,size(weight_vals,4))),4)),[2,1,3]));
%          end%if
%          labeledWeights_subplot_str = ...
%              [num2str(label, "%d")];
%          title(labeledWeights_subplot_str);
%          axis off
%        end%for %% label
%        saveas(labeledWeights_fig,  [weights_dir, filesep, labeledWeights_str, ".png"], "png");
%      end%if  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

    end%for %% i_checkpoint
  end%for %% i_weights
end%if  %% plot_weights

%plot_weights = 1;
%if plot_weights
%   weights_list = ...
%       { ...
%        ["V1S2ToLeftError_W"]; ...
%        ["V1S4ToLeftError_W"]; ... 
%        ["V1S8ToLeftError_W"]; ... 
%        ["V1S2ToRightError_W"]; ...
%        ["V1S4ToRightError_W"]; ...
%        ["V1S8ToRightError_W"]; ...
%        };
%   pre_list = ...
%       { ...
%        ["V1S2_A"]; ...
%        ["V1S4_A"]; ...
%        ["V1S8_A"]; ...
%        ["V1S2_A"]; ...
%        ["V1S4_A"]; ...
%        ["V1S8_A"]; ...
%        };
%   sparse_ndx = ...
%        {   ...
%        1;  ...
%        2;  ...
%        3;  ...
%        1;  ...
%        2;  ...
%        3;  ...
%        };
%   numDepthHist = 10;
%
%   num_weights_list = size(weights_list, 1);
%   weights_hdr = cell(num_weights_list, 1);
%   pre_hdr = cell(num_weights_list, 1);
%   weights_dir = [output_dir, filesep, "weights"];
%   mkdir(weights_dir);
%   for i_weights = 1:num_weights_list
%      weights_file = [checkpoint_path, filesep, weights_list{i_weights}, '.pvp'];
%      if ~exist(weights_file, "file")
%        error(["file does not exist: ", weights_file]);
%      end%if
%      i_pre = i_weights;
%      pre_file = [checkpoint_path, filesep, pre_list{i_pre}, '.pvp'];
%      if ~exist(pre_file, "file")
%        error(["file does not exist: ", pre_file]);
%      end%if
%      pre_fid = fopen(pre_file);
%      pre_hdr{i_pre} = readpvpheader(pre_fid);
%      fclose(pre_fid);
%      [weights_struct, weights_hdr_tmp] = ...
%      readpvpfile(weights_file);
%      %i_arbor = 1;
%      for i_arbor = 1:numarbors
%         i_frame = 1;
%         if weights_hdr_tmp.nfp == 1
%            weight_vals = squeeze(weights_struct{i_frame}.values{i_arbor});
%         else
%            weight_vals = weights_struct{i_frame}.values{i_arbor};
%            %stepval = 1/weights_hdr_tmp.nfp;
%            %rangeval = stepval/2;
%            %[maxmat, idx] = max(weight_vals, [], 3);
%            %idx = squeeze(idx);
%            %weight_vals = (idx - 1) * stepval + rangeval;
%         end%if
%         weight_time = squeeze(weights_struct{i_frame}.time);
%         if analyze_Sparse_flag
%           pre_hist_rank = Sparse_hist_rank_array{sparse_ndx{i_weights}};
%         else
%           pre_hist_rank = (1:pre_hdr{i_pre}.nf);
%         end%if
%       %% make tableau of all patches
%       %%keyboard;
%       i_patch = 1;
%       num_weights_dims = ndims(weight_vals);
%       num_patches = size(weight_vals, num_weights_dims);
%       %num_patches = 64;
%       num_patches_rows = floor(sqrt(num_patches));
%       num_patches_cols = ceil(num_patches / num_patches_rows);
%       num_weights_colors = 1;
%       if num_weights_dims == 4
%         num_weights_colors = size(weight_vals,3);
%       end%if
%       weights_fig = figure;
%       set(weights_fig, "name", ["Weights_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)]);
%       depthHistVal = [];
%       for j_patch = 1  : num_patches
%         i_patch = pre_hist_rank(j_patch);
%         subplot(num_patches_rows, num_patches_cols, j_patch);
%
%         if num_weights_colors == 1 patch_tmp = squeeze(weight_vals(:,:,i_patch));
%         else patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
%         end%if
%         if weights_hdr_tmp.nfp == 1
%            patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
%            min_patch = min(patch_tmp2(:));
%            max_patch = max(patch_tmp2(:));
%            patch_tmp2 = uint8(127.5 + 127.5*(flipdim(patch_tmp2,1) ./ (max(abs(patch_tmp2(:))) + (max(abs(patch_tmp2(:)))==0))))';
%            imagesc(patch_tmp2); 
%            colormap(gray);
%            box off
%            axis off
%            axis image
%            %%drawnow;
%         else
%            plotme = squeeze(weight_vals(:, :, :, i_patch)); 
%            assert(length(size(plotme)) == 2);
%
%            if i_patch <= numDepthHist
%               depthHistVal = [depthHistVal; find(plotme == max(plotme(:)))];
%            end
%
%            [nf, temp] = size(plotme);
%            plotme = repmat(plotme, 1, nf);
%            
%            box off
%            axis on
%            axis nolabel
%            imagesc(plotme);
%            colormap(gray);
%            axis image
%         end%if
%       end%for
%       weights_dir = [output_dir, filesep, "weights"];
%       mkdir(weights_dir);
%       saveas(weights_fig, [weights_dir, filesep, "Weights_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)], "png");
%
%       %% make histogram of all weights
%       weights_hist_fig = figure;
%       [weights_hist, weights_hist_bins] = hist(weight_vals(:), 100);
%       bar(weights_hist_bins, log(weights_hist+1));
%       set(weights_hist_fig, "name", ["weights_Histogram_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)]);
%       saveas(weights_hist_fig, [weights_dir, filesep, "weights_hist_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)], "png");
%
%       if(~isempty(depthHistVal))
%          %% make histogram of depths
%          depth_hist_fig = figure;
%          [depth_hist, depth_hist_bins] = hist(depthHistVal(:), 10);
%          bar(depth_hist_bins, log(depth_hist+1));
%          set(depth_hist_fig, "name", ["depth_Histogram_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)]);
%          saveas(depth_hist_fig, [weights_dir, filesep, "depth_hist_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)], "png");
%       end
%    end%for %% i_arbors
%  end%for %% i_weights
%end%if  %% plot_weights

%%keyboard;
plot_weights0_2 =false;
plot_weights0_2_flag = false;
plot_labelWeights_flag = false;
rank_xCorr = false;
max_patches = 132;
if plot_weights0_2
  weights1_2_list = {};
   checkpoints_list = {output_dir};
   checkpoint_weights_movie = false;

   weights1_2_list = ...
       {...
          ["w46_"], ["V2ToV1S2Error"];...
          ["w47_"], ["V2ToV1S4Error"];...
          ["w48_"], ["V2ToV1S8Error"];...
          ["w46_"], ["V2ToV1S2Error"];...
          ["w47_"], ["V2ToV1S4Error"];...
          ["w48_"], ["V2ToV1S8Error"];...
       };
   post1_2_list = ...
       {...
          ["a16_"], ["V1S2"];...
          ["a17_"], ["V1S4"];...
          ["a18_"], ["V1S8"];...
          ["a16_"], ["V1S2"];...
          ["a17_"], ["V1S4"];...
          ["a18_"], ["V1S8"];...
       };
   %% list of weights from layer1 to image
   weights0_1_list = ...
       {...
          ["w4_"], ["V1S2ToLeftError"];...
          ["w5_"], ["V1S4ToLeftError"];...
          ["w6_"], ["V1S8ToLeftError"];...
          ["w22_"], ["V1S2ToRightError"];...
          ["w23_"], ["V1S4ToRightError"];...
          ["w24_"], ["V1S8ToRightError"];...
       };
   image_list = ...
       {...
          ["a2_"], ["LeftRescale"];...
          ["a2_"], ["LeftRescale"];...
          ["a2_"], ["LeftRescale"];...
          ["a10_"], ["RightRescale"];...
          ["a10_"], ["RightRescale"];...
          ["a10_"], ["RightRescale"];...
       };
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [4, 4, 4, 4, 4, 4];
    num_checkpoints = size(checkpoints_list,1);

  num_weights1_2_list = size(weights1_2_list,1);
  if num_weights1_2_list == 0
    break;
  end%if
  if ~exist("weights1_2_pad_size") || length(weights1_2_pad_size(:)) < num_weights1_2_list
    weights1_2_pad_size = zeros(1, num_weights1_2_list);
  end%if
  %% get image header (to get image dimensions)
  i_image = 1;
  image_file = ...
      [output_dir, filesep, image_list{i_image,1}, image_list{i_image,2}, ".pvp"]
  if ~exist(image_file, "file")
    i_checkpoint = 1;
    image_file = ...
        [checkpoints_list{i_checkpoint,:}, filesep, image_list{i_image,1}, image_list{i_image,2}, ".pvp"]
  end%if
  if ~exist(image_file, "file")
    error(["file does not exist: ", image_file]);
  end%if
  image_fid = fopen(image_file);
  image_hdr = readpvpheader(image_fid);
  fclose(image_fid);

  weights1_2_hdr = cell(num_weights1_2_list,1);
  pre1_2_hdr = cell(num_weights1_2_list,1);
  post1_2_hdr = cell(num_weights1_2_list,1);

  weights1_2_movie_dir = [output_dir, filesep, "weights1_2_movie"]
  [status, msg, msgid] = mkdir(weights1_2_movie_dir);
  if status ~= 1
    warning(["mkdir(", weights1_2_movie_dir, ")", " msg = ", msg]);
  end%if 
  weights1_2_dir = [output_dir, filesep, "weights1_2"]
  [status, msg, msgid] = mkdir(weights1_2_dir);
  if status ~= 1
    warning(["mkdir(", weights1_2_dir, ")", " msg = ", msg]);
  end%if 
  for i_weights1_2 = 1 : num_weights1_2_list

    max_weight1_2_time = 0;
    max_checkpoint = 0;
    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};

      %% get weight 2->1 file
      weights1_2_file = ...
          [checkpoint_dir, filesep, weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, ".pvp"]
      if ~exist(weights1_2_file, "file")
        warning(["file does not exist: ", weights1_2_file]);
        continue;
      end%if
      weights1_2_fid = fopen(weights1_2_file);
      weights1_2_hdr{i_weights1_2} = readpvpheader(weights1_2_fid);    
      fclose(weights1_2_fid);

      weight1_2_time = weights1_2_hdr{i_weights1_2}.time;
      if weight1_2_time > max_weight1_2_time
        max_weight1_2_time = weight1_2_time;
        max_checkpoint = i_checkpoint;
      end%if
    end%for %% i_checkpoint

    for i_checkpoint = 1 : num_checkpoints
      if i_checkpoint ~= max_checkpoint 
        %%continue;
      end%if
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights1_2_file = ...
          [checkpoint_dir, filesep, weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, ".pvp"]
      if ~exist(weights1_2_file, "file")
        warning(["file does not exist: ", weights1_2_file]);
        continue;
      end%if
      weights1_2_fid = fopen(weights1_2_file);
      weights1_2_hdr{i_weights1_2} = readpvpheader(weights1_2_fid);    
      fclose(weights1_2_fid);

      weights1_2_filedata = dir(weights1_2_file);
      weights1_2_framesize = ...
          weights1_2_hdr{i_weights1_2}.recordsize*weights1_2_hdr{i_weights1_2}.numrecords+weights1_2_hdr{i_weights1_2}.headersize;
      tot_weights1_2_frames = weights1_2_filedata(1).bytes/weights1_2_framesize;
      weights1_2_nxp = weights1_2_hdr{i_weights1_2}.additional(1);
      weights1_2_nyp = weights1_2_hdr{i_weights1_2}.additional(2);
      weights1_2_nfp = weights1_2_hdr{i_weights1_2}.additional(3);

      %% read 2 -> 1 weights
      num_weights1_2 = 1;
      progress_step = ceil(tot_weights1_2_frames / 10);
      [weights1_2_struct, weights1_2_hdr_tmp] = ...
          readpvpfile(weights1_2_file, progress_step, tot_weights1_2_frames, tot_weights1_2_frames-num_weights1_2+1);
      i_frame = num_weights1_2;
      i_arbor = 1;
      weights1_2_vals = squeeze(weights1_2_struct{i_frame}.values{i_arbor});
      weights1_2_time = squeeze(weights1_2_struct{i_frame}.time);
      weights1_2_name = ...
          [weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, "_", num2str(weights1_2_time, "%08d")];
      
      %% get weight 1->0 file
      i_weights0_1 = i_weights1_2;
      weights0_1_file = ...
          [checkpoint_dir, filesep, weights0_1_list{i_weights0_1,1}, weights0_1_list{i_weights0_1,2}, ".pvp"]
      if ~exist(weights0_1_file, "file")
        warning(["file does not exist: ", weights0_1_file]);
        continue;
      end%if
      weights0_1_fid = fopen(weights0_1_file);
      weights0_1_hdr{i_weights0_1} = readpvpheader(weights0_1_fid);    
      fclose(weights0_1_fid);
      weights0_1_filedata = dir(weights0_1_file);
      weights0_1_framesize = ...
          weights0_1_hdr{i_weights0_1}.recordsize*weights0_1_hdr{i_weights0_1}.numrecords+weights0_1_hdr{i_weights0_1}.headersize;
      tot_weights0_1_frames = weights0_1_filedata(1).bytes/weights0_1_framesize;
      weights0_1_nxp = weights0_1_hdr{i_weights0_1}.additional(1);
      weights0_1_nyp = weights0_1_hdr{i_weights0_1}.additional(2);
      weights0_1_nfp = weights0_1_hdr{i_weights0_1}.additional(3);

      %% get post header (to get post layer dimensions)
      i_post1_2 = i_weights1_2;
      post1_2_file = [checkpoint_dir, filesep, post1_2_list{i_post1_2,1}, post1_2_list{i_post1_2,2}, ".pvp"]
      if ~exist(post1_2_file, "file")
        warning(["file does not exist: ", post1_2_file]);
        continue;
      end%if
      post1_2_fid = fopen(post1_2_file);
      post1_2_hdr{i_post1_2} = readpvpheader(post1_2_fid);
      fclose(post1_2_fid);
      post1_2_nf = post1_2_hdr{i_post1_2}.nf;

      %% read 1 -> 0 weights
      num_weights0_1 = 1;
      progress_step = ceil(tot_weights0_1_frames / 10);
      [weights0_1_struct, weights0_1_hdr_tmp] = ...
          readpvpfile(weights0_1_file, progress_step, tot_weights0_1_frames, tot_weights0_1_frames-num_weights0_1+1);
      i_frame = num_weights0_1;
      i_arbor = 1;
      weights0_1_vals = squeeze(weights0_1_struct{i_frame}.values{i_arbor});
      weights0_1_time = squeeze(weights0_1_struct{i_frame}.time);
      
      %% get rank order of presynaptic elements
      tmp_ndx = sparse_ndx(i_weights1_2);
      if analyze_Sparse_flag
        tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
        tmp_rank = [];
      end%if
      if rank_xCorr
         rank = [iidx{1}, jidx{1}];
         pre_hist_rank = rank'(:);
         pre_hist_rank = pre_hist_rank(1:weights1_2_hdr{i_weights1_2}.nf);
      elseif analyze_Sparse_flag && ~isempty(tmp_rank)
        pre_hist_rank = tmp_rank;
      else
        pre_hist_rank = (1:weights1_2_hdr{i_weights1_2}.nf);
      end%if

      if exist("labelWeights_list") && length(labelWeights_list) >= i_weights1_2 && ...
            ~isempty(labelWeights_list{i_weights1_2}) && ...
            plot_labelWeights_flag && ...
            i_checkpoint == max_checkpoint
        labelWeights_file = ...
            [checkpoint_dir, filesep, ...
             labelWeights_list{i_weights1_2,1}, labelWeights_list{i_weights1_2,2}, ".pvp"]
        if ~exist(labelWeights_file, "file")
          warning(["file does not exist: ", labelWeights_file]);
          continue;
        end%if
        labelWeights_fid = fopen(labelWeights_file);
        labelWeights_hdr{i_weights1_2} = readpvpheader(labelWeights_fid);    
        fclose(labelWeights_fid);
        num_labelWeights = 1;
        labelWeights_filedata = dir(labelWeights_file);
        labelWeights_framesize = ...
            labelWeights_hdr{i_weights1_2}.recordsize * ...
            labelWeights_hdr{i_weights1_2}.numrecords+labelWeights_hdr{i_weights1_2}.headersize;
        tot_labelWeights_frames = labelWeights_filedata(1).bytes/labelWeights_framesize;
        [labelWeights_struct, labelWeights_hdr_tmp] = ...
            readpvpfile(labelWeights_file, ...
                        progress_step, ...
                        tot_labelWeights_frames, ...
                        tot_labelWeights_frames-num_labelWeights+1);
        labelWeights_vals = squeeze(labelWeights_struct{i_frame}.values{i_arbor});
        labelWeights_time = squeeze(labelWeights_struct{i_frame}.time);
        labeledWeights0_2 = cell(size(labelWeights_vals,1),1);
      else
        labelWeights_vals = [];
        labelWeights_time = [];
      end%if


      %% compute layer 2 -> 1 patch size in pixels
      image2post_nx_ratio = image_hdr.nxGlobal / post1_2_hdr{i_post1_2}.nxGlobal;
      image2post_ny_ratio = image_hdr.nyGlobal / post1_2_hdr{i_post1_2}.nyGlobal;
      weights0_1_overlapp_x = weights0_1_nxp - image2post_nx_ratio;
      weights0_1_overlapp_y = weights0_1_nyp - image2post_ny_ratio;
      weights0_2_nxp = ...
          weights0_1_nxp + ...
          (weights1_2_nxp - 1) * (weights0_1_nxp - weights0_1_overlapp_x); 
      weights0_2_nyp = ...
          weights0_1_nyp + ...
          (weights1_2_nyp - 1) * (weights0_1_nyp - weights0_1_overlapp_y); 

      %% make tableau of all patches
      %%keyboard;
      i_patch = 1;
      num_weights1_2_dims = ndims(weights1_2_vals);
      num_patches0_2 = size(weights1_2_vals, num_weights1_2_dims);
      num_patches0_2 = min(num_patches0_2, max_patches);
      %% algorithms assumes weights1_2 are one to many
      num_patches0_2_rows = floor(sqrt(num_patches0_2)-1);
      num_patches0_2_cols = ceil(num_patches0_2 / num_patches0_2_rows);
      %% for one to many connections: dimensions of weights1_2 are:
      %% weights1_2(nxp, nyp, nf_post, nf_pre)
      if plot_weights0_2_flag && i_checkpoint == max_checkpoint
        weights1_2_fig = figure;
        set(weights1_2_fig, "name", weights1_2_name);
      end%if
      max_shrinkage = 8; %% 
      weight_patch0_2_array = [];

      for kf_pre1_2_rank = 1  : num_patches0_2
        kf_pre1_2 = pre_hist_rank(kf_pre1_2_rank);
        if plot_weights0_2_flag && i_checkpoint == max_checkpoint
          subplot(num_patches0_2_rows, num_patches0_2_cols, kf_pre1_2_rank); 
        end%if
        if ndims(weights1_2_vals) == 4
          patch1_2_tmp = squeeze(weights1_2_vals(:,:,:,kf_pre1_2));
        elseif ndims(weights1_2_vals) == 3
          patch1_2_tmp = squeeze(weights1_2_vals(:,:,kf_pre1_2));
          patch1_2_tmp = reshape(patch1_2_tmp, [1,1,1,size(weights1_2_vals,2)]);
        elseif ndims(weights1_2_vals) == 2
          patch1_2_tmp = squeeze(weights1_2_vals(:,kf_pre1_2));
          patch1_2_tmp = reshape(patch1_2_tmp, [1,1,1,size(weights1_2_vals,2)]);
        end%if
        %% patch0_2_array stores the sum over all post layer 1 neurons, weighted by weights1_2, 
        %% of image patches for each columun of weights0_1 for pre layer 2 neuron kf_pre
        patch0_2_array = cell(size(weights1_2_vals,1),size(weights1_2_vals,2));
        %% patch0_2 stores the complete image patch of the layer 2 neuron kf_pre
        patch0_2 = zeros(weights0_2_nyp, weights0_2_nxp, weights0_1_nfp);
        %% loop over weights1_2 rows and columns
        for weights1_2_patch_row = 1 : weights1_2_nyp
          for weights1_2_patch_col = 1 : weights1_2_nxp
            patch0_2_array{weights1_2_patch_row, weights1_2_patch_col} = ...
                zeros([weights0_1_nxp, weights0_1_nyp, weights0_1_nfp]);
            %% accumulate weights0_1 patches for each post feature separately for each weights0_1 column 
            for kf_post1_2 = 1 : post1_2_nf
              patch1_2_weight = patch1_2_tmp(weights1_2_patch_row, weights1_2_patch_col, kf_post1_2);
              if patch1_2_weight == 0
                continue;
              end%if
              if weights0_1_nfp == 1
                weights0_1_patch = squeeze(weights0_1_vals(:,:,kf_post1_2));
              else
                weights0_1_patch = squeeze(weights0_1_vals(:,:,:,kf_post1_2));
              end%if
              %%  store weights0_1_patch by column
              patch0_2_array{weights1_2_patch_row, weights1_2_patch_col} = ...
                  patch0_2_array{weights1_2_patch_row, weights1_2_patch_col} + ...
                  patch1_2_weight .* ...
                  weights0_1_patch;
            end%for %% kf_post1_2
            row_start = 1+image2post_ny_ratio*(weights1_2_patch_row-1);
            row_end = image2post_ny_ratio*(weights1_2_patch_row-1)+weights0_1_nyp;
            col_start = 1+image2post_nx_ratio*(weights1_2_patch_col-1);
            col_end = image2post_nx_ratio*(weights1_2_patch_col-1)+weights0_1_nxp;
            patch0_2(row_start:row_end, col_start:col_end, :) = ...
                patch0_2(row_start:row_end, col_start:col_end, :) + ...
                patch0_2_array{weights1_2_patch_row, weights1_2_patch_col};
          end%for %% weights1_2_patch_col
        end%for %% weights1_2_patch_row
        patch_tmp2 = flipdim(permute(patch0_2, [2,1,3]),1);
        patch_tmp3 = patch_tmp2;
        weights0_2_nyp_shrunken = size(patch_tmp3, 1);
        patch_tmp4 = patch_tmp3(1, :, :);
        while ~any(patch_tmp4(:)) %% && ((weights0_2_nyp - weights0_2_nyp_shrunken) <= max_shrinkage/2)
          weights0_2_nyp_shrunken = weights0_2_nyp_shrunken - 1;
          patch_tmp3 = patch_tmp3(2:weights0_2_nyp_shrunken, :, :);
          patch_tmp4 = patch_tmp3(1, :, :);
        end%while
        weights0_2_nyp_shrunken = size(patch_tmp3, 1);
        patch_tmp4 = patch_tmp3(weights0_2_nyp_shrunken, :, :);
        while ~any(patch_tmp4(:))
          weights0_2_nyp_shrunken = weights0_2_nyp_shrunken - 1;
          patch_tmp3 = patch_tmp3(1:weights0_2_nyp_shrunken, :, :);
          patch_tmp4 = patch_tmp3(weights0_2_nyp_shrunken, :, :);
        end%while
        weights0_2_nxp_shrunken = size(patch_tmp3, 2);
        patch_tmp4 = patch_tmp3(:, 1, :);
        while ~any(patch_tmp4(:)) %% && ((weights0_2_nyp - weights0_2_nyp_shrunken) <= max_shrinkage/2)
          weights0_2_nxp_shrunken = weights0_2_nxp_shrunken - 1;
          patch_tmp3 = patch_tmp3(:, 2:weights0_2_nxp_shrunken, :);
          patch_tmp4 = patch_tmp3(:, 1, :);
        end%while
        weights0_2_nxp_shrunken = size(patch_tmp3, 2);
        patch_tmp4 = patch_tmp3(:, weights0_2_nxp_shrunken, :);
        while ~any(patch_tmp4(:))
          weights0_2_nxp_shrunken = weights0_2_nxp_shrunken - 1;
          patch_tmp3 = patch_tmp3(:, 1:weights0_2_nxp_shrunken, :);
          patch_tmp4 = patch_tmp3(:, weights0_2_nxp_shrunken, :);
        end%while
        min_patch = min(patch_tmp3(:));
        max_patch = max(patch_tmp3(:));
        %%patch_tmp5 = ...
        %%    uint8((flipdim(patch_tmp3,1) - min_patch) * 255 / ...
        %%          (max_patch - min_patch + ((max_patch - min_patch)==0)));
        patch_tmp5 = ...
            uint8(127.5 + 127.5*(flipdim(patch_tmp3,1) ./ (max(abs(patch_tmp3(:))) + (max(abs(patch_tmp3(:)))==0))));
                  
        pad_size = weights1_2_pad_size(i_weights1_2);
        padded_patch_size = size(patch_tmp5);
        padded_patch_size(1) = padded_patch_size(1) + 2*pad_size;
        padded_patch_size(2) = padded_patch_size(2) + 2*pad_size;
        patch_tmp6 = repmat(uint8(128),padded_patch_size);
        if ndims(patch_tmp5) == 3
          patch_tmp6(pad_size+1:padded_patch_size(1)-pad_size,pad_size+1:padded_patch_size(2)-pad_size,:) = uint8(patch_tmp5);
        else
          patch_tmp6(pad_size+1:padded_patch_size(1)-pad_size,pad_size+1:padded_patch_size(2)-pad_size) = uint8(patch_tmp5);
        end%if
        
        if plot_weights0_2_flag && i_checkpoint == max_checkpoint
          %% pad by 8 as test
          image(patch_tmp6); 
          if weights0_1_nfp == 1
            colormap(gray);
          end%if
          box off
          axis off
          axis image
        end%if
        if plot_labelWeights_flag && i_checkpoint == max_checkpoint
          if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
            [~, max_label] = max(squeeze(labelWeights_vals(:,kf_pre1_2)));
            text(weights0_2_nyp_shrunken/2, -weights0_2_nxp_shrunken/6, num2str(max_label-1), "color", [1 0 0]);
          end%if %% ~empty(labelWeights_vals)
          %%drawnow;
        end%if %% plot_weights0_2_flag && i_checkpoint == max_checkpoint

        if isempty(weight_patch0_2_array)
          weight_patch0_2_array = ...
              zeros(num_patches0_2_rows*(weights0_2_nyp_shrunken+2*pad_size), ...
                    num_patches0_2_cols*(weights0_2_nxp_shrunken+2*pad_size), weights0_1_nfp);
        end%if
        col_ndx = 1 + mod(kf_pre1_2_rank-1, num_patches0_2_cols);
        row_ndx = 1 + floor((kf_pre1_2_rank-1) / num_patches0_2_cols);
        weight_patch0_2_array((1+(row_ndx-1)*(weights0_2_nyp_shrunken+2*pad_size)):(row_ndx*(weights0_2_nyp_shrunken+2*pad_size)), ...
                              (1+(col_ndx-1)*(weights0_2_nxp_shrunken+2*pad_size)):(col_ndx*(weights0_2_nxp_shrunken+2*pad_size)),:) = ...
            patch_tmp6;

        %% Plot the average movie weights for a label %%
        if plot_labelWeights_flag && i_checkpoint == max_checkpoint
          if ~isempty(labelWeights_vals) 
            if ~isempty(labeledWeights0_2{max_label})
              labeledWeights0_2{max_label} = labeledWeights0_2{max_label} + double(patch_tmp6);
            else
              labeledWeights0_2{max_label} = double(patch_tmp6);
            end%if
          end%if %%  ~isempty(labelWeights_vals) 
        end%if %% plot_weights0_2_flag && i_checkpoint == max_checkpoint

      end%for %% kf_pre1_2_ank

      if plot_weights0_2_flag && i_checkpoint == max_checkpoint
        saveas(weights1_2_fig, [weights1_2_dir, filesep, weights1_2_name, ".png"], "png");
      end%if
      if plot_labelWeights_flag && i_checkpoint == max_checkpoint && ~isempty(labelWeights_vals) 
        labeledWeights_str = ...
            ["labeledWeights_", ...
             weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, ...
             "_", num2str(weight_time, "%08d")];
        labeledWeights_fig = figure("name", labeledWeights_str);
        rows_labeledWeights = ceil(sqrt(size(labelWeights_vals,1)));
        cols_labeledWeights = ceil(size(labelWeights_vals,1) / rows_labeledWeights);
        for label = 1:size(labelWeights_vals,1)
          subplot(rows_labeledWeights, cols_labeledWeights, label);
          labeledWeights_subplot_str = ...
              [num2str(label, "%d")];
          imagesc(squeeze(labeledWeights0_2{label}));
          axis off
          title(labeledWeights_subplot_str);
        end%for %% label = 1:size(labelWeights_vals,1)
        saveas(labeledWeights_fig,  [weights_dir, filesep, labeledWeights_str, ".png"], "png");
      end%if %%  ~isempty(labelWeights_vals) 

      imwrite(uint8(weight_patch0_2_array), [weights1_2_movie_dir, filesep, weights1_2_name, ".png"], "png");
      if i_checkpoint == max_checkpoint
        save("-mat", ...
             [weights1_2_movie_dir, filesep, weights1_2_name, ".mat"], ...
             "weight_patch0_2_array");
      end%if


      %% make histogram of all weights
      if plot_weights0_2_flag && i_checkpoint == max_checkpoint
        weights1_2_hist_fig = figure;
        [weights1_2_hist, weights1_2_hist_bins] = hist(weights1_2_vals(:), 100);
        bar(weights1_2_hist_bins, log(weights1_2_hist+1));
        set(weights1_2_hist_fig, "name", ...
            ["Hist_", weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, "_", ...
             num2str(weights1_2_time, "%08d")]);
        saveas(weights1_2_hist_fig, ...
               [weights1_2_dir, filesep, "weights1_2_hist_", weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, "_", ...
                num2str(weights1_2_time, "%08d")], "png");
      end%if

      %% plot average labelWeights for each label
      if ~isempty(labelWeights_vals) && ...
            ~isempty(labelWeights_time) && ...
            plot_weights0_2_flag && ...
            i_checkpoint == max_checkpoint

        %% plot label weights as matrix of column vectors
        ranked_labelWeights = labelWeights_vals(:, pre_hist_rank(1:num_patches0_2));
        [~, max_label] = max(ranked_labelWeights,[],1);
        [max_label_sorted, max_label_ndx] = sort(max_label);
        label_weights_fig = figure;
        imagesc(ranked_labelWeights(:,max_label_ndx))
        label_weights_str = ...
            ["LabelWeights_", labelWeights_list{i_weights1_2,1}, labelWeights_list{i_weights1_2,2}, ...
             "_", num2str(labelWeights_time, "%08d")];
        %%title(label_weights_fig, label_weights_str);
        figure(label_weights_fig, "name", label_weights_str); 
        title(label_weights_str);
        saveas(label_weights_fig, [weights_dir, filesep, label_weights_str, ".png"] , "png");

      end%if  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

    end%for %% i_checkpoint

  end%for %% i_weights
keyboard  
end%if  %% plot_weights

