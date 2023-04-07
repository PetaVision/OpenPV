clear all;
close all;
%setenv('GNUTERM','X11')
% addpath('../../../mlab/util/');

script_dir = fileparts(mfilename("fullpath"));
% script_dir is the directory containing this script m-file
addpath(script_dir)
top_dir = cd(cd([script_dir filesep ".."]));
% top_dir is the directory containing script_dir

output_dir = [top_dir filesep 'output'];
checkpoint_dir = [output_dir filesep 'Checkpoints'];
analysis_dir = [top_dir filesep 'Analysis']; % Files are written to this dir
mkdir(analysis_dir);

max_history = 100000000;
numarbors = 1;


analyze_Recon_flag = true;
if analyze_Recon_flag
   Input_list = ...
   {['Input']};
   Recon_list = ...
   {['InputRecon']};

   frameSkip = 10; %Print every 10th frame

   numRecons = size(Input_list, 1);
   assert(numRecons == size(Recon_list, 1));

   recons_dir = [analysis_dir, filesep 'Recons'];
   mkdir(recons_dir);
   for i_recon = 1:numRecons
      inputPath = [output_dir, filesep, Input_list{i_recon}, '.pvp'];
      [inputData, inputHdr] = readpvpfile(inputPath);
      reconPath = [output_dir, filesep, Recon_list{i_recon}, '.pvp'];
      [reconData, reconHdr] = readpvpfile(reconPath);
      numFrames = min(length(inputData), length(reconData));
      for i_frame = 1:frameSkip:numFrames
         %readpvpfile returns in [x, y, f]. Octave expects [y, x, f]
         inputImage = permute(inputData{i_frame}.values, [2, 1, 3]);
         reconImage = permute(reconData{i_frame}.values, [2, 1, 3]);
         time = inputData{i_frame}.time;
         batch = mod(i_frame-1, inputHdr.nbatch);
         assert(time == reconData{i_frame}.time);
         %Normalize
         scaled_inputImage = (inputImage - min(inputImage(:)))/(max(inputImage(:))-min(inputImage(:)));
         scaled_reconImage = (reconImage - min(reconImage(:)))/(max(reconImage(:))-min(reconImage(:)));
         %Concat images
         outImg = [scaled_inputImage; scaled_reconImage];
         %Write image
         outName = sprintf('%s/recon_%06d_%03d.png', recons_dir, time, batch)
         imwrite(outImg, outName);
      end
   end
end

plot_flag = 1;
analyze_Sparse_flag = true;
if analyze_Sparse_flag
    Sparse_label = 'V1';
    Sparse_file = [output_dir, filesep, Sparse_label, ".pvp"];

    plot_flag = 1;

    fraction_Sparse_frames_read = 1;
    min_Sparse_skip = 1;
    fraction_Sparse_progress = 1;
    num_procs = 8;
    Sparse_dir = [analysis_dir filesep 'Sparse'];

  [Sparse_hdr, ...
   Sparse_hist_rank_array, ...
   Sparse_times_array, ...
   Sparse_percent_active_array, ...
   Sparse_percent_change_array, ...
   Sparse_std_array] = ...
      analyzeSparse(Sparse_file, ...
                    Sparse_label, ...
                    Sparse_dir, ...
                    plot_flag, ...
                    fraction_Sparse_frames_read, ...
                    min_Sparse_skip, ...
                    fraction_Sparse_progress, ...
                    num_procs);
end

analyze_nonSparse_flag = true;
if analyze_nonSparse_flag
    nonSparse_list = ...
        {[''], ['InputError']; ...
         };
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(16, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {...
         [''], ['Input']; ...
         }; ...
    nonSparse_norm_strength = [1 1];
    Sparse_std_ndx = [0 0];
    plot_flag = true;

  if ~exist('Sparse_std_ndx')
    Sparse_std_ndx = zeros(num_nonSparse_list,1);
  end
  if ~exist('nonSparse_norm_strength')
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
  end

  fraction_nonSparse_frames_read = 1;
  min_nonSparse_skip = 1;
  fraction_nonSparse_progress = 10;
  nonSparse_dir = [analysis_dir, filesep, "nonSparse"];
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
         nonSparse_dir, ...
         plot_flag, ...
         fraction_nonSparse_frames_read, ...
         min_nonSparse_skip, ...
         fraction_nonSparse_progress);

end %% analyze_nonSparse_flag

plot_flag = false;
plot_weights = true;
if plot_weights
   weights_list = ...
   { ...
   ['V1ToInputError_W']; ...
   };
   pre_list = ...
   { ...
   ['V1_A']; ...
   };
   sparse_ndx = ...
   [   ...
   1;  ...
   ];

   checkpoints_list = {dir(checkpoint_dir).name};
   %Remove hidden files
   for i = length(checkpoints_list):-1:1
      % remove folders starting with .
      fname = checkpoints_list{i};
      if fname(1) == '.'
         checkpoints_list(i) = [ ];
      end
   end

   num_checkpoints = length(checkpoints_list);
   checkpoint_weights_movie = true;
   no_clobber = false;

   num_weights_list = size(weights_list,1);
   weights_hdr = cell(num_weights_list,1);
   pre_hdr = cell(num_weights_list,1);
   if checkpoint_weights_movie
      weights_movie_dir = [analysis_dir, filesep, 'weights_movie']
      [status, msg, msgid] = mkdir(weights_movie_dir);
      if status ~= 1
         warning(['mkdir(', weights_movie_dir, ')', ' msg = ', msg]);
      end 
   end
   if(plot_flag)
      weights_dir = [output_dir, filesep, 'weights']
      [status, msg, msgid] = mkdir(weights_dir);
      if status ~= 1
         warning(['mkdir(', weights_dir, ')', ' msg = ', msg]);
      end 
   end
   for i_weights = 1 : num_weights_list
      max_weight_time = 0;
      max_checkpoint = 0;
      for i_checkpoint = 1 : num_checkpoints
         checkpoint_path = [checkpoint_dir, filesep, checkpoints_list{i_checkpoint}];
         weights_file = [checkpoint_path, filesep, weights_list{i_weights,1}, '.pvp'];
         if ~exist(weights_file, 'file')
            warning(['file does not exist: ', weights_file]);
            continue;
         end
         weights_fid = fopen(weights_file);
         weights_hdr{i_weights} = readpvpheader(weights_fid);    
         fclose(weights_fid);

         weight_time = weights_hdr{i_weights}.time;
         if weight_time > max_weight_time
              max_weight_time = weight_time;
              max_checkpoint = i_checkpoint;
         end
      end %% i_checkpoint

      for i_checkpoint = 1 : num_checkpoints
         checkpoint_path = [checkpoint_dir, filesep, checkpoints_list{i_checkpoint}];
         weights_file = [checkpoint_path, filesep, weights_list{i_weights,1}, '.pvp'];
         if ~exist(weights_file, 'file')
            warning(['file does not exist: ', weights_file]);
            continue;
         end
         weights_fid = fopen(weights_file);
         weights_hdr{i_weights} = readpvpheader(weights_fid);    
         fclose(weights_fid);
         weights_filedata = dir(weights_file);
         patchsize = weights_hdr{i_weights}.nxp * weights_hdr{i_weights}.nyp * weights_hdr{i_weights}.nfp;
         numpatches = weights_hdr{i_weights}.numPatches;
         datasize = weights_hdr{i_weights}.datasize;
         weights_framesize = (patchsize * datasize + 8) * numpatches + weights_hdr{i_weights}.headersize;
         tot_weights_frames = weights_filedata(1).bytes/weights_framesize;
         num_weights = 1;
         progress_step = ceil(tot_weights_frames / 10);
         [weights_struct, weights_hdr_tmp] = ...
         readpvpfile(weights_file, progress_step, tot_weights_frames, tot_weights_frames-num_weights+1);
         i_frame = num_weights;
         i_arbor = 1;
         weight_vals = squeeze(weights_struct{i_frame}.values{i_arbor});
         weight_time = squeeze(weights_struct{i_frame}.time);
         weights_name =  [weights_list{i_weights,1}, '_', num2str(weight_time, '%08d')];
         if no_clobber && exist([weights_movie_dir, filesep, weights_name, '.png']) && i_checkpoint ~= max_checkpoint
            continue;
         end
         tmp_ndx = sparse_ndx(i_weights);
         if analyze_Sparse_flag
           pre_hist_rank = Sparse_hist_rank_array;
         else
           pre_hist_rank = (1:weights_hdr{i_weights}.nf);
         end

         %% make tableau of all patches
         weight_patch_array = weights_tableau(weight_vals, pre_hist_rank);

         imwrite(uint8(weight_patch_array), [weights_movie_dir, filesep, weights_name, '.png'], 'png');
         %% make histogram of all weights
         if plot_flag && i_checkpoint == max_checkpoint
            weights_hist_fig = figure;
            [weights_hist, weights_hist_bins] = hist(weight_vals(:), 100);
            bar(weights_hist_bins, log(weights_hist+1));
            set(weights_hist_fig, 'name', ...
            ['Hist_',  weights_list{i_weights,1}, '_', num2str(weight_time, '%08d')]);
            saveas(weights_hist_fig, ...
            [weights_dir, filesep, 'weights_hist_', num2str(weight_time, '%08d')], 'png');
         end
      end %% i_checkpoint
   end %% i_weights
end  %% plot_weights
printf("Finished\n");
