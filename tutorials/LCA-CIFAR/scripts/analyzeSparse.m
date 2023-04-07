function [Sparse_hdr, ...
          Sparse_hist_rank, ...
          Sparse_times, ...
          Sparse_percent_active, ...
          Sparse_std, ...
          Sparse_struct_array] = ...
      analyzeSparse(Sparse_file_input, ... % file to read data from
                    Sparse_label, ... % label used in filenames for analysis files
                    Sparse_dir, ... % directory to write analysis files into
                    plot_Sparse_flag, ...
                    fraction_Sparse_frames_read, ...
                    min_Sparse_skip, ...
                    fraction_Sparse_progress, ...
                    num_procs)
  %% analyze sparse activity pvp file.
  %% Sparse_hist_rank_array returns the rank order of neurons based on firing frequency, from most
  %% to least active.
  %% Sparse_times gives the simTime for each sparse activity frame included in the analysis
  %% Sparse_percent_active gives the percentage of neurons active as a function of time in any given layer
  %% equal to the number of neurons whose activity changed over the number of neurons active in either frame

  if ~exist("plot_Sparse_flag") || isempty(plot_Sparse_flag)
    plot_Sparse_flag = false;
  end%if
  if ~exist("min_Sparse_skip") || isempty(min_Sparse_skip)
    min_Sparse_skip = 0;  %% 0 -> skip no frames
  end%if
  if ~exist("fraction_Sparse_frames_read") || isempty(fraction_Sparse_frames_read)
    fraction_Sparse_frames_read = 1;  %% 1 -> read all frames
  end%if
  if ~exist("fraction_Sparse_progress") || isempty(fraction_Sparse_progress)
    fraction_Sparse_progress = 10;
  end%if
  if ~exist("num_procs") || isempty(num_procs)
    num_procs = 1;
  end%if

  % Initialize return values
  Sparse_hdr = [];
  Sparse_hist_rank = [];
  Sparse_times = [];
  Sparse_percent_active = [];
  Sparse_std = [];
  Sparse_struct_array = [];

  if ~exist(Sparse_file_input, "file")
    warning(["file does not exist: ", Sparse_file_input]);
    continue
  end%if

  Sparse_fid = fopen(Sparse_file_input);
  Sparse_hdr = readpvpheader(Sparse_fid);
  fclose(Sparse_fid);
  tot_Sparse_frames = Sparse_hdr.nbands;
  %% number of activity frames to analyze, counting backward from last frame, maximum is tot_Sparse_frames
  num_Sparse_skip = tot_Sparse_frames - fix(tot_Sparse_frames/fraction_Sparse_frames_read);
  num_Sparse_skip = max(num_Sparse_skip, min_Sparse_skip);
  progress_step = ceil(tot_Sparse_frames / fraction_Sparse_progress);

  %% analyze sparse (or non-sparse) activity from pvp file
  nx_Sparse = Sparse_hdr.nx;
  ny_Sparse = Sparse_hdr.ny;
  nf_Sparse = Sparse_hdr.nf;
  n_Sparse = nx_Sparse * ny_Sparse * nf_Sparse;
  Sparse_hist = zeros(nf_Sparse+1,1);
  Sparse_hist_bins = 1:nf_Sparse;

  n_Sparse_cell = cell(1);
  n_Sparse_cell{1} = n_Sparse;
  nf_Sparse_cell = cell(1);
  nf_Sparse_cell{1} = nf_Sparse;

  [Sparse_struct, Sparse_hdr_tmp] = ...
      readpvpfile(Sparse_file_input, progress_step, tot_Sparse_frames, num_Sparse_skip,1);
  num_Sparse_frames = size(Sparse_struct,1);

  if num_procs == 1
    [Sparse_times_list, ...
     Sparse_percent_active_list, ...
     Sparse_std_list, ...
     Sparse_hist_frames_list] = ...
        cellfun(@calcSparsePVPArray, ...
                Sparse_struct, ...
                n_Sparse_cell, ...
                nf_Sparse_cell, ...
                "UniformOutput", false);
  elseif num_procs > 1
    [Sparse_times_list, ...
     Sparse_percent_active_list, ...
     Sparse_std_list, ...
     Sparse_hist_frames_list] = ...
        parcellfun(num_procs, ...
                   @calcSparsePVPArray, ...
                   Sparse_struct, ...
                   n_Sparse_cell, ...
                   nf_Sparse_cell, ...
                   "UniformOutput", false);
  end%if  %% num_procs

  Sparse_times = cell2mat(Sparse_times_list);

  % Neither the display period nor the batch size are specifically stored
  % in the .pvp file. Instead we infer them from the sequence of times.
  % There should be batch_size copies of display_period, then batch_size
  % copies of 2*display_period, and so on.
  Sparse_times_unique = unique(Sparse_times);
  num_display_periods = numel(Sparse_times_unique);
  batch_size = ceil(numel(Sparse_times) / num_display_periods);

  for i_times = 1:numel(Sparse_times)
    i_display_period = ceil(i_times / batch_size);
    correct_timestep = Sparse_times(1) * i_display_period;
    assert(Sparse_times(i_times) == correct_timestep);
  end % for i_times

  Sparse_times = reshape(Sparse_times, batch_size, num_display_periods);

  Sparse_percent_active = cell2mat(Sparse_percent_active_list);
  Sparse_percent_active = reshape(Sparse_percent_active, batch_size, num_display_periods);
  Sparse_std = cell2mat(Sparse_std_list);
  Sparse_std = reshape(Sparse_std, batch_size, num_display_periods);

  if ~isempty(Sparse_hist_frames_list)
    Sparse_hist_frames = cell2mat(Sparse_hist_frames_list);
    Sparse_hist_frames = reshape(Sparse_hist_frames, [batch_size, num_display_periods, nf_Sparse]);
  else
    Sparse_hist_frames = zeros(batch_size, num_display_periods, nf_Sparse);
  end%if

  num_Sparse_frames = size(Sparse_times,1);
  Sparse_hist = sum(Sparse_hist_frames,1);
  if length(Sparse_hist) < nf_Sparse
    continue;
  end%if
  Sparse_hist = Sparse_hist(1:nf_Sparse) / ((num_Sparse_frames) * nx_Sparse * ny_Sparse);
  [Sparse_hist_sorted, Sparse_hist_rank] = sort(Sparse_hist(:), 1, "descend");

  Sparse_filename_id = [Sparse_label, "_", ...
                        num2str(Sparse_times(num_Sparse_frames), "%08d")];

  [status, msg, msgid] = mkdir(Sparse_dir);
  if status ~= 1
    warning(["mkdir(", Sparse_dir, ")", " msg = ", msg]);
  end%if

  save("-mat", ...
       [Sparse_dir, filesep, "Sparse_hist_bins_", Sparse_filename_id, ".mat"], ...
       "Sparse_hist_bins");
  save("-mat", ...
       [Sparse_dir, filesep, "Sparse_hist_sorted_", Sparse_filename_id, ".mat"], ...
       "Sparse_hist_sorted");
  save("-mat", ...
       [Sparse_dir, filesep, "Sparse_hist_rank_", Sparse_filename_id, ".mat"], ...
       "Sparse_hist_rank");
  save("-mat", ...
       [Sparse_dir, filesep, "Sparse_percent_active_", Sparse_filename_id, ".mat"], ...
       "Sparse_times", "Sparse_percent_active");
  save("-mat", ...
       [Sparse_dir, filesep, "Sparse_std_", Sparse_filename_id, ".mat"], ...
       "Sparse_times", "Sparse_std");

  num_Sparse_frames = length(Sparse_times);
  if plot_Sparse_flag
    Sparse_hist_fig = figure;
    Sparse_hist_hndl = bar(Sparse_hist_bins(:), Sparse_hist_sorted(:));
    axis tight;
    title('Percent Active by Feature');
    set(Sparse_hist_fig, "name", ["Hist_", Sparse_filename_id]);
    saveas(Sparse_hist_fig, ...
           [Sparse_dir, filesep, "Hist_", Sparse_filename_id], "png");

    Sparse_percent_active_fig = figure;
    Sparse_percent_active_hndl = plot(Sparse_times(:), Sparse_percent_active(:), '.');
    hold on;
    plot(Sparse_times(1,:), mean(Sparse_percent_active, 1), 'ko-');
    hold off;
    axis tight;
    set(Sparse_percent_active_hndl, "linewidth", 1.5);
    set(Sparse_percent_active_fig, "name", ["percent_active_", Sparse_label, "_", ...
                                            num2str(Sparse_times(num_Sparse_frames), "%08d")]);
    title('Percent Active Over Time');
    saveas(Sparse_percent_active_fig, ...
           [Sparse_dir, filesep, "percent_active_", Sparse_label], "png");
  end%if  %% plot_Sparse_flag

  Sparse_median_active = median(Sparse_percent_active(:));
  disp([Sparse_filename_id, ...
        " median_active = ", num2str(Sparse_median_active)]);

endfunction
