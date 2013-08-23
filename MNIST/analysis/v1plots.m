%% v1rank = function v1plots(V1_file, statsdir)

function v1rank = v1plots(V1_file, statsdir)

  %%%%%% PATHS %%%%%%%%%%
  [V1_struct, V1_hdr] = readpvpfile(V1_file, [], [], []);
  n_V1 = V1_hdr.nx * V1_hdr.ny * V1_hdr.nf;
  num_frames = size(V1_struct,1);
  start_frame = 1; %%
  
  %%%%%%%% INITIALIZATION %%%%%%%%%%%%%
  V1_hist = zeros(V1_hdr.nf+1,1);
  V1_hist_edges = [0:1:V1_hdr.nf]+0.5;
  V1_current = zeros(n_V1,1);
  V1_abs_change = zeros(num_frames,1);
  V1_percent_change = zeros(num_frames,1);
  V1_current_active = 0;
  V1_tot_active = zeros(num_frames,1);
  V1_times = zeros(num_frames,1);

  %%%%%%%% CALCULATIONS: ACTIVITY OVER TIME %%%%%%%%%%%%%%%
  for i_frame = 1 : 1 : num_frames
    V1_times(i_frame) = squeeze(V1_struct{i_frame}.time);
    V1_active_ndx = squeeze(V1_struct{i_frame}.values);
    V1_previous = V1_current;
    V1_current = full(sparse(V1_active_ndx+1,1,1,n_V1,1,n_V1));
    V1_abs_change(i_frame) = sum(V1_current(:) ~= V1_previous(:));
    V1_previous_active = V1_current_active;
    V1_current_active = nnz(V1_current(:));
    V1_tot_active(i_frame) = V1_current_active;
    V1_max_active = max(V1_current_active, V1_previous_active);
    V1_percent_change(i_frame) = ...
	V1_abs_change(i_frame) / (V1_max_active + (V1_max_active==0));
    V1_active_kf = mod(V1_active_ndx, V1_hdr.nf) + 1;
    if V1_max_active > 0
      V1_hist_frame = histc(V1_active_kf, V1_hist_edges);
    else
      V1_hist_frame = zeros(V1_hdr.nf+1,1);
    endif
    V1_hist = V1_hist + V1_hist_frame;
  endfor %% i_frame

  %%%%%%%%%% PLOTS: ACTIVITY %%%%%%%%%%%%%%%%%
  V1_hist = V1_hist(1:V1_hdr.nf);
  V1_hist = V1_hist / (num_frames * V1_hdr.nx * V1_hdr.ny); %% (sum(V1_hist(:)) + (nnz(V1_hist)==0));
  [V1_hist_sorted, V1_hist_rank] = sort(V1_hist, 1, "descend");
  v1rank = V1_hist_rank;
  V1_hist_title = ["V1_hist", ".png"];
  V1_hist_fig = figure;
  V1_hist_bins = 1:V1_hdr.nf;
  V1_hist_hndl = bar(V1_hist_bins, V1_hist_sorted); axis tight;
  set(V1_hist_fig, "name", ["V1_hist_", num2str(V1_times(num_frames), "%i")]);
  saveas(V1_hist_fig, ...
	 [statsdir, filesep, ...
	  "V1_rank_", num2str(V1_times(num_frames), "%i")], "png");

  V1_abs_change_title = ["V1_abs_change", ".png"];
  V1_abs_change_fig = figure;
  V1_abs_change_hndl = plot(V1_times, V1_abs_change); axis tight;
  set(V1_abs_change_fig, "name", ["V1_abs_change"]);
  saveas(V1_abs_change_fig, ...
	 [statsdir, filesep, "V1_abs_change", num2str(V1_times(num_frames), "%i")], "png");

  V1_percent_change_title = ["V1_percent_change", ".png"];
  V1_percent_change_fig = figure;
  V1_percent_change_hndl = plot(V1_times, V1_percent_change); axis tight;
  set(V1_percent_change_fig, "name", ["V1_percent_change"]);
  saveas(V1_percent_change_fig, ...
	 [statsdir, filesep, "V1_percent_change", num2str(V1_times(num_frames), "%i")], "png");
  V1_mean_change = mean(V1_percent_change(:));
  disp(["V1_mean_change = ", num2str(V1_mean_change)]);

  V1_tot_active_title = ["V1_tot_active", ".png"];
  V1_tot_active_fig = figure;
  V1_tot_active_hndl = plot(V1_times, V1_tot_active/n_V1); axis tight;
  set(V1_tot_active_fig, "name", ["V1_tot_active"]);
  saveas(V1_tot_active_fig, ...
	 [statsdir, filesep, "V1_tot_active", num2str(V1_times(num_frames), "%i")], "png");

  V1_mean_active = mean(V1_tot_active(:)/n_V1);
  disp(["V1_mean_active = ", num2str(V1_mean_active)]);
endfunction