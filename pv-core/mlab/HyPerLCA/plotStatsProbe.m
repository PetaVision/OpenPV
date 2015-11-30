  if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_list = ...
        {["Error"],["_Stats.txt"]; ...
         ["V1"],["_Stats.txt"];
         ["Error2"],["_Stats.txt"]; ...
         ["V2"],["_Stats.txt"];
         ["Error1_2"],["_Stats.txt"]; ...
         ["V1Infra"],["_Stats.txt"]};
  elseif strcmp(run_type, "noPulvinar") || strcmp(run_type, "TopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_list = ...
        {["Error"],["_Stats.txt"]; ...
         ["V1"],["_Stats.txt"];
         ["V2"],["_Stats.txt"];
         ["Error1_2"],["_Stats.txt"]; ...
         ["V1Infra"],["_Stats.txt"]};
  elseif strcmp(run_type, "lateral")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_list = ...
	{["Error"],["_Stats.txt"]; ...
	 ["V1"],["_Stats.txt"];
	 ["Error2"],["_Stats.txt"]; ...
	 ["V2"],["_Stats.txt"];
	 ["Error1_2"],["_Stats.txt"]; ...
	 ["V2"],["_Stats.txt"]; ...
	 ["V2"],["_Stats.txt"]};
  elseif strcmp(run_type, "CIFAR_deep")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_list = ...
	{["LabelError"],["_Stats.txt"]; ...
	 ["Error"],["_Stats.txt"]; ...
	 ["Error1_2"],["_Stats.txt"]; ...
	 ["V1"],["_Stats.txt"];...
	 ["V2"],["_Stats.txt"]};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  StatsProbe_vs_time_dir = [output_dir, filesep, "StatsProbe_vs_time"]
  [status, msg, msgid] = mkdir(StatsProbe_vs_time_dir);
  if status ~= 1
    warning(["mkdir(", StatsProbe_vs_time_dir, ")", " msg = ", msg]);
  endif 
  num_StatsProbe_list = size(StatsProbe_list,1);

  StatsProbe_sigma_flag = ones(1,num_StatsProbe_list);
  if strcmp(run_type, "color_deep")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([2,4,6]) = 0;
  elseif strcmp(run_type, "noPulvinar") || strcmp(run_type, "TopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([2,3,5]) = 0;
  elseif strcmp(run_type, "lateral")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([2,4,6,7]) = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif  strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([4]) = 0;
    StatsProbe_sigma_flag([5]) = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  StatsProbe_nnz_flag = ~StatsProbe_sigma_flag;
  for i_StatsProbe = 1 : num_StatsProbe_list
    StatsProbe_file = [output_dir, filesep, StatsProbe_list{i_StatsProbe,1}, StatsProbe_list{i_StatsProbe,2}]
    if ~exist(StatsProbe_file,"file")
      warning(["StatsProbe_file does not exist: ", StatsProbe_file]);
      continue;
    endif
    [status, wc_output] = system(["cat ",StatsProbe_file," | wc"], true, "sync");
    if status ~= 0
      error(["system call to compute num lines failed in file: ", StatsProbe_file, " with status: ", num2str(status)]);
    endif
    wc_array = strsplit(wc_output, " ", true);
    StatsProbe_num_lines = str2num(wc_array{1});
    StatsProbe_fid = fopen(StatsProbe_file, "r");
    StatsProbe_line = fgets(StatsProbe_fid);
    StatsProbe_sigma_vals = [];
    StatsProbe_nnz_vals = [];
    last_StatsProbe_line = StatsProbe_num_lines - 2;
    last_StatsProbe_line = min(last_StatsProbe_line, max_StatsProbe_line);
    first_StatsProbe_line = max([(last_StatsProbe_line - StatsProbe_plot_lines), 1]);
    StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    for i_line = 1:first_StatsProbe_line-1
      StatsProbe_line = fgets(StatsProbe_fid);
    endfor
    %% extract N
    StatsProbe_N_ndx1 = strfind(StatsProbe_line, "N==");
    StatsProbe_N_ndx2 = strfind(StatsProbe_line, "Total==");
    StatsProbe_N_str = StatsProbe_line(StatsProbe_N_ndx1+3:StatsProbe_N_ndx2-2);
    StatsProbe_N = str2num(StatsProbe_N_str);
    for i_line = first_StatsProbe_line:last_StatsProbe_line
      StatsProbe_line = fgets(StatsProbe_fid);
      %% extract time
      StatsProbe_time_ndx1 = strfind(StatsProbe_line, "t==");
      StatsProbe_time_ndx2 = strfind(StatsProbe_line, "N==");
      StatsProbe_time_str = StatsProbe_line(StatsProbe_time_ndx1+3:StatsProbe_time_ndx2-2);
      StatsProbe_time_vals(i_line-first_StatsProbe_line+1) = str2num(StatsProbe_time_str);
      %% extract sigma
      StatsProbe_sigma_ndx1 = strfind(StatsProbe_line, "sigma==");
      StatsProbe_sigma_ndx2 = strfind(StatsProbe_line, "nnz==");
      StatsProbe_sigma_str = StatsProbe_line(StatsProbe_sigma_ndx1+7:StatsProbe_sigma_ndx2-2);
      StatsProbe_sigma_vals(i_line-first_StatsProbe_line+1) = str2num(StatsProbe_sigma_str);
      %% extract nnz
      StatsProbe_nnz_ndx1 = strfind(StatsProbe_line, "nnz==");
      StatsProbe_nnz_ndx2 = length(StatsProbe_line); 
      StatsProbe_nnz_str = StatsProbe_line(StatsProbe_nnz_ndx1+5:StatsProbe_nnz_ndx2-1);
      StatsProbe_nnz_vals(i_line-first_StatsProbe_line+1) = str2num(StatsProbe_nnz_str);
    endfor %%i_line
    fclose(StatsProbe_fid);
    if plot_flag
      StatsProbe_vs_time_fig(i_StatsProbe) = figure;
    endif
    if StatsProbe_nnz_flag(i_StatsProbe)
      if plot_flag
	StatsProbe_vs_time_hndl = plot(StatsProbe_time_vals, StatsProbe_nnz_vals/StatsProbe_N);
	axis tight
	set(StatsProbe_vs_time_fig(i_StatsProbe), "name", [StatsProbe_list{i_StatsProbe,1}, " nnz"]);
	saveas(StatsProbe_vs_time_fig(i_StatsProbe), ...
	       [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
		"_nnz_vs_time_", num2str(StatsProbe_time_vals(end), "%08d")], "png");
      else
	%% don't know how to imwrite a scatter plot
      endif
    else
      if plot_flag
	StatsProbe_vs_time_hndl = plot(StatsProbe_time_vals, StatsProbe_sigma_vals); axis tight;
	axis tight
	set(StatsProbe_vs_time_fig(i_StatsProbe), "name", [StatsProbe_list{i_StatsProbe,1}, " sigma"]);
	saveas(StatsProbe_vs_time_fig(i_StatsProbe), ...
	       [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
		"_sigma_vs_time_", num2str(StatsProbe_time_vals(end), "%08d")], "png");
      else
	%% don't know how to imwrite a scatter plot
      endif
      save("-mat", ...
	   [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
	    "_sigma_vs_time_", num2str(StatsProbe_time_vals(end), "%08d"), ".mat"], ...
	   "StatsProbe_time_vals", "StatsProbe_sigma_vals");
    endif %% 
    drawnow;
  endfor %% i_StatsProbe
