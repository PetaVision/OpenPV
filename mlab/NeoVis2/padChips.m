
function [tot_chips, ...
	  tot_DoG, ...
	  tot_canny, ...
	  tot_cropped, ...
	  tot_mean, ...
	  tot_std, ...
	  tot_border_artifact_top, ...
	  tot_border_artifact_bottom, ...
	  tot_border_artifact_left, ...
	  tot_border_artifact_right, ...
	  ave_original_size, ...
	  ave_cropped_size, ...
	  std_original_size, ...
	  std_cropped_size, ...
	  max_cropped_size, ...
	  min_cropped_size, ...
	  tot_time] = ...
      padChips(chip_path, ...
	       clip_name_param, ...
	       PetaVision_path, ...
	       DoG_flag_param, ...
	       DoG_struct_param, ...
	       canny_flag_param, ...
	       canny_struct_param, ...
	       pad_size_param, ...
	       num_procs)
  
  %% perform edge filtering on DARPA NeoVis2 target chips, 
  %% mirror BCs used to pad images before edge extraction.
  %% also performs edge extraction on mask images if present

  global DoG_flag
  global canny_flag
  global DoG_dir
  global DoG_struct
  global canny_dir
  global canny_struct
  global image_margin
  global pad_size
  global cropped_dir
  global rejected_dir
  global border_artifact_thresh
  global image_size_thresh

  global VERBOSE_FLAG
  if ~exist("VERBOSE_FLAG") || isempty(VERBOSE_FLAG)
    VERBOSE_FLAG = false;
  endif
 
  more off;
  begin_time = time();

  %%keyboard;
  num_argin = 0;
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("chip_path") || isempty(chip_path)
    chip_path = "~/NeoVision2/neovision-programs-petavision/Heli/Training/mask/Car_original/"
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("clip_name_param") || isempty(clip_name_param)
      clip_name = "Car"; %% "Plane"; %% "Car_bootstrap1"; %%  "distractor_bootstrap1"; %%  "distractor"; %%     "051"; %%   
  else
    clip_name = clip_name_param;  
    %% "Person"; 
    %% "Cyclist"; 
    %% "Plane"; 
    %% "Boat"; 
    %% "Container"; 
    %% "Helicopter"; 
    %% "Car"; %%  
  endif
  disp(["clip_name = ", clip_name]);
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("PetaVision_path") || isempty(PetaVision_path)
    PetaVision_path = "~/NeoVision/neovision-programs-petavision/Heli/Training/";  %% 
  endif
  if ~exist(PetaVision_path,"dir")
    error(["does not exist: PetaVision_path = ", PetaVision_path]);
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("DoG_flag_param") || isempty(DoG_flag_param)
    if isempty(DoG_flag)
      DoG_flag = 1;  %% 
    endif
  else
    DoG_flag = DoG_flag_param;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("DoG_struct_param") || isempty(DoG_struct_param)
    if isempty(DoG_struct)
      DoG_struct = struct;  %% 
      DoG_struct.amp_center_DoG = 1;
      DoG_struct.sigma_center_DoG = 1;
      DoG_struct.amp_surround_DoG = 1;
      DoG_struct.sigma_surround_DoG = 2 * DoG_struct.sigma_center_DoG;
    endif
  else
    DoG_struct = DoG_struct_param;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("canny_flag_param") || isempty(canny_flag_param)
    if isempty(canny_flag)
      canny_flag = 1;  %% 
    endif
  else
    canny_flag = canny_flag_param;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("canny_struct_param") || isempty(canny_struct_param)
    if isempty(canny_struct)
      canny_struct = struct;  %% 
      canny_struct.sigma_canny = 1;
    endif
  else
    canny_struct = canny_struct_param;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pad_size_param") || isempty(pad_size_param)
    if isempty(pad_size)
      pad_size = [256 256]; %% [720 720]; %% [1080 1920]; %% 
    endif
  else
    pad_size = pad_size_param;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("num_procs") || isempty(num_procs)
    num_procs = 8;  %% 
  endif
  
  setenv('GNUTERM', 'x11');

  local_dir = pwd;

  tot_chips = 0;
  tot_unread = 0;
  tot_rejected = 0;
  tot_DoG = 0;
  tot_canny = 0;
  tot_mean = 0;
  tot_std = 0;
  tot_border_artifact_top = 0;
  tot_border_artifact_bottom = 0;
  tot_border_artifact_left = 0;
  tot_border_artifact_right = 0;
  ave_original_size = zeros(1,2);
  ave_cropped_size = zeros(1,2);
  std_original_size = zeros(1,2);
  std_cropped_size = zeros(1,2);
  max_cropped_size = zeros(1,2);
  min_cropped_size = pad_size;
  tot_cropped = 0;
  tot_rejected = 0;
  cropped_list = {};
  
  %% path to generic image processing routines
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);

  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);

  chip_dir = [chip_path]; %%, "chips", filesep];
  if ~exist(chip_dir, "dir")
    error(["~exist(chip_dir): ", chip_dir]);
  endif
  target_dir = ...
      [chip_dir, clip_name, filesep];  %%
  if ~exist(target_dir, "dir")
    error(["~exist(target_dir): ", target_dir]);
  endif

  %%keyboard;
  cropped_path = [PetaVision_path, "cropped", filesep];
  mkdir(cropped_path);
  cropped_dir = [cropped_path, clip_name, filesep];
  mkdir(cropped_dir);

  rejected_path = [PetaVision_path, "rejected", filesep];
  mkdir(rejected_path);
  rejected_dir = [rejected_path, clip_name, filesep];
  mkdir(rejected_dir);

  log_path = [PetaVision_path, "log", filesep];
  mkdir(log_path);
  log_dir = [log_path, clip_name, filesep];
  mkdir(log_dir);

  list_path = [PetaVision_path, "list", filesep];
  mkdir(list_path);
  list_dir = [list_path, clip_name, filesep];
  mkdir(list_dir);

  if DoG_flag
    DoG_folder = [PetaVision_path, "DoG", filesep];
    mkdir(DoG_folder);
    DoG_dir = [DoG_folder, clip_name, filesep];
    mkdir(DoG_dir);
  endif %% DoG_flag
  if canny_flag
    canny_folder = [PetaVision_path, "canny", filesep];
    mkdir(canny_folder);
    canny_dir = [canny_folder, clip_name, filesep];
    mkdir(canny_dir);
  endif %% canny_flag
  hist_folder = [PetaVision_path, "hist", filesep];
  mkdir(hist_folder);
  hist_path = [hist_folder, clip_name, filesep];
  mkdir(hist_path);

  image_type = ".png";
  image_margin = 8;

  border_artifact_thresh =1000000; %% 1.25; %% use 1.25 for DARPA HeliChips
  image_size_thresh = 1000; %% in bytes

  target_path = ...
      [target_dir, '*', image_type];
  target_pathnames = glob(target_path);
  num_chips = size(target_pathnames,1);
  disp(['num_chips = ', num2str(num_chips)]);

  mean_hist = zeros(1, num_chips);
  std_hist = zeros(1, num_chips);
  std_mean_hist = zeros(1, num_chips);
    
  %%keyboard;
  if num_procs > 1
    [status_info] = ...
	parcellfun(num_procs, @padChipKernel, target_pathnames, "UniformOutput", false);
  else
    [status_info] = ...
	cellfun(@padChipKernel, target_pathnames, "UniformOutput", false);
  endif

  for i_chip = 1 : num_chips
      tot_unread = tot_unread + status_info{i_chip}.unread_flag;
      tot_rejected = tot_rejected + status_info{i_chip}.rejected_flag;
      if status_info{i_chip}.rejected_flag
	continue;
      endif
      tot_DoG = tot_DoG + status_info{i_chip}.DoG_flag;
      tot_canny = tot_canny + status_info{i_chip}.canny_flag;
      tot_mean = tot_mean + status_info{i_chip}.mean;
      tot_std = tot_std + status_info{i_chip}.std;
      mean_hist(i_chip) = status_info{i_chip}.mean;
      std_hist(i_chip) = status_info{i_chip}.std;
      std_mean_hist(i_chip) = std_hist(i_chip) / (mean_hist(i_chip) + (mean_hist(i_chip)==0));
      %% cropped images
      tot_border_artifact_top = tot_border_artifact_top + status_info{i_chip}.num_border_artifact_top;
      tot_border_artifact_bottom = tot_border_artifact_bottom + status_info{i_chip}.num_border_artifact_bottom;
      tot_border_artifact_left = tot_border_artifact_left + status_info{i_chip}.num_border_artifact_left;
      tot_border_artifact_right = tot_border_artifact_right + status_info{i_chip}.num_border_artifact_right;
      ave_original_size = ave_original_size + status_info{i_chip}.original_size(1:2);
      ave_cropped_size = ave_cropped_size + status_info{i_chip}.cropped_size;
      std_original_size = std_original_size + (status_info{i_chip}.original_size(1:2)).^2;
      std_cropped_size = std_cropped_size + (status_info{i_chip}.cropped_size).^2;
      if any(status_info{i_chip}.cropped_size ~= status_info{i_chip}.original_size(1:2))
	tot_cropped = tot_cropped + 1;
	cropped_list{tot_cropped} = status_info{i_chip}.chipname;
      endif
      max_cropped_size = max(max_cropped_size, status_info{i_chip}.cropped_size);
      min_cropped_size = min(min_cropped_size, status_info{i_chip}.cropped_size);
   endfor %% i_chip

   tot_chips = num_chips - tot_rejected;

plot_hist = 0;
if plot_hist
   fig_list = [];
   num_hist_bins = 20;

   [hist_mean_count, hist_mean_bins] = ...
       hist(mean_hist, num_hist_bins);
   fh = figure;
   fig_name = "hist_mean";
   fig_path = [hist_path, fig_name];
   set(fh, "Name", fig_name);
   fig_list = [fig_list, fh];
   bar(hist_mean_bins, hist_mean_count);
   print(fh, fig_path, "-dpng");

   [hist_std_count, hist_std_bins] = ...
       hist(std_hist, num_hist_bins);
   fh = figure;
   fig_name = "hist_std";
   fig_path = [hist_path, fig_name];
   set(fh, "Name", fig_name);
   fig_list = [fig_list, fh];
   bar(hist_std_bins, hist_std_count);
   print(fh, fig_path, "-dpng");
   
   [hist_std_mean_count, hist_std_mean_bins] = ...
       hist(std_mean_hist, num_hist_bins);
   fh = figure;
   fig_name = "hist_std_mean";
   fig_path = [hist_path, fig_name];
   set(fh, "Name", fig_name);
   fig_list = [fig_list, fh];
   bar(hist_std_mean_bins, hist_std_mean_count);
   print(fh, fig_path, "-dpng");
 endif
  
   ave_mean = tot_mean / tot_chips;
   disp(["ave_mean = ", ...
	 num2str(ave_mean)]);
   ave_std = tot_std / tot_chips;
   disp(["ave_std = ", ...
	 num2str(ave_std)]);

   ave_original_size =  ave_original_size / tot_chips;
   disp(["ave_original_size = ", ...
	 num2str(ave_original_size)]);
   std_original_size =  sqrt((std_original_size / tot_chips) - ave_original_size.^2);
   disp(["std_original_size = ", ...
	 num2str(std_original_size)]);

   ave_cropped_size =  ave_cropped_size / tot_chips;
   disp(["ave_cropped_size = ", ...
	 num2str(ave_cropped_size)]);
   std_cropped_size =  sqrt((std_cropped_size / tot_chips) - ave_cropped_size.^2);
   disp(["std_cropped_size = ", ...
	 num2str(std_cropped_size)]);
   disp(["max_cropped_size = ", ...
	 num2str(max_cropped_size)]);
   disp(["min_cropped_size = ", ...
	 num2str(min_cropped_size)]);

   disp(["tot_border_artifact_top = ", ...
	 num2str(tot_border_artifact_top)]);
   disp(["tot_border_artifact_bottom = ", ...
	 num2str(tot_border_artifact_bottom)]);
   disp(["tot_border_artifact_left = ", ...
	 num2str(tot_border_artifact_left)]);
   disp(["tot_border_artifact_right = ", ...
	 num2str(tot_border_artifact_right)]);
 
   disp(["tot_chips = ", ...
	 num2str(tot_chips)]);
   disp(["tot_unread = ", ...
	 num2str(tot_unread)]);
   disp(["tot_rejected = ", ...
	 num2str(tot_rejected)]);
   disp(["tot_DoG = ", ...
	 num2str(tot_DoG)]);
   disp(["tot_canny = ", ...
	 num2str(tot_canny)]);
   disp(["tot_cropped = ", ...
	 num2str(tot_cropped)]);

   log_filename = [log_dir, "log.txt"];
   save("-ascii", ...
	log_filename, ...
	"tot_unread", ...
	"tot_rejected", ...
	"tot_chips", ...
	"tot_DoG", ...
	"tot_canny", ...
	"tot_cropped", ...
	"tot_mean", ...
	"tot_std", ...
	"tot_border_artifact_top", ...
	"tot_border_artifact_bottom", ...
	"tot_border_artifact_left", ...
	"tot_border_artifact_right", ...
	"ave_original_size", ...
	"ave_cropped_size", ...
	"std_original_size", ...
	"std_cropped_size", ...
	"max_cropped_size", ...
	"min_cropped_size", ...
	"tot_time"); 

   list_filename = [list_dir, "cropped_list.txt"];
   [fid, msg_error] = fopen(list_filename, "w");
   if fid == -1
     disp(msg_error);
     disp(["list_filename = ", list_filename]);
   else
     for i_crop = 1 : tot_cropped
       fputs(fid, cropped_list{i_crop});
     endfor
     fclose(fid);
   endif
  
  end_time = time();
  tot_time = end_time - begin_time;
  disp(["tot_time = ", num2str(tot_time)]);


endfunction %% padChips
