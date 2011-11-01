function [CSV_struct] = pvp_makeCSVFileKernel(frame_pathname, pvp_time, pvp_activity, true_CSV_struct)

  global NFEATURES NCOLS NROWS N
  global pvp_patch_size
  global pvp_density_thresh
  global pvp_training_flag
  global ODD_dir

  CSV_struct = [];
  if isempty(pvp_activity)
    return;
  endif

  CSV_struct = struct;
  CSV_struct.frame_filename = strFolderFromPath(frame_pathname);
  CSV_struct.pvp_time = pvp_time;
  CSV_struct.num_active = nnz(pvp_activity);
  CSV_struct.mean_activity = mean(pvp_activity(:));
  CSV_struct.sum_activity = sum(pvp_activity(:));
  
  %%full_activity = full(pvp_activity);
  [pvp_image] = ...
      pvp_reconstructSparse(frame_pathname, ...
			    pvp_time, ...
			    pvp_activity);

  if pvp_training_flag
    [pvp_num_active_BB_mask, ...
     pvp_num_active_BB_notmask, ...
     pvp_num_BB_mask, ...
     pvp_num_BB_notmask] = ...
	pvp_numActiveInBoundingBox(pvp_activity, ...
				   true_CSV_struct);
    CSV_struct.num_active_BB_mask = pvp_num_active_BB_mask;
    CSV_struct.num_active_BB_notmask = pvp_num_active_BB_notmask;
    CSV_struct.num_BB_mask = pvp_num_BB_mask;
    CSV_struct.num_BB_notmask = pvp_num_BB_notmask;
  endif

  pvp_size = size(pvp_image);
  [hit_list, miss_list] = pvp_testPatches(pvp_activity);
  CSV_struct.hit_list = hit_list;
  CSV_struct.miss_list = miss_list;

  [pvp_image] = pvp_drawBoundingBox(pvp_image, hit_list);

  
  %%CSV_struct.pvp_image = pvp_image;
  pvp_image_title = CSV_struct.frame_filename;
  pvp_image_pathname = [ODD_dir, pvp_image_title];
  %%pvp_fig = figure;
  %%imagesc(pvp_image_tmp);
  %%print(pvp_fig, pvp_image_pathname, "-dpng");
  %%close all;
  imwrite(pvp_image, pvp_image_pathname);
  
  
endfunction %% pvp_makeCSVFileKernel



