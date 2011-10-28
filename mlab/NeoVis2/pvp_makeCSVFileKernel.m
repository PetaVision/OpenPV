function [CSV_struct] = pvp_makeCSVFileKernel(frame_pathname, pvp_time, pvp_activity)

  global NFEATURES NCOLS NROWS N
  CSV_struct = struct;
  CSV_struct.frame_filename = strFolderFromPath(frame_pathname);
  CSV_struct.pvp_time = pvp_time;
  CSV_struct.mean_activity = mean(pvp_activity(:));
  CSV_struct.sum_activity = sum(pvp_activity(:));
  
  %%full_activity = full(pvp_activity);
  [pvp_image, pvp_num_active] = ...
      pvp_reconstructSparse(frame_pathname, pvp_time, pvp_activity);
  CSV_struct.pvp_image = pvp_image;
  CSV_struct.pvp_num_active = pvp_num_active;

  pvp_size = size(pvp_image);

  %% patch & bounding box coordinates go counter clockwise from bottom left
  CSV_struct.patch_X1 = 0.0;
  CSV_struct.patch_Y1 = 0.0;
  CSV_struct.patch_X2 =  pvp_size(1);
  CSV_struct.patch_Y2 = 0.0;
  CSV_struct.patch_X3 = pvp_size(1);
  CSV_struct.patch_Y3 = pvp_size(2);
  CSV_struct.patch_X4 = 0.0;
  CSV_struct.patch_Y4 =  pvp_size(2);
  CSV_struct.confidence = 0.0;
  CXV_struct.site_info = [];
  CSV_struct.BoundingBox_X1 = 0.0;
  CSV_struct.BoundingBox_Y1 = 0.0;
  CSV_struct.BoundingBox_X2 = 0.0;
  CSV_struct.BoundingBox_X2 = 0.0;
  CSV_struct.BoundingBox_X3 = 0.0;
  CSV_struct.BoundingBox_Y3 = 0.0;
  CSV_struct.BoundingBox_X4 = 0.0;
  CSV_struct.BoundingBox_X4 = 0.0;
  
  
endfunction %% pvp_makeCSVFileKernel



