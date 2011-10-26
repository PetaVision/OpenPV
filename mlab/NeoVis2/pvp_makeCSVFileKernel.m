function [CSV_struct] = pvp_makeCSVFileKernel(frame_ID, pvp_time, pvp_activity)

  global NFEATURES NCOLS NROWS N
  CSV_struct = struct;
  CSV_struct.frame_ID = frame_ID;
  CSV_struct.pvp_time = pvp_time;
  CSV_struct.mean_activity = mean(pvp_activity(:));
  CSV_struct.sum_activity = sum(pvp_activity(:));
  
  %%full_activity = full(pvp_activity);
  [pvp_image, pvp_num_active] = ...
      pvp_reconstructSparse(frame_ID, pvp_time, pvp_activity);
  CSV_struct.pvp_image = pvp_image;
  CSV_struct.pvp_num_active = pvp_num_active;
  
endfunction %% pvp_makeCSVFileKernel



