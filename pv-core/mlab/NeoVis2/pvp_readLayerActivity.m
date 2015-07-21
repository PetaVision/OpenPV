function [pvp_time,...
	  pvp_activity] = ...
      pvp_readLayerActivity(pvp_fid, pvp_frame, pvp_layer, pvp_header)

  NX_PROCS = pvp_header(pvp_index.NX_PROCS);
  NY_PROCS = pvp_header(pvp_index.NY_PROCS);

  NX_LOCAL = pvp_header(pvp_index.NX);  
  NY_LOCAL = pvp_header(pvp_index.NY);

  NCOLS = pvp_header(pvp_index.NX_GLOBAL);  
  NROWS = pvp_header(pvp_index.NY_GLOBAL);
  NX_GLOBAL = NCOLS;
  NY_GLOBAL = NROWS;
  NFEATURES = pvp_header(pvp_index.NF);
  N = NROWS * NCOLS * NFEATURES;
  NO = floor( NFEATURES / NK );
  N_LOCAL = NX_LOCAL * NY_LOCAL * NFEATURES;
  
  pvp_frame_offset = pvp_frame * ( N + 2 ) * 4;  % N activity vals + double time

  pvp_status = fseek(fid, 4*(NUM_BIN_PARAMS-2), "bof");  % read integer header
  pvp_status = fseek(fid, 8, "cof"); % read time (double)
  pvp_status = fseek(fid, pvp_frame_offset, "cof");
  if ( pvp_status == -1 )
    disp(["fseek(fid, pvp_frame_offset, ""cof"") == -1 in pvp_frame: ", num2str(pvp_frame)]);
    return;
  endif

  pvp_activity = zeros(NFEATURES, NCOLS, NROWS);
  pvp_time = fread(fid,1,"float64");
  disp(["pvp_time = ", num2str(pvp_time)]);
  %%keyboard;
  for y_proc = 1 : NX_PROCS
    for x_proc = 1 : NY_PROCS
      [activity_local, countF] = fread(fid, N_LOCAL, "float32");
      if countF ~= N_LOCAL
	disp(["countF ~= N_LOCAL:", "countF = ", num2str(countF), "; N_LOCAL = ", num2str(N_LOCAL)]);
      endif
      pvp_activity(:, NX_LOCAL*(x_proc-1)+1:NX_LOCAL*x_proc, NY_LOCAL*(y_proc-1)+1:NY_LOCAL*y_proc) = ...
	  reshape(activity_local, [NFEATURES, NX_LOCAL, NY_LOCAL]);
    endfor %% x_proc
  endfor %% y_proc

  %% make sparse
  pvp_activity = sparse(reshape(pvp_activity, [N,1]));
  


 






