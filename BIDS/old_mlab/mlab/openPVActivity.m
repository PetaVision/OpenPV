function [pv_file pv_image] = openPVActivity(pv_path,frame_offset,pv_layer)
    pv_image = 0;

    % pv_layer is looking at the petavision file a(pv_layer-1).pvp
    pv_layer += 1; % there is a minus one in pvp_openActivtyFile

    filename = ['a', num2str(pv_layer-1),'.pvp'];
    %disp(['openPVActivity: PetaVision file: ',pv_path,filename])

    [pv_fid, ...
       pv_header, ...
       pv_index ] = ...
          pvp_openActivityFile(pv_path, pv_layer);
      [layerID] = neoVisLayerID(pv_layer);
      
    if pv_fid == -1
      pv_file = 0;
      error(['openPVActivity: PetaVision file ',pv_path,filename,' not found.']);
    else
      pv_file = 1; 
    endif

    if pv_file
        global NFEATURES NCOLS NROWS N
        NCOLS = pv_header(pv_index.NX_GLOBAL);
        NROWS = pv_header(pv_index.NY_GLOBAL);
        NFEATURES = pv_header(pv_index.NF);
        N = NFEATURES * NCOLS * NROWS;

        tot_frames = 450;

        pv_offset = zeros(tot_frames, 1);
        pv_time = cell(tot_frames, 1);
        pv_activity = cell(tot_frames, 1);
        frame_pathnames = cell(tot_frames, 1);

        pv_frame_offset = frame_offset; %starts at frame 0
        pv_frame_skip   = 1000;
        num_frames      = 450;

        %disp(['openPVActivity: size of pv_time (should equal # of frames) = ',num2str(size(pv_time)(1))])
        %disp(['openPVActivity: pv_frame_offset = ',num2str(pv_frame_offset)])
        
        pv_offset_tmp = 0;
        i_frame = 0;
        for j_frame = pv_frame_offset : pv_frame_skip : num_frames
            i_frame = i_frame + 1;
            pv_frame = j_frame + pv_layer - 1;
            [pv_time{i_frame},...
                pv_activity{i_frame}, ...
                pv_offset(i_frame)] = ...
                pvp_readSparseLayerActivity(pv_fid, pv_frame, pv_header, pv_index, pv_offset_tmp);
            if pv_offset(i_frame) == -1
              break;
            endif
            pv_offset_tmp = pv_offset(i_frame);
            %disp(['openPVActivity: frame = ', num2str(i_frame)])
            %disp(['openPVActivity: pv_time = ', num2str(pv_time{i_frame})])
            %disp(['openPVActivity: mean(pv_activty) = ', num2str(mean(pv_activity{i_frame}(:)))])
        endfor
        fclose(pv_fid);

        i_frame=1; %NEED TO FIX LATER

        features = full(pv_activity{i_frame});
        features = reshape(features,[NFEATURES,NCOLS,NROWS]);
        pv_image = zeros(NCOLS,NROWS);

        pv_image(:,:) = features(1,:,:)+features(2,:,:)+features(3,:,:)+features(4,:,:)+features(5,:,:)+features(6,:,:)+features(7,:,:)+features(8,:,:);
       
        pv_image = flipud(rot90(pv_image));

        %disp(['openPVActivity: NCOLS = ',num2str(NCOLS)])
        %disp(['openPVActivity: NROWS = ',num2str(NROWS)])
        %disp(['openPVActivity: NFEATURES = ',num2str(NFEATURES)])
        %disp(['openPVActivity: Number of points = ',num2str(nnz(pv_image))])
    endif
endfunction
