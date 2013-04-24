%function [data,hdr] = readpvpfile(filename,progressperiod)
% Usage:[data,hdr] = readpvpfile(filename)
% filename is a pvp file (any type)
% data is a cell array containing the data.
%     In general, data has one element for each time step written.
%     Each element is a struct containing the fields 'time' and 'values'
%     For activities, values is an nx-by-ny-by-nf array.
%     For weights, values is a cell array, each element is an nxp-by-nyp-by-nfp array.

%%%%%%%%
%%DELETE
%%%%%%%%
clear all; close all; more off;

global output_path;  output_path  = '/Users/slundquist/Documents/workspace/BIDS/output/';
global filename;     filename     = '/Users/slundquist/Documents/workspace/BIDS/output/BIDS_Clone.pvp';
global rootname;     rootname     = '00';

global OUT_FILE_EXT; OUT_FILE_EXT = 'png';             %either png or jpg for now

global MOVIE_FLAG;   MOVIE_FLAG   = 1;                 %set 1 to make a movie, set -1 to not make a movie
global FNUM_SPEC;    FNUM_SPEC    = '-1';               %can be '-1', 'int(frame)', or 'start:int:end'
global GRAPH_FLAG;   GRAPH_FLAG   = 1;                 %set to 1 to plot Histograms and ROC Curves, -1 to not
global GRAPH_SPEC;   GRAPH_SPEC   = [237,437,438,638]; %set to [no_stim_start, no_stim_end, stim_start, stim_end]
global PRINT_FLAG;   PRINT_FLAG   = 0;                 %set to 1 to print all fiures to output dir, -1 to not
%%%%%%%%

%% Parse some flags
if ~strcmpi(FNUM_SPEC,'-1') %if flag is not set to -1
    fnum_flag = 1;

    disp_frames = str2num(FNUM_SPEC);
    if gt(length(disp_frames),1) %flag delimits start-frame:interval:end-frame
        frame_of_interest = -1; %will only be set if user gave a single frame as input to FNUM_SPEC
    elseif eq(length(disp_frames),1) %single number is given
        frame_of_interest = disp_frames+1;
    else
        error(['FNUM_SPEC is not correct. FNUM_SPEC = ',FNUM_SPEC])
    end%if gt(length(disp_frames),1)
else
    fnum_flag = -1;
    frame_of_interest = -1;
end%if strcmpi

%% ROC Graph stuff
if gt(GRAPH_FLAG,0)
   no_stim_length = GRAPH_SPEC(2) - GRAPH_SPEC(1);
   stim_length    = GRAPH_SPEC(4) - GRAPH_SPEC(3);
   if (lt(no_stim_length,0) || lt(stim_length,0) || ne(no_stim_length,stim_length))
      die('readpvpfile: Graph spec not properly formatted!');
   end
   half_length = stim_length;
   clear stim_length;
   clear no_stim_length;
end


filedata = dir(filename);
if length(filedata) ~= 1
    error('readpvpfile:notonefile',...
          'Path %s should expand to exactly one file; in this case there are %d',...
          filename,length(filedata));
end

if filedata(1).bytes < 1
    error('readpvpfile:fileempty',...
          'File %s is empty',filename);
end%if filedata(1).bytes

fid = fopen(filename);

errorident = '';
errorstring = '';

hdr = readpvpheader(fid);

%% Correct for max frame
if gt(fnum_flag,0)
    if gt(length(disp_frames),1)
        if gt(max(disp_frames(:)),hdr.nbands-1)
            disp_frames(end) = hdr.nbands-1;
        end%if gt(max...
    end%if gt(length...
end%if fnum_flag


switch hdr.filetype
    case 1 % PVP_FILE_TYPE
        framesize = hdr.recordsize*hdr.numrecords;
        numframes = (filedata(1).bytes - hdr.headersize)/framesize;
    case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking
        disp(['PVP File Type: Spiking'])
        numframes = hdr.nbands;
        % framesize is variable
    case 3 % PVP_WGT_FILE_TYPE % HyPerConns that aren't KernelConns
        framesize = hdr.recordsize*hdr.numrecords*hdr.nbands+hdr.headersize;
        numframes = filedata(1).bytes/framesize;
    case 4 % PVP_NONSPIKING_ACT_FILE_TYPE
        nxprocs = hdr.nxGlobal/hdr.nx;
        nyprocs = hdr.nyGlobal/hdr.ny;
        framesize = hdr.recordsize*hdr.datasize*nxprocs*nyprocs+8;
        numframes = hdr.nbands;
    case 5 % PVP_KERNEL_FILE_TYPE
        framesize = hdr.recordsize*hdr.nbands+hdr.headersize;
        numframes = filedata(1).bytes/framesize;
    otherwise
        errorident = 'readpvpfile:badfiletype';
        errorstring = sprintf('readpvpfile:File %s has unrecognized file type %d',filename,hdr.filetype);
end

if isempty(errorstring)
    if(numframes ~= round(numframes) || numframes <= 0)
        errorident = 'readpvpfile:badfilelength';
        errorstring = sprintf('readpvpfile:File %s has file length inconsistent with header',filename);
    end
end

if isempty(errorstring)
    data = cell(numframes,1);
    switch hdr.datatype
        case 1 % PV_BYTE_TYPE
            precision = 'uchar';
        case 2 % PV_INT_TYPE
            precision = 'int32';
        case 3 % PV_FLOAT_TYPE
            precision = 'float32';
    end
    switch hdr.filetype
        case 1 % PVP_FILE_TYPE, used in HyPerCol::exitRunLoop
            numvalues = hdr.recordsize/hdr.datasize;
            for f=1:numframes
                data{f} = struct('time',hdr.time,'values',[]);
                data{f}.time = hdr.time;
                Y = zeros(numvalues,hdr.numrecords);
                for r=1:hdr.numrecords
                    Y(:,r) = fread(fid, numvalues, precision);
                end
                Z = zeros(hdr.nxGlobal,hdr.nyGlobal,hdr.nf);
                r=0;
                for y=1:hdr.nyprocs
                    yidx = (1:hdr.ny)+(y-1)*hdr.ny;
                    for x=1:hdr.nxprocs
                        xidx = (1:hdr.nx)+(x-1)*hdr.nx;
                        r = r+1;
                        for feature=1:hdr.nf
                            Z(xidx,yidx,feature) = reshape(Y(feature:hdr.nf:hdr.nx*hdr.ny*hdr.nf,r),hdr.nx,hdr.ny);
                        end
                    end
                end
                data{f}.values = Z;
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                    end
                end
            end
        case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking; I'm not using yet
            status = fseek(fid,4*hdr.numparams,"bof"); %hdr.numparams = number of parameters specified in the header file
            if ne(status,0)
                error('readpvpfile: unable to seek to the end of the header for pvp file ',filedata)
            end%if ne(status,0)while pvp_time < pvp_frame

            integrated_image = zeros([hdr.nyGlobal,hdr.nxGlobal]);

            if gt(GRAPH_FLAG,0)
               integrated_half1   = zeros([hdr.nyGlobal,hdr.nxGlobal]);
               integrated_half0   = zeros([hdr.nyGlobal,hdr.nxGlobal]);
            end

            spike_count = zeros([numframes 1]);

            for frame=0:numframes-1
                if feof(fid) 
                  break;
                end%if feof

                %%%%%%%
                %% Verify time step
                %%%%%%%
                [time, count] = fread(fid,1,"float64"); %count is the number of items read
                if eq(count,0)
                    error('readpvpfile: Did not read any elements into time from pvp file')
                end%if eq(count,0)
                if ne(time,frame) %pvp files start at time_step = 0
                    error(['readpvpfile: Not reading the correct frame. frame = ',num2str(frame),...
                        'time = ',num2str(time)])
                end%if ne(time,frame)

                %%%%%%%
                %% Get the total number of spikes in this time step
                %%%%%%%
                [num_spikes, count] = fread(fid,1,"int32");
                if eq(count,0)
                    error('readpvpfile: Did not read any elements into num_spikes from pvp file')
                end%if eq(count,0)

                %%%%%%%
                %% Get the index for each spike for this time step
                %%%%%%%
                [spike_idx, count] = fread(fid,num_spikes,"int32"); %spike_idx is a vector of indices for each spike
                if ne(count,num_spikes)
                    error(['readpvpfile: The number of elements read in does not equal the number of spikes fired.',...
                        char(10),'numbewhile pvp_time < pvp_framer of elements = ',num2str(count),'number of spikes = ',num2str(num_spikes)])
                end%if ne(count,num_spikes)

                %%%%%%%
                %% Get sparse activity
                %%%%%%%
                sparse_size = hdr.nf*hdr.nxGlobal*hdr.nyGlobal;
                activity = sparse(spike_idx+1, 1, 1, sparse_size, 1, num_spikes); %spike_idx+1 because pvp starts idx at 0

                features = full(activity);
                features = reshape(features,[hdr.nf,hdr.nxGlobal,hdr.nyGlobal]); %nf = num_features, nx = num_col, ny = num_rows

                %%%%%%%
                %% Create PVP image & integrated image(s)
                %%%%%%%
                pvp_image = squeeze(sum(features,1));
                pvp_image = flipud(rot90(pvp_image));

                integrated_image += pvp_image;


                %%%%%%%
                %% Count spikes
                %%%%%%%
                spike_count(frame+1) = length(find(pvp_image));

                if gt(GRAPH_FLAG,0)
                   if (gt(frame,GRAPH_SPEC(1)) && lt(frame,GRAPH_SPEC(2)))
                       integrated_half0 += pvp_image;
                    elseif (gt(frame,GRAPH_SPEC(3)) && lt(frame,GRAPH_SPEC(4)))
                       integrated_half1 += pvp_image;
                   end
                end
               
                %%%%%%%
                %% Set up frame string for printing
                %%%%%%%
                if lt(frame,10)
                    frame_str = ['00',num2str(frame)];
                elseif ge(frame,10) && lt(frame,100)
                    frame_str = ['0', num2str(frame)];
                elseif gt(frame,99)
                    frame_str = num2str(frame);
                end%if lt(frame,10)

                if eq(MOVIE_FLAG,1)
                    movie_path = [output_path, 'Movie/'];
                    if ne(exist(movie_path,'dir'),7) %if exists func doesn't return a 7, then movie_path is not a dir
                        mkdir(movie_path);
                    end%if ne(exist(),7)

                    inst_movie_path = [movie_path,'Instantaneous_Frames/'];
                    if ne(exist(inst_movie_path,'dir'),7) %if exists func doesn't return a 7, then inst_movie_path is not a dir
                        mkdir(inst_movie_path);
                    end%if ne(exist(),7)

                    print_movie_filename = [inst_movie_path,rootname,'_',frame_str,'.',OUT_FILE_EXT];

                    try
                        imwrite(pvp_image,print_movie_filename,OUT_FILE_EXT)
                    catch
                        disp(['readpvpfile: WARNING. Could not print file: ',char(10),print_movie_filename])
                    end%_try_catch
                end%if eq(MOVIE_FLAG,1)

                if gt(fnum_flag,0)
                    if eq(frame,frame_of_interest) || find(disp_frames==frame)
                        figs_disp_flag = 1;

                        tim_fig_id = figure;
                            imshow(pvp_image)
                            axis image
                            %axis off

                        int_fig_id = figure;
                            imagesc(integrated_image)
                            axis image
                            %axis off
                            colorbar

                        fig_ids = [2];
                        fig_ids(1) = tim_fig_id;
                        fig_ids(2) = int_fig_id;
                    else
                        figs_disp_flag = 0;
                    end%if eq(frame,frame_of_interest)
                end %gt(fnum_flag,0)

                if gt(PRINT_FLAG,0)
                    if figs_disp_flag
                        print_fig_path = [output_path, 'Figures/'];
                        if ne(exist(print_fig_path,'dir'),7) %if exists func doesn't return a 7, then print_fig_path is not a dir
                            mkdir(print_fig_path);
                        end%if ne(exist(),7)

                        print_fig_filename = [print_fig_path,rootname,'_','frame',frame_str,'_fig'];
                        try
                            for fig_idx = 1:length(fig_ids)
                                curr_fig_id = fig_ids(fig_idx);
                                if ne(curr_fig_id,-1)
                                   print(curr_fig_id,['-d',OUT_FILE_EXT],[print_fig_filename,num2str(curr_fig_id),'.',OUT_FILE_EXT]);
                                   close(curr_fig_id);
                                end
                            end%for fig_idx
                        catch
                            disp(['readpvpfile: Couldn''t print figure ',print_fig_filename])
                        end%_try_catch
                    end%if fig_id
                end%if PRINT_FLAG
            end%for frame

            if gt(GRAPH_FLAG,0)
                %%%%%%%
                %% Count spikes per node per half
                %%%%%%%
                num_spikes_nostim = sum(spike_count(GRAPH_SPEC(1):GRAPH_SPEC(2)))/half_length;
                num_spikes_stim   = sum(spike_count(GRAPH_SPEC(3):GRAPH_SPEC(4)))/half_length;

                num_bins = 500;
                p_set = zeros(2,num_bins);
                mask = ones([hdr.nyGlobal,hdr.nxGlobal]);
                
                %% Find number of spikes in half
                [rows0 cols0 counts0] = find(integrated_half0.*mask);
                [rows1 cols1 counts1] = find(integrated_half1.*mask);
                if (eq(length(counts0),0) || eq(length(counts1),0))
                  Pd = 0;
                  Pf = 0;
                else
                   tot_counts = [counts1;counts0];

                   [freq_counts, bin_loc] = hist(tot_counts,num_bins);
                   
                   h0 = hist(counts0,bin_loc,1);
                   h1 = hist(counts1,bin_loc,1);

                   figure
                   hold on
                   fid0 = bar(bin_loc,h0);
                   fid1 = bar(bin_loc,h1);
                   hold off
                   set(fid0,'facecolor',[0 0 1])
                   set(fid0,'edgecolor',[0 0 1])
                   set(fid1,'facecolor',[1 0 0])
                   set(fid1,'edgecolor',[1 0 0])
                   xlabel('Number of spikes')
                   ylabel('Normalized value number of nodes')
                   title('Histogram Plot for BIDS nodes')

                   Pd = cumsum(h1);
                   Pf = cumsum(h0);
                end %length(counts0)

                p_set(1,:) = Pd;
                p_set(2,:) = Pf;

                figure
                hold on
                plot([0,1],[0,1],'k')
                plot(p_set(1,:),p_set(2,:),'Color','red')
                text(.05, .95, ['Area under Roc Curve: ', num2str(trapz(Pd, Pf))], 'Color', 'k');
                hold off
                xlim([0 1])
                ylim([0 1])
                legend('Chance','No Stimulus','Location','SouthEast')
                ylabel('Probability of Detection')
                xlabel('Probability of False Alarm')
                title('ROC Plot for BIDS nodes')

                %%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%
                %lim = 256/2 - 0.5; %center
                %[X Y] = meshgrid(-1*lim:lim,-1*lim:lim);
                %R = sqrt(X.^2+Y.^2);
                %
                %thickness = 10;
                %num_bins = 500;
                %radius_set = [1, 20, 40, 60, 80, 100];
                %p_set = zeros(2,length(radius_set),num_bins);
                %for i = 1:length(radius_set)
                %    radius = radius_set(i);
                %    mask = R>radius & R<radius+thickness;
                %
                %    %figure
                %    %imagesc(double(mask))
                %    %axis(image)
                %
                %    ring0 = integrated_half0.*mask;
                %    ring1 = integrated_half1.*mask;
                %
                %    [rows0 cols0 counts0] = find(ring0);
                %    [rows1 cols1 counts1] = find(ring1);
                %
                %    if (eq(length(counts0),0) || eq(length(counts1),0))
                %        Pd = 0;
                %        Pf = 0;
                %    else
                %
                %        tot_counts = [counts1;counts0];
                %        [freq_counts, bin_loc] = hist(tot_counts,num_bins);
                %        
                %        h0 = hist(counts0,bin_loc,1);
                %        h1 = hist(counts1,bin_loc,1);
                %
                %        figure
                %        hold on
                %        fid0 = bar(bin_loc,h0,0.8);
                %        fid1 = bar(bin_loc,h1,1);
                %        hold off
                %        set(fid0,'facecolor',[0 0 1])
                %        set(fid0,'edgecolor',[0 0 1])
                %        set(fid1,'facecolor',[1 0 0])
                %        set(fid1,'edgecolor',[1 0 0])
                %
                %        Pd = cumsum(h1);
                %        Pf = cumsum(h0);
                %    end 
                %    p_set(1,i,:) = Pd;
                %    p_set(2,i,:) = Pf;
                %end
                %
                %figure
                %cmap = prism(length(radius_set));  %DOES NOT WORK IN MATLAB
                %hold on
                %plot([0,1],[0,1],'k')
                %for i = 1:length(radius_set)
                %    plot(1 - [0;squeeze(p_set(1,i,:));1],1 - [0;squeeze(p_set(2,i,:));1],'Color',cmap(i,:))
                %end
                %hold off
                %xlim([0 1])
                %ylim([0 1])
                %hleg = legend('Chance','r = 1     ','r = 20   ','r = 40   ','r = 60   ','r = 80   ','r = 100 ','Location','SouthEastOutside');
                %ylabel('Probability of Detection')
                %xlabel('Probability of False Alarm')
                %title('ROC Plot for BIDS nodes')
                %%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%

            end %gt(GRAPH_FLAG,0)

            if eq(MOVIE_FLAG,1)
                system(['ffmpeg -r 12 -f image2 -i ',inst_movie_path,'00_%03d.png -sameq -y ',movie_path,'pvp_instantaneous_movie.mp4']);
                %system(['rm -rf ',inst_movie_path]);
            end%if eq(MOVIE_FLAG,1)

        case 3 % PVP_WGT_FILE_TYPE
            fseek(fid,0,'bof');
            for f=1:numframes
                hdr = readpvpheader(fid,ftell(fid));
                hdr = rmfield(hdr,'additional');
                numextrabytes = hdr.headersize - 80;
                fseek(fid,-numextrabytes,'cof');
                wgt_extra = [fread(fid,3,'int32');fread(fid,2,'float32');fread(fid,1,'int32')];
                numextra = numextrabytes/4 - 6;
                hdr.nxp = wgt_extra(1);
                hdr.nyp = wgt_extra(2);
                hdr.nfp = wgt_extra(3);
                hdr.wMin = wgt_extra(4);
                hdr.wMax = wgt_extra(5);
                hdr.numPatches = wgt_extra(6);
                if numextra > 0
                    hdr.additional = fread(fid,numextra,'int32');
                end
                data{f} = struct('time',hdr.time,'values',[],'nx',[],'ny',[],'offset',[]);
                data{f}.values = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data{f}.nx = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data{f}.ny = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data{f}.offset = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);                
                for arbor=1:hdr.nbands
                    for y=1:hdr.nyprocs
                        for x=1:hdr.nxprocs
                            cellindex = sub2ind([hdr.nxprocs hdr.nyprocs hdr.nbands],x,y,arbor);
                            % octave has trouble with multidim cell arrays
                            patchesperproc = hdr.numPatches/(hdr.nxprocs*hdr.nyprocs);
                            data{f}.values{cellindex} = nan(hdr.nxp,hdr.nyp,hdr.nfp,patchesperproc);
                            for p=1:patchesperproc
                                patchnx = fread(fid,1,'uint16');
                                patchny = fread(fid,1,'uint16');
                                patchoffset = fread(fid,1,'uint32');
                                Z = fread(fid,hdr.nxp*hdr.nyp*hdr.nfp,precision);
                                tempdata = reshape(Z(1:hdr.nfp*hdr.nxp*hdr.nyp),hdr.nfp,hdr.nxp,hdr.nyp);
                                tempdata = permute(tempdata,[2 3 1]);
                                % Need to move shrunken patches
                                data{f}.values{cellindex}(:,:,:,p) = tempdata;
                                data{f}.nx{cellindex} = patchnx;
                                data{f}.ny{cellindex} = patchny;
                                data{f}.offset{cellindex} = patchoffset;
                            end
                        end
                    end
                end
            end
        case 4 % PVP_NONSPIKING_ACT_FILE_TYPE
            for f=1:numframes
                data{f} = struct('time',0,'values',zeros(hdr.nxGlobal,hdr.nyGlobal,hdr.nf));
                data{f}.time = fread(fid,1,'float64');
                for y=1:nyprocs
                    yidx = (1:hdr.ny)+(y-1)*hdr.ny;
                    for x=1:nxprocs
                        xidx = (1:hdr.nx)+(x-1)*hdr.nx;
                        Z = fread(fid,hdr.recordsize,precision);
                        data{f}.values(xidx,yidx,:) = permute(reshape(Z,hdr.nf,hdr.nx,hdr.ny),[2 3 1]);
                    end
                end
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                    end
                end
            end
        case 5 % PVP_KERNEL_FILE_TYPE
            fseek(fid,0,'bof'); % there's a header in every frame, unlike other file types
            % So go back to the beginning and read the header in each frame.
            for f=1:numframes
                hdr = readpvpheader(fid,ftell(fid));
                hdr = rmfield(hdr,'additional');
                numextrabytes = hdr.headersize - 80;
                fseek(fid,-numextrabytes,'cof');
                wgt_extra = [fread(fid,3,'int32');fread(fid,2,'float32');fread(fid,1,'int32')];
                numextra = numextrabytes/4 - 6;
                hdr.nxp = wgt_extra(1);
                hdr.nyp = wgt_extra(2);
                hdr.nfp = wgt_extra(3);
                hdr.wMin = wgt_extra(4);
                hdr.wMax = wgt_extra(5);
                hdr.numPatches = wgt_extra(6);
                if numextra > 0
                    hdr.additional = fread(fid,numextra,'int32');
                end
                data{f} = struct('time',hdr.time,'values',[]);
                data{f}.values = cell(1,1,hdr.nbands);
                for arbor=1:hdr.nbands
                    cellindex = sub2ind([1 1 hdr.nbands],1,1,arbor);
                    % octave has trouble with multidim cell arrays
                    data{f}.values{cellindex} = nan(hdr.nxp,hdr.nyp,hdr.nfp,hdr.numPatches);
                    for p=1:hdr.numPatches
                        fread(fid,1,'uint16'); % patch->nx
                        fread(fid,1,'uint16'); % patch->ny
                        fread(fid,1,'uint32'); % patch->offset
                        Z = fread(fid,hdr.nfp*hdr.nxp*hdr.nyp,precision);
                        tempdata = reshape(Z,hdr.nfp,hdr.nxp,hdr.nyp);
                        tempdata = permute(tempdata,[2 3 1]);
                                                
                        data{f}.values{cellindex}(:,:,:,p) = tempdata;
                    end
                    if hdr.datatype==1 % byte-type.  If float-type, no rescaling took place.
                        data{f}.values{cellindex} = data{f}.values{1}/255*(hdr.wMax-hdr.wMin)+hdr.wMin;
                    elseif hdr.datatype ~= 3
                        error('readpvpfile:baddatatype',...
                            'Weight file type requires hdr.datatype of 1 or 3; received %d',...
                            hdr.datatype);
                    end
                end
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                    end
                end
            end
        otherwise
            assert(0); % This possibility should have been weeded out above
    end
end

fclose(fid);

if ~isempty(errorident)
    error(errorident,errorstring);
end

