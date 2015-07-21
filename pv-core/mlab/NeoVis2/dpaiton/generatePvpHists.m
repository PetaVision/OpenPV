%%%%%%%%%%%%%%%%%%%%%%
%% getPvpHists.m
%%   Dylan Paiton
%%   Los Alamos National Laboratory
%%
%% Inputs:
%%   pvp_filename - exact path/name to the .pvp PetaVision file that you wish to analyze
%%   csv_filename - CSV file for analysis - This should specify the bounding boxes within the given frame for analysis
%%   (csv_object) - Object in CSV file that you want to analyze. If no object is specified, it will search for all objects.
%% 
%% Outputs:
%%   numEdges - MxN array where M is the number of items in the CSV file and N is the number of features in the pvp file
%%   times    - Array containing the time step for each frame
%%
%%   NOTE: CSV file should be in the format:
%%         Frame,BoundingBox_X1,BoundingBox_Y1,BoundingBox_X2,BoundingBox_Y2,BoundingBox_X3,BoundingBox_Y3,BoundingBox_X4,BoundingBox_Y4,ObjectType,Occlusion,Ambiguous,Confidence,SiteInfo,Version
%%
%%%%%%%%%%%%%%%%%%%%%%

function [numEdges, times] = generatePvpHists(pvp_filename,csv_filename,csv_object)

    %pvp_filename = '~/Documents/Work/LANL/NeoVision/repo/neovision-programs-scripts/Matlab_Clustering/PetaVision_Activity/heli/challenge/026/Car5/canny2/a7.pvp';
    %csv_filename = '~/Documents/MATLAB/vlfeat-0.9.16/apps/data/NeoVis/training/Tiles/Car/targets.csv';
    %csv_object = 'Car';


    %%Variable input
    if gt(nargin,3) || lt(nargin,2)
        error('generatePvpHist: Incorrect number of input arguments.')
    end

    %%Input CSV file
    if exist('csv_object','var')
        [csvBoxes, ~, numCSVItems numCSVFrames] = getCSVBoxes(csv_filename,csv_object);
    else
        [csvBoxes, ~, numCSVItems numCSVFrames] = getCSVBoxes(csv_filename);
    end
    frameIndices = [csvBoxes(:).Frame];
    
    %%Input PVP file
    if exist(pvp_filename,'file')
       [data,hdr] = readpvpfile(pvp_filename);
    else
        error(['generatePvpHists: ~exist(pvp_filename,"file") in pvp file: ', pvp_filename]);
    end

    numPvFrames = length(data);
    N           = hdr.nx * hdr.ny * hdr.nf;

    if exist('data','var') && exist('hdr','var')
        times    = zeros(numPvFrames,1); 
        numEdges = zeros(numCSVItems,hdr.nf);
        csvIdx = 1;
        for frame = 1 : 1 : numPvFrames %%Assumes PVP file has contiguous frames
            times(frame) = squeeze(data{frame}.time);
            active_ndx   = squeeze(data{frame}.values);
            vec_mat      = full(sparse(active_ndx+1,1,1,N,1,N)); %%Column vector. PetaVision increments in order: nf, nx, ny
            rs_mat       = reshape(vec_mat,hdr.nf,hdr.nx,hdr.ny);
            full_mat     = permute(rs_mat,[3 2 1]); %%Matrix is now [ny, nx, nf]
            csvIdxList   = find(frameIndices==frame-1);
            for csvFrameIdx = csvIdxList
                csvMask = createCSVMask(csvBoxes(csvFrameIdx),hdr.ny,hdr.nx,'box',threshold=0,sub_box=true);
                for fIdx = 1:hdr.nf
                    numEdges(csvFrameIdx,fIdx) = length(find(logical(csvMask*full_mat)));
                end
            end
        end
    end
end
