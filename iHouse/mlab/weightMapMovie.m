clear all; close all; more off; clc;
system("clear");

%%Hard coded global vars that should be in header stuff later
preScaleX  = 1;
preScaleY  = 1;
nbPre      = 3;
nxGlobal   = 32;
nyGlobal   = 32;

%postScaleX = 4;
%postScaleY = 4;
%nbPost     = 13;

postScaleX = 2;
postScaleY = 2;
nbPost     = 8;

%%%%%%%%%%%%%%%%%%%
%% Path information
%%%%%%%%%%%%%%%%%%%
%rootDir       = '/home/dpaiton';
%workspaceDir  = [rootDir,'/workspace/iHouse'];
rootDir       = '/Users/dpaiton';
workspaceDir  = [rootDir,'/Documents/Work/LANL/workspace/iHouse'];

chkptDir      = [workspaceDir,'/checkpoints/'];

%outputDir     = [chkptDir,'/analysisInter/'];
outputDir     = [chkptDir,'/analysisStellate/'];

cellMovOutDir = [outputDir, 'cell_mov/'];
%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%

checkPoints = dir(chkptDir);
checkPoints = checkPoints(3:end); %Get rid of '.' and '..'

ONweightfile                               = ['RetinaONtoStellate_W.pvp'];
%ONweightfile                               = ['RetinaONtoInter_W.pvp'];
OFFweightfile                              = ['RetinaOFFtoStellate_W.pvp'];
%OFFweightfile                               = ['RetinaOFFtoInter_W.pvp'];

%%Global variables
global CELL_LOC;  CELL_LOC = {[20,23],[12,15]}; %%{[startRow,endRow],[startCol,endCol]} - assumes increment is 1, from post perspective
global FNUM_ALL;  FNUM_ALL = 1;                %1 for all frames, 0 for FNUM_SPEC
global FNUM_SPEC; FNUM_SPEC= {...              %start:int:end frames
   [1:100]...
};

global WEIGHTS_IMAGE_SC; WEIGHTS_IMAGE_SC = -1; %-1 for autoscale
global VIEW_FIGS;        VIEW_FIGS  = 0;        %Display figures in xterm
global WRITE_FIGS;       WRITE_FIGS = 1;        %Write figures to file
global GRAY_SC;          GRAY_SC    = 0;        %Figures in grayscale

NUM_PROCS = 1;%nproc();


assert(length(CELL_LOC)==2);
assert(length(CELL_LOC{1})==length(CELL_LOC{2}) && length(CELL_LOC{1}==2));
assert(CELL_LOC{1}(1)<CELL_LOC{1}(2));
assert(CELL_LOC{2}(1)<CELL_LOC{2}(2));

cellMatX       = [CELL_LOC{1}(1):CELL_LOC{1}(2)];
cellMatY       = [CELL_LOC{2}(1):CELL_LOC{2}(2)];
numCellPointsX = length(cellMatX);
numCellPointsY = length(cellMatY);

%%%%%%%%%%%%%%%%%%%%
%%% MAKE DIRECTORIES
%%%%%%%%%%%%%%%%%%%%
chkptDirExist = exist(chkptDir, 'dir');
assert(chkptDirExist == 7);
if (exist(outputDir, 'dir') ~= 7)
   mkdir(outputDir);
end
if (exist(cellMovOutDir, 'dir') ~= 7)
   mkdir(cellMovOutDir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SORT CHECKPOINT ARRAY
%%%%%%%%%%%%%%%%%%%%%%%%%
for checkPointIdx = 1:length({checkPoints.name}) %Loop through each sub-dir of checkpoints dir
    checkPoints(checkPointIdx).num = str2num([checkPoints(checkPointIdx).name](11:end));
end

checkPointsC = struct2cell(checkPoints); 
checkPointsC = checkPointsC';
numField = size(checkPointsC,2);
[sorted, sortIdx] = sort([checkPointsC{:,numField}]);
checkPoints = checkPoints(sortIdx);
clear checkPointsC;

%%%%%%%%%%%%%%%%%%%%%%%
%%% GET PVP FILE DATA
%%%%%%%%%%%%%%%%%%%%%%%
numFun = 0;
for checkPointIdx = 1:length({checkPoints.name}) %Loop through each sub-dir of checkpoints dir
    %ON
    numFun += 1;
    fileNames{numFun} = [chkptDir,checkPoints(checkPointIdx).name,'/',ONweightfile];
    output_paths{numFun} = '';  %No output path needed for weights
    makeMov{numFun} = 0;
    %OFF
    numFun += 1;
    fileNames{numFun} = [chkptDir,checkPoints(checkPointIdx).name,'/',OFFweightfile];
    output_paths{numFun} = '';  %No output path needed for weights
    makeMov{numFun} = 0;
end

disp('weightMapMovie: Reading pvp files')
fflush(1);
if NUM_PROCS == 1
   [data hdr] = cellfun(@readpvpfile, fileNames, output_paths, makeMov, 'UniformOutput', 0);
else
   [data hdr] = parcellfun(NUM_PROCS, @readpvpfile, fileNames, output_paths, makeMov, 'UniformOutput', 0);
end

numFun = 0;
for checkPointIdx = 1:length({checkPoints.name}) %Loop through each sub-dir of checkpoints dir
    %ON
    numFun += 1;
    %%<HARD CODED STUFF>%%
    hdr{numFun}.preScaleX   = preScaleX;
    hdr{numFun}.preScaleY   = preScaleY;
    hdr{numFun}.postScaleX  = postScaleX;
    hdr{numFun}.postScaleY  = postScaleY;
    hdr{numFun}.nbPost      = nbPost;
    hdr{numFun}.nbPre       = hdr{numFun}.nb;
    hdr{numFun}.nxGlobal    = nxGlobal;
    hdr{numFun}.nyGlobal    = nyGlobal;
    %OFF
    numFun += 1;
    hdr{numFun}.preScaleX  = preScaleX;
    hdr{numFun}.preScaleY  = preScaleY;
    hdr{numFun}.postScaleX = postScaleX;
    hdr{numFun}.postScaleY = postScaleY;
    hdr{numFun}.nbPost     = nbPost;
    hdr{numFun}.nbPre      = hdr{numFun}.nb;
    hdr{numFun}.nxGlobal   = nxGlobal;
    hdr{numFun}.nyGlobal   = nyGlobal;
    %%<\HARD CODED STUFF>%%
end

if NUM_PROCS == 1
   [postData postHdr] = cellfun(@preWeightsToPost, data, hdr, 'UniformOutput', 0);
else
   [postData postHdr] = parcellfun(NUM_PROCS, @preWeightsToPost, data, hdr, 'UniformOutput', 0);
end

numFun = 0;
for checkPointIdx = 1:length({checkPoints.name}) %Loop through each sub-dir of checkpoints dir
    %ON
    numFun += 1;
    weightDataOn = postData{numFun};
    weightHdrOn  = postHdr{numFun};

    %OFF
    numFun += 1;
    weightDataOff = postData{numFun};
    weightHdrOff  = postHdr{numFun};

    global preScaleX;  preScaleX  = weightHdrOn.preScaleX;
    global preScaleY;  preScaleY  = weightHdrOn.preScaleY;
    global postScaleX; postScaleX = weightHdrOn.postScaleX;
    global postScaleY; postScaleY = weightHdrOn.postScaleY;
    global sizeX;      sizeX      = weightHdrOn.nx;
    global sizeY;      sizeY      = weightHdrOn.ny;

    %%%%%%%%%%%%%%%%%%%%%%%
    %%% ERROR CHECK
    %%%%%%%%%%%%%%%%%%%%%%%
    assert(weightHdrOn.nbands     == weightHdrOff.nbands);     %Num arbors
    assert(weightHdrOn.nfp        == weightHdrOff.nfp);        %Num features pre
    assert(weightHdrOn.nxprocs    == weightHdrOff.nxprocs);    %Num processors X
    assert(weightHdrOn.nyprocs    == weightHdrOff.nyprocs);    %Num processors Y
    assert(weightHdrOn.nxp        == weightHdrOff.nxp);        %nxp (post-weight patch size X)
    assert(weightHdrOn.nyp        == weightHdrOff.nyp);        %nyp (post-weight patch size Y)
    assert(weightHdrOn.nx         == weightHdrOff.nx);         %MPI partition size X
    assert(weightHdrOn.ny         == weightHdrOff.ny);         %MPI partition size Y
    assert(weightHdrOn.nbPre      == weightHdrOff.nbPre);      %Margin width Pre
    assert(weightHdrOn.nbPost     == weightHdrOff.nbPost);     %Margin width Post
    assert(weightHdrOn.nxGlobal   == weightHdrOff.nxGlobal);   %HyPerCol size X
    assert(weightHdrOn.nyGlobal   == weightHdrOff.nyGlobal);   %HyPerCol size Y
    assert(weightHdrOn.preScaleX  == weightHdrOff.preScaleX);  %pre synaptic layer scale relative to HyPerCol X
    assert(weightHdrOn.preScaleY  == weightHdrOff.preScaleY);  %pre synaptic layer scale relative to HyPerCol Y
    assert(weightHdrOn.postScaleX == weightHdrOff.postScaleX); %post synaptic layer scale relative to HyPerCol X
    assert(weightHdrOn.postScaleY == weightHdrOff.postScaleY); %post synaptic layer scale relative to HyPerCol Y

    assert(CELL_LOC{1}(1) >= 0);
    assert(CELL_LOC{1}(2) <= weightHdrOn.postScaleY*weightHdrOn.nyGlobal-1);
    assert(CELL_LOC{2}(1) >= 0);
    assert(CELL_LOC{2}(2) <= weightHdrOn.postScaleX*weightHdrOn.nxGlobal-1);

    %%%%%%%%%%%%%%%%%%%%%%%
    %%% COMPUTE CELL MAP
    %%%%%%%%%%%%%%%%%%%%%%%
    numTimeSteps = length(weightDataOff);
    assert(numTimeSteps == length(weightDataOn));
    numArbors = weightHdrOn.nbands;

    outCell = cell(numArbors,numTimeSteps,numCellPointsY,numCellPointsX);
    for arborIdx = 1:numArbors
        for time = 1:numTimeSteps %Not sure if this will work if there are multiple time-steps
            for postIdxX = 1:numCellPointsX
                for postIdxY = 1:numCellPointsY
                    outCell{arborIdx,time,postIdxY,postIdxX} = cellMap(weightDataOn{time}.values, weightDataOff{time}.values, arborIdx, [cellMatX(postIdxX),cellMatY(postIdxY)]);
                end
            end
            outMat = reshape([outCell{arborIdx,time,:,:}],weightHdrOn.nyp/weightHdrOn.postScaleY*weightHdrOn.ny,weightHdrOn.nxp/weightHdrOn.postScaleX*weightHdrOn.nx);
            printImage(outMat, checkPointIdx, arborIdx, cellMovOutDir, WEIGHTS_IMAGE_SC, ['Cell\_Map\_X\_[',num2str(min(cellMatX)),':',num2str(max(cellMatY)),']\_Y\_[',num2str(min(cellMatY)),':',num2str(max(cellMatY)),']']) 
        end
    end
end
