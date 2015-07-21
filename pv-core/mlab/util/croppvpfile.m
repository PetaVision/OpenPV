%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% Crop PVP activity file to smaller dimensions
%% 
%%    Dylan Paiton
%%    Jan 5, 2014
%%
%% Input: PVP file, offsetX, offsetY, nX, nY, (progressPeriod)
%%    PVP file - File to be cropped
%%    offsetX  - How much to stride in the X direction
%%    offsetY  - How much to stride in the Y direction
%%    nX       - X size of the cropped portion in number of neurons
%%    nY       - Y size of the cropped portion in number of neurons
%%
%%    (progressPeriod) - Optional, prints message to screen every progressPeriod frames 
%%
%% Output: Cropped PVP activity file
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%function croppvpfile(filename, offsetX, offsetY, nX, nY, progressPeriod)
%
%   if nargin < 5
%      error('croppvpfile requires inputs: filename, offsetX, offsetY, nX, nY');
%   end%if
%
%   if nargin < 6 || ~exist('progressPeriod','var') || isempty(progressPeriod)
%      progressPeriod = -1;
%   end%if

   close all; clear all;

   workspace_path = '/Users/dpaiton/Documents/workspace';
   pvp_file1 = [workspace_path,filesep,'PetaVision/mlab/PhysioMap/data/PV_Data/a5_V1_Clone_lat.pvp'];
   pvp_file2 = [workspace_path,filesep,'PetaVision/mlab/PhysioMap/data/PV_Data/a5_V1_Clone_nolat.pvp'];

   addpath([workspace_path,filesep,'PetaVision/mlab/util']);

   filename = pvp_file2;
   nX       = 16;
   nY       = 16;
   offsetX  = ((480/4) / 2) - nX/2;
   offsetY  = ((480/4) / 2) - nY/2;
   

   fid = fopen(filename);

   hdr = readpvpheader(fid);
   newData = cell(hdr.nbands,1);

   for writeTime = 1:1:hdr.nbands
      if exist('progressPeriod','var')
          if ~mod(writeTime,progressPeriod)
              fprintf(1,'File %s: frame %d of %d\n',filename, writeTime, hdr.nbands);
              if exist('fflush')
                 fflush(1);
              end%if
          end%if
      end%if

      [oldData, hdr] = readpvpfile(filename,0,writeTime,writeTime);

      newData{writeTime}.time = oldData{1}.time;
      newData{writeTime}.values = oldData{1}.values(offsetX:offsetX+nX-1,offsetY:offsetY+nY-1,:);

   end%for

   filename_split = strsplit(filename,'.');

   newFilename = [filename_split{1},'_crp_',num2str(nX),'x',num2str(nY),'y.',filename_split{2}];

   writepvpactivityfile(newFilename,newData);

%end%function
