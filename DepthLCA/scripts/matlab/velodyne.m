%function run_demoVelodyne (base_dir,calib_dir)
% KITTI RAW DATA DEVELOPMENT KIT
% 
% Demonstrates projection of the velodyne points into the image plane
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)
% calib_dir ... absolute path to directory that contains calibration files

% clear and close everything
clear all; close all; dbstop error; clc;

addpath(genpath('devkit/'));

disp('======= KITTI Depth Generator =======');

% options (modify this to select your sequence)
global base_dir; base_dir  = '/nh/compneuro/Data/KITTI/2011_09_26/2011_09_26_drive_0005_sync';
global calib_dir; calib_dir = '/nh/compneuro/Data/KITTI/2011_09_26';
global out_dir; out_dir = '/nh/compneuro/Data/Depth/depth_data_5';
global pixBorder; pixBorder = 50; %Pixel border for better interpolation
global vertCutOffset; vertCutOffset = 30;
verboseLevel = 1; %verbose level for parcellfun

numFrame     = 153; % 0-based index
%Multithreading
%numproc = 1;
%numproc = nproc()
numproc = 30

if ~exist(out_dir, 'dir')
  mkdir(out_dir);
end

function generateInput(frames)
  global base_dir;
  global calib_dir;
  global pixBorder;
  global vertCutOffset;
  global out_dir;
  %For frames
  for i=1:size(frames,2)
    frame = frames(i);
    %For the two cameras
    for cam=0:1
      depthOutDir = sprintf('%s/depth_%02d', out_dir, cam);
      imgOutDir = sprintf('%s/image_%02d', out_dir, cam);
      if ~exist(depthOutDir, 'dir')
        mkdir(depthOutDir);
      end
      if ~exist(imgOutDir, 'dir')
        mkdir(imgOutDir);
      end

      % load calibration
      calib = loadCalibrationCamToCam(fullfile(calib_dir,'calib_cam_to_cam.txt'));
      Tr_velo_to_cam = loadCalibrationRigid(fullfile(calib_dir,'calib_velo_to_cam.txt'));

      % compute projection matrix velodyne->image plane
      R_cam_to_rect = eye(4);
      R_cam_to_rect(1:3,1:3) = calib.R_rect{1};
      P_velo_to_img = calib.P_rect{cam+1}*R_cam_to_rect*Tr_velo_to_cam;

      % load and display image
      img = imread(sprintf('%s/image_%02d/data/%010d.png',base_dir,cam,frame));

      % load velodyne points
      fid = fopen(sprintf('%s/velodyne_points/data/%010d.bin',base_dir,frame),'rb');
      velo = fread(fid,[4 inf],'single')';
      %velo = velo(1:5:end,:); % remove every 5th point for display speed
      fclose(fid);

      % remove all points behind image plane (approximation
      idx = velo(:,1)<5;
      velo(idx,:) = [];

      % project to image plane (exclude luminance)
      velo_img = project(velo(:,1:3),P_velo_to_img);

      % plot points
      cols = jet;
      X = velo_img(:,1);
      Y = velo_img(:,2);
      Z = velo(:,1);

      %Remove all points not on the image, added border, assuming 1 indexed
      idxs = X>=(1-pixBorder) & X<=(size(img,2) + pixBorder) & Y>=(1-pixBorder) & Y<=(size(img, 1)+pixBorder);
      X = X(idxs);
      Y = Y(idxs);
      Z = Z(idxs);

      %plot3(X,Y,Z,'.')

      %Interpololate data
      [XI, YI] = meshgrid(1-pixBorder:1:size(img, 2)+pixBorder, 1-pixBorder:1:size(img,1)+pixBorder);
      ZI = griddata(X, Y, Z, XI, YI, 'linear');

      %Make sure image is the same size
      %Crop border, assuming 1 indexed
      ZI = ZI(pixBorder+1:end-pixBorder, pixBorder+1:end-pixBorder);

      %Cut image in half vertically
      ZI = ZI(round(size(img, 1)/2) - vertCutOffset:end, :);
      newImg = img(round(size(img, 1)/2) - vertCutOffset:end, :);
      %newRImg = rImg(round(size(lImg, 1)/2) - vertCutOffset:end, :);

      %Normalize ZI
      nZI = (ZI - min(ZI(:)))/(max(ZI(:)) - min(ZI(:)));

      imwrite(nZI, sprintf('%s/depth_%03d.png', depthOutDir, frame));
      imwrite(newImg, sprintf('%s/img_%03d.png', imgOutDir, frame));
    end %end cam loop
  end %end frames loop
end %end function

%Divide numFrames into numproc - 1
frames = 0:numFrame;
frameCell = cell(numproc, 1);
%Perfect split
if mod(numFrame+1, numproc) == 0
  framesPerProc = floor((numFrame+1)/(numproc));
  for ci=1:numproc
    frameCell(ci)=frames((ci-1)*framesPerProc+1:ci*framesPerProc);
  end
else
  framesPerProc = floor((numFrame+1)/(numproc-1));
  lastFramesProc = mod(numFrame+1, numproc-1);
  for ci=1:numproc-1
    frameCell(ci)=frames((ci-1)*framesPerProc+1:ci*framesPerProc);
  end
  frameCell(numproc)=frames(end-lastFramesProc+1:end);
end

%Run parcellfun
if(numproc == 1)
  cellfun(@generateInput, frameCell);
else
  parcellfun(numproc, @generateInput, frameCell, "VerboseLevel", verboseLevel);
end
