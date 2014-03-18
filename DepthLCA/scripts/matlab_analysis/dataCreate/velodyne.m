%function run_demoVelodyne (base_dir,calib_dir)
% KITTI RAW DATA DEVELOPMENT KIT
% 
% Demonstrates projection of the velodyne points into the image plane
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)
% calib_dir ... absolute path to directory that contains calibration files

% clear and close everything
function velodyne(base_dir_param, out_dir_param, numFrame_param)
    %clear all; close all; dbstop error; clc;

    % options (modify this to select your sequence)
    %global base_dir; base_dir  = '/nh/compneuro/Data/KITTI/2011_09_26/2011_09_26_drive_0011_sync';
    global base_dir; base_dir  = base_dir_param;
    global calib_dir; calib_dir = '/nh/compneuro/Data/KITTI/2011_09_26';
    %global out_dir; out_dir = '/nh/compneuro/Data/Depth/depth_data_11';
    global out_dir; out_dir = out_dir_param;
    global pixBorder; pixBorder = 50; %Pixel border for better interpolation
    global vertCutOffset; vertCutOffset = 30;
    global horLineWidth; horLineWidth = 200;
    global maxThreshPercent; maxThreshPercent = 0; %Remove top 10% of data and set to max distance
    verboseLevel = 2; %verbose level for parcellfun

    %numFrame     = 232; % 0-based index
    numFrame = numFrame_param;
    %Multithreading
    %numproc = 1;
    %numproc = nproc()
    %numproc =15; 

    if ~exist(out_dir, 'dir')
      mkdir(out_dir);
    end

    %Divide numFrames into numproc - 1
    frames = 0:numFrame;
    generateInput(frames);

    %frameCell = cell(numproc, 1);
    %%Perfect split
    %if mod(numFrame+1, numproc) == 0
    %  framesPerProc = floor((numFrame+1)/(numproc));
    %  for ci=1:numproc
    %    frameCell(ci)=frames((ci-1)*framesPerProc+1:ci*framesPerProc);
    %  end
    %else
    %  framesPerProc = floor((numFrame+1)/(numproc-1));
    %  lastFramesProc = mod(numFrame+1, numproc-1);
    %  for ci=1:numproc-1
    %    frameCell(ci)=frames((ci-1)*framesPerProc+1:ci*framesPerProc);
    %  end
    %  frameCell(numproc)=frames(end-lastFramesProc+1:end);
    %end

    %Run parcellfun
    %if(numproc == 1)
    %  cellfun(@generateInput, frameCell);
    %else
    %  parcellfun(numproc, @generateInput, frameCell, "VerboseLevel", verboseLevel);
    %end
end

function generateInput(frames)
  global base_dir;
  global calib_dir;
  global pixBorder;
  global vertCutOffset;
  global out_dir;
  global maxThreshPercent;
  %For frames
  parfor i=1:size(frames,2)
    frame = frames(i);
    %For the two cameras
    for cam=2:3
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


      %Check that both files exist
      inFile = sprintf('%s/image_%02d/data/%010d.png',base_dir,cam,frame);
      velFile = sprintf('%s/velodyne_points/data/%010d.bin',base_dir,frame);
      disp(inFile)
      if (exist(inFile) == 0) | (exist(velFile) == 0)
         disp([inFile, ' does not exist, skipping']);
         continue;
      end%if

      % load and display image
      img = imread(inFile);

      % load velodyne points
      fid = fopen(velFile,'rb');
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

      %Add new datapoints at the middle of the image to prevent weird interpolations at the center (infinity) 
      %yPoint = round(size(img, 1) / 2)-vertCutOffset;
      %xCenter = round(size(img, 2) / 2);
      %%Use maximum depth value that was in the dataset
      %zVal = max(Z);
      %%Make the column vector of new points
      %offsets = -round(horLineWidth/2):round(horLineWidth/2)';
      %offsets = offsets';
      %addX = xCenter + offsets;
      %addY = repmat(yPoint, length(offsets), 1);
      %addZ = repmat(zVal, length(offsets), 1);
      %%Add the points
      %X = [X;addX];
      %Y = [Y;addY];
      %Z = [Z;addZ];
      
      %plot3(X,Y,Z,'.')

      %Interpololate data
      [XI, YI] = meshgrid(1-pixBorder:1:size(img, 2)+pixBorder, 1-pixBorder:1:size(img,1)+pixBorder);
      ZI = griddata(X, Y, Z, XI, YI, 'linear');

      %Make sure image is the same size
      %Crop border, assuming 1 indexed
      ZI = ZI(pixBorder+1:end-pixBorder, pixBorder+1:end-pixBorder);

      %Cut image in half vertically
      ZI = ZI(round(size(img, 1)/2) - vertCutOffset:end, :);
      newImg = img(round(size(img, 1)/2) - vertCutOffset:end, :, :);
      %newRImg = rImg(round(size(lImg, 1)/2) - vertCutOffset:end, :);

      %Set ZI to log scale
      nZI = log(ZI/6.5)/log(8.5);

      %Normalize ZI
      %nZI = (ZI - mean(ZI(:))).*(1./std(ZI(:)));
      %nZI = (ZI - min(ZI(:)))/(max(ZI(:)) - min(ZI(:)));

      %Set all NaN's to black
      nZI(isnan(nZI)) = 0;

      %Bin up last few values to be 1 to prevent weird interpolations at infinity
      threshval = 1-maxThreshPercent;
      nZI(find(nZI >= threshval)) = 1;

      keyboard

      %imwrite is trunkatin the data to be between 0 and 1
      imwrite(nZI, sprintf('%s/%04d.png', depthOutDir, frame));
      imwrite(newImg, sprintf('%s/%04d.png', imgOutDir, frame));
    end %end cam loop
  end %end frames loop
end %end function

