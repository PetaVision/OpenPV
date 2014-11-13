clear all;
close all;
workspace_path = "/home/slundquist/workspace";
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);

readFilepath = '/nh/compneuro/Data/MLPLCA/LCA/tile_test/a1_InputV1.pvp';
actualFilepath = '/nh/compneuro/Data/repo/neovision-programs-petavision/LCA/Heli/TrainingPlusFormative/heli_V1/a2_V1.pvp'

[actualData, actualHdr] = readpvpfile(actualFilepath, 0, 20, 0);
[readData, readHdr] = readpvpfile(readFilepath, 0, 20, 0);

actualVals = actualData{3}.values;
readVals = readData{3}.values;

%Make actualVals full
N = actualHdr.nx * actualHdr.ny * actualHdr.nf;
active_ndx = actualVals(:, 1);
active_vals = actualVals(:, 2);
actualMat = full(sparse(active_ndx+1, 1, active_vals, N, 1, N));
actualMat = reshape(actualMat, actualHdr.nf, actualHdr.nx, actualHdr.ny);
actualMat = permute(actualMat, [2 3 1]);

actualOut = sum(actualMat, 3);

N = readHdr.nx * readHdr.ny * readHdr.nf;
active_ndx = readVals(:, 1);
active_vals = readVals(:, 2);
readMat = full(sparse(active_ndx+1, 1, active_vals, N, 1, N));
readMat = reshape(readMat, readHdr.nf, readHdr.nx, readHdr.ny);
readMat = permute(readMat, [2 3 1]);

readOut = sum(readMat, 3);
keyboard
