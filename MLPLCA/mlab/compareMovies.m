clear all;
close all;
workspace_path = "/home/slundquist/workspace";
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);

actualFilepath = '/nh/compneuro/Data/CIFAR/LCA/data_batch_all15/a2_S1.pvp';
readFilepath = '/nh/compneuro/Data/MLPLCA/LCA/movie_only/a0_InputS1.pvp';

[actualData, actualHdr] = readpvpfile(actualFilepath, 0, 20, 0);

%These are sparse
actualVals = actualData{3}.values;

%Make actualVals full
N = actualHdr.nx * actualHdr.ny * actualHdr.nf;
active_ndx = actualVals(:, 1);
active_vals = actualVals(:, 2);
actualMat = full(sparse(active_ndx+1, 1, active_vals, N, 1, N));
actualMat = reshape(actualMat, actualHdr.nf, actualHdr.nx, actualHdr.ny);
actualMat = permute(actualMat, [2 3 1]);

actualOut = sum(actualMat, 3);

%These are full
[readData, readHdr] = readpvpfile(readFilepath, 0, 20, 0);
readVals = readData{1}.values;

readOut = sum(readVals, 3);

keyboard
