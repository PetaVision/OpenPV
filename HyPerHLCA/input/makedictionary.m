% A very crude way of generating a dictionary
% Reads the .png files in the current directory (assumes they're grayscale)
% Selects patches from the images and chooses a dictionary from the patches.
% The patchsize, dictionary size and number of samples are specified in the
% constants below.

% define constants
numsamples = 10000;
dictionarysize = 1000;
patchsizex = 16;
patchsizey = 16;
imagedir = '~/Workspace/NMC/CNS/AnimalDB/Distractors/diffgauss_on';

fprintf(1,'reading images\n'); fflush(1);
imglist = dir([imagedir '/*.png']);
numimages = numel(imglist);
A = imread([imagedir '/' imglist(1).name]);
[imagey, imagex] = size(A);
clear A;
images = zeros(imagey,imagex,numimages);
for k=1:numimages
    images(:,:,k) = double(imread([imagedir '/' imglist(k).name]))/255;
end

fprintf(1,'selecting patches at random\n'); fflush(1);
patches = zeros(patchsizey, patchsizex, numsamples);
for k=1:numsamples
    r = randi(imagey-patchsizey+1, imagex-patchsizex+1, numimages);
    patches(:,:,k) = images(r(1)+(0:patchsizex-1), r(2)+(0:patchsizey-1), r(3));
    patches(:,:,k) = patches(:,:,k)/sqrt(sum( (patches(:,:,k)(:)).^2 ));
end

fprintf(1, 'creating dictionary\n'); fflush(1);
dictionary = zeros(patchsizey, patchsizex, dictionarysize);
% select first dictionary element at random
dictionary(:,:,1) = patches(:,:,randi(numsamples));
% Iteratively select the rest of dictionary.
% dictionary element k is chosen to be as far as possible from dictionary elements 1 through k-1
for k=2:dictionarysize
    Q = reshape(patches, patchsizex*patchsizey, numsamples)'*reshape(dictionary(:,:,1:k-1),patchsizex*patchsizey,k-1);
    Qmax = max(Q,[],2);
    c = find(Qmax==min(Qmax),1);
    dictionary(:,:,k) = patches(:,:,c);
    fprintf(1,'%d of %d\n', k, dictionarysize); fflush(1);
end

dictionarymatrix = reshape(dictionary, patchsizex*patchsizey, dictionarysize);
innerproductmatrix = dictionarymatrix'*dictionarymatrix;