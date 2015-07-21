numsamples = 10000;
imsize = [256 256];
patchsize = [8 8];
DBdir = '~/Workspace/NMC/CNS/AnimalDB/Targets/original/';
ISdir = 'imagespace/';
assert(exist(DBdir,'dir'));
if ~exist(ISdir,'dir')
    mkdir(ISdir);
end
assert(exist(ISdir,'dir'));
DB = dir([DBdir '*.jpg']);
numimages = numel(DB);
assert(numimages>0);
lastpatch = imsize-patchsize+1;
imrnd = ceil(numimages*rand(numsamples,1));
xrnd = floor(lastpatch(2)*rand(numsamples,1));
yrnd = floor(lastpatch(1)*rand(numsamples,1));

fieldsize = length(num2str(numsamples-1));
fid = fopen('filenames.txt','w');
assert(fid>0);
for k=1:numsamples
    filepath = [DBdir DB(imrnd(k)).name];
    A = imread(filepath);
    patch = A(yrnd(k)+(1:patchsize(1)),xrnd(k)+(1:patchsize(1)),:);
    patch = mean(double(patch),3)/255;
    outfname = sprintf('%spatch%05d.png',ISdir,k-1);
    imwrite(patch,outfname);
    fprintf(fid,'./input/unsupervised8x8natural/%s\n',outfname);
end
fclose(fid);
