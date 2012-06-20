function result = makemnistplusposdata(trainnum, testnum, imagex, imagey, toplayerx, toplayery, seed)
% result = makemnistplusposdata(trainnum, testnum, imagex, imagey, toplayerx, toplayery, seed)
%
% trainnum is the number of training samples
% testnum is the number of testingsamples
% imagex,imagey is the size of the generated images
% toplayerx,toplayery is100 the size of the top layer of the hierarchy
%     (it will actually be toplayerx-by-toplayery-by-10)
% seed is the seed for the random number generator (optional)

if nargin==0
    help(mfilename);
end

if exist('seed','var'), seedrng(seed); end
if ~exist('train','dir'), mkdir train; end
if ~exist('test','dir'), mkdir test; end

trainimages = readgzdata('train-images-idx3-ubyte.gz');
result.trainimages = readidx3data(trainimages);
fprintf(1,'Training images have been read\n');

trainlabels = readgzdata('train-labels-idx1-ubyte.gz');
result.trainlabels = readidx1data(trainlabels);
fprintf(1,'Training labels have been read\n');

[m,n,p] = size(result.trainimages);
assert(m<=imagex && n<=imagey);
result.trainsampleindices = randi([1 p],1,trainnum);
fid = fopen('train/trainsampleindices.txt','w');
fprintf(fid,'%d\n',result.trainsampleindices);
fclose(fid);
result.trainsamplelabels = result.trainlabels(result.trainsampleindices);
fid = fopen('train/trainsamplelabels.txt','w');
fprintf(fid,'%d\n',result.trainsamplelabels);
fclose(fid);
result.trainsamplexpos = randi([1 imagex-m+1],1,trainnum);
fid = fopen('train/trainsamplexpos.txt','w');
fprintf(fid,'%d\n',result.trainsamplexpos);
fclose(fid);
result.trainsampleypos = randi([1 imagey-n+1],1,trainnum);
fid = fopen('train/trainsampleypos.txt','w');
fprintf(fid,'%d\n',result.trainsampleypos);
fclose(fid);
result.trainsample = zeros(imagex, imagey, 1, trainnum);
result.toplayer = zeros(toplayerx, toplayery, 10, trainnum);
[toplayerX,toplayerY] = ndgrid(1:toplayerx,1:toplayery);

imagefid = fopen('train/trainsampleimagelist.txt','w');
trainingfid = fopen('train/traintoplayerlist.txt','w');
for k=1:trainnum
    x = result.trainsamplexpos(k);
    y = result.trainsampleypos(k);
    result.trainsample(x:x+m-1,y:y+n-1,1,k) = result.trainimages(:,:,result.trainsampleindices(k));
    imagefilename = sprintf('train/trainsampleimage%05d.png',k);
    imwrite(result.trainsample(:,:,1,k),imagefilename);
    fprintf(imagefid,'%s\n',imagefilename);
    scalefactorx = toplayerx/imagex;
    scalefactory = toplayery/imagey;
    toporiginx = 0.5+scalefactorx*(x+(m-1)/2-1);
    toporiginy = 0.5+scalefactory*(y+(n-1)/2-1);
    result.toplayer(:,:,result.trainsamplelabels(k)+1,k) = ...
        exp(-((toplayerX-toporiginx).^2+(toplayerY-toporiginy).^2)/2)/sqrt(2*pi);
    trainingfilename = sprintf('train/traintoplayer%05d.pvp',k);
    writepvpfile(result.toplayer(:,:,:,k),trainingfilename);
    fprintf(trainingfid,'%s\n',trainingfilename);
    if mod(k,1000)==0
        fprintf(1,'Sample image %d of %d has been generated\n',k,trainnum);
    end
end
fclose(imagefid);
fclose(trainingfid);

testimages = readgzdata('t10k-images-idx3-ubyte.gz');
result.testimages = readidx3data(testimages);
fprintf(1,'Test images have been read\n');

testlabels = readgzdata('t10k-labels-idx1-ubyte.gz');
result.testlabels = readidx1data(testlabels);
fprintf(1,'Test labels have been read\n');

[m,n,p] = size(result.testimages);
assert(m<=imagex && n<=imagey);
result.testsampleindices = randi([1 p],1,testnum);
fid = fopen('test/testsampleindices.txt','w');
fprintf(fid,'%d\n',result.testsampleindices);
fclose(fid);
result.testsamplelabels = result.testlabels(result.testsampleindices);
fid = fopen('test/testsamplelabels.txt','w');
fprintf(fid,'%d\n',result.testsamplelabels);
fclose(fid);
result.testsamplexpos = randi([1 imagex-m+1],1,testnum);
fid = fopen('test/testsamplexpos.txt','w');
fprintf(fid,'%d\n',result.testsamplexpos);
fclose(fid);
result.testsampleypos = randi([1 imagey-n+1],1,testnum);
fid = fopen('test/testsampleypos.txt','w');
fprintf(fid,'%d\n',result.testsampleypos);
fclose(fid);
result.testsample = zeros(imagex, imagey, 1, testnum);
imagefid = fopen('test/testsampleimagelist.txt','w');
for k=1:testnum
    x = result.testsamplexpos(k);
    y = result.testsampleypos(k);
    result.testsample(x:x+m-1,y:y+n-1,1,k) = result.testimages(:,:,result.testsampleindices(k));
    filename = sprintf('test/testsampleimage%05d.png',k);
    imwrite(result.testsample(:,:,1,k),filename);
    if mod(k,1000)==0
        fprintf(1,'Sample image %d of %d has been generated\n',k,testnum);
    end
end
fclose(imagefid);
