close all
clear all
clc

firstpvp=1;
showframe=36; %25-14; %36;
dnoff=3;
cbar0ff=12;
compl0ff=13;
mt0ff=14;
shrunkframesize=128;

%actpath='/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a';
%actpath='/Users/kpeterson/Documents/editfiles/output/a';
actpath='/Users/kpeterson/Documents/workspace/kris/output/laterMTtest/a';
%actpath='/Users/kpeterson/Documents/editfiles/downscalled3/output/a';
%actpath='/Users/kpeterson/Documents/editfiles/newresults/output/a';

%input time domain response:
inActFile=[actpath, num2str(firstpvp), '.pvp'];
inActMat=readpvpfile(inActFile);

time=length(inActMat);
[ysize xsize fsize]=size(inActMat{1}.values);
ystart=ysize/2-shrunkframesize/2;
yfinish=ysize/2+shrunkframesize/2-1;
xstart=xsize/2-shrunkframesize/2;
xfinish=xsize/2+shrunkframesize/2-1;
inAct=zeros(time, shrunkframesize, shrunkframesize);

for(t=[1:time]) 
    inAct(t,[1:shrunkframesize],[1:shrunkframesize])=squeeze(inActMat{t}.values([ystart:yfinish],[xstart:xfinish],1));
end
%firstpvp=1;
%complex1 output:
cbar1File=[actpath, num2str(firstpvp+5), '.pvp']
cbar1Mat=readpvpfile(cbar1File);
[t1 t2 fsize]=size(cbar1Mat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
cbar1=zeros(time, shrunkframesize, shrunkframesize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        cbar1(t,[1:shrunkframesize],[1:shrunkframesize],feat)=squeeze(cbar1Mat{t}.values([ystart:yfinish],[xstart:xfinish],feat));
    end
end
cbar1=(cbar1>0).*cbar1;
annotateimage(inAct, cbar1, pi/2, pi/2, showframe, dnoff, cbar0ff, 'Complex 1', 1,1)


%complex2 output:
cbar2File=[actpath, num2str(firstpvp+10), '.pvp']
cbar2Mat=readpvpfile(cbar2File);
[t1 t2 fsize]=size(cbar2Mat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
cbar2=zeros(time, shrunkframesize, shrunkframesize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        cbar2(t,[1:shrunkframesize],[1:shrunkframesize],feat)=squeeze(cbar2Mat{t}.values([ystart:yfinish],[xstart:xfinish],feat));
    end
end
cbar2=(cbar2>0).*cbar2;
annotateimage(inAct, cbar2, pi/4, pi/2, showframe, dnoff, cbar0ff, 'Complex 2', 1,2)

%complex3 output:
cbar3File=[actpath, num2str(firstpvp+15), '.pvp']
cbar3Mat=readpvpfile(cbar3File);
[t1 t2 fsize]=size(cbar3Mat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
cbar3=zeros(time, shrunkframesize, shrunkframesize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        cbar3(t,[1:shrunkframesize],[1:shrunkframesize],feat)=squeeze(cbar3Mat{t}.values([ystart:yfinish],[xstart:xfinish],feat));
    end
end
cbar3=(cbar3>0).*cbar3;
annotateimage(inAct, cbar3, pi/4, pi/4, showframe, dnoff, cbar0ff, 'Complex 3', 1,3)

%complex4 output:
cbar4File=[actpath, num2str(firstpvp+21), '.pvp']
cbar4Mat=readpvpfile(cbar4File);
[t1 t2 fsize]=size(cbar4Mat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
cbar4=zeros(time, shrunkframesize, shrunkframesize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        cbar4(t,[1:shrunkframesize],[1:shrunkframesize],feat)=squeeze(cbar4Mat{t}.values([ystart:yfinish],[xstart:xfinish],feat));
    end
end
cbar4=(cbar4>0).*cbar4;
annotateimage(inAct, cbar4, pi/2, pi/2, showframe, dnoff, cbar0ff, 'Complex 4', 1,4)


%MT output:
%outActMTFile=[actpath, num2str(firstpvp+3), '.pvp'];
%outActMTFile=[actpath, num2str(firstpvp+5), '.pvp'];33
%outActMTFile=[actpath, num2str(firstpvp+16), '.pvp'];
outActMTFile=[actpath, num2str(firstpvp+21), '.pvp']
outActMTMat=readpvpfile(outActMTFile);


[t1 t2 fsize]=size(outActMTMat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
outActMT=zeros(time, shrunkframesize, shrunkframesize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        outActMT(t,[1:shrunkframesize],[1:shrunkframesize],feat)=squeeze(outActMTMat{t}.values([ystart:yfinish],[xstart:xfinish],feat));
    end
end
outActMT=(outActMT>0).*outActMT;
%outActMT=(outActMT~=NaN).*outActMT;
outActMT(isnan(outActMT))=0;
outActMT(isinf(outActMT))=0;%maxMT=0.5; %25
annotateimage(inAct, outActMT, pi/2, pi/2, showframe, dnoff, mt0ff, 'MT 1', 1,5)

%MT2 output:
%outActMTFile=[actpath, num2str(firstpvp+3), '.pvp'];
%outActMTFile=[actpath, num2str(firstpvp+5), '.pvp'];33
%outActMTFile=[actpath, num2str(firstpvp+16), '.pvp'];
outActMT2File=[actpath, num2str(firstpvp+22), '.pvp']
outActMT2Mat=readpvpfile(outActMT2File);


[t1 t2 fsize]=size(outActMT2Mat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
outActMT2=zeros(time, shrunkframesize, shrunkframesize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        outActMT2(t,[1:shrunkframesize],[1:shrunkframesize],feat)=squeeze(outActMT2Mat{t}.values([ystart:yfinish],[xstart:xfinish],feat));
    end
end
outActMT2=(outActMT2>0).*outActMT2;
%outActMT2=(outActMT2~=NaN).*outActMT2;
outActMT2(isnan(outActMT2))=0;
outActMT2(isinf(outActMT2))=0;
%maxMT=0.5; %25
annotateimage(inAct, outActMT2, pi/4, pi/2, showframe, dnoff, mt0ff, 'MT 2', 1,6)

%MT3 output:
%outActMTFile=[actpath, num2str(firstpvp+3), '.pvp'];
%outActMTFile=[actpath, num2str(firstpvp+5), '.pvp'];33
%outActMTFile=[actpath, num2str(firstpvp+16), '.pvp'];
% outActMT3File=[actpath, num2str(firstpvp+23), '.pvp']
% outActMT3Mat=readpvpfile(outActMT3File);
% 
% 
% [t1 t2 fsize]=size(outActMT3Mat{t}.values(:,:,:));
% %outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
% outActMT3=zeros(time, ysize, xsize, fsize);
% 
% for(feat=[1:fsize]) 
%     for(t=[1:time])
%         outActMT3(t,[1:ysize],[1:xsize],feat)=squeeze(outActMT3Mat{t}.values(:,:,feat));
%     end
% end
% outActMT3=(outActMT3>0).*outActMT3;
% %maxMT=0.5; %25
% annotateimage(inAct, outActMT3, pi/4, pi/4, showframe, dnoff, mt0ff, 'MT 3', 1)


