close all
clear all
clc

firstpvp=2;
showframe=36; %25-14; %36;
dnoff=3;
cbar0ff=12;
compl0ff=13;
mt0ff=14;

%actpath='/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a';
actpath='/Users/kpeterson/Documents/editfiles/output/a';
%actpath='/Users/kpeterson/Documents/editfiles/downscalled3/output/a';
%actpath='/Users/kpeterson/Documents/editfiles/newresults/output/a';

%input time domain response:
inActFile=[actpath, num2str(firstpvp), '.pvp'];
inActMat=readpvpfile(inActFile);

time=length(inActMat);
[ysize xsize fsize]=size(inActMat{1}.values);
inAct=zeros(time, ysize, xsize);

for(t=[1:time]) 
    inAct(t,[1:ysize],[1:xsize])=squeeze(inActMat{t}.values(:,:,1));
end
firstpvp=1;
%complex1 output:
cbar1File=[actpath, num2str(firstpvp+6), '.pvp']
cbar1Mat=readpvpfile(cbar1File);
[t1 t2 fsize]=size(cbar1Mat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
cbar1=zeros(time, ysize, xsize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        cbar1(t,[1:ysize],[1:xsize],feat)=squeeze(cbar1Mat{t}.values(:,:,feat));
    end
end
cbar1=(cbar1>0).*cbar1;
annotateimage(inAct, cbar1, pi/2, pi/2, showframe, dnoff, cbar0ff)


%complex2 output:
cbar2File=[actpath, num2str(firstpvp+11), '.pvp']
cbar2Mat=readpvpfile(cbar2File);
[t1 t2 fsize]=size(cbar2Mat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
cbar2=zeros(time, ysize, xsize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        cbar2(t,[1:ysize],[1:xsize],feat)=squeeze(cbar2Mat{t}.values(:,:,feat));
    end
end
cbar2=(cbar2>0).*cbar2;
annotateimage(inAct, cbar2, pi/4, pi/2, showframe, dnoff, cbar0ff)

%complex3 output:
cbar3File=[actpath, num2str(firstpvp+16), '.pvp']
cbar3Mat=readpvpfile(cbar3File);
[t1 t2 fsize]=size(cbar3Mat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
cbar3=zeros(time, ysize, xsize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        cbar3(t,[1:ysize],[1:xsize],feat)=squeeze(cbar3Mat{t}.values(:,:,feat));
    end
end
cbar3=(cbar3>0).*cbar3;
annotateimage(inAct, cbar3, pi/4, pi/4, showframe, dnoff, cbar0ff)

%MT output:
%outActMTFile=[actpath, num2str(firstpvp+3), '.pvp'];
%outActMTFile=[actpath, num2str(firstpvp+5), '.pvp'];33
%outActMTFile=[actpath, num2str(firstpvp+16), '.pvp'];
outActMTFile=[actpath, num2str(firstpvp+17), '.pvp']
outActMTMat=readpvpfile(outActMTFile);


[t1 t2 fsize]=size(outActMTMat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
outActMT=zeros(time, ysize, xsize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        outActMT(t,[1:ysize],[1:xsize],feat)=squeeze(outActMTMat{t}.values(:,:,feat));
    end
end
outActMT=(outActMT>0).*outActMT;
%maxMT=0.5; %25
annotateimage(inAct, outActMT, pi/2, pi/2, showframe, dnoff, mt0ff)




