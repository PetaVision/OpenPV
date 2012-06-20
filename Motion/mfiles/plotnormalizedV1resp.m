close all
clear all
clc

firstpvp=2;

%input time domain response:
inActFile=['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a', num2str(firstpvp), '.pvp'];
inActMat=readpvpfile(inActFile);

time=length(inActMat);
[ysize xsize fsize]=size(inActMat{1}.values);
NFFTx=2^nextpow2(xsize);
NFFTy=2^nextpow2(ysize);
NFFTt=256;%2^nextpow2(time);
Fx=xsize;
Fy=ysize;
Ft=1;
ft=linspace(-Ft/2,Ft/2,NFFTt);
fx=linspace(-Fx/2,Fx/2,NFFTx);
fy=linspace(-Fy/2,Fy/2,NFFTy);


inAct=zeros(NFFTt,NFFTy,NFFTx);

for(t=[1:time]) 
    inAct(t,[1:ysize],[1:xsize])=squeeze(inActMat{t}.values(:,:,1));
end

%figure; surf(squeeze(inAct(5,[1:ysize],[1:xsize])))
figure; plot(squeeze(inAct([1:time], ysize/2,ysize/2)))
figure; plot(squeeze(inAct(time/2, [1:ysize],ysize/2)))


%input freq domain
fftinAct = fft2(squeeze(inAct(:,:,ysize/2)));
fftinActShift=fftshift(fftinAct);
magnitudeIn=abs(squeeze(fftinActShift));
%plot weight impulse response:
load '/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/weightsImpulseResp.mat';
weightmag=magnitude;
clear magnitude;

[impT, impX]=size(weightmag);
[XX,YY]=meshgrid([1:NFFTy/impX:NFFTy], [1:NFFTt/impT:NFFTt]);
weightmagscaled=interp2(XX,YY, weightmag, [1:NFFTy],[1:NFFTt]');
figure; imagesc(fx, ft, weightmagscaled/max(max(weightmagscaled))+magnitudeIn/max(max(magnitudeIn)))



%output time domain
outActSimpleAFile=['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a', num2str(firstpvp+1), '.pvp'];
outActSimpleAMat=readpvpfile(outActSimpleAFile);


[t1 t2 fsize]=size(outActSimpleAMat{t}.values(:,:,:));
outActSimpleA=zeros(NFFTt,NFFTy,NFFTx,fsize);
for(feat=[1:fsize]) 
    for(t=[1:time])
        outActSimpleA(t,[1:ysize],[1:xsize],feat)=squeeze(outActSimpleAMat{t}.values(:,:,feat));
    end
end
figure; hold on;

plot((squeeze(outActSimpleA([1:time], ysize/2,ysize/2, 1))), 'r')
plot((squeeze(outActSimpleA([1:time], ysize/2,ysize/2, 2))), 'b')
plot((squeeze(outActSimpleA([1:time], ysize/2,ysize/2, 3))), 'g')
plot((squeeze(outActSimpleA([1:time], ysize/2,ysize/2, 4))), 'c')

outActSimpleBFile=['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a', num2str(firstpvp+2), '.pvp'];
outActSimpleBMat=readpvpfile(outActSimpleBFile);


[t1 t2 fsize]=size(outActSimpleBMat{t}.values(:,:,:));
outActSimpleB=zeros(NFFTt,NFFTy,NFFTx,fsize);
for(feat=[1:fsize]) 
    for(t=[1:time])
        outActSimpleB(t,[1:ysize],[1:xsize],feat)=squeeze(outActSimpleBMat{t}.values(:,:,feat));
    end
end
%figure; hold on;

plot((squeeze(outActSimpleB([1:time], ysize/2,ysize/2, 1))), 'k')
plot((squeeze(outActSimpleB([1:time], ysize/2,ysize/2, 2))), 'm')
plot((squeeze(outActSimpleB([1:time], ysize/2,ysize/2, 3))), 'y')
plot((squeeze(outActSimpleB([1:time], ysize/2,ysize/2, 4))), 'c*')

outActComplexFile=['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a', num2str(firstpvp+3), '.pvp'];
outActComplexMat=readpvpfile(outActComplexFile);


[t1 t2 fsize]=size(outActComplexMat{t}.values(:,:,:));
outActComplex=zeros(NFFTt,NFFTy,NFFTx,fsize);
for(feat=[1:fsize]) 
    for(t=[1:time])
        outActComplex(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexMat{t}.values(:,:,feat));
    end
end

figure; hold on;

plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 1))), 'k')
plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 2))), 'm')
plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 3))), 'r')
plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 4))), 'c')

%plot sum layer
outActComplexFile=['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a', num2str(firstpvp+4), '.pvp'];
outActComplexMat=readpvpfile(outActComplexFile);


[t1 t2 fsize]=size(outActComplexMat{t}.values(:,:,:));
outActComplex=zeros(NFFTt,NFFTy,NFFTx,fsize);
for(feat=[1:fsize]) 
    for(t=[1:time])
        outActComplex(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexMat{t}.values(:,:,feat));
    end
end

figure; hold on;

plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 1))), 'k')
plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 2))), 'm')
plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 3))), 'r')
plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 4))), 'c')

%plot normalized layer:
outActComplexFile=['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a', num2str(firstpvp+5), '.pvp'];
outActComplexMat=readpvpfile(outActComplexFile);


[t1 t2 fsize]=size(outActComplexMat{t}.values(:,:,:));
outActComplex=zeros(NFFTt,NFFTy,NFFTx,fsize);
for(feat=[1:fsize]) 
    for(t=[1:time])
        outActComplex(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexMat{t}.values(:,:,feat));
    end
end

figure; hold on;

plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 1))), 'k')
plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 2))), 'm')
plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 3))), 'r')
plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 4))), 'c')


%output frequency domain:
fftoutActSimpleA = fftn(outActSimpleA); 
fftoutActSimpleAShift=fftshift(fftoutActSimpleA);
magnitudeSimpleA=abs(squeeze(fftoutActSimpleAShift(:,:,NFFTy/2)));
figure; imagesc(ft,fx,magnitudeSimpleA); %view(0,90);


