close all
clear all
clc

firstpvp=2;
showframe=36;
%actpath='/Users/kpeterson/Documents/editfiles/output/a';
actpath='/Users/kpeterson/Documents/editfiles/newresults/output/a';

%input time domain response:
inActFile=[actpath, num2str(firstpvp), '.pvp'];
inActMat=readpvpfile(inActFile);

time=length(inActMat);
[ysize xsize fsize]=size(inActMat{1}.values);
% NFFTx=2^nextpow2(xsize);
% NFFTy=2^nextpow2(ysize);
% NFFTt=256;%2^nextpow2(time);
% Fx=xsize;
% Fy=ysize;
% Ft=1;
% ft=linspace(-Ft/2,Ft/2,NFFTt);
% fx=linspace(-Fx/2,Fx/2,NFFTx);
% fy=linspace(-Fy/2,Fy/2,NFFTy);


%inAct=zeros(NFFTt,NFFTy,NFFTx);
inAct=zeros(time, ysize, xsize);

for(t=[1:time]) 
    inAct(t,[1:ysize],[1:xsize])=squeeze(inActMat{t}.values(:,:,1));
end
figure;imagesc((squeeze(inAct(showframe, :,:))))
colormap(gray)


% %figure; surf(squeeze(inAct(5,[1:ysize],[1:xsize])))
% figure; plot(squeeze(inAct([1:time], ysize/2,ysize/2)))
% figure; plot(squeeze(inAct(time/2, [1:ysize],ysize/2)))
% 
% 
% %input freq domain
% fftinAct = fft2(squeeze(inAct(:,:,ysize/2)));
% fftinActShift=fftshift(fftinAct);
% magnitudeIn=abs(squeeze(fftinActShift));
% %plot weight impulse response:
% load '/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/weightsImpulseResp.mat';
% weightmag=magnitude;
% clear magnitude;
% 
% [impT, impX]=size(weightmag);
% [XX,YY]=meshgrid([1:NFFTy/impX:NFFTy], [1:NFFTt/impT:NFFTt]);
% weightmagscaled=interp2(XX,YY, weightmag, [1:NFFTy],[1:NFFTt]');
% figure; imagesc(fx, ft, weightmagscaled/max(max(weightmagscaled))+magnitudeIn/max(max(magnitudeIn)))



% %output time domain
% outActSimpleAFile=['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a', num2str(firstpvp+1), '.pvp'];
% outActSimpleAMat=readpvpfile(outActSimpleAFile);
% 
% 
% [t1 t2 fsize]=size(outActSimpleAMat{t}.values(:,:,:));
% outActSimpleA=zeros(NFFTt,NFFTy,NFFTx,fsize);
% for(feat=[1:fsize]) 
%     for(t=[1:time])
%         outActSimpleA(t,[1:ysize],[1:xsize],feat)=squeeze(outActSimpleAMat{t}.values(:,:,feat));
%     end
% end
% figure; hold on;
% 
% plot((squeeze(outActSimpleA([1:time], ysize/2,ysize/2, 1))), 'r')
% plot((squeeze(outActSimpleA([1:time], ysize/2,ysize/2, 2))), 'b')
% plot((squeeze(outActSimpleA([1:time], ysize/2,ysize/2, 3))), 'g')
% plot((squeeze(outActSimpleA([1:time], ysize/2,ysize/2, 4))), 'c')
% 
% outActSimpleBFile=['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a', num2str(firstpvp+2), '.pvp'];
% outActSimpleBMat=readpvpfile(outActSimpleBFile);
% 
% 
% [t1 t2 fsize]=size(outActSimpleBMat{t}.values(:,:,:));
% outActSimpleB=zeros(NFFTt,NFFTy,NFFTx,fsize);
% for(feat=[1:fsize]) 
%     for(t=[1:time])
%         outActSimpleB(t,[1:ysize],[1:xsize],feat)=squeeze(outActSimpleBMat{t}.values(:,:,feat));
%     end
% end
% %figure; hold on;
% 
% plot((squeeze(outActSimpleB([1:time], ysize/2,ysize/2, 1))), 'k')
% plot((squeeze(outActSimpleB([1:time], ysize/2,ysize/2, 2))), 'm')
% plot((squeeze(outActSimpleB([1:time], ysize/2,ysize/2, 3))), 'y')
% plot((squeeze(outActSimpleB([1:time], ysize/2,ysize/2, 4))), 'c*')

outActComplexFile=[actpath, num2str(firstpvp+3), '.pvp'];
outActComplexMat=readpvpfile(outActComplexFile);


[t1 t2 fsize]=size(outActComplexMat{t}.values(:,:,:));
%outActComplex=zeros(NFFTt,NFFTy,NFFTx,fsize);
outActComplex=zeros(time, ysize, xsize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        outActComplex(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexMat{t}.values(:,:,feat));
    end
end

%figure; hold on;
figure;imagesc((squeeze(outActComplex(showframe, :,:,1))))
colormap(gray)
figure;imagesc((squeeze(outActComplex(showframe, :,:,2))))
colormap(gray)
figure;imagesc((squeeze(outActComplex(showframe, :,:,3))))
colormap(gray)
figure;imagesc((squeeze(outActComplex(showframe, :,:,4))))
colormap(gray)

% plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 1))), 'k')
% plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 2))), 'm')
% plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 3))), 'r')
% plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 4))), 'c')

% %plot sum layer
% outActComplexFile=['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/a', num2str(firstpvp+4), '.pvp'];
% outActComplexMat=readpvpfile(outActComplexFile);
% 
% 
% [t1 t2 fsize]=size(outActComplexMat{t}.values(:,:,:));
% outActComplex=zeros(NFFTt,NFFTy,NFFTx,fsize);
% for(feat=[1:fsize]) 
%     for(t=[1:time])
%         outActComplex(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexMat{t}.values(:,:,feat));
%     end
% end
% 
% figure; hold on;
% 
% plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 1))), 'k')
% plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 2))), 'm')
% plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 3))), 'r')
% plot((squeeze(outActComplex([1:time], ysize/2,ysize/2, 4))), 'c')

%plot normalized layer:
outActComplexBarFile=[actpath, num2str(firstpvp+5), '.pvp'];
outActComplexBarMat=readpvpfile(outActComplexBarFile);


[t1 t2 fsize]=size(outActComplexBarMat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
outActComplexBar=zeros(time, ysize, xsize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        outActComplexBar(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexBarMat{t}.values(:,:,feat));
    end
end

%figure; hold on;
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,1))))
title('bar feature 1');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,2))))
title('bar feature 2');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,3))))
title('bar feature 3');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,4))))
title('bar feature 4');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,3)))-(squeeze(outActComplexBar(showframe, :,:,1))))
title('bar feature 3-1');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,4)))-(squeeze(outActComplexBar(showframe, :,:,2))))
title('bar feature 4-2');
colormap(gray)

% plot((squeeze(outActComplexBar([1:time], ysize/2,ysize/2, 1))), 'k')
% plot((squeeze(outActComplexBar([1:time], ysize/2,ysize/2, 2))), 'm')
% plot((squeeze(outActComplexBar([1:time], ysize/2,ysize/2, 3))), 'r')
% plot((squeeze(outActComplexBar([1:time], ysize/2,ysize/2, 4))), 'c')


%output frequency domain:
% fftoutActSimpleA = fftn(outActSimpleA); 
% fftoutActSimpleAShift=fftshift(fftoutActSimpleA);
% magnitudeSimpleA=abs(squeeze(fftoutActSimpleAShift(:,:,NFFTy/2)));
% figure; imagesc(ft,fx,magnitudeSimpleA); %view(0,90);

% for(t=[150:time])
% figure(101);imagesc(squeeze(outActComplexBar(t,[289:340],[20:70],3))); colormap(gray); 
% figure(102);imagesc(squeeze(outActComplex(t,[289:340],[20:70],3))); colormap(gray); 
% figure(103);imagesc(squeeze(inAct(t,[289:340],[20:70],1))); colormap(gray); pause
% end


%B output:
% outActComplexFile=[actpath, num2str(firstpvp+8), '.pvp'];
% outActComplexMat=readpvpfile(outActComplexFile);
% 
% 
% [t1 t2 fsize]=size(outActComplexMat{t}.values(:,:,:));
% %outActComplex=zeros(NFFTt,NFFTy,NFFTx,fsize);
% outActComplex=zeros(time, ysize, xsize, fsize);
% 
% for(feat=[1:fsize]) 
%     for(t=[1:time])
%         outActComplex(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexMat{t}.values(:,:,feat));
%     end
% end
% 
% %figure; hold on;
% figure;imagesc((squeeze(outActComplex(showframe, :,:,1))))
% colormap(gray)
% figure;imagesc((squeeze(outActComplex(showframe, :,:,2))))
% colormap(gray)
% figure;imagesc((squeeze(outActComplex(showframe, :,:,3))))
% colormap(gray)
% figure;imagesc((squeeze(outActComplex(showframe, :,:,4))))
% colormap(gray)
% 
% 
% %plot normalized layer:
% outActComplexBarFile=[actpath, num2str(firstpvp+10), '.pvp'];
% outActComplexBarMat=readpvpfile(outActComplexBarFile);
% 
% 
% [t1 t2 fsize]=size(outActComplexBarMat{t}.values(:,:,:));
% %outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
% outActComplexBar=zeros(time, ysize, xsize, fsize);
% 
% for(feat=[1:fsize]) 
%     for(t=[1:time])
%         outActComplexBar(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexBarMat{t}.values(:,:,feat));
%     end
% end
% 
% %figure; hold on;
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,1))))
% title('sqrt(2) bar feature 1');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,2))))
% title('sqrt(2) bar feature 2');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,3))))
% title('sqrt(2) bar feature 3');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,4))))
% title('sqrt(2) bar feature 4');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,3)))-(squeeze(outActComplexBar(showframe, :,:,1))))
% title('sqrt(2) bar feature 3-1');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,4)))-(squeeze(outActComplexBar(showframe, :,:,2))))
% title('sqrt(2) bar feature 4-2');
% colormap(gray)
% 
% % simpleAFileName = ['/Users/kpeterson/Documents/editfiles/output/', 'w26_last.pvp'];
% % simpleBFileName = ['/Users/kpeterson/Documents/editfiles/output/', 'w27_last.pvp'];
% % simpleCFileName = ['/Users/kpeterson/Documents/editfiles/output/', 'w28_last.pvp'];
% % wtot=0;
% % feature = 3;
% % 
% % simpleA=readpvpfile(simpleAFileName);
% % simpleB=readpvpfile(simpleBFileName);
% % simpleC=readpvpfile(simpleCFileName);
% % 
% % [temp1 temp2 numarbors] = size(simpleA{1}.values);
% % 
% % 
% % for(t=[1:numarbors])
% % %     simpleAW(t,:,:)=squeeze(simpleA{1}.values{t}(:,:,1));
% % %     simpleBW(t,:,:)=squeeze(-simpleB{1}.values{t}(:,:,1));
% % %     simpleCW(t,:,:)=squeeze(-simpleC{1}.values{t}(:,:,1));
% %     simpleW(t,:,:)=squeeze(simpleA{1}.values{t}(:,:,1))-squeeze(simpleB{1}.values{t}(:,:,1))-squeeze(simpleC{1}.values{t}(:,:,1));
% % end
% % [t x y] = size(simpleW);
% % 
% % X=linspace(-floor(x/2),floor(x/2),x);
% % Y=linspace(-floor(y/2),floor(y/2),y);
% % T=linspace(0,-(t-1),t);
% % figure; surf(X,Y,squeeze(simpleW(1,:,:)));view(0,90);
% % figure; surf(X,Y,squeeze(simpleW(5,:,:)));view(0,90);
% % figure; surf(X,Y,squeeze(simpleW(10,:,:)));view(0,90);
% % figure; surf(X,T,squeeze(simpleW(:,:,8)));view(0,90);
% % % figure(25); surf(X,T,squeeze(simpleAW(:,8,:)));view(0,90);hold on;
% % % figure(25); surf(X,T,squeeze(simpleBW(:,8,:)));view(0,90);
% % % figure(25); surf(X,T,squeeze(simpleCW(:,8,:)));view(0,90);
% % figure; plot(T,squeeze(simpleW(:,8,8)));
% 
% %0 speed output:
% outActComplexFile=[actpath, num2str(firstpvp+13), '.pvp'];
% outActComplexMat=readpvpfile(outActComplexFile);
% 
% 
% [t1 t2 fsize]=size(outActComplexMat{t}.values(:,:,:));
% %outActComplex=zeros(NFFTt,NFFTy,NFFTx,fsize);
% outActComplex=zeros(time, ysize, xsize, fsize);
% 
% for(feat=[1:fsize]) 
%     for(t=[1:time])
%         outActComplex(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexMat{t}.values(:,:,feat));
%     end
% end
% 
% %figure; hold on;
% figure;imagesc((squeeze(outActComplex(showframe, :,:,1))))
% colormap(gray)
% figure;imagesc((squeeze(outActComplex(showframe, :,:,2))))
% colormap(gray)
% figure;imagesc((squeeze(outActComplex(showframe, :,:,3))))
% colormap(gray)
% figure;imagesc((squeeze(outActComplex(showframe, :,:,4))))
% colormap(gray)
% 
% 
% %plot normalized layer:
% outActComplexBarFile=[actpath, num2str(firstpvp+15), '.pvp'];
% outActComplexBarMat=readpvpfile(outActComplexBarFile);
% 
% 
% [t1 t2 fsize]=size(outActComplexBarMat{t}.values(:,:,:));
% %outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
% outActComplexBar=zeros(time, ysize, xsize, fsize);
% 
% for(feat=[1:fsize]) 
%     for(t=[1:time])
%         outActComplexBar(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexBarMat{t}.values(:,:,feat));
%     end
% end
% 
% %figure; hold on;
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,1))))
% title('0 bar feature 1');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,2))))
% title('0 bar feature 2');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,3))))
% title('0 bar feature 3');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,4))))
% title('0 bar feature 4');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,3)))-(squeeze(outActComplexBar(showframe, :,:,1))))
% title('0 bar feature 3-1');
% colormap(gray)
% figure;imagesc((squeeze(outActComplexBar(showframe, :,:,4)))-(squeeze(outActComplexBar(showframe, :,:,2))))
% title('0 bar feature 4-2');
% colormap(gray)

%MT output:

outActComplexBarFile=[actpath, num2str(firstpvp+16), '.pvp'];
outActComplexBarMat=readpvpfile(outActComplexBarFile);


[t1 t2 fsize]=size(outActComplexBarMat{t}.values(:,:,:));
%outActComplexBar=zeros(NFFTt,NFFTy,NFFTx,fsize);
outActComplexBar=zeros(time, ysize, xsize, fsize);

for(feat=[1:fsize]) 
    for(t=[1:time])
        outActComplexBar(t,[1:ysize],[1:xsize],feat)=squeeze(outActComplexBarMat{t}.values(:,:,feat));
    end
end

%figure; hold on;
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,1))))
title('MT feature 1');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,2))))
title('MT feature 2');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,3))))
title('MT feature 3');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,4))))
title('MT feature 4');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,3)))-(squeeze(outActComplexBar(showframe, :,:,1))))
title('MT feature 3-1');
colormap(gray)
figure;imagesc((squeeze(outActComplexBar(showframe, :,:,4)))-(squeeze(outActComplexBar(showframe, :,:,2))))
title('MT feature 4-2');
colormap(gray)