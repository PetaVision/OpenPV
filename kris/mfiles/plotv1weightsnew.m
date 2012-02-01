close all
clear all
clc


simpleAFileName = '/Users/kpeterson/Documents/workspace/kris/output/seprecfields/w1_last.pvp';
simpleBFileName = '/Users/kpeterson/Documents/workspace/kris/output/seprecfields/w2_last.pvp';
simpleCFileName = '/Users/kpeterson/Documents/workspace/kris/output/seprecfields/w3_last.pvp';
wtot=0;
feature = 3;

simpleA=readpvpfile(simpleAFileName);
simpleB=readpvpfile(simpleBFileName);
simpleC=readpvpfile(simpleCFileName);

[temp1 temp2 numarbors] = size(simpleA{1}.values);


for(t=[1:numarbors])
    simpleAW(t,:,:)=squeeze(simpleA{1}.values{t}(:,:,1));
    simpleBW(t,:,:)=squeeze(-simpleB{1}.values{t}(:,:,1));
    simpleCW(t,:,:)=squeeze(-simpleC{1}.values{t}(:,:,1));
    simpleW(t,:,:)=squeeze(simpleA{1}.values{t}(:,:,2))-squeeze(simpleB{1}.values{t}(:,:,2))-squeeze(simpleC{1}.values{t}(:,:,2));
end
[t x y] = size(simpleW);

X=linspace(-floor(x/2),floor(x/2),x);
Y=linspace(-floor(y/2),floor(y/2),y);
T=linspace(0,-(t-1),t);
figure; surf(X,Y,squeeze(simpleW(1,:,:)));view(0,90);
figure; surf(X,Y,squeeze(simpleW(5,:,:)));view(0,90);
figure; surf(X,Y,squeeze(simpleW(10,:,:)));view(0,90);
figure; surf(X,T,squeeze(simpleW(:,:,8)));view(0,90);
figure(25); surf(X,T,squeeze(simpleAW(:,8,:)));view(0,90);hold on;
figure(25); surf(X,T,squeeze(simpleBW(:,8,:)));view(0,90);
figure(25); surf(X,T,squeeze(simpleCW(:,8,:)));view(0,90);
figure; plot(T,squeeze(simpleW(:,8,8)));

%FFT of weights
NFFTt=2^nextpow2(t);
NFFTy=2^nextpow2(y);
%Fx=r;
%Ft=c;
Fy=y;
Ft=t;

fftsimpleW = fft2(squeeze(simpleW(:,7,:)),NFFTy,NFFTt);
ft=linspace(-Ft/2,Ft/2,NFFTt);
fy=linspace(-Fy/2,Fy/2,NFFTy);
shifted=fftshift(fftsimpleW);
figure; surf(ft,fy, abs(shifted));%view(0,90);

%fft of impulse response:
outActFile='/Users/kpeterson/Documents/workspace/kris/output/seprecfields/a2.pvp';
outActMat=readpvpfile(outActFile);
time=length(outActMat);
[ysize xsize fsize]=size(outActMat{1}.values);
NFFTx=2^nextpow2(xsize);
NFFTy=2^nextpow2(ysize);
NFFTt=16*2^nextpow2(time);
Fx=xsize;
Fy=ysize;
Ft=time;

% xxx=[1:NFFTx];
% yyy=[1:NFFTy];
% ttt=[1:NFFTt];
% for xxx=[1:NFFTx]
%     for yyy=[1:NFFTy]
%         for ttt=[1:NFFTt]
%             hammw(xxx,yyy,ttt)=(0.54-0.46*cos(2*pi*(xxx/NFFTx)))*(0.54-0.46*cos(2*pi*(yyy/NFFTy)))*(0.54-0.46*cos(2*pi*(ttt/NFFTt)));
%         end
%     end
% end


outAct=zeros(NFFTy,NFFTx,NFFTt);
for(feat=[1:4])
    for(t=[1:time])
        outAct(t,[1:ysize],[1:xsize])=squeeze(outActMat{t}.values(:,:,feat));
    end
    
    %outAct=outAct.*hammw;
    
    fftoutAct = fftn(outAct);
    ft=linspace(-Ft/2,Ft/2,NFFTt);
    fx=linspace(-Fx/2,Fx/2,NFFTx);
    fy=linspace(-Fy/2,Fy/2,NFFTy);
    
    fftoutActShift=fftshift(fftoutAct);
    if((feat==1)||(feat==3))
        magnitude=abs(squeeze(fftoutActShift(:,NFFTy/2,:)));
    else
        magnitude=abs(squeeze(fftoutActShift(:,:,NFFTy/2)));
    end
    figure(20+feat); surf(ft,fx,magnitude); view(0,90);
    save(['/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/weightsImpulseResp', num2str(feat), '.mat'],'magnitude');
end
inActFile='/Users/kpeterson/Documents/workspace/kris/output/seprecfields/a1.pvp';
inActMat=readpvpfile(inActFile);
inAct=zeros(NFFTy,NFFTx,NFFTt);

for(t=[1:time]) 
    inAct(t,[1:ysize],[1:xsize])=squeeze(inActMat{t}.values(:,:,1));
end


fftinAct = fftn(inAct);
fftinActShift=fftshift(fftinAct);
magnitudeIn=abs(squeeze(fftinActShift(:,NFFTy/2,:)));
figure; surf(ft,fx,magnitudeIn); %view(0,90);

figure; surf(ft,fx,magnitude./magnitudeIn); view(0,90);

outActCrop=squeeze(outAct([7:16],NFFTy/2,[122:137]));
figure; surf(outActCrop)
