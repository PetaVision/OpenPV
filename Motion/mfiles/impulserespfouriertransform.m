function impulserespfouriertransform(actMat, label, baseFigNum, startRot, rotInc)

%outActFile='/Users/kpeterson/Documents/workspace/kris/output/seprecfields/a2.pvp';
%outActMat=readpvpfile(outActFile);
time=length(actMat);
[ysize xsize fsize]=size(actMat{1}.values);
NFFTx=2^nextpow2(xsize);
NFFTy=2^nextpow2(ysize);
NFFTt=2^nextpow2(time);
Fx=xsize;
Fy=ysize;
Ft=1;
ft=linspace(-Ft/2,Ft/2,NFFTt);
%fx=linspace(-Fx/2,Fx/2,NFFTx);
%fy=linspace(-Fy/2,Fy/2,NFFTy);
fx=linspace(-256,256,NFFTx);
fy=linspace(-256,256,NFFTy);



outAct=zeros(NFFTt,NFFTy,NFFTx);
for(feat=[1:fsize])
    for(t=[1:time])
        outAct(t,[1:ysize],[1:xsize])=squeeze(actMat{t}.values(:,:,feat));
    end
    
    
    fftoutAct = fftn(outAct);    
    fftoutActShift=fftshift(fftoutAct);
    rotation=startRot+(feat-1)*rotInc;
    if(mod(rotation,pi)==0)
        magnitude=abs(squeeze(fftoutActShift(:,NFFTy/2,:)));
        figure(baseFigNum+feat); imagesc(fx,ft,magnitude); view(0,90); title([label, ' feature ', num2str(feat)]);
    elseif(mod(rotation,pi/2)==0)
        magnitude=abs(squeeze(fftoutActShift(:,:,NFFTx/2)));
        figure(baseFigNum+feat); imagesc(fx,ft,magnitude); view(0,90); title([label, ' feature ', num2str(feat)]);
    else
        %[fxp ftp fyp]=meshgrid(fx*cos(rotation), ft, fy*cos(rotation));
        %[fyp ftp fxp]=meshgrid(fy*cos(rotation), ft, fx*cos(rotation));
        [fx3d ft3d fy3d]=meshgrid(fx, ft, fy);
        fxp3d=fx3d*cos(rotation)+fy3d*sin(rotation);
        fyp3d=-fx3d*sin(rotation)+fy3d*cos(rotation);
        ftp3d=ft3d;
%         fyp=fy*cos(rotation)+fx*sin(rotation);
%         fxp=-fy*sin(rotation)+fx*cos(rotation);
%         ftp=ft;
%         [fx3d ft3d fy3d]=meshgrid(fx, ft, fy);
%         [fxp3d ftp3d fyp3d]=meshgrid(fxp, ftp, fyp);
        
        %[fyp ftp fxp]=meshgrid(-fx*sin(rotation)+fy*cos(rotation), ft, fx*cos(rotation)+fy*sin(rotation));
        %rotated=interp3(fy, ft, fx, fftoutActShift, fyp, ftp, fxp);
        %rotated=interp3(fy, ft, fx, fftoutActShift, fxp, ftp, fyp);
        magnitude=abs(fftoutActShift);
        rotated=interp3(fx3d, ft3d, fy3d, magnitude, fxp3d, ftp3d, fyp3d);
        %rotated=interp3(fy, ft, fx, magnitude, fyp, ftp, fxp);
        %size(rotated)
        %magnitude=abs(squeeze(rotated(:,NFFTy/2,:)));
        %magnitude=abs(squeeze(rotated(:,:,NFFTx/2)));
        figure(baseFigNum+feat); imagesc(fx,ft,squeeze(rotated(:,NFFTx/2,:)));title([label, ' feature ', num2str(feat)]);
    end
end