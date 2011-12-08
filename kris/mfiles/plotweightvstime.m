%close all
clear all
clc


path1 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a';
path2 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w2_a';
fileend = '_last.pvp';
wtot=0;
for(x=[0:9])
    filename1 = [path1, num2str(x), fileend];
    filename2 = [path2, num2str(x), fileend];
    temp1 = readweights(filename1);
    temp2 = readweights(filename2);
    w1{x+1} = temp1{1}(:,:,3);
    w2{x+1} = -temp2{1}(:,:,3);
    %wtot = wtot + w1{x+1} + w2{x+1};
%     wtot = wtot + w1{x+1};
%     figure(x+1);hold on;
%     xx=x+1;
%     maxxx(xx) = max(max(w1{x+1}))
%     surf(w1{x+1} + w2{x+1});
%     %surf(w1{x+1});
%     %surf(w2{x+1});
%     view(90,90);
%     www(10-x) = w1{x+1}(15,15) + w2{x+1}(15,15)
    
    wxt(10-x,:) = w1{x+1}(:,15)' + w2{x+1}(:,15)'
end


figure; surf([-14:14], [-9:0],wxt)
 view(0,90);
 xlabel('position')
ylabel('time')


[r c] = size(wxt);
NFFTx=2^nextpow2(r);
NFFTt=2^nextpow2(c);
Fx=r;
Ft=c;

FFTFXT = fft2(wxt,NFFTx,NFFTt);
ft=linspace(-Ft/2,Ft/2,NFFTt);
fx=linspace(-Fx/2,Fx/2,NFFTx);
figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
shifted=fftshift(FFTFXT);
surf(ft,fx, abs(shifted))
