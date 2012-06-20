%function [wx, wt] = fftofstimuli(filename)

clear all
close all
clc


b=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a2.pvp');
%b=readactivities(filename);
bb=double(squeeze(b(128,:,1,:)));

%figure;
%surf(bb)

[r c] = size(bb);
NFFTx=2^nextpow2(r);
NFFTt=2^nextpow2(c);
%Fx=r;
%Ft=c;
Fx=29;
Ft=9;

FFTFXT = fft2(bb,NFFTx,NFFTt);
ft=linspace(-Ft/2,Ft/2,NFFTt);
fx=linspace(-Fx/2,Fx/2,NFFTx);
 %figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
 shifted=fftshift(FFTFXT);
 surf(ft,fx, abs(shifted))
 %figure;
 %surf(ft(:),fx([1:128]), abs(shifted([1:128],:)))


peakvalue = max(max(abs(shifted([1:128],:))));
[rr cc] = find(abs(shifted([1:128],:))==peakvalue);
wx = fx(rr)
wt = ft(cc)


