close all
clear all
clc

 a=readactivities('~/Documents/workspace/kris/output/krisplay/a2.pvp');
 
a11=readweights('~/Documents/workspace/kris/output/krisplay/w1_last.pvp');
a12=readweights('~/Documents/workspace/kris/output/krisplay/w2_last.pvp');
a13=readweights('~/Documents/workspace/kris/output/krisplay/w3_last.pvp');
a14=readweights('~/Documents/workspace/kris/output/krisplay/w4_last.pvp');
a15=readweights('~/Documents/workspace/kris/output/krisplay/w5_last.pvp');

a1=readweights('~/Documents/workspace/kris/output/krisplay/w6_last.pvp');
a2=readweights('~/Documents/workspace/kris/output/krisplay/w7_last.pvp');
a3=readweights('~/Documents/workspace/kris/output/krisplay/w8_last.pvp');
a4=readweights('~/Documents/workspace/kris/output/krisplay/w9_last.pvp');
a5=readweights('~/Documents/workspace/kris/output/krisplay/w10_last.pvp');

a6=readweights('~/Documents/workspace/kris/output/krisplay/w11_last.pvp');
a7=readweights('~/Documents/workspace/kris/output/krisplay/w12_last.pvp');
a8=readweights('~/Documents/workspace/kris/output/krisplay/w13_last.pvp');
a9=readweights('~/Documents/workspace/kris/output/krisplay/w14_last.pvp');
a10=readweights('~/Documents/workspace/kris/output/krisplay/w15_last.pvp');

for (f=[1:8])
%f=8; 
%fignum=f+8;
    figure(f); hold on; 
    
    subplot(2,1,1), hold on, surfc(a11{1}(:,:, f)-a1{1}(:,:, f)-a6{1}(:,:, f));
    subplot(2,1,1), surfc(a12{1}(:,:, f)-a2{1}(:,:, f)-a7{1}(:,:, f));
    subplot(2,1,1), surfc(a13{1}(:,:, f)-a3{1}(:,:, f)-a8{1}(:,:, f));
    subplot(2,1,1), surfc(a14{1}(:,:, f)-a4{1}(:,:, f)-a9{1}(:,:, f));
    subplot(2,1,1), surfc(a15{1}(:,:, f)-a5{1}(:,:, f)-a10{1}(:,:, f));
    view(90,90);
    
    subplot(2,1,2), plot(squeeze(a(128,128,f,:)))
end

figure;
a=readactivities('~/Documents/workspace/kris/output/krisplay/a1.pvp');
surf(squeeze(double(a(:,:,1,25))))
view(90,90)
 
%  for (i=[1:8])
%     figure(i);  
%     plot(squeeze(a(128,128,i,:)))
%     i
%     max(a(128,128,i,:))-min(a(128,128,i,:))
%  end