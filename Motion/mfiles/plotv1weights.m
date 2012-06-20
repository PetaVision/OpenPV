close all
clear all
clc

% w1_a0=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a0_last.pvp');
% w1_a1=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a1_last.pvp');
% w1_a2=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a2_last.pvp');
% w1_a3=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a3_last.pvp');
% w1_a4=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a4_last.pvp');
% w1_a5=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a5_last.pvp');
% w1_a6=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a6_last.pvp');
% w1_a7=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a7_last.pvp');
% w1_a8=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a8_last.pvp');
% w1_a9=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a9_last.pvp');
%
% figure(1); surf(w1_a0{1})
% view(90,90);
% figure(2);surf(w1_a1{1})
% view(90,90);
% figure(3);surf(w1_a2{1})
% view(90,90);
% figure(4);surf(w1_a3{1})
% view(90,90);
% figure(5);surf(w1_a4{1})
% view(90,90);
% figure(6); surf(w1_a5{1})
% view(90,90);
% figure(7);surf(w1_a6{1})
% view(90,90);
% figure(8);surf(w1_a7{1})
% view(90,90);
% figure(9);surf(w1_a8{1})
% view(90,90);
% figure(10);surf(w1_a9{1})
% view(90,90);

% path1 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w1_a';
% path2 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w2_a';
% path3 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w3_a';
% path4 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w4_a';
% path5 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w5_a';
% path6 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w6_a';
% path7 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w7_a';
path1 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w2_a';
path2 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w3_a';
path3 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w4_a';
path4 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w5_a';
path5 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w6_a';
path6 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w7_a';
path7 = '/Users/kpeterson/Documents/workspace/kris/output/3DGauss/w8_a';
fileend = '_last.pvp';
wtot=0;
feature = 3;
numarbors=11;
for(x=[0:numarbors-1])
    filename1 = [path1, num2str(x), fileend];
    filename2 = [path2, num2str(x), fileend];
    filename3 = [path3, num2str(x), fileend];
    filename4 = [path4, num2str(x), fileend];
    filename5 = [path5, num2str(x), fileend];
    filename6 = [path6, num2str(x), fileend];
    filename7 = [path7, num2str(x), fileend];
    temp1 = readweights(filename1);
    temp2 = readweights(filename2);
    temp3 = readweights(filename3);
    temp4 = readweights(filename4);
    temp5 = readweights(filename5);
    temp6 = readweights(filename6);
    temp7 = readweights(filename7);
    w1{x+1} = temp1{1}(:,:,feature);
    w2{x+1} = -temp2{1}(:,:,feature);
    w3{x+1} = -temp3{1}(:,:,feature);
    w4{x+1} = temp4{1}(:,:,feature);
    w5{x+1} = temp5{1}(:,:,feature);
    w6{x+1} = -temp6{1}(:,:,feature);
    w7{x+1} = -temp7{1}(:,:,feature);
    wtot = wtot + w1{x+1} + w2{x+1} + w3{x+1};
    %wtot = wtot + w1{x+1};
    figure(x+1);hold on;
    xx=x+1;
    maxxx(xx) = max(max(w1{x+1}))
    %surf(w1{x+1} + w2{x+1});
xlabel('Position (X)')
ylabel('Position (Y)')
    surf(w1{x+1});
    surf(w2{x+1});
    surf(w3{x+1});
    view(90,90);
    
    %figure(x+10);hold on;
    %surf(w3{x+1} + w4{x+1} + w5{x+1} + w6{x+1});
%     surf(w3{x+1});
%     surf(w4{x+1});
%     surf(w5{x+1});
%     surf(w6{x+1});
    %view(90,90);
    
    www(numarbors-x) = w1{x+1}(15,15) + w2{x+1}(15,15) + w3{x+1}(15,15)
    if(feature==2||feature==4)    
        wxt(numarbors-x,:) = w1{x+1}(:,15)' + w2{x+1}(:,15)' + w3{x+1}(:,15)'
        wxtr(numarbors-x,:) = w4{x+1}(:,15)' + w5{x+1}(:,15)' + w6{x+1}(:,15)' + w7{x+1}(:,15)'
    else
        wxt(numarbors-x,:) = w1{x+1}(15,:)' + w2{x+1}(15,:)' + w3{x+1}(15,:)'
        wxtr(numarbors-x,:) = w4{x+1}(15,:)' + w5{x+1}(15,:)' + w6{x+1}(15,:)' + w7{x+1}(15,:)'
    end
end
save('~/matlabplay/weights.mat', 'wxt', 'wxtr')
figure;
surf(wtot);
view(90,90);
% 
% figure;
% plot([0:8],maxxx);

minw2=999;
for(i=[1:numarbors])
    if(min(min(w2{i}))<minw2)
        minw2=min(min(w2{i}));
    end
end
minw2
maxw1=-999;
for(i=[1:numarbors])
    if(max(max(w1{i}))>maxw1) maxw1=max(max(w1{i}));
    end
end
maxw1


% figure; hold on;
 %a=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a2.pvp');
 a=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a3.pvp');
% aa=squeeze(a(128,128,3,:));
% aaa=aa/(max(aa)-min(aa));
% aaaa=aaa-mean(aaa);
% plot(aaaa)
% hold on
% aa2=squeeze(a(128,118,3,:));
% aaa2=aa2/(max(aa2)-min(aa2));
% aaaa2=aaa2-mean(aaa2);
% plot(aaaa2, '*')
% 
% b=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a1.pvp');
% bb=squeeze(b(128,128,1,:));
% bbb=bb/(max(bb)-min(bb));
% bbbb=bbb-mean(bbb);
% plot(bbbb, 'g')
% bb2=squeeze(b(123,128,1,:));
% bbb2=bb2/(max(bb2)-min(bb2));
% bbbb2=bbb2-mean(bbb2);
% plot(bbbb2, '-g*')
% bb3=squeeze(b(118,128,1,:));
% bbb3=bb3/(max(bb3)-min(bb3));
% bbbb3=bbb3-mean(bbb3);
% plot(bbbb3, '-g.')
% 
% tt=[zeros(1,10), www, zeros(1,10)];
% ttt=tt/(max(tt)-min(tt));
% tttt=ttt-mean(ttt);
% plot(tttt, 'k')

%disp(['amp f 1 = ', num2str(max(squeeze(a(128,128,1,[10:30])))-min(squeeze(a(128,128,1,[10:30]))))])


[r,c]=size(wxt);
figure; surf([-floor(c/2):floor(c/2)], [-r+1:0],wxt)
view(0,90);
xlabel('Position (Y)')
ylabel('Time')
figure; surf([-floor(c/2):floor(c/2)], [-r+1:0],wxtr)
view(0,90);
xlabel('Position (Y)')
ylabel('Time')

b=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a2.pvp');
%b=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a1.pvp');
bb=double(squeeze(b(:,32,1,:)));
[r c] = size(bb);
for x=[1:c]
for y=[1:r]
hammw(x,y)=(0.54-0.46*cos(2*pi*(x/29)))*(0.54-0.46*cos(2*pi*(y/255)));
end
end
bbb=bb.*hammw';
NFFTx=2^nextpow2(r);
NFFTt=2^nextpow2(c);
Fx=r;
Ft=c;

FFTFXT = fft2(bbb,NFFTx,NFFTt);
ft=linspace(-Ft/2,Ft/2,NFFTt);
fx=linspace(-Fx/2,Fx/2,NFFTx);
figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
shiftedin=fftshift(FFTFXT);
%surf(ft,fx, abs(shifted))
%surf(ft, fliplr(fx), flipud(abs(shiftedin)))
surf(ft, fliplr(fx), (abs(shiftedin)))
view(90,-90);
figure;
surf(ft([16:32]),fx([1:128]), abs(shiftedin([1:128],[16:32])))


peakvalue = max(max(abs(shiftedin([1:128],[16:32]))));
[rr cc] = find(abs(shiftedin([1:128],[16:32]))==peakvalue);
wx = fx(rr)
wt = ft(cc+15)


figure;
surf(ft([1:15]),fx([129:256]), abs(shiftedin([129:256],[1:15])))
peakvalue = max(max(abs(shiftedin([129:256],[1:15]))));
[rr cc] = find(abs(shiftedin([129:256],[1:15]))==peakvalue);
wx2 = fx(rr+128)
wt2 = ft(cc)



% [r c] = size(wxt);
% NFFTx=2^nextpow2(r);
% NFFTt=2^nextpow2(c);
% Fx=c
% Ft=r

FFTFXT = fft2(wxt,NFFTx,NFFTt);
%ft=linspace(-Ft/2,Ft/2,NFFTt);
%fx=linspace(-Fx/2,Fx/2,NFFTx);
figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
shifted=fftshift(FFTFXT);
surf(ft, fliplr(fx), flipud(abs(shifted)))
line([-wt -wt2],[wx, wx2], [-1,-1], 'Color', 'g', 'LineWidth', 4)
line([wt wt2],[wx, wx2], [-1,-1], 'Color', 'b', 'LineWidth', 4)
line([wt -wt],[-wx, wx], [-1,-1], 'Color', 'r', 'LineWidth', 4)
line([-wt2 wt2],[wx2, -wx2], [-1,-1], 'Color', 'k', 'LineWidth', 4)
% line([-11.1290 11.1290],[-42.6667, 42.6667], [-1,-1], 'Color', 'c', 'LineWidth', 4)
% line([-40*0.2344, 40*0.2344],[40, -40], [-1,-1], 'Color', 'r', 'LineWidth', 4)
% line([-40 40],[-40*wt/wx, 40*wt/wx], [-1,-1], 'Color', 'k', 'LineWidth', 4)
% line([-9.194, 11.13],[27.61 -27.61], [-1,-1], 'Color', 'k', 'LineWidth', 4)
% line([wt, -10.16],[wx 43.67], [-1,-1], 'Color', 'k', 'LineWidth', 4)
view(90,-90);
xlim([-Ft/2, Ft/2])
ylabel('Spatial Frequency W_y')
xlabel('Temporal Frequency W_t')

FFTFXTr = fft2(wxtr,NFFTx,NFFTt);
figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
shiftedr=fftshift(FFTFXTr);
surf(ft, fliplr(fx), flipud(abs(shiftedr)))
view(90,-90);


figure; hold on
surf(ft, fliplr(fx), flipud(abs(shifted))/max(max(abs(shifted)))+(abs(shiftedin))/max(max(abs(shiftedin))))
view(90,-90);

bb2=double(squeeze(b(:,:,1,15)));
ffbb2=fft2(bb2,NFFTx,NFFTx);
figure; surf(fx, (fx), (abs(fftshift(ffbb2))))


ff=fft2(w1{1}+w2{1}+w3{1},NFFTx,NFFTx);
figure; surf(fx, (fx), (abs(fftshift(ff))))
figure; hold on

plot(squeeze(a(32,32,1,:)), 'b')
disp(['amp f 1 = ', num2str(max(squeeze(a(128,128,1,[15:30])))-min(squeeze(a(128,128,1,[15:30]))))])
% plot(squeeze(a(32,32,2,:)), 'g')
% disp(['amp f 2 = ', num2str(max(squeeze(a(128,128,2,[15:30])))-min(squeeze(a(128,128,2,[15:30]))))])
% plot(squeeze(a(32,32,3,:)), 'r')
% disp(['amp f 3 = ', num2str(max(squeeze(a(128,128,3,[15:30])))-min(squeeze(a(128,128,3,[15:30]))))])
% plot(squeeze(a(32,32,4,:)), 'k')
% disp(['amp f 4 = ', num2str(max(squeeze(a(128,128,4,[15:30])))-min(squeeze(a(128,128,4,[15:30]))))])

figure; hold on
%ar=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a3.pvp');
ar=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a4.pvp');
plot(squeeze(ar(32,32,1,:)), 'b')
disp(['amp f 1 = ', num2str(max(squeeze(ar(128,128,1,[15:30])))-min(squeeze(ar(128,128,1,[15:30]))))])
% plot(squeeze(ar(32,32,2,:)), 'g')
% disp(['amp f 2 = ', num2str(max(squeeze(ar(128,128,2,[15:30])))-min(squeeze(ar(128,128,2,[15:30]))))])
% plot(squeeze(ar(32,32,3,:)), 'r')
% disp(['amp f 3 = ', num2str(max(squeeze(ar(128,128,3,[15:30])))-min(squeeze(ar(128,128,3,[15:30]))))])
% plot(squeeze(ar(32,32,4,:)), 'k')
% disp(['amp f 4 = ', num2str(max(squeeze(ar(128,128,4,[15:30])))-min(squeeze(ar(128,128,4,[15:30]))))])

figure; hold on
aar=squeeze(ar(128,128,1,:));
aaar=aar-mean(aar);
%aaaar=2*aaar/(max(aaar)-min(aaar));
%aaaar=aaar/max(aaar);
aaaaar=aaar-mean(aaar);
plot(aaaaar, 'r')
aa=squeeze(a(128,128,1,:));
aaa=aa-mean(aa);
%aaaa=2*aaa/(max(aaa)-min(aaa));
%aaaa=aaa/2050;
%aaaa=aaa/max(aaar);
aaaaa=aaa-mean(aaa);
plot(aaaaa, 'b')
%plot(aaaaa.*aaaaa+aaaaar.*aaaaar, 'k')


figure; hold on
%acomp=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a4.pvp');
acomp=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a5.pvp');
plot(squeeze(acomp(32,32,1,:)), 'b')
title('complex cell 1');
disp(['amp f 1 = ', num2str(max(squeeze(acomp(128,128,1,[15:30])))-min(squeeze(acomp(128,128,1,[15:30]))))])
% figure; hold on
% acomp2=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a7.pvp');
% plot(squeeze(acomp2(32,32,1,:)), 'b')
% title('complex cell 2');
% disp(['amp f 1 = ', num2str(max(squeeze(acomp2(128,128,1,[15:30])))-min(squeeze(acomp2(128,128,1,[15:30]))))])
% figure; hold on
% acomp3=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a10.pvp');
% plot(squeeze(acomp3(32,32,1,:)), 'b')
% title('complex cell 3');
% disp(['amp f 1 = ', num2str(max(squeeze(acomp3(128,128,1,[15:30])))-min(squeeze(acomp3(128,128,1,[15:30]))))])
% figure; hold on
% acomp4=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a13.pvp');
% plot(squeeze(acomp4(32,32,1,:)), 'b')
% title('complex cell 4');
% disp(['amp f 1 = ', num2str(max(squeeze(acomp4(128,128,1,[15:30])))-min(squeeze(acomp4(128,128,1,[15:30]))))])
% figure; hold on
% amt=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a14.pvp');
% plot(squeeze(amt(32,32,1,:)), 'b')
% title('MT cell 1');
% disp(['amp f 1 = ', num2str(max(squeeze(amt(128,128,1,[15:30])))-min(squeeze(amt(128,128,1,[15:30]))))])
% figure; hold on
% amt=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a54.pvp');
% plot(squeeze(amt(32,32,1,:)), 'b')
% title('MT cell N');
% disp(['amp f 1 = ', num2str(max(squeeze(amt(128,128,1,[15:30])))-min(squeeze(amt(128,128,1,[15:30]))))])
% figure; hold on
% amt=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a39.pvp');
% plot(squeeze(amt(32,32,1,:)), 'b')
% title('MT cell E');
% disp(['amp f 1 = ', num2str(max(squeeze(amt(128,128,1,[15:30])))-min(squeeze(amt(128,128,1,[15:30]))))])
% figure; hold on
% amt=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a40.pvp');
% plot(squeeze(amt(32,32,1,:)), 'b')
% title('MT cell S');
% disp(['amp f 1 = ', num2str(max(squeeze(amt(128,128,1,[15:30])))-min(squeeze(amt(128,128,1,[15:30]))))])
% figure; hold on
% amt=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a45.pvp');
% plot(squeeze(amt(32,32,1,:)), 'b')
% title('MT cell NW');
% figure; hold on
% amt=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a44.pvp');
% plot(squeeze(amt(32,32,1,:)), 'b')
% title('MT cell SW');
% figure; hold on
% amt=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a43.pvp');
% plot(squeeze(amt(32,32,1,:)), 'b')
% title('MT cell SE');
% disp(['amp f 1 = ', num2str(max(squeeze(amt(128,128,1,[15:30])))-min(squeeze(amt(128,128,1,[15:30]))))])
% amt=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a42.pvp');
% figure; hold on
% plot(squeeze(amt(32,32,1,:)), 'b')
% title('MT cell NE');
% disp(['amp f 1 = ', num2str(max(squeeze(amt(128,128,1,[15:30])))-min(squeeze(amt(128,128,1,[15:30]))))])
% amt=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a41.pvp');
% figure; hold on
% plot(squeeze(amt(32,32,1,:)), 'b')
% title('MT cell W');
% disp(['amp f 1 = ', num2str(max(squeeze(amt(128,128,1,[15:30])))-min(squeeze(amt(128,128,1,[15:30]))))])
% plot(squeeze(acomp(32,32,2,:)), 'g')
% disp(['amp f 2 = ', num2str(max(squeeze(acomp(128,128,2,[15:30])))-min(squeeze(acomp(128,128,2,[15:30]))))])
% plot(squeeze(acomp(32,32,3,:)), 'r')
% disp(['amp f 3 = ', num2str(max(squeeze(acomp(128,128,3,[15:30])))-min(squeeze(acomp(128,128,3,[15:30]))))])
% plot(squeeze(acomp(32,32,4,:)), 'k')
% disp(['amp f 4 = ', num2str(max(squeeze(acomp(128,128,4,[15:30])))-min(squeeze(acomp(128,128,4,[15:30]))))])

% figure; hold on
% acompcos45=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a7.pvp');
% plot(squeeze(acompcos45(32,32,1,:)), 'b')
% disp(['amp f 1 = ', num2str(max(squeeze(acomp(128,128,1,[15:30])))-min(squeeze(acomp(128,128,1,[15:30]))))])
% % plot(squeeze(acompcos45(32,32,2,:)), 'g')
% % disp(['amp f 2 = ', num2str(max(squeeze(acomp(128,128,2,[15:30])))-min(squeeze(acomp(128,128,2,[15:30]))))])
% % plot(squeeze(acompcos45(32,32,3,:)), 'r')
% % disp(['amp f 3 = ', num2str(max(squeeze(acomp(128,128,3,[15:30])))-min(squeeze(acomp(128,128,3,[15:30]))))])
% % plot(squeeze(acompcos45(32,32,4,:)), 'k')
% % disp(['amp f 4 = ', num2str(max(squeeze(acomp(128,128,4,[15:30])))-min(squeeze(acomp(128,128,4,[15:30]))))])
% 
% figure; hold on
% mtact=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a8.pvp');
% plot(squeeze(mtact(32,32,1,:)), 'b')
% disp(['amp f 1 = ', num2str(max(squeeze(mtact(128,128,1,[15:30])))-min(squeeze(acomp(128,128,1,[15:30]))))])
% % plot(squeeze(mtact(32,32,2,:)), 'g')
% % disp(['amp f 2 = ', num2str(max(squeeze(acomp(128,128,2,[15:30])))-min(squeeze(acomp(128,128,2,[15:30]))))])
% % plot(squeeze(mtact(32,32,3,:)), 'r')
% % disp(['amp f 3 = ', num2str(max(squeeze(acomp(128,128,3,[15:30])))-min(squeeze(acomp(128,128,3,[15:30]))))])
% % plot(squeeze(mtact(32,32,4,:)), 'k')
% % disp(['amp f 4 = ', num2str(max(squeeze(acomp(128,128,4,[15:30])))-min(squeeze(acomp(128,128,4,[15:30]))))])
% mean(squeeze(mtact(32,32,1,[15:30])))



