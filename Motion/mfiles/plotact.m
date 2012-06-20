close all
clear all
clc

c=readpvpfile('~/Documents/workspace/kris/output/krisplay/L1ON_A_last.pvp');

for (i=[1:5])
    
    figure(i)
    %surfc([1:256], [1:256], double(squeeze(c(:,:,1,i))))
    surfc(c{1}.values(:,:, i))
end