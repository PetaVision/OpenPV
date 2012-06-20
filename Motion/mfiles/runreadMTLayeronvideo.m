close all
clear all
clc

firstpvp=2;
showframe=36; %25-14; %36;
dnoff=3;
cbar0ff=12;
compl0ff=13;
mt0ff=14;
%basepath='/Users/kpeterson/Documents/workdir/';
basepath='/Volumes/Fantom2/new results/TowerTraining/';
%basepath='/Volumes/Fantom2/new results/TowerNew/';
strt=340; %12

for(clipcnt=[50:50])
    %actpath='/Users/kpeterson/Documents/workdir/026/026_000/a';
    %actpath='/Users/kpeterson/Documents/workdir/050/050_000/a';
    actpath=[basepath, num2str(clipcnt,'%3.3d'),'/', num2str(clipcnt,'%3.3d'), '_000/a'];
    
    
    %input time domain response:
    inActFile=[actpath, num2str(firstpvp), '.pvp'];
    inActMat=readpvpfile(inActFile);
    
    time=length(inActMat);
    %time=358
    [ysize xsize fsize]=size(inActMat{1}.values);
    inAct=zeros(time, ysize, xsize);
    
    for(t=[1:time])
        inAct(t,[1:ysize],[1:xsize])=squeeze(inActMat{t}.values(:,:,1));
    end
    clear inActMat;
    
    %MT output:
    outActMTFile=[actpath, num2str(firstpvp+21), '.pvp']
    outActMTMat=readpvpfile(outActMTFile);
    
    
    [t1 t2 fsize]=size(outActMTMat{t}.values(:,:,:));
    outActMT=zeros(time, ysize, xsize, fsize);
    
    for(feat=[1:fsize])
        for(t=[1:time])
            outActMT(t,[1:ysize],[1:xsize],feat)=squeeze(outActMTMat{t}.values(:,:,feat));
        end
    end
    clear outActMTMat;
    outActMT=(outActMT>0).*outActMT;
    
    %MT2 output:
    outActMT2File=[actpath, num2str(firstpvp+22), '.pvp']
    outActMT2Mat=readpvpfile(outActMT2File);
    
    
    [t1 t2 fsize]=size(outActMT2Mat{t}.values(:,:,:));
    outActMT2=zeros(time, ysize, xsize, fsize);
    
    for(feat=[1:fsize])
        for(t=[1:time])
            outActMT2(t,[1:ysize],[1:xsize],feat)=squeeze(outActMT2Mat{t}.values(:,:,feat));
        end
    end
    clear outActMT2Mat;
    outActMT2=(outActMT2>0).*outActMT2;
    
    dn=permute(inAct,[2, 3, 4, 1]);
    clear inAct;
    mt1=permute(outActMT,[2, 3, 4, 1]);
    clear outActMT;
    mt2=permute(outActMT2,[2, 3, 4, 1]);
    clear outActMT2;
    
    
    
    outpath=[basepath, num2str(clipcnt,'%3.3d'),'/', num2str(clipcnt,'%3.3d'), '_000/mt1/'];
    mkdir(outpath);
    for(t=[strt:time-mt0ff])
        readMTLayer(dn, mt1, [], t, outpath);
    end
    
    outpath=[basepath, num2str(clipcnt,'%3.3d'),'/', num2str(clipcnt,'%3.3d'), '_000/mt1amt2/'];
    mkdir(outpath);
    for(t=[strt:time-mt0ff])
        readMTLayer(dn, mt1, mt2, t, outpath);
    end
    
end