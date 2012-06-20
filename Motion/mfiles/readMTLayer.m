function [mask] = readMTLayer(downsampledinput, mt1, mt2, time, path)

% downsampledinput=readactivities([path,'a2.pvp']);
% mt1=readactivities([path, 'a', num2str(12), '.pvp']);
% mt2=readactivities([path, 'a', num2str(13), '.pvp']);
% readMTLayer(downsampledinput, mt1, mt2, frame)

% mt1=permute(outActComplexBar,[2, 3, 4, 1]);
% size(mt1)
% dn=permute(inAct,[2, 3, 4, 1]);
% size(dn)
% readMTLayer(dn, mt1, [], showframe, './')

mt1pvpnum=12; %27;
dnoff=3;
mt0ff=14;
threshold=0; %0.005;
startFile=0;

% downsampledinput=readactivities([path,'a2.pvp']);
% mt1=readactivities([path, 'a', num2str(12), '.pvp']);
% mt2=readactivities([path, 'a', num2str(13), '.pvp']);

[xsize, ysize, fsize, tsize] = size(downsampledinput);

% mtdiff1=mt1(:,:,3,[25:tsize])-mt1(:,:,1,[25:tsize]);
% mtdiff2=mt1(:,:,4,[25:tsize])-mt1(:,:,2,[25:tsize]);
% if(~isempty(mt2))
%     mtdiff3=mt2(:,:,3,[25:tsize])-mt2(:,:,1,[25:tsize]);
%     mtdiff4=mt2(:,:,4,[25:tsize])-mt2(:,:,2,[25:tsize]);
% end
% 
% globalMeanMTDiff1=mean(mean(mean(mtdiff1)));
% globalMeanMTDiff2=mean(mean(mean(mtdiff2)));
% if(~isempty(mt2))
%     globalMeanMTDiff3=mean(mean(mean(mtdiff3)));
%     globalMeanMTDiff4=mean(mean(mean(mtdiff4)));
% end
% 
imageAtTime=squeeze(flipud(rot90(downsampledinput(:,:,1,time+dnoff))));
% 
% 
% 
% 
% mtdiffAtTime1=mt1(:,:,3,time+mt0ff)-mt1(:,:,1,time+mt0ff);
% minMTDiff1=min(min(mtdiffAtTime1));
% mtdiffAtTime1=mtdiffAtTime1-minMTDiff1;
% 
% 
% mtdiffAtTime2=mt1(:,:,4,time+mt0ff)-mt1(:,:,2,time+mt0ff);
% minMTDiff2=min(min(mtdiffAtTime2));
% mtdiffAtTime2=mtdiffAtTime2-minMTDiff2;
% 
% if(~isempty(mt2))
%     mtdiffAtTime3=mt2(:,:,3,time+mt0ff)-mt2(:,:,1,time+mt0ff);
%     minMTDiff3=min(min(mtdiffAtTime3));
%     mtdiffAtTime3=mtdiffAtTime3-minMTDiff3;
%     
%     mtdiffAtTime4=mt2(:,:,4,time+mt0ff)-mt2(:,:,2,time+mt0ff);
%     minMTDiff4=min(min(mtdiffAtTime4));
%     mtdiffAtTime4=mtdiffAtTime4-minMTDiff4;
% end
% thresh1=globalMeanMTDiff1-minMTDiff1-threshold;
% thresh2=globalMeanMTDiff2-minMTDiff2-threshold;
% if(~isempty(mt2))
%     thresh3=globalMeanMTDiff3-minMTDiff3-threshold;
%     thresh4=globalMeanMTDiff4-minMTDiff4-threshold;
% end

mask=zeros(size(imageAtTime));

mask1=mt1(:,:,1,time+mt0ff)>threshold;
mask = mask + flipud(rot90(mask1(:,:)));
mask2=mt1(:,:,2,time+mt0ff)>threshold;
mask = mask + flipud(rot90(mask2(:,:)));
mask3=mt1(:,:,3,time+mt0ff)>threshold;
mask = mask + flipud(rot90(mask3(:,:)));
mask4=mt1(:,:,4,time+mt0ff)>threshold;
mask = mask + flipud(rot90(mask4(:,:)));
if(~isempty(mt2))
    mask1=mt2(:,:,1,time+mt0ff)>threshold;
    mask = mask + flipud(rot90(mask1(:,:)));
    mask2=mt2(:,:,2,time+mt0ff)>threshold;
    mask = mask + flipud(rot90(mask2(:,:)));
    mask3=mt2(:,:,3,time+mt0ff)>threshold;
    mask = mask + flipud(rot90(mask3(:,:)));
    mask4=mt2(:,:,4,time+mt0ff)>threshold;
    mask = mask + flipud(rot90(mask4(:,:)));
end
mask = mask>0;
% mask1=mtdiffAtTime1<thresh1;
% mask = mask + flipud(rot90(mask1(:,:)));
% 
% mask2=mtdiffAtTime2<thresh2;
% mask = mask + flipud(rot90(mask2(:,:)));
% 
% if(~isempty(mt2))
%     mask3=mtdiffAtTime3<thresh3;
%     mask = mask + flipud(rot90(mask3(:,:)));
%     
%     mask4=mtdiffAtTime4<thresh4;
%     mask = mask + flipud(rot90(mask4(:,:)));
% end
invmask = 1-mask;

im=uint8((imageAtTime/max(max(imageAtTime)))*128);
[x, y] = size(im);

finalimage=zeros(x,y,3);
finalimage(:,:,1)=im.*uint8(invmask) + uint8(mask*255);
finalimage(:,:,2)=im.*uint8(invmask);
finalimage(:,:,3)=im.*uint8(invmask);
finalimage=(double(finalimage)/255);

figure(1); imagesc(finalimage);
%figure; imagesc(finalimage);

%size(mask)
upscaledimage=zeros(4*size(mask));
[xsize ysize]=size(upscaledimage);

for(x=[1:xsize])
    for(y=[1:ysize])
        upscaledimage(x,y)=mask(ceil(x/4),ceil(y/4));
        %upscaledimage(x,y,2)=mask(ceil(x/4),ceil(y/4),2);
        %upscaledimage(x,y,3)=mask(ceil(x/4),ceil(y/4),3);
    end
end
%figure(2); imagesc(upscaledimage);
filename=[path, 'PM-', num2str(startFile+time,'%5.5d'), '.png']
imwrite(upscaledimage, filename, 'png', 'bitdepth', 2);



mask=upscaledimage;




