close all
clear all
clc
cm_per_inch = 1/0.3937;

% for (i=[1:256])
%     for (n=[1:256])
%         if(n<129)
%             a(i,n)=double(1);
%         else
%             a(i,n)=double(-1);
%         end
%     end
% end

stripecnt=8;
width=256/stripecnt;
black=ones(256, width);
white=zeros(256, width);

FigH = figure('Position', [1 1 256 256], 'Clipping', 'off' );
data=[];
for(i=[1:stripecnt/2])
    data = [data, black, white];
end

h=pcolor(data);
colormap(gray(2))
shading flat
axis off
axis tight
box on


%outputfile = '/Users/krispeterson/Documents/workspace/kris/input/1stvertlinedetectorplay/halfblacknwhite.png';
outputfile = '/Users/kpeterson/Documents/presentations/ibmfigs/halfblacknwhite.png';
imwrite(data, outputfile);


