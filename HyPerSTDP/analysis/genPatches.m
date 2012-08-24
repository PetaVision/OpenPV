clear all;

load('/Users/rcosta/Documents/workspace/HyPerSTDP/input/OlshausenField/IMAGES.mat');

nsamples = 1000; %Samples per image
s = 12; % n x n

xmax = size(IMAGES,1)-s-1;
ymax = size(IMAGES,2)-s-1;

for i=1:size(IMAGES,3)
    for j=1:nsamples
        x=round(rand()*xmax+1);
        y=round(rand()*ymax+1);
        imwrite(IMAGES(x:x+s,y:y+s, i), ["/Users/rcosta/Documents/workspace/HyPerSTDP/input/OlshausenField/whitened/12x12/" num2str(s) "x" num2str(s) "_i" num2str(i) "_s" num2str(j),".jpg"]);
    end
end