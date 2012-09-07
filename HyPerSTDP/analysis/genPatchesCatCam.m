clear all;

path = '/Users/rcosta/Documents/workspace/HyPerSTDP/input/catcam/1/orig/';
im_list = dir(path);


s = 32; % n x n

for i=3:size(im_list,1)
    img = imread([path im_list(i).name]);
    
    xmax = size(img,1)-s-1;
    ymax = size(img,2)-s-1;
    
    x = round(xmax/2-s/2); %center
    y = round(ymax/2-s/2); %center
    
    imwrite(img(x:x+(s-1),y:y+(s-1)), ["/Users/rcosta/Documents/workspace/HyPerSTDP/input/catcam/1/32x32/" num2str(i-2) "_" num2str(s) "x" num2str(s) "_i1.jpg"]);
end