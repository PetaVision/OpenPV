d = 10
max = 360;
im = imread('/Users/rcosta/Documents/workspace/HyPerSTDP/input/0deg.png');

for r=0:d:max    
    temp=imrotate(im,r,'nearest','crop');
    figure; imshow(temp);
    imwrite(temp,["/Users/rcosta/Documents/workspace/HyPerSTDP/input/orient/r_" num2str(r) "_deg.png"],'png');
end