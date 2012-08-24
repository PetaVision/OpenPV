d = 10
max = 360;
im = imread('/Users/rcosta/Documents/workspace/HyPerSTDP/input/0deg.png');
size = [12 12];

for r=0:d:max    
    temp = imresize(imrotate(im,r,'nearest','crop'), size);
    figure; imshow(temp);
    imwrite(temp,["/Users/rcosta/Documents/workspace/HyPerSTDP/input/orient/r_" num2str(r) "_deg.png"],'png');
end