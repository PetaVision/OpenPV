

list = dir('/Users/rcosta/Documents/workspace/HyPerSTDP/input/OlshausenField/whitened/12x12/');

inc_img = zeros(13,13);
for i=4:length(list)
    im = imread(['/Users/rcosta/Documents/workspace/HyPerSTDP/input/OlshausenField/whitened/12x12/' list(i).name]);
    inc_img = inc_img + double(im);
end

imagesc(inc_img)