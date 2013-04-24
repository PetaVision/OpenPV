% reads and plots the patches.

pv_dir = '/nh/home/manghel/petavision/workspace/pv-craig/';
out_dir = [pv_dir 'src/output/'];
    
weights_file = [out_dir 'weights.tif'];

    
NX = 8;
NY = 8;
NO = 8;    % number of orientations/features

for i=1:NO
    pixels = imread(weights_file,i);
    if ~isempty(pixels)
        fprintf('patch %d:\n',i);
        subplot(2,4,i);
        imagesc(pixels);% plots patch as an image
        colormap(gray);
    else
        fprintf('empty patch: end of weights file\n');
        break;
    end
    
end