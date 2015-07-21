%% generate imgs for an expanding square

out_dir = '../input/';

out_pic = ones([256 256])*128;

num_scales = 10;
cone_x     = 129;
cone_y     = 129;

for i = 1:num_scales

    out_pic(cone_x-i+1:cone_x+i,cone_y-i+1:cone_y+i) = 256;
    imwrite(uint8(out_pic),[out_dir,'whitespot',num2str(2*i),'x',num2str(2*i),'.png'])

    out_pic(cone_x-i+1:cone_x+i,cone_y-i+1:cone_y+i) = 0;
    imwrite(uint8(out_pic),[out_dir,'blackspot',num2str(2*i),'x',num2str(2*i),'.png'])

    out_pic(:,:) = 128;
end

