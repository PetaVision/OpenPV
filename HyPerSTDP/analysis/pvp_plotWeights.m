%Function for plotting STDP weights

debug_on_error(0);
close all; more off;
addpath('/Users/rcosta/Documents/workspace/HyPerSTDP/mlab/')

output_path = '/Users/rcosta/Documents/workspace/HyPerSTDP/output/';
  
%names = {'w4_last.pvp', 'w5_last.pvp'} %pre point of view
names = {'w4_post.pvp'} %post point of view
post = 1;

for i=1:length(names)
    name = names{i};

    filename = [output_path, name];
    plot_on = 0;

    %1. Read file
    [data hdr]=readpvpfile(filename, output_path, name, post);


    if(plot_on)
     for f=1:size(data,1)
        nx=floor(sqrt(size(data{f}.values{1},4)));
        ny=floor(sqrt(size(data{f}.values{1},4)));
    
        wm = zeros(hdr.nxp*nx, hdr.nyp*ny);
        c = 1;
        for x=0:(nx-1)
            for y=0:(ny-1)
                wm(x*hdr.nxp+1:x*hdr.nxp+hdr.nxp, y*hdr.nyp+1:y*hdr.nyp+hdr.nyp) = reshape(data{f}.values{1}(:,:,1,c), hdr.nxp, hdr.nyp);
                c=c+1;
            end%for
        end%for


        fig=figure;
        imagesc(wm);
        colormap(gray);
        title(['STDP Weights, t=' num2str(data{f}.time) 'ms']);
        colorbar
        pause
        close(fig);
     end    
    end
end
