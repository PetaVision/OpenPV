%Function for plotting STDP weights

setenv("GNUTERM", "x11");

debug_on_error(0);
close all; more off;
addpath('/Users/rcosta/Documents/workspace/HyPerSTDP/mlab/')

%output_path = '/Users/rcosta/Documents/workspace/HyPerSTDP/output/orient_simple/';
output_path = '/Users/rcosta/Documents/workspace/HyPerSTDP/output/orient_36r/';
%output_path = '/Users/rcosta/Documents/workspace/HyPerSTDP/output/orient_1r/';

%names = {'w4_last.pvp', 'w5_last.pvp'} %pre point of view
names = {'w4_post.pvp'} %post point of view
%names = {'Last/S1InhtoS1_W.pvp'}
%names = {'Last/ImagetoRetinaONCenter_W.pvp'}
timestep = 100;
post = 1;

for i=1:length(names)
    name = names{i};

    filename = [output_path, name];
    plot_on = 0;

    %1. Read file
    [data hdr wm]=readpvpfile(filename, output_path, name, post);

    %2. Plot changes in weights over frames
    if(post)
        nx=hdr.nx; %sqrt(size(data{f}.values{1},4));
        ny=hdr.ny; %sqrt(size(data{f}.values{1},4));
    else
        nx=hdr.nx+hdr.nb*2; %sqrt(size(data{f}.values{1},4));
        ny=hdr.ny+hdr.nb*2; %sqrt(size(data{f}.values{1},4));
    end

    colors = {};
    figure
    c=1;
    deltaw_total = zeros(1,size(data,1)-1);
        for x=1:nx
          for y=1:ny
            deltaw = zeros(1,size(data,1)-1);
            for f=2:size(data,1)
                deltaw(f-1) = mean(mean(abs(data{f-1}.values{1,1}(:,:,1,c)-data{f}.values{1,1}(:,:,1,c))));
            end
            c=c+1;
            deltaw_total = deltaw_total+deltaw;
            plot((2:size(data,1)).*timestep, deltaw);
            hold on;
          end
        end
        plot((2:size(data,1)).*timestep, deltaw_total./(nx*ny),'-r','LineWidth',2);
        ylabel('\Deltaw');
        xlabel('time (ms)');

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
