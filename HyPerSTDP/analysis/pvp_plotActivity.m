%Function for plotting activity from the STDP experiment

debug_on_error(0);
clear all; close all; more off;
addpath('/Users/rcosta/Documents/workspace/HyPerSTDP/mlab/')

output_path = '/Users/rcosta/Documents/workspace/HyPerSTDP/output/orient_simple/';
  
%filename = {'S1Inh.pvp', 'S1.pvp', 'RetinaON.pvp'};
%filename = {'S1.pvp'};
filename = {'RetinaON.pvp'};
plot_on = 0;

%1. Read file
for i=1:length(filename)
    path = [output_path, filename{i}];
    [data hdr] = readpvpfile(path, output_path, filename{i});
end


% for f=1:size(data,1)
%     nx=sqrt(size(data{f}.values{1},4));
%     ny=sqrt(size(data{f}.values{1},4));
%     
% 	wm = zeros(hdr.nxp*nx, hdr.nyp*ny);
% 	c = 1;
% 	for x=0:(nx-1)
% 	    for y=0:(ny-1)
%             wm(x*hdr.nxp+1:x*hdr.nxp+hdr.nxp, y*hdr.nyp+1:y*hdr.nyp+hdr.nyp) = reshape(data{f}.values{1}(:,:,1,c), hdr.nxp, hdr.nyp);
%             c=c+1;
% 	    end%for
% 	end%for
% 
%     if(plot_on)
%         fig=figure;
%         imagesc(wm);
%         colormap(gray);
%         title(['STDP Weights, t=' num2str(data{f}.time) 'ms']);
%         colorbar
%         pause
%         close(fig);
%     end    
% end