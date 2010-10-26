function h = showoneweight(W)
% h = showoneweight(W)
%
% Plots the weights loaded from readweights.m
% W is the output of a call to readweights.m
% h is the function handle of the figure
%
% The figure created has m by n subplots, where
% m is the number of features and n is the number
% of kernel patches.

setenv('GNUTERM','x11);
nfeatures = size(W,3);
npatches = size(W,4);
for p = 1:npatches
    for f=1:nfeatures
        h = subplot(nfeatures, npatches, (p-1)*nfeatures+f);
        imagesc(W(:,:,f,p),[0 255]);
        a = axis;
        axis equal;
        axis(a);
        colormap(gray);
        title(sprintf('feature %d, patch %d',f,p));
    end%for f
end%for p

h = get(h,'Parent');
