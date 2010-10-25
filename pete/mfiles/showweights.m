% A script m-file to display the weights in w0_last.pvp through
% w14_last.pvp, each in a separate figure.

weightfiledirectory='../output';
W = cell(15,1);
for conn = 0:14
    k = conn+1;
    W{k} = readweights(sprintf('%s/w%d_last.pvp',weightfiledirectory,k-1));
    figure(k);
    set(k,'position',[1+32*conn,35,1024,1024]);
    showoneweight(W{k});
end
