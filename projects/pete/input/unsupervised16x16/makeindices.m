seed = 17;
N = 25000;
seedrng(17);
r = randi([0,31],[N,1]);
fid = fopen('filenames.txt','w');
for k=1:N
    fprintf(fid,'input/unsupervised16x16/imagespace/image%02d.png\n',r(k));
end
fclose(fid);