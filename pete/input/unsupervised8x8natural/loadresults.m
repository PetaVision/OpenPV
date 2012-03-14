outputdir = '../../output/unsupervised8x8natural';
a1 = readpvpfile(sprintf('%s/a1_Retina.pvp',outputdir),1000);
retina = zeros(numel(a1{1}.values),numel(a1));
for k=1:numel(a1), retina(:,k)=a1{k}.values(:); end
clear a1
a3 = readpvpfile(sprintf('%s/a3_V1.pvp',outputdir),1000);
v1 = zeros(numel(a3{1}.values),numel(a3));
for k=1:numel(a3), v1(:,k) = a3{k}.values(:); end
clear a3
W = readpvpfile(sprintf('%s/w2.pvp',outputdir),100);
weights = zeros([size(squeeze(W{1}.values{1})) numel(W)]);
for k=1:numel(W), weights(:,:,k) = squeeze(W{k}.values{1}); end
clear W
fid = fopen('../../output/unsupervised8x8natural/TotalEnergy.txt');
E = fscanf(fid,'time = %f, column = %f\n',inf);
E = reshape(E,2,numel(E)/2)';
fclose(fid);
clear fid;