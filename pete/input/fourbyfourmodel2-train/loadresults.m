%outputdir = '../../output/fourbyfourmodel2-train/';
outputdir = '../../output/debugging/';
a1 = readpvpfile([outputdir 'a1_Retina.pvp']);
R = zeros(numel(a1{1}.values),numel(a1));
for k=1:numel(a1), R(:,k) = a1{k}.values(:); end
clear a1
disp('retina loaded into variable R');

a3 = readpvpfile([outputdir 'a3_Layer A.pvp']);
layerA = zeros(numel(a3{1}.values),numel(a3));
for k=1:numel(a3), layerA(:,k) = a3{k}.values(:); end
clear a3
disp('A layer loaded into variable layerA');

a8 = readpvpfile([outputdir 'a8_Layer B.pvp']);
layerB = zeros(numel(a8{1}.values),numel(a8));
for k=1:numel(a8), layerB(:,k) = a8{k}.values(:); end
clear a8
disp('B layer loaded into variable layerB');

W2 = readpvpfile([outputdir 'w2.pvp']);
wgtA = zeros(8,16,size(R,2));
wgtupdateperiod = size(R,2)/numel(W2);
for k=1:numel(W2)
    wgtA(:,:,(k-1)*wgtupdateperiod+(1:wgtupdateperiod)) = ...
            repmat(squeeze(W2{k}.values{1}),[1 1 wgtupdateperiod]);
end
clear W2
disp('Weights A loaded into variable wgtA');

W10 = readpvpfile([outputdir 'w10.pvp']);
wgtB = zeros(2,8,size(layerA,2));
wgtupdateperiod = size(layerA,2)/numel(W10);
for k=1:numel(W10)
    wgtB(:,:,(k-1)*wgtupdateperiod+(1:wgtupdateperiod)) = ...
            repmat(squeeze(W10{k}.values{1}),[1 1 wgtupdateperiod]);
end
clear W10
disp('Weights B loaded into variable wgtB');

fid = fopen([outputdir 'TotalEnergy.txt']);
E = fscanf(fid, 'time = %f, column = %f\n', inf);
E = reshape(E,2,numel(E)/2)';
fclose(fid);
clear fid
disp('Total Energy loaded into variable E');

clear k wgtupdateperiod outputdir
