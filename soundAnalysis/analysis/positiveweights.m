addpath("~/newvision/trunk/mlab/util/");
pluspvpfile = "~/newvision/sandbox/soundAnalysis/servers/output/Checkpoint59670/A1ToPositiveError_W.pvp"


plusWcell = readpvpfile(pluspvpfile);

NF = size(plusWcell{1}.values{1,1,1},4);
numdelays = size(plusWcell{1}.values{1,1,1},3);
numfreqs = size(plusWcell{1}.values{1,1,1},1)



minusW = zeros(numfreqs,1,numdelays,NF);


for(k = 1:numel(plusWcell))

plusW(:,:,:,:) = plusWcell{1}.values{1,1,1}(:,:,:,:);

end

W = plusW;






for(feature = 1:NF)

weight = zeros(numdelays,numfreqs);

for(time = 1:numdelays)

weight(time, :) = W(:,:,time,feature);

end

weightrescaled = (weight - min(weight(:))) / (max(weight(:)) - min(weight(:)));

subplot(ceil(sqrt(NF)),ceil(sqrt(NF)),feature);
imagesc(weightrescaled);
axis off;
colormap(gray);

[value, location] = max(abs(weightrescaled(:)));

[R,C] = ind2sub(size(weightrescaled),location);


C

dlmwrite('positivefreqs.txt',C,"-append");

end

print -dpng positiveweights.png

