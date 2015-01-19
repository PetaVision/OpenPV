addpath("/Users/MLD/newvision/PetaVision/mlab/util/");
pluspvpfile = "/Users/MLD/newvision/sandbox/soundAnalysis/newupdate/checkpoints/Checkpoint77350/A1ToPositiveError_W.pvp"
minuspvpfile = "/Users/MLD/newvision/sandbox/soundAnalysis/newupdate/checkpoints/Checkpoint77350/A1ToNegativeError_W.pvp"

plusWcell = readpvpfile(pluspvpfile);
minusWcell = readpvpfile(minuspvpfile);

NF = size(plusWcell{1}.values{1,1,1},4);
numdelays = size(plusWcell{1}.values{1,1,1},3);
numfreqs = size(plusWcell{1}.values{1,1,1},1)


plusW = zeros(numfreqs,1,numdelays,NF);
minusW = zeros(numfreqs,1,numdelays,NF);

for(k = 1:numel(plusWcell))

    plusW(:,:,:,:) = plusWcell{1}.values{1,1,1}(:,:,:,:);

end

for(k = 1:numel(minusWcell))

    minusW(:,:,:,:) = minusWcell{1}.values{1,1,1}(:,:,:,:);

end

W = plusW - minusW;

%%%%%%%%%%%%%%%%%sparse stuff

%%apvpfile = "~/newvision/sandbox/soundAnalysis/biglebowski32nf/a6_A1.pvp";

%%a1 = readpvpfile(apvpfile);

%%time = numel(a1)
%%nf = numel(a1{1}.values(1,1,:))

%%output = zeros(time,1);
%%sparse = zeros(nf,1);

%%size(a1{1}.values)

  %%  for(k = 1:time)

    %%    output(k) = nnz(a1{k}.values)/nf;

      %%  for (j = 1:nf)

        %%    sparse(j) = sparse(j) + ((a1{k}.values(1,1,j) > 0) / time);

    %%end

%%end


%%[newsparse, oldsparse] = sort(sparse,1,"descend");

%%dlmwrite('sparserank.txt',oldsparse);

%%bar(newsparse);
%%xlabel("weights")
%%ylabel("activity")
%%print("sparsesorted.png");


%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for(feature = 1:NF)

    weight = zeros(numfreqs,numdelays);

    for(time = 1:numdelays)

        weight(:,time) = W(:,:,time,feature);

    end

    %%weightrescaled = (weight - min(weight(:))) / (max(weight(:)) - min(weight(:)));

    weightrescaled = 127.5 + 127.5 * (weight / max(abs(weight(:))));

    subplot(ceil(sqrt(NF)),ceil(sqrt(NF)),feature);
    imagesc(weightrescaled);
    %%xlabel(feature, 'FontSize', 4);
    %%title( feature, 'FontSize', 6 )
    axis off;
    %%colormap(gray);

    [value, location] = max(abs(weightrescaled(:)));

    [R,C] = ind2sub(size(weightrescaled),location);



    dlmwrite('freqs.txt',R,"-append");

end

%%print -dpng soundbyte.png;


print("soundbyte.png","-dpng")

