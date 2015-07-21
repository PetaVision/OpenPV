%%  Mohit Dubey
%%  plots nnz of a1 over time
%%

addpath("~/newvision/PetaVision/mlab/util/");
inputpvpfile = "~/newvision/sandbox/soundAnalysis/tau20ms/a6_A1.pvp";

a1 = readpvpfile(inputpvpfile);

time = numel(a1)
nf = numel(a1{1}.values(1,1,:))

output = zeros(time,1);
sparse = zeros(nf,1);

size(a1{1}.values)

for(k = 1:time)

    output(k) = nnz(a1{k}.values)/nf;

    for (j = 1:nf)

        sparse(j) = sparse(j) + a1{k}.values(1,1,j);

    end

end


plot(output);
xlabel("timesteps");
ylabel("%sparse");
print("sparsity.png");

plot(sparse);
xlabel("features")
ylabel("activity")
print("sparserank.png");


[newsparse, oldsparse] = sort(sparse,1,"ascend");

dlmwrite('sparserank.txt',oldsparse,"-append");



