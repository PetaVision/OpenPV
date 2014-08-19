%%  Mohit Dubey
%%  plots nnz of a1 over time
%%

addpath("~/newvision/trunk/mlab/util/");
inputpvpfile = "~/newvision/sandbox/soundAnalysis/servers/output/a6_A1.pvp";

a1 = readpvpfile(inputpvpfile);

time = numel(a1)

output = zeros(time,1);

numel(a1{7}.values)

for(k = 1:time)

    output(k) = numel(a1{k}.values);

end

output(7)

plot(output);

print("sparsity.png");