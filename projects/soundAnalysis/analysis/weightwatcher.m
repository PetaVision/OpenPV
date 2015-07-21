addpath("~/newvision/trunk/mlab/util/");
pvpfile = "/Users/MLD/newvision/sandbox/soundAnalysis/output/a6_A1.pvp";
outfilename = "weightwatcher.png";

data = readpvpfile(pvpfile);

length = numel(data);

output = zeros(length,1);

for(k = 1:length)

    output(k) = data{k}.values(2,2);

end

plot(output);

print("weightwatcher.png");

wavwrite(output, "weightwatcher.wav");