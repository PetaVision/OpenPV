setenv("GNUTERM","X11");

addpath("~/Desktop/newvision/trunk/mlab/util/");
pvpfile = "/Users/JEC/Desktop/newvision/sandbox/soundAnalysis/output/a5_inverseWeights.pvp";

sound = readpvpfile(pvpfile);

output = zeros(218,1);

for(time = 1:218)
    output(time) = sound{time}.values;
end

plot(output);

print("outweight.png");
replot;