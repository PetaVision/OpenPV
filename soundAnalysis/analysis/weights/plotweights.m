

addpath("~/newvision/trunk/mlab/util/");
pvpfile = "~/newvision/sandbox/soundAnalysis/rectifiedweights/a5_inverseWeights.pvp";

sound = readpvpfile(pvpfile);

output = zeros(218,1);

for(time = 1:218)
    output(time) = sound{time}.values;
end

plot(output);

print("outweight.png");

wavwrite(output,4410,"outweight.wav");