setenv("GNUTERM","X11");

addpath("~/Desktop/newvision/trunk/mlab/util/");
pvpfile = "/Users/JEC/Desktop/newvision/sandbox/soundAnalysis/output/a5_inverseCochlear.pvp";

sound = readpvpfile(pvpfile);

output = zeros(1000,1);

for(time = 1:1000)
    output(time) = sound {time}.values;
end

plot(output);

print("outplot.png");
replot;