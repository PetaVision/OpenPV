setenv("GNUTERM","X11");

addpath("~/Desktop/newvision/trunk/mlab/util/");
pvpfile = "/Users/JEC/Desktop/newvision/sandbox/soundAnalysis/output/a10_inverseCochlear.pvp";

sound = readpvpfile(pvpfile);

output = zeros(1324,1);

for(time = 1:1324)
    output(time) = sound {time}.values;
end

plot(output);

print("outplot.png");
replot;