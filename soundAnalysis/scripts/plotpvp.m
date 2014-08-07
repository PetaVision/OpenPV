setenv("GNUTERM","X11");

addpath("~/Desktop/newvision/trunk/mlab/util/");
pvpfile = "/Users/JEC/Desktop/newvision/sandbox/soundAnalysis/output/a1_NewCochlear.pvp";

sound = readpvpfile(pvpfile);

output = zeros(101,1);

for(time = 4000:4100)
    output(time - 3999) = sound {time}.values;
end

plot(output);

print("outplot.png");
replot;