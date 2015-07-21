addpath("~/Desktop/newvision/trunk/mlab/util/");
pvpfile = "/Users/JEC/Desktop/newvision/sandbox/soundAnalysis/output/a2_WeightRecon.pvp";
outfilename = "outImg.png";

[data, hdr] = readpvpfile(pvpfile);

outimg = zeros(hdr.nbands, hdr.nx);

for(time = 1:length(data))
   outimg(time, :) = squeeze(data{time}.values)';
end


outimgrescaled = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));

imwrite(outimgrescaled, outfilename);