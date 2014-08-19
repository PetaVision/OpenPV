addpath("~/newvision/trunk/mlab/util/");
pvpfile = "/Users/MLD/newvision/sandbox/soundAnalysis/rectifiedweights/a4_WeightRecon.pvp";
outfilename = "weightgraph.png";

[data, hdr] = readpvpfile(pvpfile);


outimg = zeros(hdr.nbands, hdr.nx);


size(squeeze(data{16}.values))

for(time = 1:length(data))

outimg(time, :) = squeeze(data{time}.values(:,1,1))';

end


outimgrescaled = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));

imwrite(outimgrescaled, outfilename);