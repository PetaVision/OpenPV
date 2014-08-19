addpath("~/newvision/trunk/mlab/util/");
pvpfile = "/Users/MLD/newvision/sandbox/soundAnalysis/output/a1_NewCochlear.pvp";


[data, hdr] = readpvpfile(pvpfile);

outimg = zeros(hdr.nbands, hdr.nx);



for(time = 1:length(data))
outimg(time, :) = squeeze(data{time}.values)';
end


cochleagraph = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));



pvpfile = "/Users/MLD/newvision/sandbox/soundAnalysis/output/a9_FullRecon.pvp";


[data, hdr] = readpvpfile(pvpfile);


outimg = zeros(hdr.nbands, hdr.nx);


size(squeeze(data{16}.values))

for(time = 1:length(data))

outimg(time, :) = squeeze(data{time}.values(:,1,1))';

end


recongraph = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));

outimgrescaled = [cochleagraph recongraph];

outfilename = "comparegraph.png"

imwrite(outimgrescaled, outfilename);