addpath("~/newvision/trunk/mlab/util/");
pvpfile = "/Users/MLD/newvision/sandbox/soundAnalysis/servers/output/a1_NewCochlear.pvp";
outfilename = "cochleagraph.png";

[data, hdr] = readpvpfile(pvpfile);

outimg = zeros(hdr.nbands, hdr.nx);



for(time = 1:length(data))
    outimg(time, :) = squeeze(data{time}.values)';
end


outimgrescaled = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));


imwrite(outimgrescaled, outfilename);