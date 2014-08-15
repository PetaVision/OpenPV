addpath("~/Desktop/newvision/trunk/mlab/util/");
pvpfile = "/Users/JEC/Desktop/newvision/sandbox/soundAnalysis/output/a1_NewCochlear.pvp";
outfilename = "cochleagraph.png";

[data, hdr] = readpvpfile(pvpfile);

outimg = zeros(hdr.nbands, hdr.nx);



for(time = 1:length(data))
outimg(time, :) = squeeze(data{time}.values)';
end


outimgrescaled = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));

imwrite(outimgrescaled, outfilename);