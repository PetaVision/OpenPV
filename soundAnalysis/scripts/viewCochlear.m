addpath("~/Desktop/PetaVision/trunk/mlab/util/");
pvpfile = "/Users/JEC/Desktop/PetaVision/sandbox/soundAnalysis/output/a1_Cochlear.pvp";
outfilename = "outImg.png";

[data, hdr] = readpvpfile(pvpfile);

outimg = zeros(hdr.nbands, hdr.nf);

for(time = 1:length(data))
   outimg(time, :) = squeeze(data{time}.values)';
end


outimgrescaled = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));

imwrite(outimgrescaled, outfilename);

