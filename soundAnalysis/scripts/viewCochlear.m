addpath("~/workspace/PetaVision/mlab/util/");
pvpfile = "/Users/Burger/workspace/soundAnalysis/output/a1_Cochlear.pvp";
outfilename = "outImg.png";

[data, hdr] = readpvpfile(pvpfile);

outimg = zeros(hdr.nbands, hdr.nf);

for(time = 1:length(data))
   outimg(time, :) = squeeze(data{time}.values)';
end

outimg = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));
imwrite(outimg, outfilename);

