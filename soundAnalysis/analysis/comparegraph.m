addpath("~/newvision/PetaVision/mlab/util/");
pvpfile = "/Users/MLD/newvision/sandbox/soundAnalysis/servers/output2/a1_NewCochlear.pvp";


[data, hdr] = readpvpfile(pvpfile,1000,30000,1); %% filename, displayperiod, end frame, start frame

outimg = zeros(hdr.nbands - 1 , hdr.nx);



for(time = 1:length(data) - 1)

    outimg(time, :) = squeeze(data{time}.values)';

end

cochlea = outimg;


%%cochleagraph = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));



pvpfile = "/Users/MLD/newvision/sandbox/soundAnalysis/servers/output2/a9_FullRecon.pvp";


[data, hdr] = readpvpfile(pvpfile,1000,30000,1);


t = hdr.nbands

numfreqs = hdr.nx


outimg = zeros(hdr.nbands, hdr.nx);


%%size(squeeze(data{10}.values))

for(time = 1:length(data))

        outimg(time, :) = squeeze(data{time}.values(:,1,1))';

end

recon = outimg;


%%%%%RECONSTUFF

%%recongraph = (outimg - min(outimg(:))) / (max(outimg(:)) - min(outimg(:)));


%%outimgrescaled = [cochleagraph recongraph];

%%outfilename = "comparegraph.png"

%%imwrite(outimgrescaled, outfilename);


%%%ERRORSTUFF

error = zeros(29999,1);



for(time = 2:30000)

    error(time,1) = 10 * (std(recon(time,:)) / std(cochlea(time,:)));

end

plot(error);

print("error.png");

wavwrite(error, 4410, "error.wav");
