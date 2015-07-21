setenv("GNUTERM","X11");
addpath("~/Desktop/newvision/trunk/mlab/util/");

errorpvp = "/Users/JEC/Desktop/newvision/sandbox/soundAnalysis/output/a4_PositiveError.pvp";
cochlearpvp = errorpvp = "/Users/JEC/Desktop/newvision/sandbox/soundAnalysis/output/a2_PositiveCochlear.pvp";

error = readpvpfile(errorpvp);
cochlear = readpvpfile(cochlearpvp);

numfreqs = size(error{1}.values,1);
numdelays = size(error{1}.values,3);
lengthofdata = size(error,1)

numerator = zeros(lengthofdata,1);
denominator = zeros(lengthofdata,1);
output = zeros(lengthofdata,1);

innercochlearsum = zeros(lengthofdata,1);


for(numframes = 1:lengthofdata)

    for(freq = 1:numfreqs)

        innercochlearsum(numframes) = innercochlearsum(numframes) + (cochlear{numframes}.values(freq))^2;

    end

end


for(frameno = 1:lengthofdata)

    innererrorsum = zeros(numfreqs,1);

    %% compute numerator

    for(delay = 1:numdelays)

        for(freq = 1:numfreqs)

            innererrorsum(delay) = innererrorsum(delay) + (error{frameno}.values(freq,delay))^2;

        end

        numerator(frameno) = numerator(frameno) + innererrorsum(delay);

    end

    %% compute denominator


    if (frameno > numdelays)

        denominator(frameno) = sum(innercochlearsum((frameno - numdelays):frameno));

    end

    output(frameno) = numerator(frameno) / denominator(frameno);

end


plot(output);

print("outerror.png");

replot;











