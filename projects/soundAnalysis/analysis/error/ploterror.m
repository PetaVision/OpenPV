
addpath("~/newvision/trunk/mlab/util/");

errorpvp = "/Users/MLD/newvision/sandbox/soundAnalysis/servers/output/a5_NegativeError.pvp";
cochlearpvp = "/Users/MLD/newvision/sandbox/soundAnalysis/servers/output/a3_NegativeCochlear.pvp";

error = readpvpfile(errorpvp);
cochlear = readpvpfile(cochlearpvp);

numfreqs = size(error{1}.values,1)
numdelays = size(error{1}.values,3)
lengthofdata = size(error,1)

numerator = zeros(lengthofdata,1);
denominator = zeros(lengthofdata,1);
positive = zeros(lengthofdata,1);
negative = zeros(lengthofdata,1);
output = zeros(lengthofdata,1);

innercochlearsum = zeros(lengthofdata,1);


for(numframes = 1:lengthofdata)

    for(freq = 1:numfreqs)

        innercochlearsum(numframes) = innercochlearsum(numframes) + (cochlear{numframes}.values(freq))^2;

    end

end


for(frameno = 1:lengthofdata)

    innererrorsum = zeros(numdelays,1);

    %% compute numerator

    for(delay = 1:numdelays)

        for(freq = 1:numfreqs)

            innererrorsum(delay) = innererrorsum(delay) + (error{frameno}.values(freq,1,delay))^2;

        end

    numerator(frameno) = numerator(frameno) + innererrorsum(delay);

    end

    %% compute denominator


    if (frameno > numdelays)

        denominator(frameno) = sum(innercochlearsum((frameno - numdelays):frameno));

    end

positive(frameno) = sqrt(numerator(frameno) / denominator(frameno));

end



errorpvp = "/Users/MLD/newvision/sandbox/soundAnalysis/servers/output/a4_PositiveError.pvp";
cochlearpvp = "/Users/MLD/newvision/sandbox/soundAnalysis/servers/output/a2_PositiveCochlear.pvp";

error = readpvpfile(errorpvp);
cochlear = readpvpfile(cochlearpvp);

numfreqs = size(error{1}.values,1);
numdelays = size(error{1}.values,3);
lengthofdata = size(error,1)

numerator = zeros(lengthofdata,1);
denominator = zeros(lengthofdata,1);
positive = zeros(lengthofdata,1);

innercochlearsum = zeros(lengthofdata,1);


for(numframes = 1:lengthofdata)

    for(freq = 1:numfreqs)

        innercochlearsum(numframes) = innercochlearsum(numframes) + (cochlear{numframes}.values(freq))^2;

    end

end


for(frameno = 1:lengthofdata)

    innererrorsum = zeros(numdelays,1);

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

negative(frameno) = sqrt(numerator(frameno) / denominator(frameno));

output(frameno) = sqrt(positive(frameno) + negative(frameno));

end



plot(output);

print("error.png");

wavwrite(output,4410,"error.wav");


