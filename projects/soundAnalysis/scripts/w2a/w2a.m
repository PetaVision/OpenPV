%%  Mohit Dubey
%%  converts A1 weights to activity that can be read by a movie layer
%%

addpath("~/Desktop/newvision/trunk/mlab/util/");
inputpvpfile = "/Users/JEC/Desktop/newvision/sandbox/soundAnalysis/output/checkpoints/Checkpoint1323/A1ToCochlearError_W.pvp";


W = readpvpfile(inputpvpfile);

NF = size(W{1}.values{1,1,1},4);
numdelays = size(W{1}.values{1,1,1},3);


for(m = 1:NF)
    
    for(n = 1:numdelays)

        A{n}.time = .00023 * (n - 1); %% W{1}.time ?
        A{n}.values = W{1}.values{1,1,1}(:,:,n,m);

    end


    outputfilename = sprintf("feature%d.pvp",m);
    
    writepvpactivityfile(outputfilename, A);

end
    
    


