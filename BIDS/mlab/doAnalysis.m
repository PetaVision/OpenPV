function [pSet, AUC] = doAnalysis(label,fileLoc,params)
    workspacePath   = fileLoc.workspacePath;
    inputFileName   = fileLoc.inputFileName;
    outputPath      = fileLoc.outputPath;
    outFileExt      = params.outFileExt;

    MOVIE_FLAG      = params.MOVIE_FLAG;

    GRAPH_FLAG      = params.GRAPH_FLAG;
    if GRAPH_FLAG
        graphSpec   = params.graphSpec;
        numHistBins = params.numHistBins;

    addpath([workspacePath,'PetaVision/mlab/util/']);

    if ne(exist(outputPath,'dir'),7)
        mkdir(outputPath);
    end

    [data hdr] = readpvpfile(inputFileName);
    if ~exist('data','var') || ~exist('hdr','var')
        print('bidsAnalysis: readpvpfile error.')
        keyboard
    end

    numFrames = hdr.nbands-1; %First band is steady state - does not factor into calculations
    N         = hdr.nx * hdr.ny * hdr.nf;

    if GRAPH_FLAG
        noStimLength = graphSpec(2) - graphSpec(1);
        stimLength   = graphSpec(4) - graphSpec(3);
        if (lt(noStimLength,0) || lt(stimLength,0) || ne(noStimLength,stimLength))
            print('bidsAnalysis: Graph spec not properly formatted!');
            keyboard
        end
        halfLength = stimLength;
        clear stimLength;
        clear noStimLength;

        integratedHalf1 = zeros([hdr.nyGlobal,hdr.nxGlobal]);
        integratedHalf0 = zeros([hdr.nyGlobal,hdr.nxGlobal]);
    end

    times      = zeros(numFrames,1); 
    spikeCount = zeros(numFrames,1);
    frame      = 0;
    for frameIdx = 2:numFrames

        assert(data{frameIdx}.time==frame); % Make sure we are on the right time step

        spikeCount(frameIdx) = length(find(data{frameIdx}.values(:)));

        times(frameIdx)      = squeeze(data{frameIdx}.time);

        activeIdx            = squeeze(data{frameIdx}.values);
        vecMat               = full(sparse(activeIdx+1,1,1,N,1,N)); %%Column vector. PetaVision increments in order: nf, nx, ny
        rsMat                = reshape(vecMat,hdr.nf,hdr.nx,hdr.ny);
        fullMat              = permute(rsMat,[3 2 1]); %%Matrix is now [ny, nx, nf]

        if GRAPH_FLAG
            if (gt(frame,graphSpec(1)) && lt(frame,graphSpec(2)))
                integratedHalf0 = integratedHalf0 + fullMat;
            elseif (gt(frame,graphSpec(3)) && lt(frame,graphSpec(4)))
                integratedHalf1 = integratedHalf1 + fullMat;
            end
        end

        if MOVIE_FLAG
            moviePath = [outputPath,'/',label,'Movie/'];
            if ne(exist(moviePath,'dir'),7)
                mkdir(moviePath);
            end

            instMoviePath = [moviePath,'InstantaneousFrames/'];
            if ne(exist(instMoviePath,'dir'),7)
                mkdir(instMoviePath);
            end

            frameStr = sprintf('%04d',frame);
            movieFilename = [instMoviePath,frameStr,'.',outFileExt];
            try
                imwrite(fullMat,movieFilename,outFileExt)
            catch
                disp(['bidsAnalysis: WARNING. Could not print file: ',char(10),movieFilename])
            end
        end%if MOVIE_FLAG

        frame = frame + 1;
    end%frameIdx

    if GRAPH_FLAG
        figPath = [outputPath,'/',label,'Figures/'];
        if ne(exist(figPath,'dir'),7)
            mkdir(figPath);
        end

        numSpikesNoStim = sum(spikeCount(graphSpec(1):graphSpec(2)))/halfLength
        numSpikesStim   = sum(spikeCount(graphSpec(3):graphSpec(4)))/halfLength

        pSet = zeros(2,numHistBins);
        mask = ones([hdr.nyGlobal,hdr.nxGlobal]); %In case you want to histogram over a certain window

        [rows0 cols0 counts0] = find(integratedHalf0.*mask);
        [rows1 cols1 counts1] = find(integratedHalf1.*mask);

        if (eq(length(counts0),0) || eq(length(counts1),0))
            binLoc = 0;
            h0     = 0;
            h1     = 0;
            Pd     = 0;
            Pf     = 0;
        else
            totCounts = [counts1;counts0];

            [freqCounts, binLoc] = hist(totCounts,numHistBins);

            h0 = hist(counts0,binLoc,1);
            h1 = hist(counts1,binLoc,1);

            Pd = cumsum(h1);
            Pf = cumsum(h0);
        end

        pSet(1,:) = Pd;
        pSet(2,:) = Pf;

        AUC = trapz(Pd,Pf);

        %Histograms
        figure
        hold on
        fid0 = bar(binLoc,h0);
        fid1 = bar(binLoc,h1);
        hold off
        set(fid0,'facecolor',[0 0 1])
        set(fid0,'edgecolor',[0 0 1])
        set(fid1,'facecolor',[1 0 0])
        set(fid1,'edgecolor',[1 0 0])
        xlabel('Number of Spikes')
        ylabel('Normalized Number of Nodes')
        title(['Histogram Plot for BIDS Nodes - ',label])
        print([figPath,label,'_','Hist.png'])
        
    end%GRAPH_FLAG

    if MOVIE_FLAG
        system(['ffmpeg -r 12 -f image2 -i ',instMoviePath,'%04d.png -qscale 0 -y ',moviePath,'pvp_instantaneous_movie.mp4']);
        %system(['rm -rf ',inst_movie_path]);
    end
    return 
end%function
