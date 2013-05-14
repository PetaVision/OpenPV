function [pSet, AUC] = doAnalysis(label,fileLoc,params)
    workspacePath   = fileLoc.workspacePath;
    layerFileName   = fileLoc.layerFileName;
    outputPath      = fileLoc.outputPath;
    outFileExt      = params.outFileExt;

    MOVIE_FLAG      = params.MOVIE_FLAG;

    GRAPH_FLAG      = params.GRAPH_FLAG;
    if GRAPH_FLAG
        graphSpec   = params.graphSpec.*(params.displayPeriod/params.dt);
        numHistBins = params.numHistBins;
    end

    WEIGHTS_FLAG    = params.WEIGHTS_FLAG;
    if WEIGHTS_FLAG
        connFileName = fileLoc.connFileName;
    end

    addpath([workspacePath,'PetaVision/mlab/util/']);

    if ne(exist(outputPath,'dir'),7)
        mkdir(outputPath);
    end

    [layerData layerHDR] = readpvpfile(layerFileName);
    if ~exist('layerData','var') || ~exist('layerHDR','var')
        print('bidsAnalysis: readpvpfile error on layer data.')
        keyboard
    end
    
    if WEIGHTS_FLAG
        [connData connHDR] = readpvpfile(connFileName);
        if ~exist('connData','var') || ~exist('connHDR','var')
            print('bidsAnalysis: readpvpfile error on conn data.')
            keyboard
        end
        numConnFrames = length(connData)-1; %First element is steady state
        assert(isequal(connData{2}.values,connData{numConnFrames}.values)); %The weights should not change over time
    end

    numFrames = layerHDR.nbands-1; %First band is steady state - does not factor into calculations
    N         = layerHDR.nx * layerHDR.ny * layerHDR.nf;

    if GRAPH_FLAG
        noStimLength = graphSpec(2) - graphSpec(1);
        stimLength   = graphSpec(4) - graphSpec(3);
        if (lt(noStimLength,0) || lt(stimLength,0) || ne(noStimLength,stimLength))
            print('bidsAnalysis: Graph spec not properly formatted!');
            keyboard
        end
        integratedHalf1 = zeros([layerHDR.nyGlobal,layerHDR.nxGlobal]);
        integratedHalf0 = zeros([layerHDR.nyGlobal,layerHDR.nxGlobal]);
    end

    times      = zeros(numFrames,1); 
    spikeCount = zeros(numFrames,1);
    frame      = 0;
    for frameIdx = 2:numFrames
        if ne(uint64(layerData{frameIdx}.time/params.dt),uint64(frame)) % Make sure we are on the right time step
            disp('doAnalysis: ERROR: layerData{frameIdx}.time != frame')
            keyboard
        end

        spikeCount(frameIdx) = length(find(layerData{frameIdx}.values(:)));

        times(frameIdx)      = squeeze(layerData{frameIdx}.time/params.dt);

        activeIdx            = squeeze(layerData{frameIdx}.values);
        vecMat               = full(sparse(activeIdx+1,1,1,N,1,N)); %%Column vector. PetaVision increments in order: nf, nx, ny
        rsMat                = reshape(vecMat,layerHDR.nf,layerHDR.nx,layerHDR.ny);
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

        numSpikesNoStim = sum(spikeCount(graphSpec(1):graphSpec(2)));
        numSpikesStim   = sum(spikeCount(graphSpec(3):graphSpec(4)));

        disp(['numSpikesNoStim = ',num2str(numSpikesNoStim)])
        disp(['noStimRate = ',num2str(numSpikesNoStim/(noStimLength/1000)/params.numBIDSNodes)])
        disp(['numSpikesStim = ',num2str(numSpikesStim)])
        disp(['StimRate = ',num2str(numSpikesStim/(stimLength/1000)/params.numBIDSNodes)])
        disp(['----'])

        pSet = zeros(2,numHistBins+2);
        mask = ones([layerHDR.nyGlobal,layerHDR.nxGlobal]); %In case you want to histogram over a certain window

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

            Pf = [0 fliplr(1-cumsum(h0)) 1]; %We want an ascending count, not descending
            Pd = [0 fliplr(1-cumsum(h1)) 1];
        end

        AUC = trapz(Pf,Pd);

        pSet(1,:) = Pf;
        pSet(2,:) = Pd;

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
        hxLabel = xlabel('Number of Spikes');
        hyLabel = ylabel('Normalized Number of Nodes');
        hTitle = title(['Histogram Plot for BIDS Nodes - ',label]);
        set(gca, ...
            'FontName'  ,'Helvetica', ...
            'FontSize'  ,20, ...
            'TickDir'   ,'out', ...
            'XMinorTick','on', ...
            'YMinorTick','on');
        set([hxLabel, hyLabel, hTitle], ...
            'FontName','AvantGarde');
        set([hxLabel, hyLabel], ...
            'FontSize',20);
        set(hTitle, ...
            'FontSize',20, ...
            'FontWeight','bold');
        print([figPath,label,'_','Hist.png'])
        
    end%GRAPH_FLAG

    if MOVIE_FLAG
        system(['ffmpeg -r 12 -f image2 -i ',instMoviePath,'%04d.png -qscale 0 -y ',moviePath,'pvp_instantaneous_movie.mp4']);
        %system(['rm -rf ',inst_movie_path]);
    end
    return 
end%function
