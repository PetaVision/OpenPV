function [newData,newHdr] = preWeightsToPost(data,hdr)

    disp('preWeightsToPost: Rearranging weight matrix to be from post-synaptic perspective.')
    newHdr = hdr; %no changes here

    numTimeSteps = length(data{1})-1; %% PetaVision will return first time step as steady state, have to subtract 1
    if numTimeSteps == 0 %%No steady state if a single time step
        numTimeSteps = 1;
    end
    [procsX procsY numArbors] = size(data{numTimeSteps}.values);
    [patchLocSizeX patchLocSizeY numFeatures numPatches] = size(data{numTimeSteps}.values{procsX, procsY, numArbors});

    assert(procsX==hdr.nxprocs);
    assert(procsY==hdr.nyprocs);
    assert(numArbors==hdr.nbands);
    assert(patchLocSizeX==hdr.nxp);
    assert(patchLocSizeY==hdr.nyp);
    assert(numFeatures==hdr.nf);
    numHdrPatches = (hdr.nx+2*hdr.nb)*(hdr.ny+2*hdr.nb);
    assert(numPatches==numHdrPatches);

    newDataX = hdr.nxp / hdr.postScaleX;
    newDataY = hdr.nyp / hdr.postScaleY;
    newDataF = hdr.nfp;
    newDataP = hdr.nx * hdr.ny * hdr.postScaleX * hdr.postScaleY * hdr.nf;

    newData = cell(numTimeSteps,1);

    for timeStep = 1:numTimeSteps
        newData{timeStep} = struct('time',hdr.time,'values',[]);
        newData{timeStep}.time = hdr.time;

        Q = cell(procsX,procsY,numArbors);
        for arbor = 1:numArbors
            for procY = 1:procsY
                for procX = 1:procsX
                    %Loop through pre-synaptic neurons (quad nested: nxp, nyp, nf, numPatches) 
                    %For each index in loop, add value to post-synaptic matrix (as long as you nest post-synaptic in the same way as pre-synaptic you should be fine)
                    Z = zeros(newDataX,newDataY,newDataF,newDataP);
                    newXIdx = 1;
                    newYIdx = 1;
                    newFIdx = 1;
                    newPIdx = 1;
                    for xIdx = 1:patchLocSizeX
                        newXIdx += 1;
                        if newXIdx > newDataX
                            newXIdx = 1;
                        end
                        for yIdx = 1:patchLocSizeY
                            newYIdx += 1;
                            if newYIdx > newDataY
                                newYIdx = 1;
                            end
                            for fIdx = 1:numFeatures
                                newFIdx += 1;
                                if newFIdx > newDataF
                                    newFIdx = 1;
                                end
                                for pIdx = 1:numPatches
                                    newPIdx += 1;
                                    if newPIdx > newDataP
                                        newPIdx = 1;
                                    end
                                    Z(newXIdx,newYIdx,newFIdx,newPIdx) = data{timeStep}.values{procX,procY,arbor}(xIdx,yIdx,fIdx,pIdx);
                                end
                            end
                        end
                    end
                    Q(procX,procY,arbor) = Z;
                end
            end
        end
        newData{timeStep}.values = Q;
    end
end
