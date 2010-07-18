function [PATCH, patch_size, NXP, NYP] = roc_readPatch(fname, I, J, NXscaled, NYscaled)
% plot patch for neuron I, J 
global input_dir

fprintf('read patch I = %d J = %d from %s\n',I,J,fname);

filename = fname;
filename = [input_dir, filename];

fid=fopen(filename,'r','native');

scaleWeights = 1;

[time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
    readHeader(fid);
fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d ',...
    time,numPatches,NXP,NYP,NFP);
fprintf('minVal = %f maxVal = %d\n',minVal,maxVal);

if numPatches ~= NXscaled*NYscaled
    disp('mismatch between numPatches and NX*NY')
    return
end

    
patch_size = NXP*NYP;   

    
while (~feof(fid))
        
    for j=1:NYscaled
        for i=1:NXscaled
            if ~feof(fid)
                nxp = fread(fid, 1, 'uint16'); % unsigned short
                nyp = fread(fid, 1, 'uint16'); % unsigned short
                nItems = nxp*nyp*NFP;
                w = fread(fid, nItems, 'uchar'); % unsigned char

                if(i==I & j==J & ~isempty(w))
                    %fprintf('nxp = %d nyp = %d : ',nxp,nyp);
                    % scale weights: they are quantized before written
                    if scaleWeights
                        w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                    end
                    w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                    %A = reshape(w(1:patch_size), [NXP NYP]);
                    PATCH =  w(1:patch_size);
                    return
                end
            end % if ~ feof
        end
    end % loop over post-synaptic neurons

end

% End primary function
%


function [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
    %fprintf('read first header\n');
    head = fread(fid,3,'int');
    if head(3) ~= 3
       disp('incorrect file type')
       return
    end
    numParams = head(2)-8;
    fseek(fid,0,'bof'); % rewind file
    
    params = fread(fid, numParams, 'int'); 
    %pause
    NX         = params(4);
    NY         = params(5);
    NF         = params(6);
    %fprintf('numParams = %d ',numParams);
    %fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
    % read time
    time = fread(fid,1,'float64');
    %fprintf('time = %f\n',time);
    
    wgtParams = fread(fid,3,'int');
    NXP = wgtParams(1);
    NYP = wgtParams(2);
    NFP = wgtParams(3);
    
    rangeParams = fread(fid,2,'float');
    minVal      = rangeParams(1);
    maxVal      = rangeParams(2);
    
    numPatches  = fread(fid,1,'int');
    
    %fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
    %fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
    %    minVal,maxVal,numPatches);
    %pause
    
% End subfunction 
%
    
