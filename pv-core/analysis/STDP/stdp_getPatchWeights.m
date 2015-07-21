function patch = stdp_getPatchWeights(fname, xScale, yScale, kxPost, kyPost)
% plot "weights" (typically after turning on just one neuron)
% Xtarg and Ytarg contain the X and Y coordinates of the target
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as argumrnts

global input_dir  NX NY 

filename = fname;
filename = [input_dir, filename]
    

NXscaled = NX * xScale; % L1 size
NYscaled = NY * yScale;

%fprintf('scalex NX = %d scaled NY = %d\n',NX,NY);

debug = 0;

scaleWeights = 1;


if exist(filename,'file')
    
    
    fprintf('read weights from file %s\n',filename);
    
    fid = fopen(filename, 'r', 'native');

    [time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid);
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
    if numPatches ~= NXscaled*NYscaled
        disp('mismatch between numPatches and NX*NY')
        return
    end
    patch_size = NXP*NYP;
    
       
    % read  weights

    k=0;

    for j=1:NYscaled
        for i=1:NXscaled
            if ~feof(fid)
                k=k+1;
                nx = fread(fid, 1, 'uint16'); % unsigned short
                ny = fread(fid, 1, 'uint16'); % unsigned short
                nItems = nx*ny*NFP;
                if debug
                    fprintf('k = %d nx = %d ny = %d nItems = %d: ',...
                        k,nx,ny,nItems);
                end

                w = fread(fid, nItems, 'uchar'); % unsigned char
                % scale weights: they are quantized before written
                if scaleWeights
                    w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                end
                if debug
                    for r=1:patch_size
                        fprintf('%d ',w(r));
                    end
                    fprintf('\n');
                    pause
                end
                if(i==kxPost & j==kyPost & ~isempty(w) )
                    patch = reshape(w(1:patch_size),[NXP NYP])';
                    fclose(fid);
                    return
                end
            end % if ~ feof
        end
    end % loop over post-synaptic neurons

else

    disp(['Skipping, could not open ', filename]);

end

% End primary function
%


function [time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
    readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
%fprintf('read first header\n');
head = fread(fid,3,'int');
if head(3) ~= 3
    disp('incorrect file type')
    return
end
numWgtParams = 6;
numParams = head(2)-8;
fseek(fid,0,'bof'); % rewind file

params = fread(fid, numParams, 'int');
%pause
NX         = params(4);
NY         = params(5);
NF         = params(6);
fprintf('numParams = %d ',numParams);
fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
% read time
time = fread(fid,1,'float64');
fprintf('time = %f\n',time);

wgtParams = fread(fid,numWgtParams,'int');
NXP = wgtParams(1);
NYP = wgtParams(2);
NFP = wgtParams(3);
minVal      = wgtParams(4);
maxVal      = wgtParams(5);
numPatches  = wgtParams(6);
fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
    minVal,maxVal,numPatches);
%pause

% End subfunction
%


%function [time,varargout] = readHeader(fid,numParams,numWgtParams)
function [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = readHeader(fid,numParams,numWgtParams)



% NOTE: see analysis/python/PVReadWeights.py for reading params

if ~feof(fid)
    params = fread(fid, numParams, 'int')
    if numel(params)
        NX         = params(4);
        NY         = params(5);
        NF         = params(6);
        fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
        % read time
        time = fread(fid,1,'float64');
        fprintf('time = %f\n',time);

        wgtParams = fread(fid,numWgtParams,'int');
        NXP = wgtParams(1);
        NYP = wgtParams(2);
        NFP = wgtParams(3);
        minVal      = wgtParams(4);
        maxVal      = wgtParams(5);
        numPatches  = wgtParams(6);
        fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
        fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
            minVal,maxVal,numPatches);

        %varargout{1} = numPatches;
        %varargout{2} = NXP;
        %varargout{3} = NYP;
        %varargout{4} = NFP;
        %varargout{5} = minVal;
        %varargout{6} = maxVal;
        %pause
    else
        disp('eof found: return');
        time = -1;
        NXP = 0;
        NYP = 0;
        NFP = 0;
        minVal      = 0;
        maxVal      = 0;
        numPatches  = 0;
    end
else
    disp('eof found: return');
    time = -1;
    NXP = 0;
    NYP = 0;
    NFP = 0;
    minVal      = 0;
    maxVal      = 0;
    numPatches  = 0;
end
% End subfunction
%



