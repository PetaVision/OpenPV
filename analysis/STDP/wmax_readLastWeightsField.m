function [W_array, NXP, NYP] = wmax_readLastWeigthsField(fname_last, xScale, yScale)
% read the (last) weight patches for each neuron and compute
% the learning score of a set of pre-defined features
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as arguments
% NOTE: Needs to be implemented using ~/Documents/MATLAB/Kmeans


global input_dir  NX NY 


NXscaled = NX * xScale; % L1 size
NYscaled = NY * yScale;

scaleWeights = 1;

debug = 0;


%% open file pointers & figure handles

filename = [input_dir, fname_last];

fprintf('read weights from %s\n',filename);

if exist(filename,'file')
    fid = fopen(filename, 'r', 'native');
else
    disp(['Skipping, could not open ', filename]);
    return
end
 
W_array = []; % N x patch_size array where N = NX * NY 

%% read headers



[time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
    readFirstHeader(fid);
fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
    time,numPatches,NXP,NYP,NFP);


if numPatches ~= NXscaled*NYscaled
    fprintf('numPatches = %d NX = %d NY = %d NX*NY = %d\n',...
        numPatches, NXscales,NYscaled,NXscaled*NYscaled);
    disp('mismatch between numPatches and NX*NY: return');
    return
end
patch_size = NXP*NYP;
    
        
%% read the weights field (configuration)


k = 0;

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
                %pause
            end
            if(~isempty(w) & nItems ~= 0)
                w = w ./ norm(w); % normalize weights
                W_array(k,:) = w(1:patch_size);
                
                %pause
            end
        end % if ~ feof
    end
end % loop over post-synaptic neurons


%% close files
fclose(fid);


% End primary function
%


function [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
    readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
head = fread(fid,3,'int');
if head(3) ~= 3
    disp('incorrect file type')
    return
end
numParams = head(2)-8;
fseek(fid,0,'bof'); % rewind file

params = fread(fid, numParams, 'int')
%pause
NX         = params(4);
NY         = params(5);
NF         = params(6);
fprintf('numParams = %d ',numParams);
fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
% read time
time = fread(fid,1,'float64');
fprintf('time = %f\n',time);

wgtParams = fread(fid,3,'int');
NXP = wgtParams(1);
NYP = wgtParams(2);
NFP = wgtParams(3);

rangeParams = fread(fid,2,'float');
minVal      = rangeParams(1);
maxVal      = rangeParams(2);
    
numPatches  = fread(fid,1,'int');


fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
    minVal,maxVal,numPatches);
%pause

% End subfunction
%


function [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
    readHeader(fid,numParams)

% NOTE: see analysis/python/PVReadWeights.py for reading params

if ~feof(fid)
    params = fread(fid, numParams, 'int')
    if numel(params)
        NXP         = params(4);
        NYP         = params(5);
        NFP         = params(6);
        fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
        % read time
        time = fread(fid,1,'float64');
        fprintf('time = %f\n',time);

        wgtParams = fread(fid,3,'int');
        NXP = wgtParams(1);
        NYP = wgtParams(2);
        NFP = wgtParams(3);
        
        
        rangeParams = fread(fid,2,'float');
        minVal      = rangeParams(1);
        maxVal      = rangeParams(2);
    
        numPatches  = fread(fid,1,'int');
    

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

function F = compFeatures(NXP, NYP)

% these are projections for 4x4 receptive fields
% in the Bars images experiments; it includes both
% vertical and horizontal features

F = zeros(NXP*NYP,8);
V = zeros(1,NXP*NYP);

% vertical features
for p=1:4
    V(:) = 0;
    ind = p:4:16;
    V(ind) = 1;
    V = V ./ norm(V);
    %patch = reshape(V,[NXP NYP])';
    %figure(p)
    %imagesc(patch,'CDataMapping','direct');
    F(:,p) = V'; % when V gets reshaped and transposed
                    % we get the right feature
    %pause
end

% horizontal features

for p=1:4
    V(:) = 0;
    V( (p-1)*4 +1 : p*4) = 1;
    V = V ./ norm(V);
    %patch = reshape(V,[NXP NYP])';
    %figure(p)
    %imagesc(patch,'CDataMapping','direct');
    F(:,4+p) = V'; % when V gets reshaped and transposed
                    % we get the right feature
    %pause
end


% End subfunction
%