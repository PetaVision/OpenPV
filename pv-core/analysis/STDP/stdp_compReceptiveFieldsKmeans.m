function A = stdp_compReceptiveFieldsKmeans(fname, numCenters, xScale, yScale)
% cluster the "weight" fields for all pre-synaptic layers
% connected to the same post-synaptic layer; all weight patches
% get concatenated and klustering is then performed.
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as arguments
% NOTE: Needs to be implemented using ~/Documents/MATLAB/Kmeans


global input_dir  output_dir NX NY 


NXscaled = NX * xScale; % L1 size
NYscaled = NY * yScale;


scaleWeights = 1;
write_centers =0;
plot_centers = 1;

debug = 0;



%% open file pointers & figure handles
for f=1:numel(fname)

    filename = fname{f};
    filename = [input_dir, filename];

    if exist(filename,'file')
        fid{f} = fopen(filename, 'r', 'native');
    else
        disp(['Skipping, could not open ', filename]);
        return
    end
    
end


%% open output file pointers

if(write_centers)
    centers_file = [output_dir,'ReceptiveFieldsCenters',num2str(numCenters),'.dat'];
    fid_centers = fopen(centers_file,'w');
end

 
 
%% read file headers

for f=1:numel(fname)
    
    [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid{f});
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
        
    if numPatches ~= NXscaled*NYscaled
        disp('mismatch between numPatches and NX*NY: return');
        return
    end
    patch_size = NXP*NYP;
    
end

%% To insert computing Gabor filters here
        

%% read the last weights field (configuration)
W_array = []; % N x (m*patch_size) array where N = NX * NY
              % and m is the number of pre-synaptic layers 
    
      
% loop over weights file
for f=1:numel(fname)

    k = 0;
    for j=1:NYscaled
        for i=1:NXscaled
            if ~feof(fid{f})
                k=k+1;
                nx = fread(fid{f}, 1, 'uint16'); % unsigned short
                ny = fread(fid{f}, 1, 'uint16'); % unsigned short
                nItems = nx*ny*NFP;
                if debug
                    fprintf('k = %d nx = %d ny = %d nItems = %d: ',...
                        k,nx,ny,nItems);
                end

                w = fread(fid{f}, nItems, 'uchar'); % unsigned char
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
                    if f==1
                        W_array(k,:) = w(1:patch_size);
                    else
                        W_array(k,:) = W_array(k,:) - w(1:patch_size)';
                    end
                    %pause
                end
            end % if ~ feof
        end
    end % loop over post-synaptic neurons

    fclose(fid{f});
    
end % loop over pre-synaptic files

%% compute K-means

% data is n x (m*patch_size) where n = NX * NY
[centers,mincenter,mindist,q2,quality] = kmeans(W_array,numCenters);
size(centers)  % numCenters x (m*patch_size)
%size(mincenter)% n x 1
%q2
%quality

% compute cluster weights
for k=1:numCenters
    clustW(k) = sum(find(mincenter == k));
end
clustW = clustW ./sum(clustW);

% normalize centers
for k=1:numCenters
    centers(k,:) = centers(k,:) ./ norm(centers(k,:));
end

% sort centers according to weight

[sortW,sortI] = sort(clustW,'descend');

%% plot centers in reverse order of their weights
numColumns = 4; % for each pre-synaptic layer
% ex, ON and OFF pre-synaptic layers
numRows = ceil(numCenters/numColumns);

if(plot_centers)
    figure('Name',['Receptive Fields K-means Centers: time ' num2str(time)]);

    for k=1:numCenters
        fprintf('cluster %d w = %f\n',k,sortW(k));
        for f=1:numel(fname)
            patch = reshape(centers(sortI(k),:),[4 4])';
            subplot(numRows,numColumns,k);
            colormap gray
            imagesc(patch); % 'CDataMapping','direct'
            title_str = ['Center ' num2str(k) ' w = ' num2str(clustW(sortI(k)),2) ];
            title(title_str);
            colorbar
            axis square
            axis off
        end
    end

end


if(write_centers)
    fprintf(fid_centers,'\n');
end


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

