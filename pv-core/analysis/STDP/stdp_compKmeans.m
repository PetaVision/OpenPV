function A = stdp_compKmeans(fname, numClusters, xScale, yScale)
% plot last configuration of "weight" fields. 
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as argumrnts
% NOTE: Needs to be implemented using ~/Documents/MATLAB/Kmeans


global input_dir  output_dir NX NY 

filename = fname;
filename = [input_dir, filename];


NXscaled = NX * xScale; % L1 size
NYscaled = NY * yScale;

scaleWeights = 1;


write_centers = 1;
plot_centers = 1;


debug = 0;


if exist(filename,'file')
    
    W_array = [];
    
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
    
                      
        
    %% read the last weights field (configuration)
    W_array = []; % N x patch_size array where N = NX * NY


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
                    %pause
                end
                if(~isempty(w) & nItems ~= 0)
                    W_array(k,:) = w(1:patch_size);
                    %pause
                end
            end % if ~ feof
        end
    end % loop over post-synaptic neurons
   
    fclose(fid);

    %% compute K-means
    
    % data is n x patch_size where n = NX * NY
    [centers,mincenter,mindist,q2,quality] = kmeans(W_array,numCenters);
    %size(centers) % numCenters x patch_size
    %size(mincenter)% n x 1
    %q2
    %quality

    %% compute cluster weights
    figure(gcf+1);
    plot(mincenter,'ob');
    title('Min Centers');
    for k=1:numCenters
        clustW(k) = sum(find(mincenter == k));
    end
    clustW = clustW ./sum(clustW);

    %% plot centerss
    numColumns = 4; % number of row plots
    numRows = ceil(numCenters/numColumns);
    
    if(plot_centerss)
        figure('Name','Weights K-means Centers');
        
        for k=1:numK
            fprintf('cluster %d w = %f\n',k,clustW(k));
            patch = clustW(k) * reshape(centers(k,:),[4 4])';
            subplot(numRows,numColumns,k);
            colormap gray
            imagesc(patch,[0 1]); % 'CDataMapping','direct'
            title(['Center ' num2str(k) ' w = ' num2str(clustW(k),2) ] );
            %xlabel(['w = ' num2str(clustW(k))] );
            colorbar
            axis square
            axis off
        end

    end

    
    %% write centers

    if(write_centers)
        fprintf('write centers!\n')
        centers_file = [output_dir,'WeightsCenters.dat'];
        fid = fopen(centers_file,'w');
        for k=1:numCenters
            for j=1:numel(centers(k,:))
                fprintf(fid,'%f ',centers(k,j));
            end
            fprintf(fid,'\n');
        end
        fclose(fid);
    end

   
else

    disp(['Skipping, could not open ', filename]);

end

% End primary function
%


function [time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
    readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
head = fread(fid,3,'int');
if head(3) ~= 3
    disp('incorrect file type')
    return
end
numWgtParams = 6;
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


function [time,varargout] = ...
    readHeader(fid,numParams,numWgtParams)

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

        varargout{1} = numPatches;
        varargout{2} = NXP;
        varargout{3} = NYP;
        varargout{4} = NFP;
        varargout{5} = minVal;
        varargout{6} = maxVal;
        %pause
    else
        disp('eof found: return');
        time = -1;
    end
else
    disp('eof found: return');
    time = -1;
end
% End subfunction
%

