function A = stdp_compWeightsKmeans(fname, numCenters, xScale, yScale)
% plot last configuration of "weight" fields. 
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as argumrnts
% NOTE: Needs to be implemented using ~/Documents/MATLAB/Kmeans


global input_dir  output_dir NX NY 

filename = fname;
filename = [input_dir, filename];


NXscaled = NX * xScale; % L1 size
NYscaled = NY * yScale;

nxMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
nyMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
             
print_features = 0;
scaleWeights = 1;
write_centers = 0;
plot_centers = 1;
plot_weight_patches = 1; % for each center, plots the spatial 
                  % distribution of weight patches belonging to it

debug = 0;

    
if exist(filename,'file')
    
    W_array = [];
    
    fid = fopen(filename, 'r', 'native');

    [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid);
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
        
    if numPatches ~= NXscaled*NYscaled
        disp('mismatch between numPatches and NX*NY')
        return
    end
    patch_size = NXP*NYP;
    
    %% compute features matrix
    Features = compFeatures(NXP, NYP);
    % Features is a patch_size x numFeatures matrix
    if print_features
        for f=1:size(Features,2)
            fprintf('feature %d: ',f);
            for i=1:patch_size
                fprintf('%.2f ',Proj(i,f) );
            end
            fprintf('\n');
        end
        pause
    end
        
    %% read the last weights field (configuration)
    W_array = []; % N x patch_size array where N = NX * NY


    kPost=0;
    k=0;
    
    for j=1:NYscaled
        for i=1:NXscaled
            if ~feof(fid)
                kPost=kPost+1;
                nx = fread(fid, 1, 'uint16'); % unsigned short
                ny = fread(fid, 1, 'uint16'); % unsigned short
                nItems = nx*ny*NFP;
                if debug
                    fprintf('k = %d nx = %d ny = %d nItems = %d: ',...
                        kPost,nx,ny,nItems);
                end

                w = fread(fid, nItems, 'uchar'); % unsigned char
                
                % find 2D indices (0 indexing)
                %kxPost=rem(kPost-1,NXscaled);
                %kyPost=(kPost-1-kxPost)/NXscaled;
                %fprintf('patch indexes for kPost = %d ',kPost-1);
                %fprintf('(kxPost = %d, kyPost = %d)',kxPost,kyPost);
                %fprintf(' (i = %d, j = %d)\n',i,j);
                %pause
                
                kxPost = i-1; kyPost = j-1;
                % check if boundary neuron
                if kyPost >= nyMar && kyPost <= (NYscaled-nyMar-1) ...
                        && kxPost >= nxMar && kxPost <= (NXscaled-nxMar-1)
                    k=k+1;
                    kx(k) = i;
                    ky(k) = j;
                    
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
                end % margin condition
                
            end % if ~ feof
        end
    end % loop over post-synaptic neurons
   
    fclose(fid);

    %% compute K-means
    
    % data is n x patch_size where n = NX * NY
    [centers,mincenter,mindist,q2,quality] = kmeans(W_array,numCenters);
    %size(centers)  % numCenters x patch_size
    %size(mincenter)% n x 1
    %q2
    %quality

    %% compute cluster weights
    for k=1:numCenters
        clustW(k) = sum(find(mincenter == k));
    end
    clustW = clustW ./sum(clustW);

    % sort centers according to weight
    
    [sortW,sortI] = sort(clustW,'descend');
    
    %% plot centers in reverse order of their weights
    numColumns = 4; % number of row plots
    numRows = ceil(numCenters/numColumns);
    
    if(plot_centers)
        figure('Name','Weights K-means Centers');
        
        for k=1:numCenters
            fprintf('cluster %d w = %f\n',k,sortW(k));
            %patch = clustW(k) * reshape(centers(k,:),[4 4])';
            patch = reshape(centers(sortI(k),:),[4 4])';
            subplot(numRows,numColumns,k);
            colormap gray
            imagesc(patch); % 'CDataMapping','direct'
            title(['Center ' num2str(k) ' w = ' num2str(clustW(sortI(k)),2) ] );
            %xlabel(['w = ' num2str(clustW(sortI(k)))] );
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
            fprintf(fid,'%f ', sortW(k));
            for j=1:numel(centers(sortI(k),:))
                fprintf(fid,'%f ',centers(sortI(k),j));
            end
            fprintf(fid,'\n');
        end
        fclose(fid);
    end
    
    
    %% compute learning score
    learning_score = 0;
    for k=1:numCenters
        fprintf('cluster %d w = %f ',k,clustW(sortI(k)));
        [O, I ]= max(centers(sortI(k),:) * Features ) ;  
        fprintf('max_overlap = %f for feature %d\n', O, I); 
        learning_score = learning_score + sortW(k) * O;
    end
    fprintf('learning score = %f\n',learning_score);
    
    
    %% plot weight patches for each center
    
    if plot_weight_patches

        [a,b,a1,b1,NXPbor,NYPbor] = compPatches(NXP,NYP);
        fprintf('a = %d b = %d a1 = %d b1 = %d NXPbor = %d NYPbor = %d\n',...
            a,b,a1,b1,NXPbor,NYPbor);


        if scaleWeights
            PATCH = ones(NXPbor,NYPbor) * (0.5*(maxVal+minVal));
        else
            PATCH = ones(NXPbor,NYPbor) * 122;
        end
        
        A = zeros(NYscaled*NYPbor,NXscaled*NXPbor);
        for c =1:numCenters
            clustInd = find(mincenter == sortI(c));
            if scaleWeights
                A(:) = (0.5*(maxVal+minVal));
            else
                A(:) = 122;
            end
            
            for p=1:numel(clustInd)
                k= clustInd(p);
                i= (kx(k)-1)*NXPbor + NXPbor/2;
                j= (ky(k)-1)*NYPbor + NYPbor/2;
                patch = reshape(W_array(k,:),[NXP NYP]);
                PATCH(b1+1-b:b1+b,a1+1-a:a1+a) = patch';
                %patch
                %PATCH
                %pause
                A(j-b1+1:j+b1,i-a1+1:i+a1) = PATCH;
            end

            figure('Name',['Weights Field for cluster ' num2str(c)]);
            imagesc(A,'CDataMapping','direct');
            pause
        end

    end % plot_weight_patches
    
else

    disp(['Skipping, could not open ', filename]);

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

    
function [a,b,a1,b1,NXPbor,NYPbor] = compPatches(NXP,NYP)


if (mod(NXP,2) & mod(NYP,2))  % odd size patches
    a= (NXP-1)/2;    % NXP = 2a+1;
    b= (NYP-1)/2;    % NYP = 2b+1;
    NXPold = NXP;
    NYPold = NYP;
    NXPbor = NXP+2; % patch with borders
    NYPbor = NYP+2;
    a1= (NXPbor-1)/2;    % NXP = 2a+1;
    b1= (NYPbor-1)/2;    % NYP = 2b+1;

    dX = (NXPbor+1)/2;  % used in ploting the target
    dY = (NYPbor+1)/2;

else                 % even size patches

    a= NXP/2;    % NXP = 2a;
    b= NYP/2;    % NYP = 2b;
    NXPold = NXP;
    NYPold = NYP;
    NXPbor = NXP+2;   % add border pixels for visualization purposes
    NYPbor = NYP+2;
    a1=  NXPbor/2;    % NXP = 2a1;
    b1=  NYPbor/2;    % NYP = 2b1;

    dX = NXPbor/2;  % used in ploting the target
    dY = NYPbor/2;

end
        
% End subfunction
%