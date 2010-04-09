function A = stdp_plotWeightsProjections(fname, xScale, yScale,Xtarg, Ytarg)
% plot weights projected on a number of directions (features)
% Xtarg and Ytarg contain the X and Y coordinates of the target
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as argumrnts

global input_dir  NX NY 

filename = fname;
filename = [input_dir, filename];
%colormap(jet);
    
%Proj = compProjections(4,4)
%for p=1:numel(Proj)
%    Proj{p}
%end
%pause

    
NX = NX * xScale; % L1 size
NY = NY * yScale;

fprintf('scalex NX = %d scaled NY = %d\n',NX,NY);

PLOT_STEP = 1;

%figure('Name','Weights Fields');

debug = 0;
numRecords = 0;  % number of weights records (configurations)

if exist(filename,'file')
    
    W_array = [];
    
    fprintf('read weights from file %s\n',filename);
    
    fid = fopen(filename, 'r', 'native');

    [time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid);
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
    if numPatches ~= NX*NY
        disp('mismatch between numPatches and NX*NY')
        return
    end
    patch_size = NXP*NYP;
    
    
    % define projections
    % these should be the principal components returned by
    % PCA analysis or another more sophisticated clustering algorithm

    Proj = compProjections(NXP, NYP);
    %for p=1:numel(Proj)
    %    Proj{p}
    %end
    %pause

    
    % Think of NXP and NYP as defining the size of the receptive field 
    % of the neuron that receives spikes from the retina.
    % NOTE: The file may have the full header written before each record,
    % or only a time stamp
    
    first_record = 1;
    
    while (~feof(fid))
        
        % read the weights for this time step 
        %Pmax = [];
        for p=1:4,P{p}=[];end
        I = [];
        
        W_array = []; % reset every time step: this is N x patch_size array
                      % where N =NX * NY
                      
        % read header if not first record (for which header is already read)
        
        if ~first_record
            [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
                readHeader(fid,numParams,numWgtParams);
            fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                time,numPatches,NXP,NYP,NFP);
            %pause
        else
            first_record = 0;
        end
    
        % read time
        %time = fread(fid,1,'float64');
        %fprintf('time = %f\n',time); 
        
        k=0;
        
        for j=1:NY
            for i=1:NX
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
          
                    if debug 
                        for r=1:patch_size
                            fprintf('%d ',w(r));
                        end
                        fprintf('\n');
                        pause
                    end
                    if(~isempty(w) & nItems ~= 0)
                        W_array(k,:) = w(1:patch_size)./norm(w(1:patch_size));
                        %pause
                    end
                end % if ~ feof 
            end
        end % loop over post-synaptic neurons
        if ~feof(fid)
            numRecords = numRecords + 1;
            fprintf('k = %d numRecords = %d time = %f\n',...
                k,numRecords,time);
            %pause
        end
                
        
        if(~isempty(W_array))
            
            % make the matrix of patches and plot patches for this time step
            %[Pmax,I] = max(W_array * Proj, [],2);
            for p=1:4,P{p}=W_array * Proj(:,p);end

            % maxA is a N x 1 array contains the max projection
            % of each patch on the directions in Proj
            % I is a N x 1 array contains the indices
            % of the max directions

            %I = reshape(I,[NX NY])';
            if (mod(numRecords,PLOT_STEP) == 0)
                %fprintf('time = %f\n',time);

                %figure('Name',['max Projections ' num2str(time)]);
                %imagesc(I*25,'CDataMapping','direct');
                %colorbar
                %axis square
                %axis off
                
                figure('Name',['2D Projections ' num2str(time)]);
                for p=1:4
                    subplot(2,2,p)
                    imagesc(reshape(P{p},[NX NY])','CDataMapping','direct');
                    colorbar
                    axis square
                    axis off
                end
                
                figure('Name',['1D Projections ' num2str(time)]);
                for p=1:4
                    subplot(2,2,p)
                    plot(1:NX,sum(reshape(P{p},[NX NY])')/NY,'ob');
                    axis([1 64 0 1]);
                end
            end

        end
        %pause % after reading one set of weights
    end % reading from weights file
    
    fclose(fid);
    fprintf('feof reached: numRecords = %d time = %f\n',numRecords,time);

    
else
    
     disp(['Skipping, could not open ', filename]);
    
end

% End primary function
%


function [time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
    fprintf('read first header\n');
    head = fread(fid,3,'int')
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

function Proj = compProjections(NXP, NYP)

% these are projections for 4x4 receptive fields
% in the Bars images experiments

Proj = zeros(NXP*NYP,4);
for p=1:4
    V = zeros(1,16);
    ind = p:4:16;
    V(ind) = 1;
    V = V ./ norm(V);
    %patch = reshape(V,[NXP NYP])';
    %figure(p)
    %imagesc(patch,'CDataMapping','direct');
    Proj(:,p) = V'; % when V gets reshaped and transposed
                 % we get the right feature
    %pause
end


% End subfunction
%