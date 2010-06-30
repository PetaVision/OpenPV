function A = stdp_plotWeightsStability(fname, xScale, yScale,Xtarg, Ytarg)
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

    
NXscaled = NX * xScale; % L1 size
NYscaled = NY * yScale;
numNeurons = NXscaled * NYscaled;

fprintf('scaled NX = %d scaled NY = %d\n',NXscaled,NYscaled);

PLOT_STEP = 1;
print_features = 0;
write_projections = 0;

debug = 0;
numRecords = 0;  % number of weights records (configurations)

%% figure handles to plot patch projections vs space and time

h_time = figure('Name','Weights Projections vs Time');

h_space = figure('Name','Weights Projections vs Time');


if(write_projections)
    proj_file = [output_dir,'WeightsProjections.dat'];
    if exist(scores_file,'file')
        fid_proj = fopen(proj_file,'r');
        % read projections, plot, and set startTime
        data = fscanf(fid_proj, '%g %g %g', [(1+2*numNeurons) inf]);   % It has two rows now.
        data = data';
        fclose(fid_proj);
        %figure(h_time)
        %plot(data(:,1),data(:,2),sym{1});hold on
        %plot(data(:,1),data(:,3),sym{2});hold on
 
        
        fid_proj = fopen(proj_file,'a');
        startTime = data(end,1) * 1000;
    else
        fid_proj = fopen(proj_file,'w');
        startTime = 0;
    end
end
%startTime = 3000000;
fprintf('startTime = %d \n',startTime);
pause
 

if exist(filename,'file')
    
    
    fprintf('read weights from file %s\n',filename);
    
    fid = fopen(filename, 'r', 'native');

    [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid);
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
    if numPatches ~= numNeurons
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
                fprintf('%.2f ',Features(i,f) );
            end
            fprintf('\n');
        end
        pause
    end
    numFeatures = size(Features,2);
    
    % Think of NXP and NYP as defining the size of the receptive field 
    % of the neuron that receives spikes from the retina.
    % NOTE: The file may have the full header written before each record,
    % or only a time stamp
    
    W_array = zeros(numNeurons, patch_size);
    first_record = 1;
    
    while (~feof(fid))
        
        % read the weights for this time step 
        
        W_array(:) = 0; % reset every time step: this is N x patch_size array
                      % where N =NX * NY
                      
        % read header if not first record (for which header is already read)
        
        if ~first_record
            [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
                readHeader(fid,numParams);
            fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                time,numPatches,NXP,NYP,NFP);
            pause
        else
            first_record = 0;
        end

        if write_projections
           fprintf(fid_proj,'%f ', time); 
        end
        
        k=0;   
        figure(h_time);
        
        for j=1:NYscaled
            for i=1:NXscaled
                if ~feof(fid)
                    k=k+1;
                    nx = fread(fid, 1, 'uint16'); % unsigned short
                    ny = fread(fid, 1, 'uint16'); % unsigned short
                    nItems = nx*ny*NFP;                     
                    w = fread(fid, nItems, 'uchar'); % unsigned char 
                    % scale weights
                    w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                    
                    if(~isempty(w) & nItems ~= 0)
                        w = w ./norm(w);
                        [O, I ]= max(w * Features );
                        if write_projections
                            fprintf(fid_proj,'%d %f ',I, 1-O);
                        end
                        plot([time],[(k-1)*numFeatures + I + (1-O)],'ob');
                        W_array(k,:) = w(1:patch_size);
                    end
                end % if ~ feof 
            end
        end % loop over post-synaptic neurons
        
        if write_projections
            fprintf(fid_proj,'\n');
        end
        
        if ~feof(fid)
            numRecords = numRecords + 1;
            fprintf('k = %d numRecords = %d time = %f\n',...
                k,numRecords,time);
            %pause
        end
                
        
        if(~isempty(W_array))
            
            % make the matrix of patches and plot patches for this time step
            %[Pmax,I] = max(W_array * Features, [],2);
            for p=1:4,P{p}=W_array * Features(:,p);end

            % maxA is a N x 1 array contains the max projection
            % of each patch on the directions in Features
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
                    imagesc(reshape(P{p},[NXscaled NYscaled])','CDataMapping','direct');
                    colorbar
                    axis square
                    axis off
                end
                
                figure('Name',['1D Projections ' num2str(time)]);
                for p=1:4
                    subplot(2,2,p)
                    plot(1:NXscaled,sum(reshape(P{p},[NXscaled NYscaled])')/NYscaled,'ob');
                    axis([1 NXscaled 0 Inf]);
                end
            end

        end
        %pause % after reading one set of weights
    end % reading from weights file
    
    fclose(fid);
    fprintf('feof reached: numRecords = %d time = %f\n',numRecords,time);
    fclose(fid_proj);
    
else
    
     disp(['Skipping, could not open ', filename]);
    
end

% End primary function
%


function [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
    fprintf('read first header\n');
    head = fread(fid,3,'int')
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
    
    
%function [time,varargout] = readHeader(fid,numParams,numWgtParams)
function [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = readHeader(fid,numParams)



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

function Features = compFeatures(NXP, NYP)

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