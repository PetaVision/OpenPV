function A = stdp_plotWeightsField(fname, xScale, yScale,Xtarg, Ytarg)
% plot "weights" (typically after turning on just one neuron)
% Xtarg and Ytarg contain the X and Y coordinates of the target
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as arguments

% We plot a square around the neurons that have the same receptive
% field. xShare and yShare define the size of the layer patch that
% contains neurons that have the same receptive.

global input_dir  NX NY 

margins_defined = 1;

filename = fname;
filename = [input_dir, filename];
    
xShare = 4; % define the size of the layer patch that 
yShare = 4; % contains neurons that have the same receptive field.

NXlayer = NX * xScale; % L1 size
NYlayer = NY * yScale;

fprintf('scaled NX = %d scaled NY = %d\n',NXlayer,NYlayer);

PLOT_STEP = 10;
plotTarget = 0;

%figure('Name','Weights Fields');

debug = 0;
weightsChange = 0;
scaleWeights = 1;
numRecords = 0;  % number of weights records (configurations)

if exist(filename,'file')
    
    W_array = [];
    
    fprintf('read weights from file %s\n',filename);
    
    fid = fopen(filename, 'r', 'native');

    [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid);
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
    if numPatches ~= NXlayer*NYlayer
        disp('mismatch between numPatches and NX*NY')
        return
    end
    patch_size = NXP*NYP;
    
    [a,b,a1,b1,NXPbor,NYPbor] = compPatches(NXP,NYP);
    fprintf('a = %d b = %d a1 = %d b1 = %d NXPbor = %d NYPbor = %d\n',...
        a,b,a1,b1,NXPbor,NYPbor);
    
    %b_color = 1; % use to scale weights to the full range
                 % of the color map
    %a_color = (length(get(gcf,'Colormap'))-1.0)/maxVal;

    
    if weightsChange
       PATCH = zeros(NXPbor,NYPbor);
                      % PATCH contains borders
    else
        if scaleWeights
            PATCH = ones(NXPbor,NYPbor) * (0.5*(maxVal+minVal));
        else
            PATCH = ones(NXPbor,NYPbor) * 122;
        end
    end
                      
    avWeights = [];  % time averaged weights array
    
    % Think of NXP and NYP as defining the size of the receptive field 
    % of the neuron that receives spikes from the retina.
    % NOTE: The file may have the full header written before each record,
    % or only a time stamp
    
    first_record = 1;
    
    while (~feof(fid))
        
        % read the weights for this time step 
        W_array = []; % reset every time step: this is N x patch_size array
                      % where N = NXlayer * NYlayer
                      
        % read header if not first record (for which header is already read)
        
        if ~first_record
            [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
                readHeader(fid,numParams);
            fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                time,numPatches,NXP,NYP,NFP);
            %pause
            % if minVal and/or maxVal change dynamically (homeostatic
            % control) adjust the color of the patch accordingly
            if scaleWeights
                PATCH(:) = 0.5*(maxVal+minVal);
            else
                PATCH (:) = 122;
            end
        else
            first_record = 0;
        end
    
        % read time
        %time = fread(fid,1,'float64');
        %fprintf('time = %f\n',time); 
        
        k=0;
        
        for j=1:NYlayer
            for i=1:NXlayer
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
                    if(~isempty(w) & nItems ~= 0)
                        W_array(k,:) = w(1:patch_size);
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
        
        % make the matrix of patches and plot patches for this time step
        A = [];
        
        if(~isempty(W_array))
            
            
            k=0;
            % this only holds for even patches
%             for j=(NYPbor/2):NYPbor:(NYlayer*NYPbor)
%                 for i=(NXPbor/2):NXPbor:(NXlayer*NXPbor)
%                     k=k+1;
%                     %W_array(k,:)
%                     patch = reshape(W_array(k,:),[NXP NYP]);
%                     %PATCH(b1+1-b:b1+b,a1+1-a:a1+a) = patch';
%                     PATCH(2:NYP+1,2:NXP+1) = patch';
%                     %patch
%                     %PATCH
%                     %pause
%                     A(j-b1+1:j+b1,i-a1+1:i+a1) = PATCH;
%                     %imagesc(A,'CDataMapping','direct');
%                     %pause
%                 end
%             end
            
            % this only holds for even patches
            for j=1:NYlayer
                for i=1:NXlayer
                    k=k+1;
                    %W_array(k,:)
                    patch = reshape(W_array(k,:),[NXP NYP]);
                    %PATCH(b1+1-b:b1+b,a1+1-a:a1+a) = patch';
                    PATCH(2:NYP+1,2:NXP+1) = patch';
                    %patch
                    %PATCH
                    %pause
                    A(1+(j-1)*NYPbor:j*NYPbor,1+(i-1)*NXPbor:i*NXPbor) = PATCH;
                    %imagesc(A,'CDataMapping','direct');
                    %pause
                end
            end
            
            if numRecords==1 % first record (plot)
                Ainit = A;
                Aold = A;
                avWeights = A;
                %fprintf('time = %f\n',time);
                if ~weightsChange
                    figure('Name',['Weights Field ' num2str(time)]);                    
                    imagesc(A,'CDataMapping','direct');
                    %colorbar
                    axis square
                    axis off
                    hold on
                    % plot squares around the neurons that share the same
                    % receptive field
                    % this works when there are no margins and it generates
                    % a boundary layer of width 2 in this case
                    % With margins defined there is no need for a boundary
                    % layer!
                    if (margins_defined)
                        for i=0.5:xShare*NXPbor:(NXlayer*NXPbor+1)
                            plot([i, i],[0.5,NYlayer*NYPbor+0.5],'-r');
                            
                        end
                        for j=0.5:yShare*NYPbor:(NYlayer*NYPbor+1)
                            plot([0.5,NXlayer*NXPbor+0.5],[j,j],'-r');
                        end
                    else
                        for i=(0.5+2*NXPbor):xShare*NXPbor:(NXlayer*NXPbor+1)
                            plot([i, i],[0,NYlayer*NYPbor],'-r');
                            
                        end
                        for j=(0.5+2*NYPbor):yShare*NYPbor:(NYlayer*NYPbor+1)
                            plot([0,NXlayer*NXPbor],[j,j],'-r');
                        end
                    end % margins_defined
                    %pause
                end
            else            % other records (not first)
                if (mod(numRecords,PLOT_STEP) == 0)
                    %fprintf('time = %f\n',time);
                    if weightsChange
                       %figure('Name',['Weights Change Field ' num2str(time)]); 
                       imagesc(A-Ainit,'CDataMapping','direct');
                    else
                       figure('Name',['Weights Field ' num2str(time)]);
                       imagesc(A,'CDataMapping','direct');
                    end
                    %colorbar
                    axis square
                    axis off
                    hold on
                    
                    % plot squares around the neurons that share the same
                    % receptive field
                    % this works when there are no margins and it generates
                    % a boundary layer of width 2 in this case
                    % With margins defined there is no need for a boundary
                    % layer! 
                    
                    if (margins_defined)
                        for i=0.5:xShare*NXPbor:(NXlayer*NXPbor+1)
                            plot([i, i],[0.5,NYlayer*NYPbor+0.5],'-r');
                            
                        end
                        for j=0.5:yShare*NYPbor:(NYlayer*NYPbor+1)
                            plot([0.5,NXlayer*NXPbor+0.5],[j,j],'-r');
                        end
                    else
                        for i=(0.5+2*NXPbor):xShare*NXPbor:(NXlayer*NXPbor+1)
                            plot([i, i],[0,NYlayer*NYPbor],'-r');
                            
                        end
                        for j=(0.5+2*NYPbor):yShare*NYPbor:(NYlayer*NYPbor+1)
                            plot([0,NXlayer*NXPbor],[j,j],'-r');
                        end
                    end % margins_defined
                    %pause
                    
                    % plot target pixels
                    if plotTarget
                        for t=1:length(Xtarg)
                            I=Xtarg(t);
                            J=Ytarg(t);
                            plot([(J-1)*NXP+dY],[(I-1)*NXP+dX],'.r','MarkerSize',12)
                        end
                    end
                    pause(0.1)
                    hold off
                end
                Aold = A;
                avWeights = avWeights + A;
            end
        end   
        %pause % after reading one set of weights
    end % reading from weights file
    
    fclose(fid);
    fprintf('feof reached: numRecords = %d time = %f\n',numRecords,time);
    avWeights = avWeights / numRecords;
    figure('Name','Time Averaged Weights');
    imagesc(avWeights,'CDataMapping','direct');
    %colorbar
    axis square
    axis off
    
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
    fprintf('minVal = %f maxVal = %f numPatches = %d\n',...
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
    NXPbor = NXP+2; % patch with borders
    NYPbor = NYP+2;
    a1= (NXPbor-1)/2;    % NXP = 2a1+1;
    b1= (NYPbor-1)/2;    % NYP = 2b1+1;

    dX = (NXPbor+1)/2;  % used in ploting the target
    dY = (NYPbor+1)/2;

else                 % even size patches

    a= NXP/2;    % NXP = 2a;
    b= NYP/2;    % NYP = 2b;
    NXPbor = NXP+2;   % add border pixels for visualization purposes
    NYPbor = NYP+2;
    a1=  NXPbor/2;    % NXP = 2a1;
    b1=  NYPbor/2;    % NYP = 2b1;

    dX = NXPbor/2;  % used in ploting the target
    dY = NYPbor/2;

end
        
% End subfunction
%
