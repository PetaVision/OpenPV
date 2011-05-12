function A = stdp_plotReceptiveField(fname, xScale, yScale,Xtarg, Ytarg)
% plot "weights" (typically after turning on just one neuron)
% Xtarg and Ytarg contain the X and Y coordinates of the target
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as arguments

% We plot a square around the neurons that have the same receptive
% field. xShare and yShare define the size of the layer patch that
% contains neurons that have the same receptive.

global input_dir  NX NY 

filename = fname;
filename = [input_dir, filename];
    
xShare = 4; % define the size of the layer patch that 
yShare = 4; % contains neurons that have the same receptive field.

NXlayer = NX * xScale; % L1 size
NYlayer = NY * yScale;

fprintf('scaled NX = %d scaled NY = %d\n',NXlayer,NYlayer);

PLOT_STEP = 1;

debug = 0;
scaleWeights = 1;
numRecords = 0;  % number of weights records (configurations)


%% open file pointers & figure handles
for f=1:numel(fname)

    filename = fname{f};
    fprintf('read weights from file %s\n',filename);
    filename = [input_dir, filename];

    if exist(filename,'file')        
        fid{f} = fopen(filename, 'r', 'native');
    else
        disp(['Skipping, could not open ', filename]);
        return
    end
    
end

%% read file headers

for f=1:numel(fname)
    
    [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid{f});
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
    if numPatches ~= NXlayer*NYlayer
        disp('mismatch between numPatches and NX*NY')
        return
    end
    patch_size = NXP*NYP;
    
end

%% determine patch geometry
[a,b,a1,b1,NXPbor,NYPbor] = compPatches(NXP,NYP);
fprintf('a = %d b = %d a1 = %d b1 = %d NXPbor = %d NYPbor = %d\n',...
    a,b,a1,b1,NXPbor,NYPbor);

%b_color = 1; % use to scale weights to the full range
% of the color map
%a_color = (length(get(gcf,'Colormap'))-1.0)/maxVal;


if scaleWeights
    PATCH = ones(NXPbor,NYPbor) * (0.5*(maxVal+minVal));
else
    PATCH = ones(NXPbor,NYPbor) * 0;
end


% Think of NXP and NYP as defining the size of the receptive field
% of the neuron that receives spikes from the retina.
% NOTE: The file may have the full header written before each record,
% or only a time stamp

first_record = 1;

while (~feof(fid{1}) && ~feof(fid{2}) )
    
    % read the weights for this time step
    
    
    % read header if not first record (for which header is already read)
    
    if ~first_record
        % loop over weights file
        for f=1:numel(fname)
            [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
                readHeader(fid{f},numParams);
            fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                time,numPatches,NXP,NYP,NFP);
            %pause
        end
    else
        first_record = 0;
    end
    
    % read time
    %time = fread(fid,1,'float64');
    %fprintf('time = %f\n',time);
    W_array = []; % reset every time step: this is N x patch_size array
                  % where N = NXlayer * NYlayer
                  
    for f=1:numel(fname)    
        k=0;
        for j=1:NYlayer
            for i=1:NXlayer
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
                        pause
                    end
                    if(~isempty(w) & nItems ~= 0)
                        if f==1
                            W_array(k,:) = w(1:patch_size);
                        else
%                             for p=1:size(W_array,2)
%                                 fprintf('%f ',W_array(k,p))
%                             end
%                             fprintf('\n');
%                             for p=1:numel(w)
%                                fprintf('%f ',w(p))
%                             end
%                             fprintf('\n');
%                             size(w)
%                             pause
                            W_array(k,:) = W_array(k,:) - w(1:patch_size)';
                        end
                        %pause
                    end
                end % if ~ feof
            end
        end % loop over post-synaptic neurons
    end % loop over files
    
    if ~feof(fid{1}) && ~feof(fid{2}) 
        numRecords = numRecords + 1;
        fprintf('k = %d numRecords = %d time = %f\n',...
            k,numRecords,time);
        %pause
    end
    
    % make the matrix of patches and plot patches for this time step
    A = [];
    
    if(~isempty(W_array))
        
        
        k=0;
        for j=(NYPbor/2):NYPbor:(NYlayer*NYPbor)
            for i=(NXPbor/2):NXPbor:(NXlayer*NXPbor)
                k=k+1;
                %W_array(k,:)
                patch = reshape(W_array(k,:),[NXP NYP]);
                PATCH(b1+1-b:b1+b,a1+1-a:a1+a) = patch';
                %patch
                %PATCH
                %pause
                A(j-b1+1:j+b1,i-a1+1:i+a1) = PATCH;
                %imagesc(A,'CDataMapping','direct');
                %pause
            end
        end
        
        
        if (mod(numRecords,PLOT_STEP) == 0)
            %fprintf('time = %f\n',time);
            
            figure('Name',['Weights Field ' num2str(time)]);
            imagesc(A,'CDataMapping','direct');
            
            %colorbar
            axis square
            axis off
            hold on
            
            % plot squares around the neurons that share the same
            % receptive field
            for i=(0.5+2*NXPbor):xShare*NXPbor:(NXlayer*NXPbor+1)
                plot([i, i],[0,NYlayer*NYPbor],'-r');
                
            end
            for j=(0.5+2*NXPbor):yShare*NYPbor:(NYlayer*NYPbor+1)
                plot([0,NXlayer*NXPbor],[j,j],'-r');
            end
            %pause
            
            pause(0.1)
            hold off
        end
        
        
    end
    %pause % after reading one set of weights
end % reading from weights file

for f=1:numel(fname)
    fclose(fid{f});
end

fprintf('feof reached: numRecords = %d time = %f\n',numRecords,time);



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
