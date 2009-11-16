function A = stdp_plotWeightsOnly(fname, xScale, yScale,Xtarg, Ytarg)
% plot "weights" (typically after turning on just one neuron)
% Xtarg and Ytarg contain the X and Y coordinates of the target
% xScale and yScale are scale factors for this layer

global input_dir n_time_steps % NX NY 

filename = fname;
filename = [input_dir, filename];
colormap(jet);
    
NX = 32;        % retina size
NY = 32;

NX = NX * xScale; % L1 size
NY = NY * yScale;

PLOT_STEP = 1;
plotTarget = 0;

figure('Name','Weights Fields');

debug = 0;
weightsChange = 1;

bufSize = 4; % see pv_write_patch() in io.c
nPad = 2;    % size of the layer padding
nf = 1;      % number of features

if exist(filename,'file')
    
    W_array = [];
    
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
    
    [a,b,a1,b1,NXPbor,NYPbor] = compPatches(NXP,NYP)
    fprintf('a = %d b = %d a1 = %d b1 = %d NXPbor = %d NYPbor = %d\n',...
        a,b,a1,b1,NXPbor,NYPbor);
    
    b_color = 1; % use to scale weights to the full range
                 % of the color map
    a_color = (length(get(gcf,'Colormap'))-1.0)/maxVal;

    numRecords = 0;   % number of weights records (configs)
    
    if weightsChange
       PATCH = zeros(NXPbor,NYPbor);
                      % PATCH contains borders
    else
        PATCH = ones(NXPbor,NYPbor) * (0.5*(maxVal+minVal));
    end
                      
    avWeights = [];  % time averaged weights array
    
    % Think of NXP and NYP as defining the size of the neuron's
    % receptive field that receives spikes from the retina.
    while (~feof(fid))
        
        % read the weights for this time step 
        W_array = []; % reset every time step: this is N x patch_size array
                      % where N =NX * NY
                      
        if numRecords > 0
          [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
              readHeader(fid,numParams,numWgtParams);
          if time >= 0
             fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                time,numPatches,NXP,NYP,NFP);
          else
              disp('eof found')
              break
          end
        end
        
        k=0;
        
        for j=1:NY
            for i=1:NX
                if ~feof(fid)
                    k=k+1;
                    nx = fread(fid, 1, 'uint16'); % unsigned short
                    ny = fread(fid, 1, 'uint16'); % unsigned short
                    nItems = nx*ny*NFP;
                    if debug & n_time_steps >= 0
                        fprintf('k = %d nx = %d ny = %d nItems = %d: ',...
                            k,nx,ny,nItems);
                    end
                    
                    w = fread(fid, nItems, 'uchar'); % unsigned char
                    % scale weights: they are quantized before written
                    %w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
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
        if ~feof(fid)
            numRecords = numRecords + 1;
            fprintf('k = %d numRecords = %d time = %f\n',...
                k,numRecords,time);
        end
        
        % make the matrix of patches and plot patches for this time step
        A = [];
        
        if(~isempty(W_array))
            
            
            k=0;
            for j=(NYPbor/2):NYPbor:(NY*NYPbor)
                for i=(NXPbor/2):NXPbor:(NX*NXPbor)
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
            
            if numRecords==1
                Ainit = A;
                Aold = A;
                avWeights = A;
                fprintf('time = %f\n',time);
                imagesc(A,'CDataMapping','direct');
                colorbar
                axis square
                axis off
                hold on
            else
                %imagesc(a_color*A+b_color);
                %imagesc(A-Ainit,'CDataMapping','direct');
                if (mod(numRecords,PLOT_STEP) == 0)
                    %fprintf('time = %f\n',time);
                    if weightsChange
                       imagesc(A-Ainit,'CDataMapping','direct');
                    else
                       imagesc(A-Ainit,'CDataMapping','direct');
                    end
                    colorbar
                    axis square
                    axis off
                    hold on
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
        pause % after reading one set of weights
    end % reading from weights file
    fclose(fid);
    fprintf('feof reached: numRecords = %d time = %f\n',numRecords,time);
    avWeights = avWeights / numRecords;
    figure('Name','Time Averaged Weights');
    imagesc(avWeights,'CDataMapping','direct');
    colorbar
    axis square
    axis off
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
    NXP         = params(4);
    NYP         = params(5);
    NFP         = params(6);
    fprintf('numParams = %d ',numParams);
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
    %pause
    
% End subfunction 
%
    
    
function [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
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