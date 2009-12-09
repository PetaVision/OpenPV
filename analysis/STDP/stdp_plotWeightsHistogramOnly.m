function A = stdp_plotWeightsHistogramOnly(fname, xScale,yScale, TSTEP)
% At each time step plots the weights histopgram.
% Returns the last weights distribution.
% NOTE: The weights are quantized in the range [0,255].

global input_dir  % NX NY 

filename = fname;
filename = [input_dir, filename];

NX = 32;        % retina size
NY = 32;

NX = NX * xScale; % L1 size
NY = NY * yScale;

N=NX*NY;

debug = 0;
edges = 0:255;      % bin locations

if exist(filename,'file')
    
    fid=fopen(filename,'r','native');

    [time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid);
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
    if numPatches ~= NX*NY
        disp('mismatch between numPatches and NX*NY')
        return
    end
    patch_size = NXP*NYP;
    
    recordID = 0; % number of weights records (configs)
   
    % NOTE: The file may have the full header written before each record,
    % or only a time stamp
    while (~feof(fid))
        W_array = []; % reset every time step: this is N x patch_size array
                      % where N =NX x NY
        
        % read header if not first record (for which header is already read)
        
        if recordID
            [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
                readHeader(fid,numParams,numWgtParams);
            fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                time,numPatches,NXP,NYP,NFP);
            %pause
        end
        
        % read time
        %time = fread(fid,1,'float64');
        %fprintf('time = %f\n',time);
                
        k = 0;
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
                        %pause
                    end
                    if(~isempty(w) & nItems ~= 0)
                        W_array(k,:) = w(1:patch_size);
                        %pause
                    end
                end % if ~feof
            end
        end % loop over post-synaptic neurons
        if ~feof(fid)
            recordID = recordID + 1;
            fprintf('k = %d recordID = %d time = %f\n',...
                k,recordID,time);
        end
        
        % plot the weights histogram for the first time step
        
        if ( recordID == 1 && ~isempty(W_array) )
            [m,n]=size(W_array);
            fprintf('%d %d %d\n',recordID,m,n);
            A = reshape(W_array, [1 (N*patch_size)] ) ;
            %ind = find(A > 0.0);
            %[n,xout] = hist(A(ind),nbins);
            n = histc(A,edges);
            %plot(xout,n,'-g','LineWidth',3);
            figure('Name', ['Time ' num2str(time) ' Weights Histogram']);
            bar(edges,n);
            %hist(A,nbins);
            %hold on
            pause(0.1)
            
        end
        
        
        % plot the weights histogram for this time step
        
        if ( ~mod(recordID-1,TSTEP) && ~isempty(W_array) )
            [m,n]=size(W_array);
            fprintf('%d %d %d \n',recordID,m,n);
            A = reshape(W_array, [1 (N*patch_size)] ) ;
            %ind = find(A > 0.0);
            %[n,xout] = hist(A(ind),nbins);
            n = histc(A,edges);
            %plot(xout,n,'-r');
            figure('Name', ['Time ' num2str(time) ' Weights Histogram']);
            bar(edges,n,'r');
            %hold on
            pause(0.1)
            
        end
        
    end
    fclose(fid);


else
    
     disp(['Skipping, could not open ', filename]);
    
end

%
% End primary function


function [time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
    fprintf('read first header\n');
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
    
    
function [time,varargout] = readHeader(fid,numParams,numWgtParams)

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
        varargout{1} = numPatches;
        varargout{2} = 0;
        varargout{3} = 0;
        varargout{4} = 0;
        varargout{5} = 0;
        varargout{6} = 0;
    end
else
   disp('eof found: return'); 
   time = -1;
end
% End subfunction 
%
        
    

