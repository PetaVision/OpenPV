function A = stdp_plotWeightsHistogramOnly(fname, xScale,yScale, TSTEP)
% At each time step plots the weights histopgram.
% Returns the last weights distribution.
% NOTE: The weights are quantized in the range [0,255].

global input_dir  NX NY 

scaleWeights = 1;
hist_change = 0;
debug = 0;

filename = fname;
filename = [input_dir, filename];
fprintf('read weights from %s\n',filename);

NXlayer = NX * xScale; % L1 size
NYlayer = NY * yScale;
fprintf('NXlayer = %d NYlayer = %d\n', NXlayer, NYlayer);
%pause

N=NXlayer*NYlayer;

nbins = 100;
TSTEP = 1;  % plot every TSTEP

% if scaleWeights
%     edges = 0:0.01:0.75; % scaled weights
% else
%     edges = 0:255;      % bin locations
% end

if exist(filename,'file')
    
    fid=fopen(filename,'r','native');

    [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid);
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
    if numPatches ~= NXlayer*NYlayer
        disp('mismatch between numPatches and NXlayer * NYlayer')
        return
    end
    patch_size = NXP*NYP;
    
    if scaleWeights
        edges = 0:0.01:maxVal; % scaled weights
    else
        edges = 0:255;      % bin locations
    end

    recordID = 0; % number of weights records (configs)
   
    % NOTE: The file may have the full header written before each record,
    % or only a time stamp
    while (~feof(fid))
        W_array = []; % reset every time step: this is N x patch_size array
                      % where N =NX x NY
        
        % read header if not first record (for which header is already read)
        
        if recordID
            [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
                readHeader(fid,numParams);
           
            if time > 0
                
                fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                    time,numPatches,NXP,NYP,NFP);
                %pause
            else
                break; % EOF found
            end
        end
        
        % read time
        %time = fread(fid,1,'float64');
        %fprintf('time = %f\n',time);
                
        k = 0;
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
                    % scale weights
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
            %n = histc(A,edges);
            %plot(xout,n,'-g','LineWidth',3);
            %figure('Name', ['Time ' num2str(time) ' Weights Histogram']);
            %bar(edges,n);
            hist(A,nbins);
            title(['Time ' num2str(time) ' Weights Histogram']);
            %hold on
            pause(1)
            % store first histogram
            if hist_change
                n = histc(A,edges);
                n0 = n;
            end
        end
        
        
        % plot the weights histogram for this time step
        
        if ( ~mod(recordID,TSTEP) && ~isempty(W_array) )
            [m,n]=size(W_array);
            fprintf('%d %d %d \n',recordID,m,n);
            A = reshape(W_array, [1 (N*patch_size)] ) ;
            %ind = find(A > 0.0);
            %[n,xout] = hist(A(ind),nbins);
            %n = histc(A,edges);
            %plot(xout,n,'-r');
            %figure('Name', ['Time ' num2str(time) ' Weights Histogram']);
            if hist_change
                n = histc(A,edges);
                bar(edges,n-n0,'r');
            else
                %bar(edges,n,'r');
                hist(A,nbins);
                title(['Time ' num2str(time) ' Weights Histogram']);
            end
            %hold on
            pause(1)
            
        end
        
    end
    fclose(fid);


else
    
     disp(['Skipping, could not open ', filename]);
    
end

%
% End primary function


function [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
    fprintf('read first header\n');
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
        
    else
        disp('empty params -> eof found: return');
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
        
    

