function [PATCH, patch_size, NXP, NYP] = stdp_plotPatch(fname, I, J, NX, NY, plot_title,fh)
% plot patch for neuron I, J 
global input_dir

if ~exist('fh','var')          % tests if 'fh' is a variable in the workspace
                               % returns 0 if 'fh' does not exists
    fh = figure('Name',plot_title);
else
    set(fh, 'Name', plot_title);
end

filename = fname;
filename = [input_dir, filename];
N = NX * NY;

fid=fopen(filename,'r','native');

scaleWeights = 0;

[time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
    readFirstHeader(fid);
fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d ',...
    time,numPatches,NXP,NYP,NFP);
fprintf('minVal = %f maxVal = %d\n',minVal,maxVal);

if numPatches ~= NX*NY
    disp('mismatch between numPatches and NX*NY')
    return
end

    
patch_size = NXP*NYP;   

[a,b,a1,b1,NXPbor,NYPbor] = compPatches(NXP,NYP);
    fprintf('a = %d b = %d a1 = %d b1 = %d NXPbor = %d NYPbor = %d\n',...
        a,b,a1,b1,NXPbor,NYPbor);
    
%colormap('gray');
colormap(jet);
b_color = 1;     % use to scale weights to the full range
                 % of the color map
a_color = (length(get(gcf,'Colormap'))-1.0)/maxVal;

PATCH = [];

nts = 0;

first_record = 1;
    
while (~feof(fid))
    
    % read header if not first record (for which header is already read)

    if ~first_record
        [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
            readHeader(fid,numParams,numWgtParams);
        if time < 0
            fprintf('eof reached\n');
            break
        else
            fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                time,numPatches,NXP,NYP,NFP);
            %pause
        end
    else
        first_record = 0;
    end
        
    for j=1:NY
        for i=1:NX
            if ~feof(fid)
                nxp = fread(fid, 1, 'uint16'); % unsigned short
                nyp = fread(fid, 1, 'uint16'); % unsigned short
                nItems = nxp*nyp*NFP;
                w = fread(fid, nItems, 'uchar'); % unsigned char

                if(i==I & j==J & ~isempty(w))
                    %fprintf('nxp = %d nyp = %d : ',nxp,nyp);
                    % scale weights: they are quantized before written
                    if scaleWeights
                        w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                    end
                    w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                    %A = reshape(w(1:patch_size), [NXP NYP]);
                    PATCH =  [PATCH; w(1:patch_size)'];
                end
            end % if ~ feof
        end
    end % loop over post-synaptic neurons
    nts = nts + 1;
    if (mod(nts,50) == 0)
        %imagesc(a_color*A+b_color);
        %imagesc(A', 'CDataMapping','direct'); 
        % NOTE: It seems that I need A' here!!!
        %colorbar
        %pause(0.1)
        %fprintf('%d\n',nts);
    end
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
    pause
    
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
        numPatches = 0;
        NXP = 0;
        NYP=0;
        NFP=0;
        minVal=0;
        maxVal=0;
    end
else
   disp('eof found: return'); 
   time = -1;
   numPatches = 0;
   NXP = 0;
   NYP=0;
   NFP=0;
   minVal=0;
   maxVal=0;
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
