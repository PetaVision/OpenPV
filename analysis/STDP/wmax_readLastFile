function D = wmax_readFile(fname, l_name, xScale, yScale)
% reds and plots last configuration of "wMax" fields. 
% the wMax thresholds are written by HyPerConn::outputState()
% which calls write() in fileio.cpp
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as argumrnts
% NOTE: Needs to be implemented using ~/Documents/MATLAB/Kmeans


global input_dir  NX NY 

filename = fname;
filename = [input_dir, filename];

fprintf('read %s from %s\n',l_name, filename);

xShare = 4; % define the size of the layer patch that 
yShare = 4; % contains neurons that have the same receptive field.

nPad = 0;

NXscaled = NX * xScale; % L1 size
NYscaled = NY * yScale;

% data lives in the extended space? maybe not!
% need to pass an extended boolean variable

numItems = (NXscaled + 2*nPad)*(NYscaled + 2*nPad);

nxMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
nyMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
             

debug = 0;
first_record = 1;
    
if exist(filename,'file')
    
    fid = fopen(filename, 'r', 'native');
    k = 0;
    
    while(~feof(fid))
        
        if first_record
            % read header and time
            [time, NXread, NYread, NFread, numParams] =  readFirstHeader(fid);
            fprintf('time = %f numParams = %d NX = %d NY = %d NF = %d\n',...
                time,numParams,NXread,NYread,NFread);
            first_record = 0;
        end
        
        %params = fread(fid, numParams, 'int')
        % read time
        if ~first_record
            time = fread(fid,1,'float64');
            if isempty(time)
                return
            else
                %fprintf('time = %f\n',time);
            end
        end
        
        % read the data field
        [data, count] = fread(fid, numItems, 'float32'); 
        size(data)
        fprintf('count = %d\n',count);
        k = k + 1;
        D(k,:) = data;
        
        if 0
            recon2D = reshape(data, [NXscaled, NYscaled] );
            %recon2D = rot90(recon2D);
            %recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            imagesc( recon2D' );  % plots recon2D as an image
            colorbar
            axis square
            axis off
            pause
        end
    end % end while loop
    
    fclose(fid);
    
else

    disp(['Skipping, could not open ', filename]);

end

% End primary function
%


function [time, NX, NY, NF, numParams] = readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
head = fread(fid,3,'int')

% type is 4 for Voltage files
if head(3) ~= 1 % PVP_FILE_TYPE
    disp('incorrect file type')
    %return
end
numParams = head(2)-2; % 18 params
fseek(fid,0,'bof'); % rewind file

params = fread(fid, numParams, 'int')
%pause
NX         = params(4);
NY         = params(5);
NF         = params(6);
%fprintf('numParams = %d ',numParams);
%fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
% read time
time = fread(fid,1,'float64');
%fprintf('time = %f\n',time);

%pause

