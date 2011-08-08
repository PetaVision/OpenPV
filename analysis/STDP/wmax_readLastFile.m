function data = wmax_readLastFile(fname, l_name, xScale, yScale)
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

%numItems = (NXscaled + 2*nPad)*(NYscaled + 2*nPad);

%numItems = NXscaled*NYscaled;

nxMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
nyMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
             

debug = 0;

% Note: We write rows first.  When we reshape, reshaping goes column first.
% For eample for a 
%     1     2     3
%     4     5     6     (1)
%     7     8     9
% we write it as a = [1 2 3, 4 5 6, 7 8 9]. When we reshape it,
% > reshape(a,[ 3 3 ] )
% we get
%
%     1     4     7
%     2     5     8    (2)
%     3     6     9
% instead of the correct answer (1).
     
if exist(filename,'file')

    fid = fopen(filename, 'r', 'native');


    % read header and time
    [time, NXread, NYread, NFread, numParams] =  readFirstHeader(fid);
    fprintf('time = %f numParams = %d NX = %d NY = %d NF = %d\n',...
        time,numParams,NXread,NYread,NFread);


    % read the data field
    [data, count] = fread(fid, NXread*NYread, 'float32');
    size(data)
    fprintf('count = %d\n',count);

    if 0
        recon2D = reshape(data, [NXread, NYread] );
        %recon2D = rot90(recon2D);
        %recon2D = 1 - recon2D;
        %figure('Name','Rate Array ');
        imagesc( recon2D' );  % plots recon2D as an image
        colorbar
        axis square
        axis off
        pause
    end


    fclose(fid);

else

    disp(['Skipping, could not open ', filename]);

end

% End primary function
%


function [time, NX, NY, NF, numParams] = readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
head = fread(fid,3,'int');

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

