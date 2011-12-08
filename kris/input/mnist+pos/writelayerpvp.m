function writelayerpvp(A,filename,listoftimes)
% writelayerpvp(A,filename)
% A is a 4-dimensional array with features in the third dimension
% and time in the fourth dimension.
% filename is, well, a filename to write the data to
% The file is a pvp file with file type PVP_NONSPIKING_ACT_FILE_TYPE.

if ~exist('listoftimes','var') || isempty(listoftimes)
    listoftimes = 0:size(A,4)-1;
end

header = zeros(20,1);
header(1) = 80; % numParams*sizeof(int)
header(2) = 20; % numParams
header(3) = 4;  % PVP_NONSPIKING_ACT_FILE_TYPE
header(4) = size(A,1);
header(5) = size(A,2);
header(6) = size(A,3);
header(7) = 1;  % Number of blocks
header(8) = numel(A)/size(A,4);
header(9) = 4;  % Data size (4=float)
header(10) = 3; % Data type (3=PV_FLOAT_TYPE)
header(11) = 1; % Number of blocks in the x direction
header(12) = 1; % Number of blocks in the y direction
header(13) = size(A,1); % Number of pixels in the x direction, over all blocks
header(14) = size(A,2); %   "    "    "    "    " y  "    "    "    "    "
header(15) = 0; % x-Origin of local block in global coordinates
header(16) = 0; % y-Origin of local block in global coordinates
header(17) = 0; % Size of padding
header(18) = size(A,3); % Number of features
header(19:20) = 0;  % Time stored as a double

numframes = size(A,4);
Afeaturefirst = permute(A,[3 1 2 4]);
Aflat = reshape(Afeaturefirst,numel(A)/numframes,numframes);

fid = fopen(filename,'w');
if fid<0
    error('writelayerpvp:badfile',...
          'File %s could not be opened for writing',filename);
end
fwrite(fid,header,'int32',0,'l');
for k=1:numframes
    fwrite(fid,listoftimes(k),'float64',0,'l'); % Time
    fwrite(fid,Aflat(:,k),'float32',0,'l');
end
fclose(fid);