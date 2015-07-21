function writepvpfile(A,filename)
% writepvpfile(A,filename)
% A is a 3-dimensional array with features in the third dimension
% filename is, well, a fielname to write the data to
% The file is a pvp file with file type PVP_FILE_TYPE (i.e., 1)

header = zeros(18,1);
header(1) = 80; % numParams*sizeof(int)
header(2) = 20; % numParams
header(3) = 1;  % PVP_FILE_TYPE
header(4) = size(A,1);
header(5) = size(A,2);
header(6) = size(A,3);
header(7) = 1;  % Number of blocks
header(8) = numel(A);
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

pvptime = 0;  % Time stored as a double

Afeaturefirst = permute(A,[3 1 2]);

fid = fopen(filename,'w');
if fid<0
    error('writelayerpvp:badfile',...
          'File %s could not be opened for writing',filename);
end
fwrite(fid,header,'int32',0,'l');
fwrite(fid,pvptime,'float64',0,'l'); % Time
fwrite(fid,Afeaturefirst(:),'float32',0,'l');
fclose(fid);