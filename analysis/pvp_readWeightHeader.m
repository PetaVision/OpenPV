function [pvp_header, pvp_index] = pvp_readWeightHeader(filename)

global NUM_BIN_PARAMS 
global NUM_WGT_PARAMS

[pvp_header, pvp_index] = pvp_readHeader(filename);

if isempty(pvp_header)
  return
end

pvp_index.WGT_NXP = pvp_index.TIME+1;
pvp_index.WGT_NYP = pvp_index.TIME+2;
pvp_index.WGT_NFP = pvp_index.TIME+3;
pvp_index.WGT_MIN = pvp_index.TIME+4;
pvp_index.WGT_MAX = pvp_index.TIME+5;
pvp_index.WGT_NUMPATCHES = pvp_index.TIME+6;

fid = fopen(filename, 'r');
if fid == -1
    pvp_header = [];
    return;
end

pvp_weight_header = zeros(NUM_WGT_PARAMS,1);
status = fseek(fid, NUM_BIN_PARAMS*sizeof(int32(0))  );
%pvp_weight_header = fread(fid, NUM_WGT_PARAMS, 'int32'); 
pvp_weight_header(1:3) = fread(fid, 3, 'int32'); 
pvp_weight_header(4:5) = fread(fid, 2, 'float32'); 
pvp_weight_header(6) = fread(fid, 1, 'int32'); 
pvp_header = [pvp_header(1:NUM_BIN_PARAMS-1); pvp_weight_header];
fclose(fid);



