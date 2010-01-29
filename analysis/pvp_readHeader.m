function [pvp_header, pvp_index] = pvp_readHeader(filename)

pvp_index = struct;
pvp_index.HEADER_SIZE  = 0+1;
pvp_index.NUM_PARAMS   = 1+1;
pvp_index.FILE_TYPE    = 2+1;
pvp_index.NX           = 3+1;
pvp_index.NY           = 4+1;
pvp_index.NF           = 5+1;
pvp_index.NUM_RECORDS  = 6+1;
pvp_index.RECORD_SIZE  = 7+1;
pvp_index.DATA_SIZE    = 8+1;
pvp_index.DATA_TYPE    = 9+1;
pvp_index.NX_PROCS    = 10+1;
pvp_index.NY_PROCS    = 11+1;
pvp_index.NX_GLOBAL   = 12+1;
pvp_index.NY_GLOBAL   = 13+1;
pvp_index.KX0         = 14+1;
pvp_index.KY0         = 15+1;
pvp_index.NPAD        = 16+1;
pvp_index.NBANDS      = 17+1;
pvp_index.TIME        = 18+1;

fid = fopen(filename, 'r');
if fid == -1
    pvp_header = [];
    return;
end
NUM_BIN_PARAMS = 18;
pvp_header = fread(fid, NUM_BIN_PARAMS, 'int32');
pvp_time = fread(fid, 1, 'double');
fclose(fid);
pvp_header(pvp_index.TIME) = pvp_time;


