function [vmem_time, vmem_G_E, vmem_G_I, vmem_V, vmem_Vth, vmem_a, vmem_index, vmem_row, vmem_col, vmem_feature, vmem_name] = ...
    ptprobe_readV(filename)

global output_path 
global N NROWS NCOLS % for the current layer
global NFEATURES  % for the current layer
global NO NK dK % for the current layer
global n_time_steps begin_step end_step time_steps

filename = [output_path, filename];
if ~exist(filename,'file')
    disp(['~exist(filename,''file'') in ptprobe file: ', filename]);
    return;
end

fid = fopen(filename, 'r');
if fid == -1
    pvp_header = [];
    return;
end

vmem_time = zeros(n_time_steps, 1);
vmem_G_E = zeros(n_time_steps, 1);
vmem_G_I = zeros(n_time_steps, 1);
vmem_V = zeros(n_time_steps, 1);
vmem_Vth = zeros(n_time_steps, 1);
vmem_a = zeros(n_time_steps, 1);

for i_step = 1:n_time_steps
    vmem_name = fscanf(fid, '%s', 1);
    vmem_time(i_step) = fscanf(fid, ' t=%f', 1);
    vmem_G_E(i_step) = fscanf(fid, ' G_E=%f', 1);
    vmem_G_I(i_step) = fscanf(fid, ' G_I=%f', 1);
    vmem_V(i_step) = fscanf(fid, ' V=%f', 1);
    vmem_Vth(i_step) = fscanf(fid, ' Vth=%f', 1);
    vmem_a(i_step) = fscanf(fid, ' a=%f\n', 1);
end
fclose(fid);
   

vmem_row_loc = findstr( vmem_name, '(' ) + 1;
vmem_col_loc2 = findstr( vmem_name, ',' );
vmem_col_loc = vmem_col_loc2(1) + 1;
vmem_feature_loc = vmem_col_loc2(2) + 1;
vmem_feature_loc2 = findstr( vmem_name, ')' ) - 1;
vmem_row_str = vmem_name(vmem_row_loc:vmem_col_loc-2);
vmem_col_str = vmem_name(vmem_col_loc:vmem_feature_loc-2);
vmem_feature_str = vmem_name(vmem_feature_loc:vmem_feature_loc2);
vmem_row = str2double( vmem_row_str );
vmem_col = str2double( vmem_col_str );
vmem_feature = str2double( vmem_feature_str );
vmem_index = ( ( vmem_row - 1 ) * NCOLS + ( vmem_col - 1 ) ) * NFEATURES;




