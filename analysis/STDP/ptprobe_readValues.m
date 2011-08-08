function [vmem_time, vmem_G_E, vmem_G_I, vmem_G_IB, vmem_V, vmem_Vth, vmem_R, vmem_Wmax, vmem_aWmax, vmem_VthRest, vmem_a] = ...
    ptprobe_readValues(filename, begin_step, end_step)

global input_dir 
%global NCOLS % for the current layer
%global NFEATURES  % for the current layer

filename = [input_dir, filename];
if ~exist(filename,'file')
    disp(['~exist(filename,''file'') in ptprobe file: ', filename]);
    return;
end
fprintf('Read probe data from %s\n',filename);

fid = fopen(filename, 'r');
if fid == -1
    pvp_header = [];
    return;
end
if 0 
    n2_time_steps = end_step - begin_step+1;
    vmem_time = zeros(n2_time_steps, 1);
    vmem_G_E = zeros(n2_time_steps, 1);
    vmem_G_I = zeros(n2_time_steps, 1);
    vmem_G_IB = zeros(n2_time_steps, 1);
    vmem_V = zeros(n2_time_steps, 1);
    vmem_Vth = zeros(n2_time_steps, 1);
    vmem_a = zeros(n2_time_steps, 1);
end
for i_step = 1:(begin_step-1)
    bla = fscanf(fid, '%s', 1);
    bla = fscanf(fid, ' t=%f', 1);
    bla = fscanf(fid, ' k=%d', 1);
    bla = fscanf(fid, ' G_E=%f', 1);
    bla = fscanf(fid, ' G_I=%f', 1);
    bla = fscanf(fid, ' G_IB=%f', 1);
    bla = fscanf(fid, ' V=%f', 1);
    bla = fscanf(fid, ' Vth=%f', 1);
    bla = fscanf(fid, ' R=%f', 1);
    bla = fscanf(fid, ' Wmax=%f', 1);
    bla = fscanf(fid, ' aWmax=%f', 1);
    bla = fscanf(fid, ' VthRest=%f', 1);
    bla = fscanf(fid, ' a=%f\n', 1);
end
i_step = 0;
while ~feof(fid)
    if ~feof(fid)
        i_step = i_step + 1;
        %fprintf('i_step = %d\n',i_step);
    else
        fprintf('i_step = %d break!\n',i_step);
        break
    end
%for i_step = 1:(end_step-begin_step+1) 
    vmem_name            = fscanf(fid, '%s', 1);
    vmem_time(i_step)    = fscanf(fid, ' t=%f', 1);
    k                    = fscanf(fid, ' k=%d', 1);
    vmem_G_E(i_step)     = fscanf(fid, ' G_E=%f', 1);
    vmem_G_I(i_step)     = fscanf(fid, ' G_I=%f', 1);
    vmem_G_IB(i_step)    = fscanf(fid, ' G_IB=%f', 1);
    vmem_V(i_step)       = fscanf(fid, ' V=%f', 1);
    vmem_Vth(i_step)     = fscanf(fid, ' Vth=%f', 1);
    vmem_R(i_step)       = fscanf(fid, ' R=%f', 1);
    vmem_Wmax(i_step)    = fscanf(fid, ' Wmax=%f', 1);
    vmem_aWmax(i_step)   = fscanf(fid, ' aWmax=%f', 1);
    vmem_VthRest(i_step) = fscanf(fid, ' VthRest=%f', 1);
    vmem_a(i_step)       = fscanf(fid, ' a=%f\n', 1);
end
fclose(fid);

max_V = max(vmem_V);
min_V = min(vmem_V);
vmem_a = max_V * ( vmem_a > 0 ) + min_V * ( vmem_a == 0 );
   

% vmem_row_loc = findstr( vmem_name, '(' ) + 1;
% vmem_col_loc2 = findstr( vmem_name, ',' );
% vmem_col_loc = vmem_col_loc2(1) + 1;
% vmem_feature_loc = vmem_col_loc2(2) + 1;
% vmem_feature_loc2 = findstr( vmem_name, ')' ) - 1;
% vmem_row_str = vmem_name(vmem_row_loc:vmem_col_loc-2);
% vmem_col_str = vmem_name(vmem_col_loc:vmem_feature_loc-2);
% vmem_feature_str = vmem_name(vmem_feature_loc:vmem_feature_loc2);
% vmem_row = str2double( vmem_row_str );
% vmem_col = str2double( vmem_col_str );
% vmem_feature = str2double( vmem_feature_str );
% vmem_index = 0;			    
%vmem_index = ( ( vmem_row ) * NCOLS + ( vmem_col ) ) * NFEATURES + vmem_feature;
			    




