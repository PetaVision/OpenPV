function [vmem_time, vmem_G_E, vmem_G_I, vmem_G_IB, vmem_V, vmem_Vth, vmem_a] = ...
    ptprobe_readV(filename)

global SPIKE_PATH
%global NCOLS % for the current layer
%global NFEATURES  % for the current layer
global BEGIN_TIME END_TIME
global DELTA_T
begin_step = floor(BEGIN_TIME / DELTA_T) + 1;
vmem_steps = ceil( ( END_TIME - BEGIN_TIME ) / DELTA_T );

vmem_time = [];
vmem_G_E = [];
vmem_G_I = [];
vmem_G_IB = [];
vmem_V = [];1
vmem_Vth = [];
vmem_a = [];

filename = [SPIKE_PATH, filename];
if ~exist(filename,'file')
    disp(['~exist(filename,''file'') in ptprobe file: ', filename]);
    return;
end

fid = fopen(filename, 'r');
if fid == -1
    pvp_header = [];
    return;
end

vmem_time = zeros(vmem_steps, 1);
vmem_G_E = zeros(vmem_steps, 1);
vmem_G_I = zeros(vmem_steps, 1);
vmem_G_IB = zeros(vmem_steps, 1);
vmem_V = zeros(vmem_steps, 1);
vmem_Vth = zeros(vmem_steps, 1);
vmem_a = zeros(vmem_steps, 1);

%%keyboard;
for i_step = 1:begin_step-1
  vmem_str = fgets(fid);
%%    vmem_name = fscanf(fid, "[\0,0-9,a-z,A-Z, ]*:]", 1);
%%    vmem_time_tmp = fscanf(fid, " t=%f", 1);
%%    vmem_time_tmp = fscanf(fid, " k=%i", 1);
%%    vmem_G_E_tmp = fscanf(fid, " G_E=%f", 1);
%%    vmem_G_I_tmp = fscanf(fid, " G_I=%f", 1);
%%    vmem_G_IB_tmp = fscanf(fid, " G_IB=%f", 1);
%%    vmem_V_tmp = fscanf(fid, " V=%f", 1);
%%    vmem_Vth_tmp = fscanf(fid, " Vth=%f", 1);
%%    vmem_R_tmp = fscanf(fid, " R= %f", 1);
%%    vmem_a_tmp = fscanf(fid, " a=%f\n", 1);
end
%%keyboard;
for i_step = 1:vmem_steps 
  vmem_str = fgets(fid);

  time_start_ndx = strfind(vmem_str, "t=");
  time_start_ndx = time_start_ndx(1) + 2;
  time_end_ndx = strfind(vmem_str, "k=");
  time_end_ndx = time_end_ndx(1) - 1;
  vmem_time(i_step) = str2num(vmem_str(time_start_ndx:time_end_ndx));

  G_E_start_ndx = strfind(vmem_str, "G_E=");
  G_E_start_ndx = G_E_start_ndx(1) + 4;
  G_E_end_ndx = strfind(vmem_str, "G_I=");
  G_E_end_ndx = G_E_end_ndx(1) - 1;
  vmem_G_E(i_step) = str2num(vmem_str(G_E_start_ndx:G_E_end_ndx));

  G_I_start_ndx = strfind(vmem_str, "G_I=");
  G_I_start_ndx = G_I_start_ndx(1) + 4;
  G_I_end_ndx = strfind(vmem_str, "G_IB=");
  G_I_end_ndx = G_I_end_ndx(1) - 1;
  vmem_G_I(i_step) = str2num(vmem_str(G_I_start_ndx:G_I_end_ndx));

  G_IB_start_ndx = strfind(vmem_str, "G_IB=");
  G_IB_start_ndx = G_IB_start_ndx(1) + 5;
  G_IB_end_ndx = strfind(vmem_str, "V=");
  G_IB_end_ndx = G_IB_end_ndx(1) - 1;
  vmem_G_IB(i_step) = str2num(vmem_str(G_IB_start_ndx:G_IB_end_ndx));

  V_start_ndx = strfind(vmem_str, "V=");
  V_start_ndx = V_start_ndx(1) + 2;
  V_end_ndx = strfind(vmem_str, "Vth=");
  V_end_ndx = V_end_ndx(1) - 1;
  vmem_V(i_step) = str2num(vmem_str(V_start_ndx:V_end_ndx));

  Vth_start_ndx = strfind(vmem_str, "Vth=");
  Vth_start_ndx = Vth_start_ndx(1) + 4;
  Vth_end_ndx = strfind(vmem_str, "a=");
  Vth_end_ndx = Vth_end_ndx(1) - 1;
  vmem_Vth(i_step) = str2num(vmem_str(Vth_start_ndx:Vth_end_ndx));

  a_start_ndx = strfind(vmem_str, "a=");
  a_start_ndx = a_start_ndx(1) + 2;
  a_end_ndx = strfind(vmem_str, "\n");
  %%a_end_ndx = a_end_ndx(1) - 1;
  vmem_a(i_step) = str2num(vmem_str(a_start_ndx:a_end_ndx));
end
fclose(fid);

max_V = max(vmem_V(:));
min_V = min(vmem_V(:));
vmem_a = max_V .* ( vmem_a > 0 ) + min_V .* ( vmem_a == 0 );
   

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
			    




