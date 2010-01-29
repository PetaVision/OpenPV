function [Vmem_data, vmem_row, vmem_col, vmem_feature, vmem_name] = ptprobe_readV(filename)

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


%    fprintf(fp, "%s t=%6.1f", msg, time);
%    fprintf(fp, " G_E=%6.3f", clayer->G_E[k]);
%    fprintf(fp, " G_I=%6.3f", clayer->G_I[k]);
%    fprintf(fp, " V=%6.3f",   clayer->V[k]);
%    fprintf(fp, " Vth=%6.3f", clayer->Vth[k]);
%    fprintf(fp, " a=%3.1f\n", clayer->activity->data[kex]);

msg_template = "%s";
time_template = " t=%6.1f";
G_E_template = " G_E=%6.3f";
G_I_template = " G_I=%6.3f";
V_template = " V=%6.3f";
Vth_template = " Vth=%6.3f";
a_template = " a=%6.1f\n";
ptprobe_template = strcat(msg_template, time_template, G_E_template, G_I_template, V_template, Vth_template, a_template );
ptprobe_size = [ n_time_steps, 7 ];
[Vmem_data, count] = fscanf(fid, ptprobe_template, ptprobe_size);

close(fid);

vmem_name = Vmem_data(1,1);
