
function [time,V] = PV_readPointLIFprobeA(filename,nsteps)

filename = ["../output/pt",filename,".txt"]
myfile = fopen(filename,"r");
if myfile == -1
  error('File not found');
  return;
endif

time = [];
G_E  = [];
G_I  = [];
G_IB = [];
V    = [];
Vth  = [];
A    = [];
location = [];

time = zeros(nsteps, 1);
G_E  = zeros(nsteps, 1);
G_I  = zeros(nsteps, 1);
G_IB = zeros(nsteps, 1);
V    = zeros(nsteps, 1);
Vth  = zeros(nsteps, 1);
A    = zeros(nsteps, 1);
location = zeros(nsteps, 1);

  for i_step = 1:nsteps
     line_str = fgets(myfile);

     name_pos = index(line_str, ":");
     name_tmp = line_str(1:name_pos-1);

     time_begin_ndx = index(line_str, "t=");
     time_end_ndx = index(line_str, "k=");
     time_tmp_str = line_str(time_begin_ndx+2:time_end_ndx-1);
     time(i_step) = str2double(time_tmp_str);
    
     location_begin_ndx = index(line_str, "k=");
     location_end_ndx = index(line_str, "G_E=");
     location_tmp_str = line_str(location_begin_ndx+2:location_end_ndx-1);
     location(i_step) = str2double(location_tmp_str);

     G_E_begin_ndx = index(line_str, "G_E=");
     G_E_end_ndx = index(line_str, "G_I=");
     G_E_tmp_str = line_str(G_E_begin_ndx+5:G_E_end_ndx-1);
     G_E(i_step) = str2double(G_E_tmp_str);

     G_I_begin_ndx = index(line_str, "G_I=");
     G_I_end_ndx = index(line_str, "G_IB=");
     G_I_tmp_str = line_str(G_I_begin_ndx+5:G_I_end_ndx-1);
     G_I(i_step) = str2double(G_I_tmp_str);

     G_IB_begin_ndx = index(line_str, "G_IB=");
     G_IB_end_ndx = index(line_str, "V=");
     G_IB_tmp_str = line_str(G_IB_begin_ndx+6:G_IB_end_ndx-1);
     G_IB(i_step) = str2double(G_IB_tmp_str);
   
     V_begin_ndx = index(line_str, "V=");
     V_end_ndx = index(line_str, "Vth=");
     V_tmp_str = line_str(V_begin_ndx+2:V_end_ndx-1);
     V(i_step) = str2double(V_tmp_str);

     Vth_begin_ndx = index(line_str, "Vth=");
     Vth_end_ndx = index(line_str, "a=");
     Vth_tmp_str = line_str(Vth_begin_ndx+4:Vth_end_ndx-2);
     Vth(i_step) = str2double(Vth_tmp_str);
        
     A_begin_ndx = index(line_str, "a=");
     A_tmp_str = line_str(A_begin_ndx+2:84);
     A(i_step) = str2double(A_tmp_str);

end


%BB = zeros(nsteps,8);
%BB(:,1) = time(i_step);
%BB(:,2) = location(i_step);
%BB(:,3) = G_E(i_step);
%BB(:,4) = G_I(i_step);
%BB(:,5) = G_IB(i_step);
%BB(:,6) = V(i_step);
%BB(:,7) = Vth(i_step);
%BB(:,8) = A(i_step);

%disp(BB);

%name = fscanf(myfile, "%s", 1)
%time = fscanf(myfile," t=%3.1f", 1)
%location = fscanf(myfile, " k=%i", 1)
%G_E  = fscanf(myfile, " G_E=%5.3f", 1)
%G_I  = fscanf(myfile, " G_I=%5.3f", 1)
%G_IB = fscanf(myfile, " G_IB=%5.3f", 1)
%V    = fscanf(myfile, " V=%6.3f", 1)
%Vth  = fscanf(myfile, " Vth=%10.3f", 1)
%R            = fscanf(myfile, " R= %f", 1)
%A_tmp    = fscanf(myfile, " a=%f\n", 1)



fclose(myfile);

