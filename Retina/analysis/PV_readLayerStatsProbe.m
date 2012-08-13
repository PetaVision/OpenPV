
function [time,Avg] = PV_readLayerStatsProbe(dirname,filename)

filename = ["../output/",dirname,"/stats",filename,".txt"]
myfile = fopen(filename,"r");
if myfile == -1
  error("No input file");
  return;
end %if

count = 0;
while ~feof(myfile)
lines = fgetl(myfile);
count = count + 1;
end %while

nLines = count;

disp(count);

fclose(myfile);

myfile = fopen(filename,"r");

N      = zeros(nLines, 1);
Total  = zeros(nLines, 1);
Min    = zeros(nLines, 1);
Avg    = zeros(nLines, 1);
Max    = zeros(nLines, 1);


  for i_step = 1:nsteps
    
     line_str = fgets(myfile);
    
        name_pos = index(line_str, ":");
        name_tmp = line_str(1:name_pos-1);

        N_begin_ndx = index(line_str, "N==");
        N_end_ndx = index(line_str, "Total=");
        N_tmp_str = line_str(N_begin_ndx+3:N_end_ndx-1);
        N(i_step) = str2double(N_tmp_str);

        Total_begin_ndx = index(line_str, "Total==");
        Total_end_ndx = index(line_str, "G_E=");
        Total_tmp_str = line_str(Total_begin_ndx+7:Total_end_ndx-1);
        Total(i_step) = str2double(Total_tmp_str);

        Min_begin_ndx = index(line_str, "Min==");
        Min_end_ndx = index(line_str, "G_E=");
        Min_tmp_str = line_str(Min_begin_ndx+5:Min_end_ndx-1);
        Min(i_step) = str2double(Min_tmp_str);

        Avg_begin_ndx = index(line_str, "Avg==");
        Avg_end_ndx = index(line_str, "Hz");
        Avg_tmp_str = line_str(Avg_begin_ndx+5:Avg_end_ndx-1);
        Avg(i_step) = str2double(Avg_tmp_str);

        Max_begin_ndx = index(line_str, "Max==");
        Max_tmp_str = line_str(Max_begin_ndx+5:93);
        Max(i_step) = str2double(Max_tmp_str);

   %name          = fscanf(myfile, '%s', 1);
   %time(i_step)  = fscanf(myfile, ' t=%f', 1);
   %N(i_step)     = fscanf(myfile, ' N=%i', 1);
   %Total(i_step) = fscanf(myfile, ' Total=%f', 1);
   %Min(i_step)   = fscanf(myfile, ' Min=%f', 1);
   %Avg(i_step)   = fscanf(myfile, ' Avg=%f', 1);
   %Max(i_step)   = fscanf(myfile, ' Hz(/dt ms) Max=%f', 1);

end

time = nLines;

fclose(myfile);

