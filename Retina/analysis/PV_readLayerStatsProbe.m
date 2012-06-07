
function [time,Avg] = PV_readLayerStatsProbe(dirname,filename,nsteps)

filename = ["../../gjkunde/output/",dirname,"/stats",filename,".txt"]
myfile = fopen(filename,"r");
if myfile == -1
  return;
endif

time   = [];
N      = [];
Total  = [];
Min    = [];
Avg    = [];
Max    = [];


time   = zeros(nsteps, 1);
N      = zeros(nsteps, 1);
Total  = zeros(nsteps, 1);
Min    = zeros(nsteps, 1);
Avg    = zeros(nsteps, 1);
Max    = zeros(nsteps, 1);




for i_step = 1:nsteps
    name          = fscanf(myfile, '%s', 1);
    time(i_step)  = fscanf(myfile, ' t=%f', 1);
    N(i_step)     = fscanf(myfile, ' N=%i', 1);
    Total(i_step) = fscanf(myfile, ' Total=%f', 1);
    Min(i_step)   = fscanf(myfile, ' Min=%f', 1);
    Avg(i_step)   = fscanf(myfile, ' Avg=%f', 1);
    Max(i_step)   = fscanf(myfile, ' Hz(/dt ms) Max=%f', 1);
end

fclose(myfile);

