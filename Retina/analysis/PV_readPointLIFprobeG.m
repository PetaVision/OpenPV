
function [time,G_E,G_I,G_IB] = PV_readPointLIFprobeG(filename,nsteps)

filename = ["../../gjkunde/output/pt",filename,".txt"]
myfile = fopen(filename,"r");
if myfile == -1
  return;
endif

time = [];
G_E  = [];
G_I  = [];
G_IB = [];
V    = [];
Vth  = [];
A    = [];


time = zeros(nsteps, 1);
G_E  = zeros(nsteps, 1);
G_I  = zeros(nsteps, 1);
G_IB = zeros(nsteps, 1);
V    = zeros(nsteps, 1);
Vth  = zeros(nsteps, 1);
A    = zeros(nsteps, 1);



for i_step = 1:nsteps
    name         = fscanf(myfile, '%s', 1);
    time(i_step) = fscanf(myfile, ' t=%f', 1);
    location     = fscanf(myfile, ' k=%i', 1);
    G_E(i_step)  = fscanf(myfile, ' G_E=%f', 1);
    G_I(i_step)  = fscanf(myfile, ' G_I=%f', 1);
    G_IB(i_step) = fscanf(myfile, ' G_IB=%f', 1);
    V(i_step)    = fscanf(myfile, ' V=%f', 1);
    Vth(i_step)  = fscanf(myfile, ' Vth=%f', 1);
    R            = fscanf(myfile, ' R= %f', 1);
    A(i_step)    = fscanf(myfile, ' a=%f\n', 1);
end

fclose(myfile);

