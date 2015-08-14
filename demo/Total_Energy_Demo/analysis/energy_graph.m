fidrecon = fopen('../output/recon_error_l2norm.txt');
assert(fidrecon>0);
recon = zeros(0,2);
fgetlresult = fgetl(fidrecon);
while ischar(fgetlresult)
   lineresults = sscanf(fgetlresult, 't = %f b = 0 numNeurons = %d, L2-norm squared = %f');
   recon = [recon; lineresults([1 3])'];
   fgetlresult = fgetl(fidrecon);
end%while
fclose(fidrecon); clear fidrecon;

fidsparsity = fopen('../output/cost_function.txt');
assert(fidsparsity>0);
sparsitypenalty = zeros(0,2);
fgetlresult = fgetl(fidsparsity);
while ischar(fgetlresult)
   lineresults = sscanf(fgetlresult, 't = %f b = 0 numNeurons = %d L1-norm = %f');
   sparsitypenalty = [sparsitypenalty; lineresults([1 3])'];
   fgetlresult = fgetl(fidsparsity);
end%while
fclose(fidsparsity); clear fidsparsity;

fidtotal = fopen('../output/total_energy.txt');
assert(fidtotal>0);
totalenergy = zeros(0,2);
fgetlresult = fgetl(fidtotal);
while ischar(fgetlresult)
   lineresults = sscanf(fgetlresult, '"Total_Energy_Probe",%f,0,%f\n');
   totalenergy = [totalenergy; lineresults'];
   fgetlresult = fgetl(fidtotal);
end%while
fclose(fidtotal); clear fidtotal;

clf;
hold on;
plot(recon(:,1), 0.5*recon(:,2), 'r');
plot(sparsitypenalty(:,1), 0.025*sparsitypenalty(:,2), 'g');
plot(totalenergy(:,1), totalenergy(:,2), 'b');
hold off;
