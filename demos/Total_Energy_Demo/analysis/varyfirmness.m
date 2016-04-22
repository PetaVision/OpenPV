[t_hardreconerror, hardreconerror] = readenergydata('../output-HardThreshold/recon_error_l2norm.txt',...
        't = %f b = 0 numNeurons = %d L2-norm squared = %f',...
        [1 3]);

[t_hardcostfunc, hardcostfunc] = readenergydata('../output-HardThreshold/cost_function.txt',...
        't = %f b = 0 numNeurons = %d L0-norm = %f',...
        [1 3]);

[t_hardtotalenergy, hardtotalenergy] = readenergydata('../output-HardThreshold/total_energy.txt',...
        '"Total_Energy_Probe",%f,0,%f\n',...
        [1 2]);

figure(1);
clf;
hold on;
plot(t_hardreconerror, 0.5*hardreconerror, 'r');
plot(t_hardcostfunc, 0.03125*hardcostfunc, 'g');
plot(t_hardtotalenergy, hardtotalenergy, 'b');
hold off;
title('HardThreshold');

[t_firmreconerror, firmreconerror] = readenergydata('../output-AlmostHardThreshold/recon_error_l2norm.txt',...
        't = %f b = 0 numNeurons = %d L2-norm squared = %f',...
        [1 3]);

[t_firmcostfunc, firmcostfunc] = readenergydata('../output-AlmostHardThreshold/cost_function.txt',...
        't = %f b = 0 numNeurons = %d Cost function = %f',...
        [1 3]);

[t_firmtotalenergy, firmtotalenergy] = readenergydata('../output-AlmostHardThreshold/total_energy.txt',...
        '"Total_Energy_Probe",%f,0,%f\n',...
        [1 2]);

figure(2);
clf;
hold on;
plot(t_firmreconerror, 0.5*firmreconerror, 'r');
plot(t_firmcostfunc, 0.25*firmcostfunc, 'g');
plot(t_firmtotalenergy, firmtotalenergy, 'b');
hold off;
title('AlmostHardThreshold');

[t_firmreconerror, firmreconerror] = readenergydata('../output-FirmThreshold/recon_error_l2norm.txt',...
        't = %f b = 0 numNeurons = %d L2-norm squared = %f',...
        [1 3]);

[t_firmcostfunc, firmcostfunc] = readenergydata('../output-FirmThreshold/cost_function.txt',...
        't = %f b = 0 numNeurons = %d Cost function = %f',...
        [1 3]);

[t_firmtotalenergy, firmtotalenergy] = readenergydata('../output-FirmThreshold/total_energy.txt',...
        '"Total_Energy_Probe",%f,0,%f\n',...
        [1 2]);

figure(3);
clf;
hold on;
plot(t_firmreconerror, 0.5*firmreconerror, 'r');
plot(t_firmcostfunc, 0.25*firmcostfunc, 'g');
plot(t_firmtotalenergy, firmtotalenergy, 'b');
hold off;
title('FirmThreshold');

[t_firmreconerror, firmreconerror] = readenergydata('../output-AlmostSoftThreshold/recon_error_l2norm.txt',...
        't = %f b = 0 numNeurons = %d L2-norm squared = %f',...
        [1 3]);

[t_firmcostfunc, firmcostfunc] = readenergydata('../output-AlmostSoftThreshold/cost_function.txt',...
        't = %f b = 0 numNeurons = %d Cost function = %f',...
        [1 3]);

[t_firmtotalenergy, firmtotalenergy] = readenergydata('../output-AlmostSoftThreshold/total_energy.txt',...
        '"Total_Energy_Probe",%f,0,%f\n',...
        [1 2]);

figure(4);
clf;
hold on;
plot(t_firmreconerror, 0.5*firmreconerror, 'r');
plot(t_firmcostfunc, 0.25*firmcostfunc, 'g');
plot(t_firmtotalenergy, firmtotalenergy, 'b');
hold off;
title('AlmostSoftThreshold');

[t_softreconerror, softreconerror] = readenergydata('../output-SoftThreshold/recon_error_l2norm.txt',...
        't = %f b = 0 numNeurons = %d L2-norm squared = %f',...
        [1 3]);

[t_softcostfunc, softcostfunc] = readenergydata('../output-SoftThreshold/cost_function.txt',...
        't = %f b = 0 numNeurons = %d L1-norm = %f',...
        [1 3]);

[t_softtotalenergy, softtotalenergy] = readenergydata('../output-SoftThreshold/total_energy.txt',...
        '"Total_Energy_Probe",%f,0,%f\n',...
        [1 2]);

figure(5);
clf;
hold on;
plot(t_softreconerror, 0.5*softreconerror, 'r');
plot(t_softcostfunc, 0.25*softcostfunc, 'g');
plot(t_softtotalenergy, softtotalenergy, 'b');
hold off;
title('SoftThreshold');
