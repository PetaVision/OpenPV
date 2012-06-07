duration = 600;
[time,Ganglion_RU1] =PV_readPointLIFprobe("GanglionRU1",duration);
[time,Ganglion_RD1] =PV_readPointLIFprobe("GanglionRD1",duration);
[time,Ganglion_RU2] =PV_readPointLIFprobe("GanglionRU2",duration);
[time,Ganglion_RD2] =PV_readPointLIFprobe("GanglionRD2",duration);
size time;
S1 = (Ganglion_RU1-Ganglion_RD1);
S2 = (Ganglion_RU2-Ganglion_RD2);
U  = (Ganglion_RU1-Ganglion_RU2);
D  = (Ganglion_RD1-Ganglion_RD2);

%[R,lag] = xcorr(Ganglion_RU1,Ganglion_RU1);

plot \
    (time,Ganglion_RU1,"1",time,Ganglion_RD1-50,"2",time,Ganglion_RU2-100,"3",time,Ganglion_RD2-150,"4",time,S1-85,"5",time,S2-185,"5",U-250,"0",time,D-300,"0");

title("Ganglion Rectangle Correlations");

t2=text (25,-50,"RU1");
set(t2,'color',[1 0 0]);
t3=text (25,-100,"RD1");
set(t3,'color',[0 1 0]);
t4=text (25,-150,"RU2");
set(t4,'color',[0 0 1])
t5=text (25,-200,"RD2");
set(t5,'color',[1 0 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
grid;
%%axis([0,600,-350,-40]);
print -dgif ../../gjkunde/octave/GanglionCorrelations.gif
