
[time,Ganglion_RU1]=pvp_readPointLIFprobe("GanglionRU1",{'V'});
[time,Ganglion_RD1]=pvp_readPointLIFprobe("GanglionRD1",{'V'});
[time,Ganglion_RU2]=pvp_readPointLIFprobe("GanglionRU2",{'V'});
[time,Ganglion_RD2]=pvp_readPointLIFprobe("GanglionRD2",{'V'});
size time;
S1 = (Ganglion_RU1-Ganglion_RD1);
S2 = (Ganglion_RU2-Ganglion_RD2);
U  = (Ganglion_RU1-Ganglion_RU2);
D  = (Ganglion_RD1-Ganglion_RD2);

%[R,lag] = xcorr(Ganglion_RU1,Ganglion_RU1);

rtime = 1:time;

plot \
    (rtime,Ganglion_RU1,"1",rtime,Ganglion_RD1-50,"2",rtime,Ganglion_RU2-100,"3",rtime,Ganglion_RD2-150,"4",rtime,S1-85,"5",rtime,S2-185,"5",U-250,"0",rtime,D-300,"0");

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
print -dgif ../octave/GanglionCorrelations.gif
