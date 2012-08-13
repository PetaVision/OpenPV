
[time,Ganglion_RU1] =PV_readPointLIFprobeA("GanglionRU1",{'A'});
[time,Ganglion_RD1] =PV_readPointLIFprobeA("GanglionRD1",{'A'});
[time,Ganglion_RU2] =PV_readPointLIFprobeA("GanglionRU2",{'A'});
[time,Ganglion_RD2] =PV_readPointLIFprobeA("GanglionRD2",{'A'});
size time;
S1 = (Ganglion_RU1-Ganglion_RD1);
S2 = (Ganglion_RU2-Ganglion_RD2);
U  = (Ganglion_RU1-Ganglion_RU2);
D  = (Ganglion_RD1-Ganglion_RD2);
clf;

rtime = 1:time;

plot \
    (rtime,Ganglion_RU1,"1",rtime,Ganglion_RD1-5,"2",rtime,Ganglion_RU2-10,"3",rtime,Ganglion_RD2-15,"4",rtime,S1-7.5,"5",rtime,S2-17.5,"5",U-25,"0",rtime,D-30,"0");

title("Ganglion Rectangle Correlations");

t2=text (25,-5,"RU1");
set(t2,'color',[1 0 0]);
t3=text (25,-10,"RD1");
set(t3,'color',[0 1 0]);
t4=text (25,-15,"RU2");
set(t4,'color',[0 0 1])
t5=text (25,-20,"RD2");
set(t5,'color',[1 0 1]);
xlabel("time [msec]");
ylabel("Spiking Activity");
grid;
%%axis([0,600,-350,-40]);
print -dgif ../octave/GanglionCorrelationsActivity.gif
