duration = 400;
[time,Ganglion_V_B] =PV_readPointLIFprobe("GanglionB",duration);
[time,Ganglion_V_P1]=PV_readPointLIFprobe("GanglionP1",duration);
[time,Ganglion_V_P3]=PV_readPointLIFprobe("GanglionP3",duration);
[time,Ganglion_V_P5]=PV_readPointLIFprobe("GanglionP5",duration);
[time,Ganglion_V_P7]=PV_readPointLIFprobe("GanglionP7",duration);
[time,Ganglion_V_P9]=PV_readPointLIFprobe("GanglionP9",duration);
plot \
    (time,Ganglion_V_B-250,"0",time,Ganglion_V_P1-200,"1",time,Ganglion_V_P3-150,"2",time,Ganglion_V_P5-100,"3",time,Ganglion_V_P7-50,"4",time,Ganglion_V_P9,"5");

title("Ganglion Patchsize Response");
t1=text (50,-300,"Black");
set(t1,'color',[0 0 0]);
t2=text (50,-250,"1x1");
set(t2,'color',[1 0 0]);
t3=text (50,-200,"3x3");
set(t3,'color',[0 1 0]);
t4=text (50,-150,"5x5");
set(t4,'color',[0 0 1])
t5=text (50,-100,"7x7");
set(t5,'color',[1 0 1]);
t5=text (50,-50,"9x9");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(40,-16,"");
%axis([0,400,-56,-48]);
grid;


