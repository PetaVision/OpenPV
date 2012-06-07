duration = 400;
[time,Horizontal_V_B] =PV_readPointLIFprobe("HorizontalB",duration);
[time,Horizontal_V_P1]=PV_readPointLIFprobe("HorizontalP1",duration);
[time,Horizontal_V_P3]=PV_readPointLIFprobe("HorizontalP3",duration);
[time,Horizontal_V_P5]=PV_readPointLIFprobe("HorizontalP5",duration);
[time,Horizontal_V_P7]=PV_readPointLIFprobe("HorizontalP7",duration);
[time,Horizontal_V_P9]=PV_readPointLIFprobe("HorizontalP9",duration);
plot \
    (time,Horizontal_V_B,"0",time,Horizontal_V_P1,"1",time,Horizontal_V_P3,"2",time,Horizontal_V_P5,"3",time,Horizontal_V_P7,"4",time,Horizontal_V_P9,"5");

title("Horizontal Patchsize Response");
t1=text (50,-53,"Black");
set(t1,'color',[0 0 0]);
t2=text (50,-52,"1x1");
set(t2,'color',[1 0 0]);
t3=text (50,-51,"3x3");
set(t3,'color',[0 1 0]);
t4=text (50,-51,"5x5");
set(t4,'color',[0 0 1])
t5=text (50,-50,"7x7");
set(t5,'color',[1 0 1]);
t5=text (50,-49,"9x9");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(40,-16,"tau = 20 msec, strength = 0.2766, no gapjunctions");
axis([0,400,-56,-48]);
grid;


