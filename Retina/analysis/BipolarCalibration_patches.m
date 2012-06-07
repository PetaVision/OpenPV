duration = 400;
[time,Bipolar_V_B] =PV_readPointLIFprobe("BipolarB",duration);
[time,Bipolar_V_P1]=PV_readPointLIFprobe("BipolarP1",duration);
[time,Bipolar_V_P3]=PV_readPointLIFprobe("BipolarP3",duration);
[time,Bipolar_V_P5]=PV_readPointLIFprobe("BipolarP5",duration);
[time,Bipolar_V_P7]=PV_readPointLIFprobe("BipolarP7",duration);
[time,Bipolar_V_P9]=PV_readPointLIFprobe("BipolarP9",duration);
plot \
    (time,Bipolar_V_B,"0",time,Bipolar_V_P1,"1",time,Bipolar_V_P3,"2",time,Bipolar_V_P5,"3",time,Bipolar_V_P7,"4",time,Bipolar_V_P9,"5");

title("Bipolar Patchsize Response");
t1=text (50,-69,"Black");
set(t1,'color',[0 0 0]);
t2=text (50,-66,"1x1");
set(t2,'color',[1 0 0]);
t3=text (50,-63,"3x3");
set(t3,'color',[0 1 0]);
t4=text (50,-60,"5x5");
set(t4,'color',[0 0 1])
t5=text (50,-57,"7x7");
set(t5,'color',[1 0 1]);
t5=text (50,-54,"9x9");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(40,-16,"tau = 20 msec, strength = 0.2766, no gapjunctions");
axis([0,400,-75,-50]);
grid;


