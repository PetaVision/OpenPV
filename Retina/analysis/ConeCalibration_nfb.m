[time,Cone_V_B] =PV_readPointLIFprobe("ConeB",200);
[time,Cone_V_02]=PV_readPointLIFprobe("Cone02",200);
[time,Cone_V_04]=PV_readPointLIFprobe("Cone04",200);
[time,Cone_V_06]=PV_readPointLIFprobe("Cone06",200);
[time,Cone_V_08]=PV_readPointLIFprobe("Cone08",200);
[time,Cone_V_10]=PV_readPointLIFprobe("Cone10",200);
plot \
    (time,Cone_V_B,"0",time,Cone_V_02,"1",time,Cone_V_04,"2",time,Cone_V_06,"3",time,Cone_V_08,"4",time,Cone_V_10,"5");

title("Cone Calibration Target: -55 to -40 mV");
t1=text (100,-56,"Black");
set(t1,'color',[0 0 0]);
t2=text (100,-52,"20%");
set(t2,'color',[1 0 0]);
t3=text (100,-49,"40%");
set(t3,'color',[0 1 0]);
t4=text (100,-46,"60%");
set(t4,'color',[0 0 1])
t5=text (100,-43,"80%");
set(t5,'color',[1 0 1]);
t5=text (100,-41,"100%");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(40,-37,"tau = 10 msec, strength = 0.1475, no feedback");


