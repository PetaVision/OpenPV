[time,Horizontal_V_B] =PV_readPointLIFprobe("HorizontalB",200);
[time,Horizontal_V_02]=PV_readPointLIFprobe("Horizontal02",200);
[time,Horizontal_V_04]=PV_readPointLIFprobe("Horizontal04",200);
[time,Horizontal_V_06]=PV_readPointLIFprobe("Horizontal06",200);
[time,Horizontal_V_08]=PV_readPointLIFprobe("Horizontal08",200);
[time,Horizontal_V_10]=PV_readPointLIFprobe("Horizontal10",200);
plot \
    (time,Horizontal_V_B,"0",time,Horizontal_V_02,"1",time,Horizontal_V_04,"2",time,Horizontal_V_06,"3",time,Horizontal_V_08,"4",time,Horizontal_V_10,"5");

title("Horizontal Calibration Target: -55 to -20 mV Swing");
t1=text (100,-57,"Black");
set(t1,'color',[0 0 0]);
t2=text (100,-41,"20%");
set(t2,'color',[1 0 0]);
t3=text (100,-32,"40%");
set(t3,'color',[0 1 0]);
t4=text (100,-27,"60%");
set(t4,'color',[0 0 1])
t5=text (100,-24,"80%");
set(t5,'color',[1 0 1]);
t5=text (100,-21.3,"100%");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(20,-16,"tau = 20 msec, strength = 0.2766, no feedback, no gapjunctions");


