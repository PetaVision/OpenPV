[time,Bipolar_V_B] =PV_readPointLIFprobe("BipolarB",200);
[time,Bipolar_V_02]=PV_readPointLIFprobe("Bipolar02",200);
[time,Bipolar_V_04]=PV_readPointLIFprobe("Bipolar04",200);
[time,Bipolar_V_06]=PV_readPointLIFprobe("Bipolar06",200);
[time,Bipolar_V_08]=PV_readPointLIFprobe("Bipolar08",200);
[time,Bipolar_V_10]=PV_readPointLIFprobe("Bipolar10",200);
plot \
    (time,Bipolar_V_B,"0",time,Bipolar_V_02,"1",time,Bipolar_V_04,"2",time,Bipolar_V_06,"3",time,Bipolar_V_08,"4",time,Bipolar_V_10,"5");

title("Bipolar Calibration Target: -70 to -50 mV Swing");
t1=text (100,-71,"Black");
set(t1,'color',[0 0 0]);
t2=text (100,-66,"20%");
set(t2,'color',[1 0 0]);
t3=text (100,-60,"40%");
set(t3,'color',[0 1 0]);
t4=text (100,-56.5,"60%");
set(t4,'color',[0 0 1])
t5=text (100,-53.5,"80%");
set(t5,'color',[1 0 1]);
t5=text (100,-51,"100%");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(54,-47,"tau = 20 msec, strength = 0.2735, no feedback");


