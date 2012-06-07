duration = 400;
[time,Cone_V_B] =PV_readPointLIFprobe("ConeB",duration);
[time,Cone_V_02]=PV_readPointLIFprobe("Cone02",duration);
[time,Cone_V_04]=PV_readPointLIFprobe("Cone04",duration);
[time,Cone_V_06]=PV_readPointLIFprobe("Cone06",duration);
[time,Cone_V_08]=PV_readPointLIFprobe("Cone08",duration);
[time,Cone_V_10]=PV_readPointLIFprobe("Cone10",duration);
plot \
    (time,Cone_V_B,"0",time,Cone_V_02,"1",time,Cone_V_04,"2",time,Cone_V_06,"3",time,Cone_V_08,"4",time,Cone_V_10,"5");

title("Cone Calibration Target: -55 to -40 mV, reduced by 50% at the peak");
t1=text (25,-54,"Black");
set(t1,'color',[0 0 0]);
t2=text (25,-52.5,"20%");
set(t2,'color',[1 0 0]);
t3=text (25,-51,"40%");
set(t3,'color',[0 1 0]);
t4=text (25,-49.5,"60%");
set(t4,'color',[0 0 1])
t5=text (25,-48,"80%");
set(t5,'color',[1 0 1]);
t5=text (25,-46.5,"100%");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(50,-45.5,"tau = 10 msec, I-C strength = 0.1475, horizontal feedback");
grid;
axis([0,400,-61,-35]);
fontsize = 20;
print -dgif ../../gjkunde/octave/ConeCalibration.gif

