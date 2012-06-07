duration = 400;
[time,Bipolar_V_B] =PV_readPointLIFprobe("BipolarB",duration);
[time,Bipolar_V_02]=PV_readPointLIFprobe("Bipolar02",duration);
[time,Bipolar_V_04]=PV_readPointLIFprobe("Bipolar04",duration);
[time,Bipolar_V_06]=PV_readPointLIFprobe("Bipolar06",duration);
[time,Bipolar_V_08]=PV_readPointLIFprobe("Bipolar08",duration);
[time,Bipolar_V_10]=PV_readPointLIFprobe("Bipolar10",duration);
plot \
    (time,Bipolar_V_B,"0",time,Bipolar_V_02,"1",time,Bipolar_V_04,"2",time,Bipolar_V_06,"3",time,Bipolar_V_08,"4",time,Bipolar_V_10,"5");

title("Bipolar Calibration");
t1=text (25,-69,"Black");
set(t1,'color',[0 0 0]);
t2=text (25,-68,"20%");
set(t2,'color',[1 0 0]);
t3=text (25,-67,"40%");
set(t3,'color',[0 1 0]);
t4=text (25,-66,"60%");
set(t4,'color',[0 0 1])
t5=text (25,-65,"80%");
set(t5,'color',[1 0 1]);
t5=text (25,-64,"100%");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(50,-45.5,"tau = 10 msec, I-C strength = 0.1475, horizontal feedback");
grid;
axis([0,400,-72,-60]);
print -dgif ../../gjkunde/octave/BipolarCalibration.gif
