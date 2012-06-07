duration = 400;
[time,Ganglion_V_B] =PV_readPointLIFprobe("GanglionB",duration);
[time,Ganglion_V_02]=PV_readPointLIFprobe("Ganglion02",duration);
[time,Ganglion_V_04]=PV_readPointLIFprobe("Ganglion04",duration);
[time,Ganglion_V_06]=PV_readPointLIFprobe("Ganglion06",duration);
[time,Ganglion_V_08]=PV_readPointLIFprobe("Ganglion08",duration);
[time,Ganglion_V_10]=PV_readPointLIFprobe("Ganglion10",duration);
plot \
    (time,Ganglion_V_B-250,"0",time,Ganglion_V_02-200,"1",time,Ganglion_V_04-150,"2",time,Ganglion_V_06-100,"3",time,Ganglion_V_08-50,"4",time,Ganglion_V_10,"5");

title("Ganglion Calibration for all amplitudes");
t1=text (25,-300,"Black");
set(t1,'color',[0 0 0]);
t2=text (25,-250,"20%");
set(t2,'color',[1 0 0]);
t3=text (25,-200,"40%");
set(t3,'color',[0 1 0]);
t4=text (25,-150,"60%");
set(t4,'color',[0 0 1])
t5=text (25,-100,"80%");
set(t5,'color',[1 0 1]);
t5=text (25,-50,"100%");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
grid;
axis([0,400,-350,-40]);
print -dgif ../../gjkunde/octave/GanglionCalibrationAll.gif
