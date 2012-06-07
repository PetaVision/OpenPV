duration = 400;
[time,Horizontal_V_B] =PV_readPointLIFprobe("HorizontalB",duration);
[time,Horizontal_V_02]=PV_readPointLIFprobe("Horizontal02",duration);
[time,Horizontal_V_04]=PV_readPointLIFprobe("Horizontal04",duration);
[time,Horizontal_V_06]=PV_readPointLIFprobe("Horizontal06",duration);
[time,Horizontal_V_08]=PV_readPointLIFprobe("Horizontal08",duration);
[time,Horizontal_V_10]=PV_readPointLIFprobe("Horizontal10",duration);
grid;
plot \
    (time,Horizontal_V_B,"0",time,Horizontal_V_02,"1",time,Horizontal_V_04,"2",time,Horizontal_V_06,"3",time,Horizontal_V_08,"4",time,Horizontal_V_10,"5");

title("Horizontal Cells, feedback inhibition to Cones and gap junctions");
t1=text (25,-53,"Black");
set(t1,'color',[0 0 0]);
t2=text (25,-50,"20%");
set(t2,'color',[1 0 0]);
t3=text (25,-47,"40%");
set(t3,'color',[0 1 0]);
t4=text (25,-44,"60%");
set(t4,'color',[0 0 1])
t5=text (25,-41,"80%");
set(t5,'color',[1 0 1]);
t5=text (25,-38,"100%");
set(t5,'color',[0 1 1]);
text(50,-28,"tau = 10 msec, I-C strength = 0.1475, horizontal feedback");
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
axis([0,400,-60,-15])
grid;

print -dgif ../../gjkunde/octave/HorizontalCalibration.gif
