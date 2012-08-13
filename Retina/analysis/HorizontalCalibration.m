
[time,Horizontal_V_B] =pvp_readPointLIFprobe("HorizontalB",{'V'});
[time,Horizontal_V_02]=pvp_readPointLIFprobe("Horizontal02",{'V'});
[time,Horizontal_V_04]=pvp_readPointLIFprobe("Horizontal04",{'V'});
[time,Horizontal_V_06]=pvp_readPointLIFprobe("Horizontal06",{'V'});
[time,Horizontal_V_08]=pvp_readPointLIFprobe("Horizontal08",{'V'});
[time,Horizontal_V_10]=pvp_readPointLIFprobe("Horizontal10",{'V'});
grid;

rtime = 1:time;

plot \
    (rtime,Horizontal_V_B,"0",rtime,Horizontal_V_02-30,"1",rtime,Horizontal_V_04-60,"2",rtime,Horizontal_V_06-90,"3",rtime,Horizontal_V_08-120,"4",rtime,Horizontal_V_10-150,"5");

n = 10;

title("Horizontal Cells, feedback inhibition to Cones and gap junctions");
t1=text (80,Horizontal_V_B(80)-n,"Black");
set(t1,'color',[0 0 0]);
t2=text (80,Horizontal_V_02(80)-30-n,"20%");
set(t2,'color',[1 0 0]);
t3=text (80,Horizontal_V_04(80)-60-n,"40%");
set(t3,'color',[0 1 0]);
t4=text (80,Horizontal_V_06(80)-90-n,"60%");
set(t4,'color',[0 0 1])
t5=text (80,Horizontal_V_08(80)-120-n,"80%");
set(t5,'color',[1 0 1]);
t6=text (80,Horizontal_V_10(80)-150-n,"100%");
set(t6,'color',[0 1 1]);
text(50,-380,"tau = 10 msec, I-C strength = 0.1475, horizontal feedback");
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
axis([0 500 -410 -50])
grid;

print -dgif ../octave/HorizontalCalibration.gif
