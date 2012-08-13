
[time,Horizontal_V_B] =pvp_readPointLIFprobe("HorizontalB",{'V'});
[time,Horizontal_V_02]=pvp_readPointLIFprobe("Horizontal02",{'V'});
[time,Horizontal_V_04]=pvp_readPointLIFprobe("Horizontal04",{'V'});
[time,Horizontal_V_06]=pvp_readPointLIFprobe("Horizontal06",{'V'});
[time,Horizontal_V_08]=pvp_readPointLIFprobe("Horizontal08",{'V'});
[time,Horizontal_V_10]=pvp_readPointLIFprobe("Horizontal10",{'V'});

rtime = 1:time;

plot \
    (rtime,Horizontal_V_B,"0",rtime,Horizontal_V_02-20,"1",rtime,Horizontal_V_04-40,"2",rtime,Horizontal_V_06-60,"3",rtime,Horizontal_V_08-80,"4",rtime,Horizontal_V_10-100,"5");

n = 10;

title("Horizontal Calibration Target: -55 to -20 mV Swing");
t1=text (100,Horizontal_V_B(100)-n,"Black");
set(t1,'color',[0 0 0]);
t2=text (100,Horizontal_V_02(100)-20-n,"20%");
set(t2,'color',[1 0 0]);
t3=text (100,Horizontal_V_04(100)-40-n,"40%");
set(t3,'color',[0 1 0]);
t4=text (100,Horizontal_V_06(100)-60-n,"60%");
set(t4,'color',[0 0 1])
t5=text (100,Horizontal_V_08(100)-80-n,"80%");
set(t5,'color',[1 0 1]);
t6=text (100,Horizontal_V_10(100)-100-n,"100%");
set(t6,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(100,-320,"tau = 20 msec, strength = 0.2766, no feedback, no gapjunctions");
axis([0 500 -370 -50])

