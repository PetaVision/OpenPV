
[time,Cone_V_B] =pvp_readPointLIFprobe("ConeB",{'V'});
[time,Cone_V_02]=pvp_readPointLIFprobe("Cone02",{'V'});
[time,Cone_V_04]=pvp_readPointLIFprobe("Cone04",{'V'});
[time,Cone_V_06]=pvp_readPointLIFprobe("Cone06",{'V'});
[time,Cone_V_08]=pvp_readPointLIFprobe("Cone08",{'V'});
[time,Cone_V_10]=pvp_readPointLIFprobe("Cone10",{'V'});

rtime = 1:time;

plot \
(rtime,Cone_V_B,"0",rtime,Cone_V_02-5,"1",rtime,Cone_V_04-10,"2",rtime,Cone_V_06-15,"3",rtime,Cone_V_08-20,"4",rtime,Cone_V_10-25,"5");

n = 2;

title("Cone Calibration Target: -55 to -40 mV");
t1=text (100,Cone_V_B(100)-n,"Black");
set(t1,'color',[0 0 0]);
t2=text (100,Cone_V_02(100)-5-n,"20%");
set(t2,'color',[1 0 0]);
t3=text (100,Cone_V_04(100)-10-n,"40%");
set(t3,'color',[0 1 0]);
t4=text (100,Cone_V_06(100)-15-n,"60%");
set(t4,'color',[0 0 1]);
t5=text (100,Cone_V_08(100)-20-n,"80%");
set(t5,'color',[1 0 1]);
t5=text (100,Cone_V_10(100)-25-n,"100%");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(40,-80,"tau = 10 msec, strength = 0.1475, no feedback");


