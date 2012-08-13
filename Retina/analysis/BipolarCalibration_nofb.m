
[time,BipolarON_V_B]=pvp_readPointLIFprobe("BipolarONB",{'V'});
[time,BipolarON_V_02]=pvp_readPointLIFprobe("BipolarON02",{'V'});
[time,BipolarON_V_04]=pvp_readPointLIFprobe("BipolarON04",{'V'});
[time,BipolarON_V_06]=pvp_readPointLIFprobe("BipolarON06",{'V'});
[time,BipolarON_V_08]=pvp_readPointLIFprobe("BipolarON08",{'V'});
[time,BipolarON_V_10]=pvp_readPointLIFprobe("BipolarON10",{'V'});

[time,BipolarOFF_V_B]=pvp_readPointLIFprobe("BipolarOFFB",{'V'});
[time,BipolarOFF_V_02]=pvp_readPointLIFprobe("BipolarOFF02",{'V'});
[time,BipolarOFF_V_04]=pvp_readPointLIFprobe("BipolarOFF04",{'V'});
[time,BipolarOFF_V_06]=pvp_readPointLIFprobe("BipolarOFF06",{'V'});
[time,BipolarOFF_V_08]=pvp_readPointLIFprobe("BipolarOFF08",{'V'});
[time,BipolarOFF_V_10]=pvp_readPointLIFprobe("BipolarOFF10",{'V'});

rtime = 1:time;

plot \
   (rtime,BipolarON_V_B,"0",rtime,BipolarOFF_V_B,"0",rtime,BipolarON_V_02+5,"1",rtime,BipolarOFF_V_02-5,"1",rtime,BipolarON_V_04+10,"2",rtime,BipolarOFF_V_04-10,"2",rtime,BipolarON_V_06+15,"3",rtime,BipolarOFF_V_06-15,"3",rtime,BipolarON_V_08+20,"4",rtime,BipolarOFF_V_08-20,"4",rtime,BipolarON_V_10+25,"5",rtime,BipolarOFF_V_10-25,"5");


n = 3;

axis([0 500 -100 -30]);
title("Bipolar Calibration Target: -70 to -50 mV Swing");
t1=text (100,BipolarON_V_B(100)-n,"Black");
set(t1,'color',[0 0 0]);
t2=text (100,BipolarON_V_02(100)+5-n,"20%");
set(t2,'color',[1 0 0]);
t3=text (100,BipolarOFF_V_02(100)-5-n, "20%");
set(t3,'color',[1 0 0]);
t4=text (100,BipolarON_V_04(100)+10-n,"40%");
set(t4,'color',[0 1 0]);
t5=text (100,BipolarOFF_V_04(100)-10-n,"40%");
set(t5,'color',[0 1 0]);
t6=text (100,BipolarON_V_06(100)+15-n,"60%");
set(t6,'color',[0 0 1])
t7=text (100,BipolarOFF_V_06(100)-15-n,"60%");
set(t7,'color',[0 0 1]);
t8=text (100,BipolarON_V_08(100)+20-n,"80%");
set(t8,'color',[1 0 1]);
t9=text (100,BipolarOFF_V_08(100)-20-n,"80%");
set(t9,'color',[1 0 1]);
t10=text (100,BipolarON_V_10(100)+25-n,"100%");
set(t10,'color',[0 1 1]);
t11=text (100,BipolarOFF_V_10(100)-25-n,"100%");
set(t11,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(100,-97,"tau = 20 msec, strength = 0.2735, no feedback");


