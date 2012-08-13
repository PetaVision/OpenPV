
[time,BipolarON_V_B] =pvp_readPointLIFprobe("BipolarONB",{'V'});
[time,BipolarON_V_P1]=pvp_readPointLIFprobe("BipolarONP1",{'V'});
[time,BipolarON_V_P3]=pvp_readPointLIFprobe("BipolarONP3",{'V'});
[time,BipolarON_V_P5]=pvp_readPointLIFprobe("BipolarONP5",{'V'});
[time,BipolarON_V_P7]=pvp_readPointLIFprobe("BipolarONP7",{'V'});
[time,BipolarON_V_P9]=pvp_readPointLIFprobe("BipolarONP9",{'V'});

[time,BipolarOFF_V_B] =pvp_readPointLIFprobe("BipolarOFFB",{'V'});
[time,BipolarOFF_V_P1]=pvp_readPointLIFprobe("BipolarOFFP1",{'V'});
[time,BipolarOFF_V_P3]=pvp_readPointLIFprobe("BipolarOFFP3",{'V'});
[time,BipolarOFF_V_P5]=pvp_readPointLIFprobe("BipolarOFFP5",{'V'});
[time,BipolarOFF_V_P7]=pvp_readPointLIFprobe("BipolarOFFP7",{'V'});
[time,BipolarOFF_V_P9]=pvp_readPointLIFprobe("BipolarOFFP9",{'V'});

rtime = 1:time;

plot \
(rtime,BipolarON_V_B,"0",rtime,BipolarOFF_V_B,"0",rtime,BipolarON_V_P1+5,"1",rtime,BipolarOFF_V_P1-5,"1",rtime,BipolarON_V_P3+10,"2",rtime,BipolarOFF_V_P3-10,"2",rtime,BipolarON_V_P5+15,"3",rtime,BipolarOFF_V_P5-15,"3",rtime,BipolarON_V_P7+20,"4",rtime,BipolarOFF_V_P7-20,"4",rtime,BipolarON_V_P9+25,"5",rtime,BipolarOFF_V_P9-25,"5");

n = 2;

axis([0 500 -100 -25]);
title("Bipolar Patchsize Response");
t1=text (50,BipolarON_V_B(50)-n,"Black");
set(t1,'color',[0 0 0]);
t2=text (50,BipolarON_V_P1(50)+5-n,"1x1");
set(t2,'color',[1 0 0]);
t3=text (50,BipolarOFF_V_P1(50)-5-n,"1x1");
set(t3,'color',[1 0 0]);
t4=text (50,BipolarON_V_P3(50)+10-n,"3x3");
set(t4,'color',[0 1 0]);
t5=text (50,BipolarOFF_V_P3(50)-10-n,"3x3");
set(t5,'color',[0 1 0]);
t6=text (50,BipolarON_V_P5(50)+15-n,"5x5");
set(t6,'color',[0 0 1]);
t7=text (50,BipolarOFF_V_P5(50)-15-n,"5x5");
set(t7,'color',[0 0 1]);
t8=text (50,BipolarON_V_P7(50)+20-n,"7x7");
set(t8,'color',[1 0 1]);
t9=text (50,BipolarOFF_V_P7(50)-20-n,"7x7");
set(t9,'color',[1 0 1]);
t10=text (50,BipolarON_V_P9(50)+25-n,"9x9");
set(t10,'color',[0 1 1]);
t11=text (50,BipolarOFF_V_P9(50)-25-n,"9x9");
set(t11,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(100,-96,"tau = 20 msec, strength = 0.2766, no gapjunctions");
grid;


