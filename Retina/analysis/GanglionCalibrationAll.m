
[time,GanglionON_V_B] =pvp_readPointLIFprobe("GanglionONB",{'V'});
[time,GanglionON_V_02]=pvp_readPointLIFprobe("GanglionON02",{'V'});
[time,GanglionON_V_04]=pvp_readPointLIFprobe("GanglionON04",{'V'});
[time,GanglionON_V_06]=pvp_readPointLIFprobe("GanglionON06",{'V'});
[time,GanglionON_V_08]=pvp_readPointLIFprobe("GanglionON08",{'V'});
[time,GanglionON_V_10]=pvp_readPointLIFprobe("GanglionON10",{'V'});


[time,GanglionOFF_V_B] =pvp_readPointLIFprobe("GanglionOFFB",{'V'});
[time,GanglionOFF_V_02]=pvp_readPointLIFprobe("GanglionOFF02",{'V'});
[time,GanglionOFF_V_04]=pvp_readPointLIFprobe("GanglionOFF04",{'V'});
[time,GanglionOFF_V_06]=pvp_readPointLIFprobe("GanglionOFF06",{'V'});
[time,GanglionOFF_V_08]=pvp_readPointLIFprobe("GanglionOFF08",{'V'});
[time,GanglionOFF_V_10]=pvp_readPointLIFprobe("GanglionOFF10",{'V'});


rtime = 1:time;

plot \
   (rtime,GanglionON_V_B+250,"0",rtime,GanglionOFF_V_B-250,"0",rtime,GanglionON_V_P1+200,"1",rtime,GanglionOFF_V_P1-200,"1",rtime,GanglionON_V_P3+150,"2",rtime,GanglionOFF_V_P3-150,"2",rtime,GanglionON_V_P5+100,"3",rtime,GanglionOFF_V_P5-100,"3",rtime,GanglionON_V_P7+50,"4",rtime,GanglionOFF_V_P7-50,"4",rtime,GanglionON_V_P9,"5",rtime,GanglionOFF_V_P9,"5");

n = 30;

title("Ganglion Calibration for all amplitudes");
t1=text (100,GanglionON_V_B(100)+250-n,"Black");
set(t1,'color',[0 0 0]);
t2=text (100,GanglionOFF_V_B(100)-250-n,"Black");
set(t2,'color',[0 0 0]);
t3=text (100,GanglionON_V_P1(100)+200-n,"20%");
set(t3,'color',[1 0 0]);
t4=text (100,GanglionOFF_V_P1(100)-200-n, "20%");
set(t4,'color',[1 0 0]);
t5=text (100,GanglionON_V_P3(100)+150-n,"40%");
set(t5,'color',[0 1 0]);
t6=text (100,GanglionOFF_V_P3(100)-150-n,"40%");
set(t6,'color',[0 1 0]);
t7=text (100,GanglionON_V_P5(100)+100-n,"60%");
set(t7,'color',[0 0 1])
t8=text (100,GanglionOFF_V_P5(100)-100-n,"60%");
set(t8,'color',[0 0 1]);
t9=text (100,GanglionON_V_P7(100)+50-n,"80%");
set(t9,'color',[1 0 1]);
t10=text (100,GanglionOFF_V_P7(100)-50-n,"80%");
set(t10,'color',[1 0 1]);
t11=text (100,GanglionON_V_P9(100)-20,"100%");
set(t11,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
grid;
axis([0 500 -350 210]);
print -dgif ../octave/GanglionCalibrationAll.gif
