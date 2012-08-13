
[time,GanglionON_V_B]=pvp_readPointLIFprobe("GanglionONB",{'V'});
[time,GanglionON_V_P1]=pvp_readPointLIFprobe("GanglionONP1",{'V'});
[time,GanglionON_V_P3]=pvp_readPointLIFprobe("GanglionONP3",{'V'});
[time,GanglionON_V_P5]=pvp_readPointLIFprobe("GanglionONP5",{'V'});
[time,GanglionON_V_P7]=pvp_readPointLIFprobe("GanglionONP7",{'V'});
[time,GanglionON_V_P9]=pvp_readPointLIFprobe("GanglionONP9",{'V'});

[time,GanglionOFF_V_B]=pvp_readPointLIFprobe("GanglionOFFB",{'V'});
[time,GanglionOFF_V_P1]=pvp_readPointLIFprobe("GanglionOFFP1",{'V'});
[time,GanglionOFF_V_P3]=pvp_readPointLIFprobe("GanglionOFFP3",{'V'});
[time,GanglionOFF_V_P5]=pvp_readPointLIFprobe("GanglionOFFP5",{'V'});
[time,GanglionOFF_V_P7]=pvp_readPointLIFprobe("GanglionOFFP7",{'V'});
[time,GanglionOFF_V_P9]=pvp_readPointLIFprobe("GanglionOFFP9",{'V'});

rtime = 1:time;

plot \
   (rtime,GanglionON_V_B+250,"0",rtime,GanglionOFF_V_B-250,"0",rtime,GanglionON_V_P1+200,"1",rtime,GanglionOFF_V_P1-200,"1",rtime,GanglionON_V_P3+150,"2",rtime,GanglionOFF_V_P3-150,"2",rtime,GanglionON_V_P5+100,"3",rtime,GanglionOFF_V_P5-100,"3",rtime,GanglionON_V_P7+50,"4",rtime,GanglionOFF_V_P7-50,"4",rtime,GanglionON_V_P9,"5",rtime,GanglionOFF_V_P9,"5");

n = 15;

title("Ganglion Patchsize Response");
t1=text (50,GanglionON_V_B(50)+250-n,"Black");
set(t1,'color',[0 0 0]);
t2=text (50,GanglionOFF_V_B(50)-250-n,"Black");
set(t2,'color',[0 0 0]);
t3=text (50,GanglionON_V_P1(50)+200-n,"1x1");
set(t3,'color',[1 0 0]);
t4=text (50,GanglionOFF_V_P1(50)-200-25,"1x1");
set(t4,'color',[1 0 0]);
t5=text (50,GanglionON_V_P3(50)+150-n,"3x3");
set(t5,'color',[0 1 0]);
t6=text (50,GanglionOFF_V_P3(50)-150-n,"3x3");
set(t6,'color',[0 1 0]);
t7=text (50,GanglionON_V_P5(50)+100-n,"5x5");
set(t7,'color',[0 0 1]);
t8=text (50,GanglionOFF_V_P5(50)-100-n,"5x5");
set(t8,'color',[0 0 1]);
t9=text (50,GanglionON_V_P7(50)+50-n,"7x7");
set(t9,'color',[1 0 1]);
t10=text (50,GanglionOFF_V_P7(50)-50-n,"7x7");
set(t10,'color',[1 0 1]);
t11=text (50,GanglionON_V_P9(50)-n,"9x9");
set(t11,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
text(40,-16,"");
axis([0 500 -380 240]);
grid;


