
[time,Horizontal_V_B]=pvp_readPointLIFprobe("HorizontalB",{'V'});
[time,Horizontal_V_P1]=pvp_readPointLIFprobe("HorizontalP1",{'V'});
[time,Horizontal_V_P3]=pvp_readPointLIFprobe("HorizontalP3",{'V'});
[time,Horizontal_V_P5]=pvp_readPointLIFprobe("HorizontalP5",{'V'});
[time,Horizontal_V_P7]=pvp_readPointLIFprobe("HorizontalP7",{'V'});
[time,Horizontal_V_P9]=pvp_readPointLIFprobe("HorizontalP9",{'V'});

rtime = 1:time;

plot \
    (rtime,Horizontal_V_B,"0",rtime,Horizontal_V_P1-30,"1",rtime,Horizontal_V_P3-60,"2",rtime,Horizontal_V_P5-90,"3",rtime,Horizontal_V_P7-120,"4",rtime,Horizontal_V_P9-150,"5");

n = 8;

title("Horizontal Patchsize Response");
t1=text (80,Horizontal_V_B(100)-10,"Black");
set(t1,'color',[0 0 0]);
t2=text (80,Horizontal_V_P1(100)-30-n,"1x1");
set(t2,'color',[1 0 0]);
t3=text (80,Horizontal_V_P3(100)-60-n,"3x3");
set(t3,'color',[0 1 0]);
t4=text (80,Horizontal_V_P5(100)-90-n,"5x5");
set(t4,'color',[0 0 1])
t5=text (80,Horizontal_V_P7(100)-120-n,"7x7");
set(t5,'color',[1 0 1]);
t6=text (80,Horizontal_V_P9(100)-150-n,"9x9");
set(t6,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
axis([0 500 -350 -50]);
grid;


