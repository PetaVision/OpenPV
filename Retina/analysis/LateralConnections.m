
[time,Cone_V]=pvp_readPointLIFprobe("Cone02",{'V'});
[time,BipolarON_V]=pvp_readPointLIFprobe("BipolarON02",{'V'});
[time,Horizontal_V]=pvp_readPointLIFprobe("Horizontal02",{'V'});
[time,GanglionON_V]=pvp_readPointLIFprobe("GanglionON02",{'V'});
[time,WFAmacrineON_V]=pvp_readPointLIFprobe("WFAmacrineON02",{'V'});
[time,PAAmacrineON_V]=pvp_readPointLIFprobe("PAAmacrineON02",{'V'});
[time,SFAmacrine_V]=pvp_readPointLIFprobe("SFAmacrine02",{'V'});

rtime = 1:time;

plot \
(rtime,Cone_V,"1",rtime,BipolarON_V-80,"2",rtime,Horizontal_V+10,"3",rtime,GanglionON_V,"4",rtime,WFAmacrineON_V,"5x",rtime,PAAmacrineON_V,"5-",rtime,SFAmacrine_V-100,"5*");
title("Lateral Connections");
t1=text (100,Cone_V(100)-8,"Cone");
set(t1,'color',[1 0 0]);
t2=text (100,BipolarON_V(100)-90,"Bipolar");
set(t2,'color',[0 1 0]);
t3=text (100,Horizontal_V(100),"Horizontal");
set(t3,'color',[0 0 1]);
t4=text (100,GanglionON_V(100)-23,"Ganglion");
set(t4,'color',[1 0 1]);

t5=text (400,-220,"WFAmacrineON: x");
set(t5,'color',[0 1 1]);
t6=text (400,-230,"PAAmacrineON: - ");
set(t6,'color',[0 1 1]);
t7=text (400,-240,"SFAmacrine: *");
set(t7,'color',[0 1 1]);



%print -dgif "../octave/ConeCalibration.gif";

