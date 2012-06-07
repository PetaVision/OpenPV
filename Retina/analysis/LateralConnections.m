
[time,Cone_V]=PV_readPointLIFprobe("Cone02",1000);
[time,Bipolar_V]=PV_readPointLIFprobe("Bipolar02",1000);
[time,Horizontal_V]=PV_readPointLIFprobe("Horizontal02",1000);
[time,Ganglion_V]=PV_readPointLIFprobe("Ganglion02",1000);
[time,Amacrine_V]=PV_readPointLIFprobe("Amacrine02",1000);
plot \
    (time,Cone_V,"1",time,Bipolar_V,"2",time,Horizontal_V,"3",time,Ganglion_V,"4",time,Amacrine_V,"5");
title("Lateral Connections");
t1=text (500,-40,"Cone");
set(t1,'color',[1 0 0])
t2=text (500,-42.5,"Bipolar");
set(t2,'color',[0 1 0])
t3=text (500,-45,"Horizontal");
set(t3,'color',[0 0 1])
t4=text (500,-47.5,"Ganglion");
set(t4,'color',[1 0 1])
t5=text (500,-50,"Amacrine");
set(t5,'color',[0 1 1])

print -dgif "../../gjkunde/octave/ConeCalibration.gif";

