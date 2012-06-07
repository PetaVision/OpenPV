function GanglionStatsClose(dirname)
duration = 1000;
[time,avgON]=PV_readLayerStatsProbe(dirname,"GanglionON",duration);
[time,avgOFF]=PV_readLayerStatsProbe(dirname,"GanglionOFF",duration);


semilogy(time,avgON,"1",time,avgOFF,"3");


titlestring = ["Ganglion Spike Response (",dirname,") to 1/f  background with luminance of 128"] 
title(titlestring,"fontsize",15);

t1=text (110,5,"ON cells");
set(t1,'color',[1 0 0],"fontsize",20)
t2=text (110,2,"OFF cells");
set(t2,'color',[0 0 1],"fontsize",20);
xlabel("time [msec]","fontsize",20);

ylabel("Average Spike Frequency in Hz (/dt ms)","fontsize",20);
grid;
axis([0,200,0.1,300]);
;
 
outname = ["../../gjkunde/octave/GanglionStats_Close_",dirname,".pdf"]
print(outname,"-dpdf");