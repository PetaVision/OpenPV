
[time,avgON]=pvp_readLayerStatsProbe("GanglionON");
[time,avgOFF]=pvp_readLayerStatsProbe("GanglionOFF");

rtime = 1:time;

semilogy(rtime,avgON,"1",rtime,avgOFF,"3");


titlestring = ["Ganglion Spike Response to 1/f  background with luminance of 128"]; 
title(titlestring,"fontsize",15);

t1=text (110,1.5,"ON cells");
set(t1,'color',[1 0 0],"fontsize",20);
t2=text (110,0.5,"OFF cells");
set(t2,'color',[0 0 1],"fontsize",20);
xlabel("time [msec]","fontsize",15);

ylabel("Average Spike Frequency in Hz (/dt ms)","fontsize",15);
grid;
axis([0 200 0.05 600]);
;
 
outname = ["../octave/GanglionStats_Close",".pdf"]
print(outname,"-dpdf");
