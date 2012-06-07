function GanglionCalibration(duration = 1000,filename)
[time,Horizontal_V_10]=PV_readPointLIFprobe("Horizontal",duration);
[time,Bipolar_V_10]=PV_readPointLIFprobe("BipolarON",duration);
[time,Ganglion_V_10,Ganglion_A_10,Ganglion_T_10]=PV_readPointLIFprobeVAT("GanglionON",duration);
[time,Amacrine_V_10,Amacrine_A_10,Amacrine_T_10]=PV_readPointLIFprobeVAT("PAAmacrineON",duration);
[time,Ganglion_E_10,Ganglion_I_10,Ganglion_IB_10]=PV_readPointLIFprobeG("GanglionON",duration);

figure

plot \
    (time,Horizontal_V_10,"2",time,Bipolar_V_10,"3",time,Ganglion_V_10,"4",time,Ganglion_T_10,"4",time,Amacrine_V_10,"5",time,Amacrine_T_10,"5",time,Ganglion_A_10*5-40,"4",time,Amacrine_A_10*5-45,"5",time,Ganglion_E_10*10-80,"4",Ganglion_I_10*10-85,"4",Ganglion_IB_10*10-90,"4");

title("Ganglion ON for center");
t4=text (time(1)-25,-55,"Bipolar");
set(t4,'color',[0 0 1])
t5=text (time(1)-25,-60,"Ganglion");
set(t5,'color',[1 0 1]);
t5=text (time(1)-25,-65,"Amacrine");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
grid;



outname = [filename,"ON.pdf"]
print(outname,"-dpdf");
outname = [filename,"ON.jpeg"]
print(outname,"-djpeg");


[time,Horizontal_V_10]=PV_readPointLIFprobe("Horizontal",duration);
[time,Bipolar_V_10]=PV_readPointLIFprobe("BipolarOFF",duration);
[time,Ganglion_V_10,Ganglion_A_10,Ganglion_T_10]=PV_readPointLIFprobeVAT("GanglionOFF",duration);
[time,Amacrine_V_10,Amacrine_A_10,Amacrine_T_10]=PV_readPointLIFprobeVAT("PAAmacrineOFF",duration);
[time,Ganglion_E_10,Ganglion_I_10,Ganglion_IB_10]=PV_readPointLIFprobeG("GanglionOFF",duration);



figure


plot \
    (time,Horizontal_V_10,"2",time,Bipolar_V_10,"3",time,Ganglion_V_10,"4",time,Ganglion_T_10,"4",time,Amacrine_V_10,"5",time,Amacrine_T_10,"5",time,Ganglion_A_10*5-40,"4",time,Amacrine_A_10*5-45,"5",time,Ganglion_E_10*10-80,"4",Ganglion_I_10*10-85,"4",Ganglion_IB_10*10-90,"4");

title("Ganglion OFF for center");
t4=text (time(1)-25,-55,"Bipolar");
set(t4,'color',[0 0 1])
t5=text (time(1)-25,-60,"Ganglion");
set(t5,'color',[1 0 1]);
t5=text (time(1)-25,-65,"Amacrine");
set(t5,'color',[0 1 1]);
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
grid;

outname = [filename,"OFF.pdf"]
print(outname,"-dpdf");

outname = [filename,"OFF.jpeg"]
print(outname,"-djpeg");