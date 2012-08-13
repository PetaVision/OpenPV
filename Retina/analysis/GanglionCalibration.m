
[time,Horizontal_V_10]=pvp_readPointLIFprobe("Horizontal",{'V'});
[time,Bipolar_V_10]=pvp_readPointLIFprobe("BipolarON",{'V'});
[time,Ganglion_V_10,Ganglion_A_10,Ganglion_T_10]=pvp_readPointLIFprobe("GanglionON",{'V','A','Vth'});
[time,Amacrine_V_10,Amacrine_A_10,Amacrine_T_10]=pvp_readPointLIFprobe("PAAmacrineON",{'V','A','Vth'});
[time,Ganglion_E_10,Ganglion_I_10,Ganglion_IB_10]=pvp_readPointLIFprobe("GanglionON",{'G_E','G_I','G_IB'});

rtime = 1:time;



subplot(1,2,1);
plot(rtime,Ganglion_V_10,"4",rtime,Ganglion_T_10+4,"4",rtime,Ganglion_A_10*5-40,"4",rtime,Ganglion_E_10*10-40,"4",rtime,Ganglion_I_10*10-90,"4",rtime,Ganglion_IB_10*10-90,"4");
axis([0 500 -95 -25]);
title("Ganglion ON Properties for center");
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
t1=text (100, Ganglion_V_10(100)-19, "Membrane Potential");
set(t1,'color',[1 0 1]);
t2=text (100,Ganglion_T_10(100)+2,"mV Threshold");
set(t2,'color',[1 0 1]);
t3=text (100,Ganglion_A_10(100)*5-42,"Activity");
set(t3,'color',[1 0 1]);

subplot(1,2,2);
plot(rtime,Horizontal_V_10,"2",rtime,Bipolar_V_10,"3",rtime,Amacrine_V_10,"5",rtime,Amacrine_T_10+15,"5",rtime,Amacrine_A_10*5-20,"5");
%(rtime,Horizontal_V_10-90,"2",rtime,Bipolar_V_10,"3",rtime,Ganglion_V_10-80,"4",rtime,Ganglion_T_10-70,"4",rtime,Amacrine_V_10,"5",rtime,Amacrine_T_10,"5",rtime,Ganglion_A_10*5-40,"4",rtime,Amacrine_A_10*5-45,"5",rtime,Ganglion_E_10*10-80,"4",rtime,Ganglion_I_10*10-85,"4",rtime,Ganglion_IB_10*10-90,"4");

title("Other cells for Ganglion ON center");
xlabel("time [msec]");
ylabel("Membrane Potential [mV]");
t1=text (70,Amacrine_A_10(100)*5-30,"Activity of Amacrine");
set(t1,'color',[0 1 1]);
t2=text (70,Amacrine_T_10(100)+7,"mV Threshold of Amacrine");
set(t2,'color',[0 1 1]);
t3=text (70,Amacrine_V_10(100)-10,"mV of Amacrine");
set(t3,'color',[0 1 1]);
t4=text (70,Bipolar_V_10(100)-10,"mV of Bipolar");
set(t4,'color',[0 0 1]);
t5=text (70,Horizontal_V_10(100)-10,"mV of Horizontal");
set(t5,'color',[0 1 0]);
grid;



%outname = [filename,"ON.pdf"]
%print(outname,"-dpdf");
%outname = [filename,"ON.jpeg"]
%print(outname,"-djpeg");


