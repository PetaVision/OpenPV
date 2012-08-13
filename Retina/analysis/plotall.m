% Plots all pt output

[time,WFAmacrineON_V] = pvp_readPointLIFprobeRun(1,"WFAmacrineON",{'V'});
[time,WFAmacrineOFF_V] = pvp_readPointLIFprobeRun(1,"WFAmacrineOFF",{'V'});
[time,SFAmacrine_V] = pvp_readPointLIFprobeRun(1,"SFAmacrine",{'V'});
[time,BipolarON_V] = pvp_readPointLIFprobeRun(1,"BipolarON",{'V'});
[time,BipolarOFF_V] = pvp_readPointLIFprobeRun(1,"BipolarOFF",{'V'});
[time,GanglionON_V] = pvp_readPointLIFprobeRun(1,"GanglionON",{'V'});
[time,GanglionOFF_V] = pvp_readPointLIFprobeRun(1,"GanglionOFF",{'V'});
[time,Horizontal_V] = pvp_readPointLIFprobeRun(1,"Horizontal",{'V'});
[time,Cone_V] = pvp_readPointLIFprobeRun(1,"Cone",{'V'});
[time,PAAmacrineON_V] = pvp_readPointLIFprobeRun(1,"PAAmacrineON",{'V'});
[time,PAAmacrineOFF_V] = pvp_readPointLIFprobeRun(1,"PAAmacrineOFF",{'V'});

rtime = 1:time;

figure(1);
clf;
plot(rtime,WFAmacrineON_V,'-r',rtime,WFAmacrineOFF_V,'-r','LineWidth',1,rtime,SFAmacrine_V,'-c',rtime,BipolarON_V,'-b',rtime,BipolarOFF_V,'-b','LineWidth',1,rtime,Horizontal_V,'-k',rtime,Cone_V,'-m',rtime,PAAmacrineON_V,'-g',rtime,PAAmacrineOFF_V,'-g','LineWidth',1);

legend('WFAmacrineON','WFAmacrineOFF','SFAmacrine','BipolarON','BipolarOFF','Horizontal','Cone','PAAmacrineON','PAAmacrineOFF');

axis([0 900 -80 -20]);

grid on

figure(2);
clf;
plot(rtime,GanglionON_V,'-g',rtime,GanglionOFF_V,'-r')
