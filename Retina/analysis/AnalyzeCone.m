 [time, C_G_GAP, C_G_I, C_V] = pvp_readLIFGapptProbe(0,"Cone",{'G_GAP','G_I','V'});
 [time, H_G_E, H_G_I, H_V] = pvp_readLIFGapptProbe(0,"Horizontal",{'G_E','G_I','V'});
 [time, Bon_G_E, Bon_G_I, Bon_V] = pvp_readLIFptProbe(0,"BipolarON",{'G_E','G_I','V'});
 [time, Gon_G_E, Gon_G_I, Gon_V] = pvp_readLIFptProbe(0,"GanglionON",{'G_E','G_I','V'});
 [time, Boff_G_E, Boff_G_I, Boff_V] = pvp_readLIFptProbe(0,"BipolarOFF",{'G_E','G_I','V'});
 [time, Goff_G_E, Goff_G_I, Goff_V] = pvp_readLIFptProbe(0,"GanglionOFF",{'G_E','G_I','V'});
 
figure(1);
titlestring = ["Cone Response"] 
title(titlestring,"fontsize",15);
plot(time,C_G_GAP,time,C_V,time,C_G_I*40-20); grid;
%%plot(time,C_G_GAP); grid;
axis([0 300 -70 10]);
print("ConeResponse.pdf","-dpdf");
print("ConeResponse.jpg","-djpg");
figure(2);
titlestring = ["Horizontal Response"] 
title(titlestring,"fontsize",15);
plot(time,H_G_E*20-20,time,H_V,time,H_G_I*40-20); grid;
axis([0 300 -70 10]);
print("HorizontalResponse.pdf","-dpdf");
print("HorizontalResponse.jpg","-djpg");
figure(3);
titlestring = ["BipolarON Response"] 
title(titlestring,"fontsize",15);
plot(time,Bon_G_E*20-20,time,Bon_V,time,Bon_G_I*40-20); grid;
axis([0 300 -70 10]);
print("BipolarONResponse.pdf","-dpdf");
print("BipolarONResponse.jpg","-djpg");
figure(4);
titlestring = ["BipolarOFF Response"] 
title(titlestring,"fontsize",15);
plot(time,Boff_G_E*20-20,time,Boff_V,time,Boff_G_I*40-20); grid;
axis([0 300 -70 10]);
print("BipolarOFFResponse.pdf","-dpdf");
print("BipolarOFFResponse.jpg","-djpg");
figure(5);
titlestring = ["Cone Sigmoid"] 
title(titlestring,"fontsize",15);
plot(C_V,H_G_E*2);grid;
figure(6);
titlestring = ["Horizontal Sigmoid"] 
title(titlestring,"fontsize",15);
plot(H_V,C_G_I*2); %grid; have to take the connection strength out
figure(7);
titlestring = ["BipolarON Sigmoid"] 
title(titlestring,"fontsize",15);
plot(B_V,G_G_E*1);grid;
figure(8);
titlestring = ["BipolarOFF Sigmoid"] 
title(titlestring,"fontsize",15);
plot(Boff_V,Goff_G_E*1);grid;
figure(9);
titlestring = ["GanglionON Vmem"] 
title(titlestring,"fontsize",15);
plot(time,Gon_V);grid;
print("GanglionONResponse.pdf","-dpdf");
print("GanglionONResponse.jpg","-djpg");
figure(10);
titlestring = ["GanglionOFF Vmem"] 
title(titlestring,"fontsize",15);
plot(time,Goff_V);grid;
print("GanglionOFFResponse.pdf","-dpdf");
print("GanglionOFFResponse.jpg","-djpg");
 

C_V(90)
C_V(190)
Vrest = -55;
ratio = (C_V(190)-C_V(90))/(C_V(90)-Vrest)

ON  = Bon_V(90)
OFF = Boff_V(90)
ONOFF =  Bon_V(90)/Boff_V(90)