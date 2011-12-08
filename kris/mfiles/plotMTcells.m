clear all;
close all;
clc;


%MTN=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a50.pvp');
MTN=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTTest4/a21.pvp');
% MTE=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a51.pvp');
% MTS=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a52.pvp');
% MTW=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a53.pvp');
%MTNE=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a54.pvp');
MTNE=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTTest4/a22.pvp');
% MTSE=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a55.pvp');
% MTSW=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a56.pvp');
%MTNW=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a57.pvp');
MTNW=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTTest4/a23.pvp');



figure; hold on
plot(squeeze(MTN(32,32,1,:)), 'b','LineWidth',4)
%title('MT cell N');
% figure; hold on
% plot(squeeze(MTE(32,32,1,:)), 'b')
% title('MT cell E');
% figure; hold on
% plot(squeeze(MTS(32,32,1,:)), 'b')
% title('MT cell S');
% figure; hold on
% plot(squeeze(MTW(32,32,1,:)), 'b')
% title('MT cell W');
%figure; hold on
plot(squeeze(MTNE(32,32,1,:)), 'k','LineWidth',4)
%title('MT cell NE');
% figure; hold on
% plot(squeeze(MTSE(32,32,1,:)), 'b')
% title('MT cell SE');
% figure; hold on
% plot(squeeze(MTSW(32,32,1,:)), 'b')
% title('MT cell SW');
%figure; hold on
plot(squeeze(MTNW(32,32,1,:)), 'r','LineWidth',4)
%title('MT cell NW');
legend('MT cell N', 'MT cell NE', 'MT cell NW');
title('MT cells Plaid stimuli', 'FontSize', 36);
set(gca, 'FontSize', 36)

 vicomp1=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTTest4/a5.pvp');
 vicomp2=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTTest4/a8.pvp');
 vicomp3=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTTest4/a11.pvp');
 vicomp13=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTTest4/a14.pvp');
 vicomp14=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTTest4/a17.pvp');
 vicomp16=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTTest4/a20.pvp');
% vicomp13=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a40.pvp');
% vicomp14=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/a43.pvp');
figure; hold on
plot(squeeze(vicomp2(32,32,1,:)), 'b','LineWidth',4)
%title('complex cell 1 NE');
%figure; hold on
plot(squeeze(vicomp3(32,32,1,:)), 'r','LineWidth',4)
%title('complex cell 1 NW');
%figure; hold on
plot(squeeze(vicomp1(32,32,1,:)), 'k','LineWidth',4)
%title('complex cell 1 N');
%figure; hold on
plot(squeeze(vicomp13(32,32,1,:)), 'c','LineWidth',4)
%title('complex cell 13 N V 2');
%figure; hold on
plot(squeeze(vicomp14(32,32,1,:)), 'y','LineWidth',4)
%title('complex cell 14 E V 2');
%figure; hold on
plot(squeeze(vicomp16(32,32,1,:)), 'g','LineWidth',4)
%title('complex cell 16 W V 2');
legend('complex cell 1 NE', 'complex cell 1 NW', 'complex cell 1 N', 'complex cell 13 N V 2', 'complex cell 14 E V 2', 'complex cell 16 W V 2');
title('V1 Complex cells Plaid stimuli', 'FontSize', 36);
set(gca, 'FontSize', 36)

 %vicomp16=readweights('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest1_onesine/w55_last.pvp');

MTN=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest3/a20.pvp');
MTNE=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest3/a21.pvp');
MTNW=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest3/a22.pvp');



figure; hold on
plot(squeeze(MTN(32,32,1,:)), 'b','LineWidth',4)
plot(squeeze(MTNE(32,32,1,:)), 'k','LineWidth',4)
plot(squeeze(MTNW(32,32,1,:)), 'r','LineWidth',4)
legend('MT cell N', 'MT cell NE', 'MT cell NW');
title('MT cells sine wave stimuli', 'FontSize', 36);
set(gca, 'FontSize', 36)

 vicomp1=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest3/a4.pvp');
 vicomp2=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest3/a7.pvp');
 vicomp3=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest3/a10.pvp');
 vicomp13=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest3/a13.pvp');
 vicomp14=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest3/a16.pvp');
 vicomp16=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/MTtest3/a19.pvp');

 figure; hold on
plot(squeeze(vicomp2(32,32,1,:)), 'b','LineWidth',4)
plot(squeeze(vicomp3(32,32,1,:)), 'r','LineWidth',4)
plot(squeeze(vicomp1(32,32,1,:)), 'k','LineWidth',4)
plot(squeeze(vicomp13(32,32,1,:)), 'c','LineWidth',4)
plot(squeeze(vicomp14(32,32,1,:)), 'y','LineWidth',4)
plot(squeeze(vicomp16(32,32,1,:)), 'g','LineWidth',4)
legend('complex cell 1 NE', 'complex cell 1 NW', 'complex cell 1 N', 'complex cell 13 N V 2', 'complex cell 14 E V 2', 'complex cell 16 W V 2');
title('V1 Complex cells sine wave stimuli', 'FontSize', 36);
set(gca, 'FontSize', 36)