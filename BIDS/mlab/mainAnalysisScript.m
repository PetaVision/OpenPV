clear all; close all; clc;

fileLoc.workspacePath = ['~/workspace/'];
fileLoc.outputPath    = [fileLoc.workspacePath,'BIDS/experimentAnalysis/'];

params.GRAPH_FLAG     = 1;                    %% Display histograms
    %% Input starts at 539, 217 frame integration window
    params.graphSpec   = [221,438,648,865];   %% Specify when the stimulus was present
    params.numHistBins = 250;                 %% For histogram - number of bins

params.outFileExt     = 'png';

params.numBIDSNodes   = 64*64*1;

params.dt             = 1.2;                  %% [ms]
params.displayPeriod  = 12;                   %% [ms]

disp('Analyzing BIDS results...')
disp(['----'])
fflush(stdout);

label                 = 'Lat2.5';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputLI2.5/BIDS_Clone.pvp'];
fileLoc.connFileName  = [fileLoc.workspacePath,'BIDS/outputLI2.5/Lateral_Excitation.pvp'];
[LI2_5pSet, LI2_5AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'NoLat2.5';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputNLI2.5/BIDS_Clone.pvp'];
[NLI2_5pSet, NLI2_5AUC] = doAnalysis(label,fileLoc,params);

label                 = 'Lat5';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputLI5/BIDS_Clone.pvp'];
fileLoc.connFileName  = [fileLoc.workspacePath,'BIDS/outputLI5/Lateral_Excitation.pvp'];
[LI5pSet, LI5AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'NoLat5';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputNLI5/BIDS_Clone.pvp'];
[NLI5pSet, NLI5AUC] = doAnalysis(label,fileLoc,params);

label                 = 'Lat10';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputLI10/BIDS_Clone.pvp'];
fileLoc.connFileName  = [fileLoc.workspacePath,'BIDS/outputLI10/Lateral_Excitation.pvp'];
[LI10pSet, LI10AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'NoLat10';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputNLI10/BIDS_Clone.pvp'];
[NLI10pSet, NLI10AUC] = doAnalysis(label,fileLoc,params);

label                 = 'Lat20';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputLI20/BIDS_Clone.pvp'];
fileLoc.connFileName  = [fileLoc.workspacePath,'BIDS/outputLI20/Lateral_Excitation.pvp'];
[LI20pSet, LI20AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'NoLat20';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputNLI20/BIDS_Clone.pvp'];
[NLI20pSet, NLI20AUC] = doAnalysis(label,fileLoc,params);

label                 = 'Lat40';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputLI40/BIDS_Clone.pvp'];
fileLoc.connFileName  = [fileLoc.workspacePath,'BIDS/outputLI40/Lateral_Excitation.pvp'];
[LI40pSet, LI40AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'NoLat40';
disp(label);
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputNLI40/BIDS_Clone.pvp'];
[NLI40pSet, NLI40AUC] = doAnalysis(label,fileLoc,params);

label                 = 'Lat60';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputLI60/BIDS_Clone.pvp'];
fileLoc.connFileName  = [fileLoc.workspacePath,'BIDS/outputLI60/Lateral_Excitation.pvp'];
[LI60pSet, LI60AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'NoLat60';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputNLI60/BIDS_Clone.pvp'];
[NLI60pSet, NLI60AUC] = doAnalysis(label,fileLoc,params);

label                 = 'Lat80';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputLI80/BIDS_Clone.pvp'];
fileLoc.connFileName  = [fileLoc.workspacePath,'BIDS/outputLI80/Lateral_Excitation.pvp'];
[LI80pSet, LI80AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'NoLat80';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputNLI80/BIDS_Clone.pvp'];
[NLI80pSet, NLI80AUC] = doAnalysis(label,fileLoc,params);

label                 = 'Lat90';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputLI90/BIDS_Clone.pvp'];
fileLoc.connFileName  = [fileLoc.workspacePath,'BIDS/outputLI90/Lateral_Excitation.pvp'];
[LI90pSet, LI90AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'NoLat90';
disp(label)
fflush(stdout);
params.MOVIE_FLAG     = 0;                    %% Create a movie from the BIDS_Clone output
params.WEIGHTS_FLAG   = 0;                    %% Display plots for respective lateral weights file
fileLoc.layerFileName = [fileLoc.workspacePath,'BIDS/outputNLI90/BIDS_Clone.pvp'];
[NLI90pSet, NLI90AUC] = doAnalysis(label,fileLoc,params);

figPath = [fileLoc.outputPath,'Figures/'];
if ne(exist(figPath,'dir'),7)
    mkdir(figPath);
end

figure
hold on
plot([0,1],[0,1],'k')
plot(NLI2_5pSet(1,:),NLI2_5pSet(2,:),'Color','r')
plot(LI2_5pSet(1,:),LI2_5pSet(2,:),'Color','b')
hold off
xlim([0 1])
ylim([0 1])
legend('Chance',...
    ['No Lateral Interactions(',num2str(NLI2_5AUC),')'],...
    ['Lateral Interactions (',num2str(LI2_5AUC),')'],...
    'Location','SouthEast')
hxLabel = xlabel('Probability of False Alarm');
hyLabel = ylabel('Probability of Detection');
hTitle  = title('Receiver Operator Characterists for BIDS Network at 2.5% SNR');
set(gca, ...
    'FontName'  ,'Helvetica', ...
    'FontSize'  ,20, ...
    'TickDir'   ,'out', ...
    'XMinorTick','on', ...
    'YMinorTick','on');
set([hxLabel, hyLabel, hTitle], ...
    'FontName','AvantGarde');
set([hxLabel, hyLabel], ...
    'FontSize',20);
set(hTitle, ...
    'FontSize',16, ...
    'FontWeight','bold');
print([figPath,'ROC1.png'])

figure
hold on
plot([0,1],[0,1],'k')
plot(NLI5pSet(1,:),NLI5pSet(2,:),'Color','r')
plot(LI5pSet(1,:),LI5pSet(2,:),'Color','b')
hold off
xlim([0 1])
ylim([0 1])
legend('Chance',...
    ['No Lateral Interactions(',num2str(NLI5AUC),')'],...
    ['Lateral Interactions (',num2str(LI5AUC),')'],...
    'Location','SouthEast')
hxLabel = xlabel('Probability of False Alarm');
hyLabel = ylabel('Probability of Detection');
hTitle  = title('Receiver Operator Characterists for BIDS Network at 5% SNR');
set(gca, ...
    'FontName'  ,'Helvetica', ...
    'FontSize'  ,20, ...
    'TickDir'   ,'out', ...
    'XMinorTick','on', ...
    'YMinorTick','on');
set([hxLabel, hyLabel, hTitle], ...
    'FontName','AvantGarde');
set([hxLabel, hyLabel], ...
    'FontSize',20);
set(hTitle, ...
    'FontSize',16, ...
    'FontWeight','bold');
print([figPath,'ROC2.png'])

figure
hold on
plot([0,1],[0,1],'k')
plot(NLI10pSet(1,:),NLI10pSet(2,:),'Color','r')
plot(LI10pSet(1,:),LI10pSet(2,:),'Color','b')
hold off
xlim([0 1])
ylim([0 1])
legend('Chance',...
    ['No Lateral Interactions(',num2str(NLI10AUC),')'],...
    ['Lateral Interactions (',num2str(LI10AUC),')'],...
    'Location','SouthEast')
hxLabel = xlabel('Probability of False Alarm');
hyLabel = ylabel('Probability of Detection');
hTitle  = title('Receiver Operator Characterists for BIDS Network at 10% SNR');
set(gca, ...
    'FontName'  ,'Helvetica', ...
    'FontSize'  ,20, ...
    'TickDir'   ,'out', ...
    'XMinorTick','on', ...
    'YMinorTick','on');
set([hxLabel, hyLabel, hTitle], ...
    'FontName','AvantGarde');
set([hxLabel, hyLabel], ...
    'FontSize',20);
set(hTitle, ...
    'FontSize',16, ...
    'FontWeight','bold');
print([figPath,'ROC3.png'])

figure
hold on
plot([0,1],[0,1],'k')
plot(NLI20pSet(1,:),NLI20pSet(2,:),'Color','r')
plot(LI20pSet(1,:),LI20pSet(2,:),'Color','b')
hold off
xlim([0 1])
ylim([0 1])
legend('Chance',...
    ['No Lateral Interactions(',num2str(NLI20AUC),')'],...
    ['Lateral Interactions (',num2str(LI20AUC),')'],...
    'Location','SouthEast')
hxLabel = xlabel('Probability of False Alarm');
hyLabel = ylabel('Probability of Detection');
hTitle  = title('Receiver Operator Characterists for BIDS Network at 20% SNR');
set(gca, ...
    'FontName'  ,'Helvetica', ...
    'FontSize'  ,20, ...
    'TickDir'   ,'out', ...
    'XMinorTick','on', ...
    'YMinorTick','on');
set([hxLabel, hyLabel, hTitle], ...
    'FontName','AvantGarde');
set([hxLabel, hyLabel], ...
    'FontSize',20);
set(hTitle, ...
    'FontSize',16, ...
    'FontWeight','bold');
print([figPath,'ROC4.png'])

figure
hold on
plot([0,1],[0,1],'k')
plot(NLI40pSet(1,:),NLI40pSet(2,:),'Color','r')
plot(LI40pSet(1,:),LI40pSet(2,:),'Color','b')
hold off
xlim([0 1])
ylim([0 1])
legend('Chance',...
    ['No Lateral Interactions(',num2str(NLI40AUC),')'],...
    ['Lateral Interactions (',num2str(LI40AUC),')'],...
    'Location','SouthEast')
hxLabel = xlabel('Probability of False Alarm');
hyLabel = ylabel('Probability of Detection');
hTitle  = title('Receiver Operator Characterists for BIDS Network at 40% SNR');
set(gca, ...
    'FontName'  ,'Helvetica', ...
    'FontSize'  ,20, ...
    'TickDir'   ,'out', ...
    'XMinorTick','on', ...
    'YMinorTick','on');
set([hxLabel, hyLabel, hTitle], ...
    'FontName','AvantGarde');
set([hxLabel, hyLabel], ...
    'FontSize',20);
set(hTitle, ...
    'FontSize',16, ...
    'FontWeight','bold');
print([figPath,'ROC5.png'])

figure
hold on
plot([0,1],[0,1],'k')
plot(NLI60pSet(1,:),NLI60pSet(2,:),'Color','r')
plot(LI60pSet(1,:),LI60pSet(2,:),'Color','b')
hold off
xlim([0 1])
ylim([0 1])
legend('Chance',...
    ['No Lateral Interactions(',num2str(NLI60AUC),')'],...
    ['Lateral Interactions (',num2str(LI60AUC),')'],...
    'Location','SouthEast')
hxLabel = xlabel('Probability of False Alarm');
hyLabel = ylabel('Probability of Detection');
hTitle  = title('Receiver Operator Characterists for BIDS Network at 60% SNR');
set(gca, ...
    'FontName'  ,'Helvetica', ...
    'FontSize'  ,20, ...
    'TickDir'   ,'out', ...
    'XMinorTick','on', ...
    'YMinorTick','on');
set([hxLabel, hyLabel, hTitle], ...
    'FontName','AvantGarde');
set([hxLabel, hyLabel], ...
    'FontSize',20);
set(hTitle, ...
    'FontSize',16, ...
    'FontWeight','bold');
print([figPath,'ROC6.png'])

figure
hold on
plot([0,1],[0,1],'k')
plot(NLI80pSet(1,:),NLI80pSet(2,:),'Color','r')
plot(LI80pSet(1,:),LI80pSet(2,:),'Color','b')
hold off
xlim([0 1])
ylim([0 1])
legend('Chance',...
    ['No Lateral Interactions(',num2str(NLI80AUC),')'],...
    ['Lateral Interactions (',num2str(LI80AUC),')'],...
    'Location','SouthEast')
hxLabel = xlabel('Probability of False Alarm');
hyLabel = ylabel('Probability of Detection');
hTitle  = title('Receiver Operator Characterists for BIDS Network at 80% SNR');
set(gca, ...
    'FontName'  ,'Helvetica', ...
    'FontSize'  ,20, ...
    'TickDir'   ,'out', ...
    'XMinorTick','on', ...
    'YMinorTick','on');
set([hxLabel, hyLabel, hTitle], ...
    'FontName','AvantGarde');
set([hxLabel, hyLabel], ...
    'FontSize',20);
set(hTitle, ...
    'FontSize',16, ...
    'FontWeight','bold');
print([figPath,'ROC7.png'])

figure
hold on
plot([0,1],[0,1],'k')
plot(NLI90pSet(1,:),NLI90pSet(2,:),'Color','r')
plot(LI90pSet(1,:),LI90pSet(2,:),'Color','b')
hold off
xlim([0 1])
ylim([0 1])
legend('Chance',...
    ['No Lateral Interactions(',num2str(NLI90AUC),')'],...
    ['Lateral Interactions (',num2str(LI90AUC),')'],...
    'Location','SouthEast')
hxLabel = xlabel('Probability of False Alarm');
hyLabel = ylabel('Probability of Detection');
hTitle  = title('Receiver Operator Characterists for BIDS Network at 90% SNR');
set(gca, ...
    'FontName'  ,'Helvetica', ...
    'FontSize'  ,20, ...
    'TickDir'   ,'out', ...
    'XMinorTick','on', ...
    'YMinorTick','on');
set([hxLabel, hyLabel, hTitle], ...
    'FontName','AvantGarde');
set([hxLabel, hyLabel], ...
    'FontSize',20);
set(hTitle, ...
    'FontSize',16, ...
    'FontWeight','bold');
print([figPath,'ROC8.png'])

x = [0 .025 .05 .1 .2 .4 .6 .8 .9 1];
y_NLI = [0 NLI2_5AUC NLI5AUC NLI10AUC NLI20AUC NLI40AUC NLI60AUC NLI80AUC NLI90AUC 1];
y_LI = [0 LI2_5AUC LI5AUC LI10AUC LI20AUC LI40AUC LI60AUC LI80AUC LI90AUC 1];
figure
hold on
plot(x, y_NLI, 'r-o');
plot(x, y_LI, 'b-o');
hold off
legend('No Lateral Interaction', 'Lateral Interaction', 'Location', 'SouthEast');
hxLabel = xlabel('Signal to Noise Ratio');
hyLabel = ylabel('Area Under Recevier Operator Characteristic (ROC) Curves');
xlim([0 1])
ylim([0.5 1])
%xlim([0.6 0.9])
%ylim([0.6 0.9])
hTitle  = title('Area Under ROC vs. Signal to Noise Ratio');
set(gca, ...
    'FontName'  ,'Helvetica', ...
    'FontSize'  ,14, ...
    'TickDir'   ,'out', ...
    'XMinorTick','on', ...
    'YMinorTick','on');
set([hxLabel, hyLabel, hTitle], ...
    'FontName','AvantGarde');
set([hxLabel, hyLabel], ...
    'FontSize',14);
set(hTitle, ...
    'FontSize',20, ...
    'FontWeight','bold');
print([figPath, 'Strength_vs_AUC.png'])

