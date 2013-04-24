clear all; close all; clc;

fileLoc.workspacePath = '/Users/dpaiton/Documents/Work/LANL/workspace/';
fileLoc.outputPath    = [fileLoc.workspacePath,'BIDS/experimentAnalysis/'];

params.GRAPH_FLAG    = 1;                       %% Display histograms
    params.graphSpec     = [237,437,438,638];   %% Specify when the stimulus was present
    params.numHistBins   = 500;                 %% For histogram - number of bins

params.outFileExt    = 'png';

label                 = 'Lat60';
params.MOVIE_FLAG     = 0;                   %% Create a movie from the BIDS_Clone output
fileLoc.inputFileName = [fileLoc.workspacePath,'BIDS/outputLI60/BIDS_Clone.pvp'];
[LI60pSet, LI60AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'Lat80';
params.MOVIE_FLAG     = 1;                   %% Create a movie from the BIDS_Clone output
fileLoc.inputFileName = [fileLoc.workspacePath,'BIDS/outputLI80/BIDS_Clone.pvp'];
[LI80pSet, LI80AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'Lat90';
params.MOVIE_FLAG     = 0;                   %% Create a movie from the BIDS_Clone output
fileLoc.inputFileName = [fileLoc.workspacePath,'BIDS/outputLI90/BIDS_Clone.pvp'];
[LI90pSet, LI90AUC]   = doAnalysis(label,fileLoc,params);

label                 = 'NoLat60';
params.MOVIE_FLAG     = 0;                   %% Create a movie from the BIDS_Clone output
fileLoc.inputFileName = [fileLoc.workspacePath,'BIDS/outputNLI60/BIDS_Clone.pvp'];
[NLI60pSet, NLI60AUC] = doAnalysis(label,fileLoc,params);

label                 = 'NoLat80';
params.MOVIE_FLAG     = 1;                   %% Create a movie from the BIDS_Clone output
fileLoc.inputFileName = [fileLoc.workspacePath,'BIDS/outputNLI80/BIDS_Clone.pvp'];
[NLI80pSet, NLI80AUC] = doAnalysis(label,fileLoc,params);

label                 = 'NoLat90';
params.MOVIE_FLAG     = 0;                   %% Create a movie from the BIDS_Clone output
fileLoc.inputFileName = [fileLoc.workspacePath,'BIDS/outputNLI90/BIDS_Clone.pvp'];
[NLI90pSet, NLI90AUC] = doAnalysis(label,fileLoc,params);

figPath = [fileLoc.outputPath,'Figures/'];
if ne(exist(figPath,'dir'),7)
    mkdir(figPath);
end

figure
hold on
plot([0,1],[0,1],'k')
plot([0 NLI80pSet(1,:) 1],[0 NLI80pSet(2,:) 1],'Color','r')
plot([0 LI80pSet(1,:) 1],[0 LI80pSet(2,:) 1],'Color','b')
hold off
xlim([0 1])
ylim([0 1])
legend('Chance',...
    ['No Lateral Interactions(',num2str(NLI80AUC),')'],...
    ['Lateral Interactions (',num2str(LI80AUC),')'],...
    'Location','SouthEast')
xlabel('Probability of Detection')
ylabel('Probability of False Alarm')
title('Receiver Operator Characterists for BIDS Network at 80% SNR')
print([figPath,'ROC.png'])

x = [0 .6 .8 .9 1];
y_NLI = [0 NLI60AUC NLI80AUC NLI90AUC 1];
y_LI = [0 LI60AUC LI80AUC LI90AUC 1];
figure
hold on
plot(x, y_NLI, 'r');
plot(x, y_LI, 'b');
hold off
legend('No Lateral Interaction', 'Lateral Interaction', 'Location', 'SouthEast');
xlabel('Signal Strength');
ylabel('Area Under Recevier Operator Characteristic Curves');
xlim([0.6 0.9])
ylim([0.6 0.9])
title('Signal Strength vs Area under ROC');
print([figPath, 'Strength_vs_AUC.png'])
