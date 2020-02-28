% errorcurve(errorFileDir, filenames, plotLimits, displayPeriod)
%
% Used to visualize how the L2, L1 error and total energy of the network
% change during training. First n, last m display periods are plotted as well as
% all settled values and a running mean of the latter. Will also save figures
% as pdf files in errorFileDir
%
% errorFileDir: usually the directory set in the outputPath param
% filenames: cell containing the filenames for files containing the 
%    L1-Error, L2-Error, Combined Energy (in that order)
% plotLimits: 2-Vector specifying first n and last m display periods to plot
% displayPeriod: set in params file
%
%ToDo: extract display Period automatically

function errorcurve(errorFileDir, filenames, plotLimits, displayPeriod)
% clear all
% close all
% clc
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; % for compatibility

%% settings

displayPeriodDetailedPlotLimit_first = plotLimits(1); % detailed data will be plotted for this many display periods at the beginning
displayPeriodDetailedPlotLimit_last = plotLimits(2); % detailed data will be plotted for this many display periods at the end
plotDetailed    = true; % the first plot


L1Filename      = filenames{1};
L2Filename      = filenames{2};
EnergyFilename  = filenames{3};

% minor settings
plotColors{1}=[1,0,0];
plotColors{2}=[0,1,0];
plotColors{3}=[0,0,1];


%% read files line by line and display them in the meantime
fprintf('\n');
disp('Loading data:');
currentMessage='Currently parsing display period no. 1';
fprintf(currentMessage);

fid = cell(3,1);
fid{1} = fopen(fullfile(errorFileDir,L1Filename));
fid{2} = fopen(fullfile(errorFileDir,L2Filename));
fid{3} = fopen(fullfile(errorFileDir,EnergyFilename));
fgetl(fid{3}); %read and discard first line, due to different displays of energy probes

close all

displayPeriod_current=0;
rescaleFactor = cell(3,1);
settledData  = cell(3,1);
currentXMin   = inf;
clear temp_lastData;
displayPeriodData_complete=cell(3,1);
while true
    displayPeriod_current=displayPeriod_current+1;
    
    % print currently parsed display period.
    for i = 1 : size(currentMessage,2)
      fprintf('\b');
    end
    currentMessage = ['Currently parsing display period no. ',num2str(displayPeriod_current)];
    fprintf(currentMessage);
    if isOctave
      fflush(stdout);
    end
    
    clear displayPeriodData;
    displayPeriodData = cell(3,1);
    if exist('temp_lastData','var')
        displayPeriodData=temp_lastData;
        loopadd=1;
    else
        loopadd=0;
    end
    for j = 1 : size(fid,1) % number of displayed probes
        for i = 1+loopadd : displayPeriod+loopadd % display period
            temp_fileLine   = fgetl(fid{j});
            if ~ischar(temp_fileLine) && temp_fileLine==-1
                break;
            end
            temp_positions = [0, strfind(temp_fileLine,',')];
            if size(temp_positions,2)<3
                continue;
            end
            displayPeriodData{j}(i,1)  = sscanf(temp_fileLine(temp_positions(1)+1:end),'%g',1);
            displayPeriodData{j}(i,2)  = sscanf(temp_fileLine(temp_positions(end)+1:end),'%f',1);
            if mod(displayPeriodData{j}(i,1)+1,displayPeriod) == 0
                settledData{j}(size(settledData{1},1)+1,1) = displayPeriodData{j}(i,1);
                settledData{j}(end,2) = displayPeriodData{j}(i,2);
            end
        end
    end
    if isempty(displayPeriodData{3})
        error('Error: No Data. Exiting.')
    end
    
    % plot first n data
    if plotDetailed ...
            && displayPeriodDetailedPlotLimit_first > 0 ...
            && displayPeriod_current == displayPeriodDetailedPlotLimit_first
        
        figName = ['First ',num2str(displayPeriodDetailedPlotLimit_first),' Error Probes, every time step'];
        detailedEnergyFigure_first = figure ('Name',figName,'NumberTitle','off');
        hold on;
        legend Location northeast
        for j = 1 : size(fid,1)
            currentXMin=min(min(displayPeriodData_complete{j}(:,1)),currentXMin);
            subplot(3,1,j)
            hold on;
            xlim([currentXMin,max(displayPeriodData_complete{j}(:,1))]);
            plot(displayPeriodData_complete{j}(:,1), displayPeriodData_complete{j}(:,2),'Color',[plotColors{j}]);
            if j==1
                legend ('L1 Probe');
            elseif j == 2
                legend ('L2 Probe');
            elseif j == 3
                legend ('Combined Energy')
            end
        end
        drawnow
        saveas(detailedEnergyFigure_first,fullfile(errorFileDir,[figName,'.pdf']));
    end
    
    % remember last value for next display
    clear temp_lastData;
    temp_lastData = cell(3,1);
    for j = 1 : size(fid,1)
        temp_lastData{j}(1,1)  = displayPeriodData{j}(end,1);
        temp_lastData{j}(1,2)  = displayPeriodData{j}(end,2);
        displayPeriodData_complete{j} = [displayPeriodData_complete{j}; displayPeriodData{j}];
    end
    
    if ~ischar(temp_fileLine) && temp_fileLine==-1
        clear temp_fileLine temp_positions;
        break;
    end
end

% plot last m data
if plotDetailed ... 
        && displayPeriodDetailedPlotLimit_last > 0
    
    if displayPeriod_current <= displayPeriodDetailedPlotLimit_last
        figName = [num2str(displayPeriod_current),' Error Probes (+ offset), every time step'];
        detailedEnergyFigure_last = figure ('Name',figName,'NumberTitle','off');
        last_n_reached=false;% size(displayPeriodData_complete{1},1)-1;
    else
        figName = ['Last ',num2str(displayPeriodDetailedPlotLimit_last),' Error Probes (+ offset), every time step'];
        detailedEnergyFigure_last = figure ('Name',figName,'NumberTitle','off');
        last_n_reached=true;% displayPeriodDetailedPlotLimit_last*displayPeriod;
    end
    hold on;
    legend Location northeast
    for j = 1 : size(fid,1)
        if last_n_reached
            last_n = displayPeriodDetailedPlotLimit_last*displayPeriod;
        else
            last_n =  size(displayPeriodData_complete{j},1)-1;
        end
        currentXMin=min(displayPeriodData_complete{j}(end-last_n:end,1));
        subplot(3,1,j)
        hold on;
        xlim([currentXMin,max(displayPeriodData_complete{j}(end-last_n:end,1))]);
        plot(displayPeriodData_complete{j}(end-last_n:end,1), displayPeriodData_complete{j}(end-last_n:end,2),'Color',[plotColors{j}]);
        
        if j==1
            legend ('L1 Probe');
        elseif j == 2
            legend ('L2 Probe');
        elseif j == 3
            legend ('Combined Energy')
        end
        
    end
    drawnow
    saveas(detailedEnergyFigure_last,fullfile(errorFileDir,[figName,'.pdf']));
end

% close files
for i = 1 : size(fid,1)
    fclose(fid{i});
end

fprintf('\n');
if isOctave
      fflush(stdout);
end

%% plot learning
if size(settledData{1},1)>1
    windowSize = ceil(size(settledData{1},1)*.3);
    if windowSize >= size(settledData{1},1)
        windowSize = size(settledData{1},1)-1;
    end
    figName = 'Error Probes, only last value of display period';
    learningEnergyFigure = figure ('Name',figName,'NumberTitle','off');
    hold on;
    smoothed=cell(3,1);
    for j = 1 : size(fid,1)
        if settledData{j}(1,1)==0
            settledData{j}=settledData{j}(2:end,:);
        end
        subplot(3,1,j)
        hold on;
        plot(settledData{j}(:,1), settledData{j}(:,2),'Color',[plotColors{j}]);
        clear temp_smoothed
        smoothed{j} = imfilter(settledData{j}(:,2), fspecial('average', [windowSize 1]));
        plot(settledData{j}(ceil(windowSize/2)+1:end-floor(windowSize/2),1), smoothed{j}(ceil(windowSize/2)+1:end-floor(windowSize/2)),'Color',[plotColors{j}]*.7,'LineWidth',2);
    end
    
    
    for j = 1 : size(fid,1)
        subplot(3,1,j)
        hold on;
        clear temp_maxVal temp_minVal temp_diff
        temp_maxVal = max(smoothed{j}(ceil(windowSize/2)+1:end-floor(windowSize/2)));
        temp_minVal = min(smoothed{j}(ceil(windowSize/2)+1:end-floor(windowSize/2)));
        temp_diffFraction = (temp_maxVal - temp_minVal) * .03;
        temp_diffFraction = max(temp_diffFraction,temp_maxVal*.0001);
        ylim([temp_minVal-temp_diffFraction, temp_maxVal+temp_diffFraction]);
        legend Location northeast
        if j==1
            legend ('L1 Probe (only settled value)', ['L1 Probe (running mean, window size ',num2str(windowSize),')']);
        elseif j == 2
            legend ('L2 Probe (only settled value)', ['L2 Probe (running mean, window size ',num2str(windowSize),')']);
        elseif j == 3
            legend ('Combined Energy Function (only settled value)', ['Combined Energy Function (running mean, window size ',num2str(windowSize),')'])
        end
    end
    drawnow
    saveas(learningEnergyFigure,fullfile(errorFileDir,[figName,'.pdf']));
end
if isOctave
  disp('Hit any key to exit and close all windows');
  pause
  close(detailedEnergyFigure_first);
  close(detailedEnergyFigure_last);
  close(learningEnergyFigure);
end
end%function
