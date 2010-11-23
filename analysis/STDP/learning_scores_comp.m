% compare the learning scores for various learning parameters


sym = {'+r','+b','or','ob','*r','*b','^r','^b'};
numCenters = 8;
plot_dW = 1;
plot_wMax = 0;
plot_inh = 0;
plot_Vth =1;

if plot_dW
%% compare scores for different learning rates

output_dir = {'/Users/manghel/Documents/STDP-sim/soren12/',
              '/Users/manghel/Documents/STDP-sim/soren11/',
              '/Users/manghel/Documents/STDP-sim/soren8/',
              '/Users/manghel/Documents/STDP-sim/soren9/'};
          
figure('Name','Variation with learning rate');
dW = [0.01, 0.02, 0.03, 0.04]; %wMax = 0.75
disp('Variation with learning rate');
for f=2:numel(output_dir)
    scores_file = [output_dir{f},'WeightsLearningScores',...
        num2str(numCenters),'.dat'];
    fprintf('dW = %f\n',dW(f));
    fid_scores = fopen(scores_file,'r');
    % read scores, plot scores
    data = fscanf(fid_scores, '%g %g %g', [3 inf]);    % It has two rows now.
    data = data';
    fclose(fid_scores);
    plot(data(:,1),data(:,2),sym{2*f-1},'MarkerFaceColor','r');hold on
    if f==2
       title('\bfLearning Score Evolution'); 
       xlabel('\bf t');
       ylabel('\bf S(t)');
    end
    plot(data(:,1),data(:,3),sym{2*f}, 'MarkerFaceColor','b');
    pause 
end
end


if plot_wMax
%% compare scores for identical learning rates but different wMax values
output_dir = {'/Users/manghel/Documents/STDP-sim/soren13/',
              '/Users/manghel/Documents/STDP-sim/soren11/',
              '/Users/manghel/Documents/STDP-sim/soren14/'};
wMax = [0.5, 0.75, 1.00]; %dW = 0.02

figure('Name','Variation with max values');
disp('Variation with max weight values');
for f=1:numel(output_dir)
    scores_file = [output_dir{f},'WeightsLearningScores',...
        num2str(numCenters),'.dat'];
    fprintf('wMax = %f\n',wMax(f));
    fid_scores = fopen(scores_file,'r');
    % read scores, plot scores
    data = fscanf(fid_scores, '%g %g %g', [3 inf]);    % It has two rows now.
    data = data';
    fclose(fid_scores);
    plot(data(:,1),data(:,2),sym{2*f-1},'MarkerFaceColor','r');hold on
    if f==1
       title('Learning Score Evolution'); 
    end
    plot(data(:,1),data(:,3),sym{2*f}, 'MarkerFaceColor','b');
    pause 
end
end

if plot_inh
%% compare scores for models with and without inhibition

output_dir = {'/Users/manghel/Documents/STDP-sim/soren11/',
              '/Users/manghel/Documents/STDP-sim/soren15/', %delay 0
              '/Users/manghel/Documents/STDP-sim/soren16/', %delay 5
              }; 
%wMax = 0.75; dW = 0.02

figure('Name','Variation with inhibition');
strText={'no inhibition','with inhibition: delay 0',...
    'with inhibition: delay 5'};
disp('Variation with inhibition');
for f=1:numel(output_dir)
    scores_file = [output_dir{f},'WeightsLearningScores',...
        num2str(numCenters),'.dat'];
    fprintf('%s\n',strText{f});
    fid_scores = fopen(scores_file,'r');
    % read scores, plot scores
    data = fscanf(fid_scores, '%g %g %g', [3 inf]);    % It has two rows now.
    data = data';
    fclose(fid_scores);
    plot(data(:,1),data(:,2),sym{2*f-1},'MarkerFaceColor','r');hold on
    if f==1
       title('Learning Score Evolution'); 
    end
    plot(data(:,1),data(:,3),sym{2*f}, 'MarkerFaceColor','b');
    pause 
end
end

if plot_Vth
%% compare scored for model with different VthRest values

output_dir = {'/Users/manghel/Documents/STDP-sim/soren11/', % Vth -55
              '/Users/manghel/Documents/STDP-sim/soren17/', %Vth -60
              }; 

figure('Name','VthRest');
strText={'VthRest -55mV','VthRest -60mV'};
disp('Variation with inhibition');
for f=1:numel(output_dir)
    scores_file = [output_dir{f},'WeightsLearningScores',...
        num2str(numCenters),'.dat'];
    fprintf('%s\n',strText{f});
    fid_scores = fopen(scores_file,'r');
    % read scores, plot scores
    data = fscanf(fid_scores, '%g %g %g', [3 inf]);    % It has two rows now.
    data = data';
    fclose(fid_scores);
    plot(data(:,1),data(:,2),sym{2*f-1},'MarkerFaceColor','r');hold on
    if f==1
       title('Learning Score Evolution'); 
    end
    plot(data(:,1),data(:,3),sym{2*f}, 'MarkerFaceColor','b');
    pause 
end
end

