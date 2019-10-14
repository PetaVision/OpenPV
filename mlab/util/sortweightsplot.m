% Sort the weigth vectors by their overall activities in the training or testing process
%  output_dir: 
%       where to find the pvp-activity file  
%       also, the weights_move folder with the plots will be placed here
%  activity_file: filename of activity layer, e.g., V1
%  checkpoint_dir: where to find the weight files
%  weight_files: string of weightfile name. 
%       If more than one weight (e.g. stereo) input is a cell of strings.
%       Cell arrangement also specifies sub arangement of kernels (e.g. 4x1, 2x2) 
%  sortby: sort by 'percentActive' or by 'meanActivity'
%  showActivityPlots: whether to show the activity spectrum
%  pauseflag: whether to pause after activity spectrum has been plotted
%  nLookBack: calculate on the basis of the last nLookBack displayPeriods
function ind=sortweightsplot(output_dir,activity_file,checkpoint_dir,weights_files,sortby,showActivityPlots,pauseflag,nLookBack)

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; % for compatibility

if ~exist('pauseflag','var')
    pauseflag=false;
end

if isempty(dir(fullfile(output_dir,'weights_movie')))
    mkdir(fullfile(output_dir,'weights_movie'));
end

disp('Reading activity file.')
[data,header]=readpvpfile(fullfile(output_dir,activity_file),1000000);

if ~exist('nLookBack')
	nLookBack=199;
else
	nLookBack=nLookBack-1;
end
lookBackForReal = max(1,size(data,1)-nLookBack) : size(data,1);
nLookBackForReal = length(lookBackForReal);
data=data(end-nLookBackForReal+1:end);

disp(['Looking back ',num2str(nLookBackForReal), ' presentations.']);
allEmpty=true;
v1pack=nan(header.nf,header.nx,header.ny,nLookBackForReal);
activityDistribution=nan(nLookBackForReal,1);
for i = 1:nLookBackForReal
		if mod(i,floor((nLookBackForReal+1)/10)) == 0
			fprintf('.')
		end
    v1container_tmp=zeros(header.nx*header.ny*header.nf,1);
    if ~isempty(data{i}.values)
        v1container_tmp(data{i}.values(:,1)+1)=data{i}.values(:,2);
        v1pack(:,:,:,i)=reshape(v1container_tmp,header.nf,header.nx,header.ny);
        activityDistribution(i)=size(data{i}.values,1);
        allEmpty=false;
    else
        v1pack(:,:,:,i)=reshape(v1container_tmp,header.nf,header.nx,header.ny);
        activityDistribution(i)=0;
    end
end
fprintf('\n')
if allEmpty
    warning('Data structure is empty. No activity (within the last ',num2str(nLookBackForReal),' display periods)?')
    warning('Plotting without sorting.')
    weightsplot(output_dir,checkpoint_dir,weights_files);
else
		disp('Done. Now sorting.')
    [counted,ind]=sortAndCount(v1pack,sortby);
    proportioned=counted/nLookBackForReal/header.nx/header.ny*100;
    
    listToWrite=[[1:length(proportioned)]',ind,proportioned];
    %listToWrite=sortrows(listToWrite,1);
    csvwrite(fullfile(output_dir,'weights_movie',[sortby,'_values','.csv']), listToWrite);
    csvwrite(fullfile(output_dir,'weights_movie',['activityDistribution','.csv']), [header.nx*header.ny*header.nf;nan;activityDistribution])
    
    if exist('showActivityPlots') && showActivityPlots==true
			disp('Done. Now generating plots.')
			
			activityplot=figure;
			bar(proportioned);
			xlabel('The index number of kernels')
			if strcmp(sortby,'percentActive')
				ylabel(['Percent active within ',num2str(nLookBackForReal),' presentations [%]' ])
			elseif strcmp(sortby,'meanActivity')
				ylabel(['Mean activity within ',num2str(nLookBackForReal),' presentations' ])
			else
				error('valid options for sorting are "percentActive" and "meanActivity".')
			end
			axis tight
			drawnow
			saveas(activityplot,fullfile(output_dir,'weights_movie',[sortby,'_spectrum','.png']))
			
			numActivePlot=figure;
			hist(activityDistribution,...
				max(1,min(20,(size(activityDistribution,1)/2))));
			xlabel([{'Activity per presentation [# active].'},{['# model neurons = ',num2str(header.nx*header.ny*header.nf)]}]);
			ylabel(['Frequency (', num2str(nLookBackForReal), ' Presentations)']);
			drawnow
			saveas(numActivePlot,fullfile(output_dir,'weights_movie',['nActiveElementsPerPresentation','.png']))
			
			percentActivePlot=figure;
			hist(activityDistribution/(header.nf*header.nx*header.ny)*100,...
				max(1,min(20,(size(activityDistribution,1)/2))));
			xlabel([{'Activity-ratio per presentation [% active].'},{['# model neurons = ',num2str(header.nx*header.ny*header.nf)]}]);
			ylabel(['Frequency (', num2str(nLookBackForReal), ' Presentations)']);
			drawnow
			saveas(percentActivePlot,fullfile(output_dir,'weights_movie',['percentActiveElementsPerPresentation','.png']))
    end
    
		disp('Done. Now calling weightsplot function.')
    weightsplot(output_dir,checkpoint_dir,weights_files,ind,[],['sortedBy_',sortby]);
    if isOctave && pauseflag && exist('showActivityPlots') && showActivityPlots==true
			disp('Hit any key to exit and close all windows');
			pause
		end
end

end





function [y,i] = sortAndCount(x,sortby)
	if strcmp(sortby,'percentActive')
		[y,i] = sort(sum(sum(sum(logical(x),2),3),4),'descend');
	elseif strcmp(sortby,'meanActivity')
		[y,i] = sort(sum(sum(sum(x,2),3),4),'descend');
	else
		error('valid options for sorting are "percentActive" and "meanActivity".')
	end
end
