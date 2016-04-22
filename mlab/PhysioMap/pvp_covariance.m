%%%%%%%%%%%%%%%%%%%%%%
%% pv_covariance.m
%%   Dylan Paiton
%%   University of California, Berkeley
%%   Jan 3, 2013
%%
%% Given 2 pvp activity files, output the covariance between them
%%
%% Inputs:  TBD
%% 
%% Outputs: TBD
%%
%%%%%%%%%%%%%%%%%%%%%%

close all; clear all; %more off;

workspace_path = '/Users/dpaiton/Documents/workspace';

clobber = false;

file1_thresh      = 0.1;   %% Typically V_thresh for one of the pvp files. Other pvp file will be normalized to this value.
prog_step         = 1000;  %% For when pvp file is read in
bin_size          = 40;    %% Time steps (usually [ms])
numCorrNeurons    = 10000; %% How many neurons to compare for correlation matrix, -1 for all neurons
numCorrIterations = 3;     %% How many times to draw random neurons & look at them

addpath([workspace_path,filesep,'PetaVision/mlab/util']);

n_pvp_file1 = [workspace_path,filesep,'PetaVision/mlab/PhysioMap/data/PV_Data/a5_V1_Clone_lat_noise_crp_16x16y.pvp'];
n_pvp_file2 = [workspace_path,filesep,'PetaVision/mlab/PhysioMap/data/PV_Data/a5_V1_Clone_nolat_noise_crp_16x16y.pvp'];
n_pvp_file1_savepath = './data/lat_noise_16x16.mat';
n_pvp_file2_savepath = './data/nolat_noise_16x16.mat';

nn_pvp_file1 = [workspace_path,filesep,'PetaVision/mlab/PhysioMap/data/PV_Data/a5_V1_Clone_lat_crp_16x16y.pvp'];
nn_pvp_file2 = [workspace_path,filesep,'PetaVision/mlab/PhysioMap/data/PV_Data/a5_V1_Clone_nolat_crp_16x16y.pvp'];
nn_pvp_file1_savepath = './data/lat_16x16.mat';
nn_pvp_file2_savepath = './data/nolat_16x16.mat';

pvp_file_struct.names     = {'noise-lat','noise-noLat','vine-lat','vine-noLat'};
pvp_file_struct.inFiles   = {n_pvp_file1,n_pvp_file2,nn_pvp_file1,nn_pvp_file2};
pvp_file_struct.saveFiles = {n_pvp_file1_savepath,n_pvp_file2_savepath,nn_pvp_file1_savepath,nn_pvp_file2_savepath};

for fileNo = 1:2:length(pvp_file_struct.inFiles)

   name1 = pvp_file_struct.names{fileNo};
   name2 = pvp_file_struct.names{fileNo+1};

   file1 = pvp_file_struct.inFiles{fileNo};
   file2 = pvp_file_struct.inFiles{fileNo+1};

   file1_save = pvp_file_struct.saveFiles{fileNo};
   file2_save = pvp_file_struct.saveFiles{fileNo+1};

   fprintf(1,['\n------File ',pvp_file_struct.names{fileNo},'------\n\n']);

   if ~exist(file1_save) || clobber
      [data1,hdr1] = readpvpfile(file1,prog_step); 
      data1 = cell2mat(data1);
      %%data1 will be [num_cells,time]
      data1 = reshape(permute(reshape([data1(:).values],[hdr1.nx,hdr1.ny,hdr1.nbands,hdr1.nf]),[3 1 2 4]),hdr1.nx*hdr1.ny*hdr1.nf,hdr1.nbands);

      data1Struct.data = data1;
      data1Struct.hdr  = hdr1;

      save(file1_save,'data1Struct','-v7.3');
   else
      load(file1_save);
      hdr1  = data1Struct.hdr;
      data1 = data1Struct.data;
   end
   fprintf(1,'Done.\n');
   clearvars data1Struct

   fprintf(1,['\n------File ',pvp_file_struct.names{fileNo+1},'------\n\n']);

   if ~exist(file2_save) || clobber
      [data2,hdr2] = readpvpfile(file2,prog_step);
      data2 = cell2mat(data2);
      %%data2 will be [num_cells,time]
      data2 = reshape(permute(reshape([data2(:).values],[hdr2.nx,hdr2.ny,hdr2.nbands,hdr2.nf]),[3 1 2 4]),hdr2.nx*hdr2.ny*hdr2.nf,hdr2.nbands);

      data2Struct.data = data2;
      data2Struct.hdr  = hdr2;

      save(file2_save,'data2Struct','-v7.3');
   else
      load(file2_save);
      hdr2 = data2Struct.hdr;
      data2 = data2Struct.data;
   end
   fprintf(1,'Done.\n');

   clearvars data2Struct

   assert(hdr1.nbands == hdr2.nbands);
   assert(hdr1.nx == hdr2.nx);
   assert(hdr1.ny == hdr2.ny);
   assert(hdr1.nf == hdr2.nf);

   % First time step is initial conditions
   data1 = data1(:,2:end);
   data2 = data2(:,2:end);

   %%%
   %%%
   %%%%THRESHOLD & BIN DATA
   %%%
   %%%

   fprintf(1,['\n------Thresholding, Scaling, & Binning Data------\n\n']);

   threshData1                     = data1;
   threshData1(data1<file1_thresh) = 0;
   threshData1                     = threshData1./max(threshData1(:));
   numTimesActive1                 = nnz(threshData1);     % Activity after thresholding

   [sortData2,sortIdx2] = sort(data2(:),'descend');

   file2_thresh = sortData2(numTimesActive1); 
   
   % Threshold second file according to num active from thresholded first file
   threshData2                     = data2;
   threshData2(data2<file2_thresh) = 0;
   threshData2                     = threshData2./max(threshData2(:));
   numTimesActive2                 = nnz(threshData2);     % Activity after thresholding

   fprintf(1,['Num time steps active for all cells in file1: ',num2str(numTimesActive1),' and file2: ',num2str(numTimesActive2),'\n\n']);

   % Now it is [num_cells, frame, time-step within frame]
   data1Binned = reshape(threshData1, [size(threshData1,1),(size(threshData1,2))/bin_size,bin_size]); %-1 is bc first time_step is initial conditions
   data2Binned = reshape(threshData2, [size(threshData2,1),(size(threshData2,2))/bin_size,bin_size]); 

   fprintf(1,'Done.\n');

   %%%
   %%%
   %%%%PSTH
   %%%
   %%%

   fprintf(1,['\n------Plot PSTH------\n\n']);

   data1Sum = sum(sum(data1Binned,3),1)./size(data1Binned,1).*100;
   figure()
   plot(data1Sum)
   title(['PSTH (% active) for ',name1])
   xlabel('time (frame)')
   ylabel('percent active')

   data2Sum = sum(sum(data2Binned,3),1)./size(data2Binned,1).*100;
   figure()
   plot(data2Sum)
   title(['PSTH (% active) for ',name2])
   xlabel('time (frame)')
   ylabel('percent active')

   fprintf(1,'Done.\n');

   [xcf,lags,bounds] = crosscorr(data1Sum,data2Sum);
   
   figure()
   stem(lags,xcf)
   hold on
   bounds1Vect = bounds(1).*ones(length(lags));
   bounds2Vect = bounds(2).*ones(length(lags));
   plot(lags,bounds1Vect,'r')
   plot(lags,bounds2Vect,'r')
   hold off
   title([name1,' PSTH vs. ',name2,' PSTH x-corr'])
   xlabel('lag')
   ylabel('sample cross correlation')

   %%
   %%
   %%%CORRELATIONS
   %% inner product between all pairs of cells
   %%
   %%

   fprintf(1,['\n------Compute Cross Correlation------\n\n']);

   % Average across bins
   data1AvgBin = mean(data1Binned,3);
   data2AvgBin = mean(data2Binned,3);

   % Sort by activity
   [sortData1,sortIdx1] = sort(data1AvgBin,1,'descend');
   [sortData2,sortIdx2] = sort(data2AvgBin,1,'descend');
   
   % Grab most active cell
   dataVect1 = sortData1(1,:);
   dataVect2 = sortData2(1,:);

   % Compute cross correlation for most active cell
   [xcf,lags,bounds] = crosscorr(dataVect1,dataVect2);

   xcfArry  = cell([size(data1AvgBin,1) size(xcf,2)]);
   for i=1:size(data1AvgBin,1)
      xcfArry{i} = squeeze(crosscorr(data1AvgBin(i,:),data2AvgBin(i,:)));
   end
   
   figure()
   stem(lags,mean(cell2mat(xcfArry)))
   hold on
   bounds1Vect = bounds(1).*ones(length(lags));
   bounds2Vect = bounds(2).*ones(length(lags));
   plot(lags,bounds1Vect,'r')
   plot(lags,bounds2Vect,'r')
   hold off
   title([name1,' vs. ',name2,' mean x-corr for all cells'])
   xlabel('lag')
   ylabel('sample cross correlation')

   fprintf(1,'Done.\n');

   clearvars data1 data2 xcfArry data1Sum data2Sum threshData1 threshData2 data1Binned data2Binned sortData1 sortData2 sortIdx1 sortIdx2
   
   fprintf(1,['\n------Compute Correlation Matrix for ',num2str(numCorrNeurons),' Random Neurons------\n\n']);
   numPVals = zeros(numCorrIterations);
   for iteration=1:numCorrIterations
      fprintf(1,'Iteration %d of %d\n\n',iteration,numCorrIterations);

      if ne(numCorrNeurons,-1)
         randIndexMatrix = randi(size(data1AvgBin,1),1,numCorrNeurons);
      else
         randIndexMatrix = [1:size(data1AvgBin)];
      end

      [rho pval] = corr(data1AvgBin(randIndexMatrix,:)',data2AvgBin(randIndexMatrix,:)');

      [val idx] = max(rho(:));
      [i j]     = ind2sub(size(rho),idx);
      fprintf(1,['Maximum correlation is ',num2str(val),' at subscrips i=',num2str(i),' j=',num2str(j),' with p-value ',num2str(pval(idx)),'\n'])

      [i j v] = find(pval<0.01);
      fprintf(1,[num2str(length(i)),' pairs have correlation with p-value less than 0.01\n']);
      [i j v] = find(pval<0.001);
      fprintf(1,[num2str(length(i)),' pairs have correlation with p-value less than 0.001\n\n']);
      pvals(iteration) = length(i);
   end
   fprintf(1,'An average of %d pairs (%d percent) had p-values less than 0.001 in %d iterations.\n\n',mean(pvals),(mean(pvals)/numel(pval))*100,numCorrIterations);

   fprintf(1,'Done.\n');

   %keyboard

end
