%%  analyzeV1.m - An analysis tool for first-layer PV implementations
%%  -Wesley Chavez
%%
%% -----Hints-----
%% Copy this file to your PV output directory (outputPath in params file) and edit lines 20-26 accordingly.
%% 
%% Write all layers and connections at the end of a display period (initialWriteTime = displayPeriod*n or displayPeriod*n - 1 and writeStep = displayPeriod*k, n and k are integers > 0)
%% Sync the write times of Input and Recon layers for comparison (writeStep and initialWriteTime in params file).
%% Sync the write times of Input and Error layers for more useful RMS values (more useful than just standard deviation of Error values).
%% Sync the write times of Error and V1 layers for error vs sparsity graph.
%% You know what, just sync all your write times.
%%
%% If you want to run this script while your PetaVision implementation is still running, don't change the order of readpvpfile commands below.
%%
%% -----ToDo-----
%% Add functionality to read from Checkpoint pvp files (no write time synchronization needed).
%% Convolve 2nd layer (V2,S2,etc.) weights with 1st layer (V1,S1) weights for visualization in image space.
%% Figure out how to plot/save everything without user input (Octave asks "-- less -- (f)orward, (b)ack, (q)uit" after plotting). 

addpath('~/workspace/PetaVision/mlab/util');

datainput = readpvpfile('a3_GanglionRescaled.pvp',10);
dataerr = readpvpfile('a4_Error.pvp',10);
[dataV1 headerV1] = readpvpfile('a5_V1.pvp',10);
datarecon = readpvpfile('a6_Recon.pvp',10);
dataw = readpvpfile('w4_V1ToError.pvp',10);


%%%% Input
for i = size(datainput,1)-4:size(datainput,1)
   t = datainput{i}.time;
   p = datainput{i}.values;
   disp (['Ganglion size: ', num2str(size(p))]);
   p = p-min(p(:));
   p = p*255/max(p(:));
   p = permute(p,[2 1 3]);
   p = uint8(p);
   imwrite(p,['Ganglion_' sprintf('%.08d',t) '.png']);
end

%%%% Recon
for i = size(datarecon,1)-4:size(datarecon,1)
   t = datarecon{i}.time;
   p = datarecon{i}.values;
   disp (['Recon size: ', num2str(size(p))]);
   p = p-min(p(:));
   p = p*255/max(p(:));
   p = permute(p,[2 1 3]);
   p = uint8(p);
   imwrite(p,['Recon_' sprintf('%.08d',t) '.png']);
end


%%%% Error
%% If write-times for input layer and error were synced, plot RMS error.  Else, plot std of error values. 


for i = 1:size(datainput,1)
   if (dataerr{i}.time == datainput{i}.time)
      syncedtimes = 1;
   else
      syncedtimes = 0;
      break;
   end
end

if (syncedtimes)
   for i = 1:size(datainput,1)
      t_err(i) = dataerr{i}.time;
      err(i) = std(dataerr{i}.values(:))/std(datainput{i}.values(:));
   end

   h_err = figure(1);
   plot(t_err,err);
   print(h_err,['RMS_Error_' sprintf('%.08d',t_err(length(t_err))) '.png']);
else
   for i = 1:size(dataerr,1)
      t_err(i) = dataerr{i}.time;
      err(i) = std(dataerr{i}.values(:));
   end

   h_err = figure(1);
   plot(t_err,err);
   print(h_err,['Std_Error_' sprintf('%.08d',t_err(length(t_err))) '.png']);
end


%%%% V1 Sparsity and activity per feature
numV1neurons = headerV1.nx*headerV1.ny*headerV1.nf;
for i = 1:size(dataV1,1)
   t_V1(i) = dataV1{i}.time;
   sparsity(i) = size(dataV1{i}.values,1)/(numV1neurons);
   if (i == size(dataV1,1))
      V1_yxf = zeros(1,numV1neurons);
      V1_yxf(dataV1{i}.values(:,1)+1) = dataV1{i}.values(:,2);
      V1_yxf = reshape(V1_yxf,[headerV1.nf headerV1.nx headerV1.ny]);
      V1_yxf = permute(V1_yxf,[3 2 1]);
      for j = 1:headerV1.nf
         meanfeatureval = mean(mean(V1_yxf));
         meanfeatureval = meanfeatureval(:)';
      end
   end   
end
h_V1 = figure(2);
plot(t_V1,sparsity);
print(h_V1,['V1_Sparsity_' sprintf('%.08d',t_V1(length(t_V1))) '.png']);

h_V1feats = figure(3);
bar(meanfeatureval);
print(h_V1feats,['MeanFeatureValues_' sprintf('%.08d',t_V1(length(t_V1))) '.png'])


%%%% Error vs Sparse    Print this graph if V1 and Error write times are synced. (blue = first write time, red = last write time)
for i = 1:size(dataerr,1)
   if (dataV1{i}.time == dataerr{i}.time)
      syncedtimes = 1;
   else
      syncedtimes = 0;
      break;
   end
end

if (syncedtimes)
   h_V1vsSparse = figure(4);
   c=linspace(0,1,length(err));
   scatter(sparsity(1:length(err)),err,[],c);
   xlabel('Sparsity');
   ylabel('Error');
   print(h_V1vsSparse,['ErrorVsSparse_' sprintf('%.08d',t_V1(length(t_V1))) '.png']);
end



%%%% Weights     Each patch is normalized individually.
t = dataw{size(dataw,1)}.time;
numpatches = size(dataw{size(dataw,1)}.values{1})(4)
for i = 1:numpatches
   patch{i} = dataw{size(dataw,1)}.values{1}(:,:,:,i);
   patch{i} = patch{i}-min(patch{i}(:));
   patch{i} = patch{i}*255/max(patch{i}(:));
   patch{i} = uint8(permute(patch{i},[2 1 3]));
end
subplot_x = ceil(sqrt(numpatches));
subplot_y = ceil(numpatches/subplot_x);

h_weights1 = figure(5);
for i = 1:numpatches
   i
   fflush(1);
   subplot(subplot_y,subplot_x,i);
   imshow(patch{i});
end
print(h_weights1,['WeightsByFeatureIndex_' sprintf('%.08d',t) '.png']);

[dontcare sortedindex] = sort(meanfeatureval);
sortedindex = fliplr(sortedindex);
h_weights2 = figure(6);
for i = 1:numpatches
   i
   fflush(1);
   subplot(subplot_y,subplot_x,i);
   imshow(patch{sortedindex(i)});
end
print(h_weights2,['WeightsByActivity_' sprintf('%.08d',t) '.png']);

