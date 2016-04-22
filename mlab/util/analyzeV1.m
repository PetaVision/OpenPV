%%  analyzeV1.m - An analysis tool for first-layer PV implementations
%%  -Wesley Chavez
%%
%% -----Hints-----
%% Copy this file to your PV output directory (outputPath in params file) and edit lines 26-30 accordingly.
%% 
%% Write all layers and connections at the end of a display period (initialWriteTime = displayPeriod*n or displayPeriod*n - 1 and writeStep = displayPeriod*k, n and k are integers > 0)
%% Sync the write times of Input and Recon layers for comparison (writeStep and initialWriteTime in params file).
%% Sync the write times of Input and Error layers for more useful RMS values (more useful than just standard deviation of Error values).
%% Sync the write times of Error and V1 layers for Error vs V1sparsity graph.
%% You know what, just sync all your write times.
%%
%% If you want to run this script while your PetaVision implementation is still running, don't change the order of readpvpfile commands.
%%
%% -----ToDo-----
%% Add functionality to read from Checkpoint pvp files (no write time synchronization needed).
%% Convolve 2nd layer (V2,S2,etc.) weights with 1st layer (V1,S1) weights for visualization in image space.
%% Figure out how to plot/save everything without user input (Octave asks "-- less -- (f)orward, (b)ack, (q)uit" after plotting). 


% Counts number of figures plotted, initialize to zero (Doesn't include input and recon imwrites)
numFigures = 0; 

addpath('~/workspace/PetaVision/mlab/util'); 

inputpvp = 'a3_GanglionRescaled.pvp'
errpvp = 'a4_Error.pvp'
V1pvp = 'a5_V1.pvp'
reconpvp = 'a6_Recon.pvp'
weightspvp = 'w4_V1ToError.pvp'

%How many of the most recent inputs/recons in the pvp files you want to write
numImagesToWrite = 5;



%%%% Input    Only last numImagesToWrite frames are written 
inputdata = readpvpfile(inputpvp,10);

% Save these for error computation
for i = 1:size(inputdata,1)
   t_input(i) = inputdata{i}.time;
   inputstd(i) = std(inputdata{i}.values(:));
end

% Normalize and write input images
for i = size(inputdata,1)-numImagesToWrite+1:size(inputdata,1)
   t = inputdata{i}.time;
   p = inputdata{i}.values;
   disp (['Ganglion size: ', num2str(size(p))]);
   p = p-min(p(:));
   p = p*255/max(p(:));
   p = permute(p,[2 1 3]);
   p = uint8(p);
   outFile = ['Ganglion_' sprintf('%.08d',t) '.png']
   imwrite(p,outFile);
end
clear inputdata;



%%%% Recon    Only last numImagesToWrite frames are read and written
fid = fopen(reconpvp,'r');
reconheader = readpvpheader(fid);
fclose(fid);
if (reconheader.nbands < numImagesToWrite)
   display('Recon pvp was only written to %d times, but numImagesToWrite is specified as %d\n', reconheader.nbands, numImagesToWrite);
end
recondata = readpvpfile(reconpvp,10, reconheader.nbands, reconheader.nbands-numImagesToWrite+1);

% Normalize and write recon images
for i = 1:size(recondata,1)
   t = recondata{i}.time;
   p = recondata{i}.values;
   disp (['Recon size: ', num2str(size(p))]);
   p = p-min(p(:));
   p = p*255/max(p(:));
   p = permute(p,[2 1 3]);
   p = uint8(p);
   outFile = ['Recon_' sprintf('%.08d',t) '.png']
   imwrite(p,outFile);
end
clear recondata;



%%%% Error
%% If write-times for input layer and error were synced, plot RMS error.  Else, plot std of error values. 
errdata = readpvpfile(errpvp,10);


for i = 1:size(t_input,2)  % If PetaVision implementation is still running, errdata might contain more frames, even if synced with input, since errpvp is read after inputpvp. 
   if (errdata{i}.time == t_input(i))
      syncedtimes = 1;
   else
      syncedtimes = 0;
      break;
   end
end

if (syncedtimes)
   for i = 1:size(t_input,2)
      t_err(i) = errdata{i}.time;
      err(i) = std(errdata{i}.values(:))/inputstd(i);
   end
   numFigures++;
   h_err = figure(numFigures);
   plot(t_err,err);
   outFile = ['RMS_Error_' sprintf('%.08d',t_err(length(t_err))) '.png']
   print(h_err,outFile);
else
   for i = 1:size(errdata,1)
      t_err(i) = errdata{i}.time;
      err(i) = std(errdata{i}.values(:));
   end
   numFigures++;
   h_err = figure(numFigures);
   plot(t_err,err);
   outFile = ['Std_Error_' sprintf('%.08d',t_err(length(t_err))) '.png']
   print(h_err,outFile);
end
clear errdata;



%%%% V1 Sparsity and activity per feature
[V1data V1header] = readpvpfile(V1pvp,10);
V1numneurons = V1header.nx*V1header.ny*V1header.nf;
for i = 1:size(V1data,1)
   t_V1(i) = V1data{i}.time;
   V1sparsity(i) = size(V1data{i}.values,1)/(V1numneurons);
   if (i == size(V1data,1))
      V1_yxf = zeros(1,V1numneurons);
      V1_yxf(V1data{i}.values(:,1)+1) = V1data{i}.values(:,2);
      V1_yxf = reshape(V1_yxf,[V1header.nf V1header.nx V1header.ny]);
      V1_yxf = permute(V1_yxf,[3 2 1]);  % Reshaped to actual size of V1 layer
      V1meanfeaturevals = mean(mean(V1_yxf));
      V1meanfeaturevals = V1meanfeaturevals(:)';
      t_V1_sortedweights = t_V1(i);
   end   
end
numFigures++;
h_V1 = figure(numFigures);
plot(t_V1,V1sparsity);
outFile = ['V1_Sparsity_' sprintf('%.08d',t_V1(length(t_V1))) '.png']
print(h_V1,outFile);

numFigures++;
h_V1featvals = figure(numFigures);
bar(V1meanfeaturevals);
outFile = ['MeanFeatureValues_' sprintf('%.08d',t_V1(length(t_V1))) '.png']
print(h_V1featvals,outFile);
clear V1data;



%%%% Error vs Sparse    Print this graph if V1 and Error write times are synced. (blue = first write time, red = last write time)
for i = 1:size(t_err,2)  % If PetaVision implementation is still running, V1data might contain more frames, even if synced with input, since V1pvp is read after errpvp.
   if (t_V1(i) == t_err(i))
      syncedtimes = 1;
   else
      syncedtimes = 0;
      break;
   end
end

if (syncedtimes)
   numFigures++;
   h_ErrorvsSparse = figure(numFigures);
   c=linspace(0,1,length(err));
   scatter(V1sparsity(1:length(err)),err,[],c);
   xlabel('Sparsity');
   ylabel('Error');
   outFile = ['ErrorVsSparse_' sprintf('%.08d',t_V1(length(t_V1))) '.png']
   print(h_ErrorvsSparse,outFile);
end



%%%% Weights     Only last weights frame is analyzed.  Each weightspatch is normalized individually.
fid = fopen(weightspvp,'r');
weightsheader = readpvpheader(fid);
fclose(fid);
weightsfiledata=dir(weightspvp);
weightsframesize = weightsheader.recordsize*weightsheader.numrecords+weightsheader.headersize;
weightsnumframes = weightsfiledata(1).bytes/weightsframesize;
weightsdata = readpvpfile(weightspvp,10,weightsnumframes,weightsnumframes);
t = weightsdata{size(weightsdata,1)}.time;
weightsnumpatches = size(weightsdata{size(weightsdata,1)}.values{1})(4)

for i = 1:weightsnumpatches
   weightspatch{i} = weightsdata{size(weightsdata,1)}.values{1}(:,:,:,i);
   weightspatch{i} = weightspatch{i}-min(weightspatch{i}(:));
   weightspatch{i} = weightspatch{i}*255/max(weightspatch{i}(:));
   weightspatch{i} = uint8(permute(weightspatch{i},[2 1 3]));
end
subplot_x = ceil(sqrt(weightsnumpatches));
subplot_y = ceil(weightsnumpatches/subplot_x);

numFigures++;
h_weightsbyindex = figure(numFigures);
for i = 1:weightsnumpatches
   i
   fflush(1);
   subplot(subplot_y,subplot_x,i);
   imshow(weightspatch{i});
end
outFile = ['WeightsByFeatureIndex_' sprintf('%.08d',t) '.png']
print(h_weightsbyindex,outFile);

[dontcare sortedindex] = sort(V1meanfeaturevals);
sortedindex = fliplr(sortedindex);
numFigures++;
h_weightsbyactivity = figure(numFigures);
for i = 1:weightsnumpatches
   i
   fflush(1);
   subplot(subplot_y,subplot_x,i);
   imshow(weightspatch{sortedindex(i)});
end
if (t == t_V1_sortedweights)
   outFile = ['WeightsByActivity_' sprintf('%.08d',t) '.png']
   print(h_weightsbyactivity,outFile);
else  % If last V1pvp write time and last weightspvp write time are not the same, specifies both
   outFile = ['Weights_' sprintf('%.08d',t) '_ByActivity@_' sprintf('%.08d',t_V1_sortedweights) '.png']
   print(h_weightsbyactivity,outFile);
end
