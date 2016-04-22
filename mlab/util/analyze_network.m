%%  analyze_network.m - An analysis tool for PetaVision implementations
%%  -Wesley Chavez 3/18/15
%%
%% -----Hints-----
%% BEFORE RUNNING THIS SCRIPT FOR THE FIRST TIME:
%% Add the PetaVision mlab/util directory to your .octaverc or .matlabrc file, which is probably in your home (~) directory.
%% Example:    addpath("~/workspace/PetaVision/mlab/util")
%%
%% IMPORTANT: 
%% Run this script from your outputPath directory.  outputPath is specified in your pv.params file.
%% Analysis figures will be written to *outputPath*/Analysis
%%
%% IN YOUR PARAMS FILE:
%% Write all layers and connections at the end of a display period (initialWriteTime = displayPeriod*n or displayPeriod*n - 1 and writeStep = displayPeriod*k, n and k are integers > 0)
%% Sync the write times of Input and Recon layers for comparison (writeStep and initialWriteTime in params file).
%% Sync the write times of Input and Error layers for more useful RMS values (more useful than just standard deviation of Error values).
%% Sync the write times of Error and Sparse layers for Sparsity vs Error figure.
%% You know what, just sync all your write times.
%%
%% If you want to run this script while your PetaVision implementation is still running, don't change the order of readpvpfile commands.
%%
%% -----ToDo-----
%% Add functionality for non-shared-weight pvp files.
%% Add functionality to read from Checkpoint pvp files (no write time synchronization needed).
%% Convolve 2nd layer (V2,S2,etc.) weights with 1st layer (V1,S1) weights for visualization in image space.  This is implemented for checkpointed weights files in PetaVision/mlab/util/printCheckpointedSecondLevelWeights.m
more off;
numImagesToWrite = 5;  % The number of inputs/recons to write

paths = path;
path_separation = pathsep;
path_separation_indices = findstr(paths,path_separation);
mlab_util_index = findstr(paths,'/mlab/util');
if (isempty(mlab_util_index))
   display('Please add the PetaVision mlab/util directory to your .octaverc or .matlabrc file, which is probably in your home (~) directory, then you may run this script.');
   display('Example:    addpath("~/workspace/PetaVision/mlab/util")');
   exit;
end
startpath = path_separation_indices(path_separation_indices < mlab_util_index)(size(path_separation_indices(path_separation_indices < mlab_util_index),2))+1;
pv_path = paths(startpath:mlab_util_index-1)
status = system(['python ' pv_path  '/plab/get_names.py ' pv_path]);
status = system(['python ' pv_path  '/plab/analysis_parse.py']);
system('rm layers.txt connections.txt');

fid = fopen('found_pvps.txt', 'r');
sparsepvps_notformatted = fgetl(fid);
errpvps_notformatted = fgetl(fid);
inputpvps_notformatted = fgetl(fid);
weightspvps_notformatted = fgetl(fid);
reconpvps_notformatted = fgetl(fid);
fclose(fid);

sparsepvps = strsplit(sparsepvps_notformatted,',')
errpvps = strsplit(errpvps_notformatted,',')
inputpvps = strsplit(inputpvps_notformatted,',')
weightspvps = strsplit(weightspvps_notformatted,',')
reconpvps = strsplit(reconpvps_notformatted,',')

prefix='a[0123456789]+_';
suffix='.pvp';
system('mkdir Analysis');

%%%% Input    Only last numImagesToWrite frames are written 
%%%% 
%%%%
disp('------Analyzing Inputs------');
input_flag = 0; % To make sure we have data for at least one input
[uniqueinputpvps dontcare input_duplicate_index] = unique(inputpvps);  % To save computation time  
for i = 1:size(uniqueinputpvps,2)
   if (isempty(uniqueinputpvps{i}))
      continue;
   end
   input_flag = 1;
   [inputdata inputheader] = readpvpfile(uniqueinputpvps{i},100);
   if (inputheader.filetype == 4)  % Dense activity pvps
      for j = 1:size(inputdata,1)
         unique_t_input{i}(j) = inputdata{j}.time;
         unique_inputstd{i}(j) = std(inputdata{j}.values(:));
      end
   elseif ((inputheader.filetype == 2) || (inputheader.filetype == 6))  % Sparse pvps
      numneurons = inputheader.nx*inputheader.ny*inputheader.nf;
      for j = 1:size(inputdata,1)
         unique_t_input{i}(j) = inputdata{j}.time;
         vals = sparse(zeros(numneurons,1));
         if (inputheader.filetype == 2)
            vals(1:size(inputdata{j}.values,1)) = 1;
         else 
            vals(1:size(inputdata{j}.values,1)) = inputdata{j}.values(:,2);
         end
         unique_inputstd{i}(j)=std(vals);
      end
   else
      display(['Failure: ' uniqueinputpvps{i} ' is a weights pvp file, not an input pvp file.']);
   end
   [startPrefix,endPrefix] = regexp(uniqueinputpvps{i},prefix);
   [startSuffix,endSuffix] = regexp(uniqueinputpvps{i},suffix);
   % Normalize and write input images
   if ((inputheader.nf == 1) || (inputheader.nf == 3)) 
      if (inputheader.nbands >= numImagesToWrite) 
         for j = inputheader.nbands-numImagesToWrite+1:inputheader.nbands
            t = inputdata{j}.time;
            p = inputdata{j}.values;
            p = p-min(p(:));
            p = p*255/max(p(:));
            p = permute(p,[2 1 3]);
            p = uint8(p);
            outFile = ['Analysis/' uniqueinputpvps{i}(endPrefix+1:startSuffix-1) '_' sprintf('%.08d',t) '.png']
            imwrite(p,outFile);
         end
      else
         for j = 1:inputheader.nbands
            t = inputdata{j}.time;
            p = inputdata{j}.values;
            p = p-min(p(:));
            p = p*255/max(p(:));
            p = permute(p,[2 1 3]);
            p = uint8(p);
            outFile = ['Analysis/' uniqueinputpvps{i}(endPrefix+1:startSuffix-1) '_' sprintf('%.08d',t) '.png']
            imwrite(p,outFile);
         end
      end
   else
      display(['Not writing ' uniqueinputpvps{i}(endPrefix+1:startSuffix-1) ' layer; nf needs to be 1 or 3 for visualization.']);
   end
end
clear inputdata;

if(input_flag)  % Saves times and standard deviations at correct indices (octave's unique() function only sorts alphabetically)
   for i = 1:size(inputpvps,2)
      t_input{i} = unique_t_input{find(ismember(uniqueinputpvps,inputpvps{i}))}; 
      inputstd{i} = unique_inputstd{find(ismember(uniqueinputpvps,inputpvps{i}))}; 
   end
end


%%%% Recon    Only last numImagesToWrite frames are read and written
%%%%
%%%%
disp('------Analyzing Recons------');
for i = 1:size(reconpvps,2)
   if (isempty(reconpvps{i}))
      continue;
   end
   fid = fopen(reconpvps{i},'r');
   reconheader = readpvpheader(fid);
   fclose(fid);
   if (reconheader.nbands >= numImagesToWrite)
      recondata = readpvpfile(reconpvps{i},100, reconheader.nbands, reconheader.nbands-numImagesToWrite+1);
   else
      recondata = readpvpfile(reconpvps{i},100);
   end
   [startPrefix,endPrefix] = regexp(reconpvps{i},prefix);
   [startSuffix,endSuffix] = regexp(reconpvps{i},suffix);
   % Normalize and write RGB recon images 
   if (reconheader.nf == 3)
      for j = 1:size(recondata,1)
         t = recondata{j}.time;
         p = recondata{j}.values;
         p = p-min(p(:));
         p = p*255/max(p(:));
         p = permute(p,[2 1 3]);
         p = uint8(p);
         outFile = ['Analysis/' reconpvps{i}(endPrefix+1:startSuffix-1) '_' sprintf('%.08d',t) '.png']
         imwrite(p,outFile);
      end
   else 
   % Normalize across all features and write a recon image for each feature.
      for j = 1:size(recondata,1)
         t = recondata{j}.time;
         p = recondata{j}.values;
         minp = min(p(:));
         maxp = max(p(:));
         for k = 1:reconheader.nf
            p_nf = p(:,:,k);
            p_nf = p_nf-minp;
            p_nf = p_nf*255/(maxp-minp);
            p_nf = permute(p_nf,[2 1 3]);
            p_nf = uint8(p_nf);
            if (reconheader.nf == 1)
               outFile = ['Analysis/' reconpvps{i}(endPrefix+1:startSuffix-1) '_' sprintf('%.08d',t) '.png']
            else
               outFile = ['Analysis/' reconpvps{i}(endPrefix+1:startSuffix-1) '_Feature_' sprintf('%.03d',k) '_'  sprintf('%.08d',t) '.png']
            end
            imwrite(p_nf,outFile);
         end
      end
   end
end
clear recondata;

%%%% Error   If write-times for input layer and error were synced, plot RMS error.  Else, plot std of error values.
%%%%
%%%%
disp('------Analyzing Errors------');
err_flag = 0;  % To make sure we have data for at least one error layer
for i = 1:size(errpvps,2)
   if (isempty(errpvps{i}))
      continue;
   end
   err_flag = 1;
   errdata = readpvpfile(errpvps{i},100);
   [startPrefix,endPrefix] = regexp(errpvps{i},prefix);
   [startSuffix,endSuffix] = regexp(errpvps{i},suffix);
   syncedtimes = 0;
   if (input_flag)
      if !(isempty(t_input{i}))
         for j = 1:min([size(errdata,1) size(t_input{i},2)])  % If PetaVision implementation is still running, errdata might contain more frames, even if synced with input, since errpvp is read after inputpvp. 
            if (errdata{j}.time == t_input{i}(j))
               syncedtimes = 1;
            else
               syncedtimes = 0;
               break;
            end
         end
      end
   end
   if (syncedtimes)
      for j = 1:min([size(errdata,1) size(t_input{i},2)])
         t_err{i}(j) = errdata{j}.time;
         err{i}(j) = std(errdata{j}.values(:))/inputstd{i}(j);
      end
      h_err = figure;
      plot(t_err{i},err{i});
      outFile = ['Analysis/' errpvps{i}(endPrefix+1:startSuffix-1) '_RMS_' sprintf('%.08d',t_err{i}(length(t_err{i}))) '.png']
      print(h_err,outFile);
   else
      for j = 1:size(errdata,1)
         t_err{i}(j) = errdata{j}.time;
         err{i}(j) = std(errdata{j}.values(:));
      end
      h_err = figure;
      plot(t_err{i},err{i});
      outFile = ['Analysis/' errpvps{i}(endPrefix+1:startSuffix-1) '_Std_' sprintf('%.08d',t_err{i}(length(t_err{i}))) '.png']
      print(h_err,outFile);
   end
end
clear errdata;


%%%% Sparsity and mean activity per feature
%%%%
%%%%
disp('------Analyzing Sparsity------');
[uniquesparsepvps dontcare sparse_duplicate_index] = unique(sparsepvps);  % To save computation time  
sparse_flag = 0;  % To make sure we have data for at least one sparse layer 
for i = 1:size(uniquesparsepvps,2)
   if (isempty(uniquesparsepvps{i}))
      continue;
   end
   sparse_flag = 1;
   [sparsedata sparseheader] = readpvpfile(uniquesparsepvps{i},100);
   [startPrefix,endPrefix] = regexp(uniquesparsepvps{i},prefix);
   [startSuffix,endSuffix] = regexp(uniquesparsepvps{i},suffix);
   numneurons = sparseheader.nx*sparseheader.ny*sparseheader.nf;
   for j = 1:size(sparsedata,1)
      unique_t_sparse{i}(j) = sparsedata{j}.time;
      unique_sparsity{i}(j) = size(sparsedata{j}.values,1)/numneurons;
      if (j == size(sparsedata,1))  % Mean feature activity at last write time
         sparse_yxf = zeros(1,numneurons);
         sparse_yxf(sparsedata{j}.values(:,1)+1) = sparsedata{j}.values(:,2);
         sparse_yxf = reshape(sparse_yxf,[sparseheader.nf sparseheader.nx sparseheader.ny]);
         sparse_yxf = permute(sparse_yxf,[3 2 1]);  % Reshaped to actual size of sparse layer
         unique_sparsemeanfeaturevals{i} = zeros(1,1,size(sparse_yxf,3));
         if (size(sparse_yxf,1) == 1 || (size(sparse_yxf,2) == 1))  % unique_sparsemeanfeaturevals should be 1 x 1 x nf
            if (size(sparse_yxf,1) == 1 && (size(sparse_yxf,2) == 1))
               unique_sparsemeanfeaturevals{i} = sparse_yxf;
            else
               unique_sparsemeanfeaturevals{i} = mean(sparse_yxf);
            end
         else
            unique_sparsemeanfeaturevals{i} = mean(mean(sparse_yxf));
         end
         unique_sparsemeanfeaturevals{i} = unique_sparsemeanfeaturevals{i}(:)';
         unique_t_sparsemeanfeaturevals{i} = unique_t_sparse{i}(j);
      end   
   end
   h_sparse = figure;
   plot(unique_t_sparse{i},unique_sparsity{i});
   outFile = ['Analysis/' uniquesparsepvps{i}(endPrefix+1:startSuffix-1) '_Sparsity_' sprintf('%.08d',unique_t_sparse{i}(length(unique_t_sparse{i}))) '.png']
   print(h_sparse,outFile);

   h_sparsefeaturevals = figure;
   bar(unique_sparsemeanfeaturevals{i});
   outFile = ['Analysis/' uniquesparsepvps{i}(endPrefix+1:startSuffix-1) '_MeanFeatureValues_' sprintf('%.08d',unique_t_sparse{i}(length(unique_t_sparse{i}))) '.png']
   print(h_sparsefeaturevals,outFile);
end
clear sparsedata;

if (sparse_flag)  % Saves sparsity, times, and mean feature values at correct indices (octave's unique() function only sorts alphabetically)
   for i = 1:size(sparsepvps,2)
      sparsity{i} = unique_sparsity{find(ismember(uniquesparsepvps,sparsepvps{i}))}; 
      t_sparse{i} = unique_t_sparse{find(ismember(uniquesparsepvps,sparsepvps{i}))}; 
      sparsemeanfeaturevals{i} = unique_sparsemeanfeaturevals{find(ismember(uniquesparsepvps,sparsepvps{i}))}; 
      t_sparsemeanfeaturevals{i} = unique_t_sparsemeanfeaturevals{find(ismember(uniquesparsepvps,sparsepvps{i}))}; 
   end
end


%%%% Error vs Sparse    Print this graph if sparse layer and error layer write times are synced. (blue data point = first write time, red data point = last write time)
%%%%
%%%%
disp('------Analyzing Sparsity vs Error------');
if(err_flag && sparse_flag)
   for i = 1:size(errpvps,2)
      syncedtimes1{i} = 0;
      if !((isempty(t_sparse{i}))||(isempty(t_err{i})))
         for j = 1:  min([size(t_sparse{i},2) size(t_err{i},2)])  % If PetaVision implementation is still running, sparse data might contain more frames, even if synced with input, since sparse pvps are read after error pvps.
            if (t_sparse{i}(j) == t_err{i}(j))
               syncedtimes1{i} = 1;
            else
               syncedtimes1{i} = 0;
               break;
            end
         end
      end
   end
   for i = 1:size(uniquesparsepvps,2)
      if (syncedtimes1{i})
         [startPrefix,endPrefix] = regexp(uniquesparsepvps{i},prefix);
         [startSuffix,endSuffix] = regexp(uniquesparsepvps{i},suffix);
         h_SparsevsError = figure;
         err_index = find(sparse_duplicate_index == i);
         c=linspace(0,1,length(err{err_index(1)}));
         for j = 1:size(err_index,2)
            hold on;
            scatter(sparsity{err_index(j)}(1:length(err{err_index(j)})),err{err_index(j)},[],c);
         end
         xlabel('Sparsity');
         ylabel('Error');
         outFile = ['Analysis/' uniquesparsepvps{i}(endPrefix+1:startSuffix-1) '_SparsityVsError_' sprintf('%.08d',t_sparse{i}(length(t_sparse{i}))) '.png']
         print(h_SparsevsError,outFile);
      end
   end
end


%%%% Weights     Only last weights frame is analyzed.  Each weightspatch is normalized individually.
%%%%
%%%%
disp('------Analyzing Weights------');
for i = 1:size(weightspvps,2)
   if (isempty(weightspvps{i}))
      continue;
   end
   prefix='w[0123456789]+_';
   [startPrefix,endPrefix] = regexp(weightspvps{i},prefix);
   [startSuffix,endSuffix] = regexp(weightspvps{i},suffix);
 
   fid = fopen(weightspvps{i},'r');
   weightsheader = readpvpheader(fid);
   fclose(fid);
   weightsfiledata=dir(weightspvps{i});
   weightsframesize = weightsheader.recordsize*weightsheader.numrecords+weightsheader.headersize;
   weightsnumframes = weightsfiledata(1).bytes/weightsframesize;
   weightsdata = readpvpfile(weightspvps{i},100,weightsnumframes,weightsnumframes);
   numcolors = size(weightsdata{1}.values{1})(3);
   if !((numcolors == 1) || (numcolors == 3))
      display(['Not writing ' weightspvps{i}(endPrefix+1:startSuffix-1) ' weights; nfp currently needs to be 1 or 3 for visualization.']);
      continue;
   end
   t = weightsdata{size(weightsdata,1)}.time;
   weightsnumpatches = size(weightsdata{size(weightsdata,1)}.values{1})(4)
   wmax=max(weightsdata{size(weightsdata,1)}.values{1}(:));
   wmin=min(weightsdata{size(weightsdata,1)}.values{1}(:));
   clear weightspatch;
   for j = 1:weightsnumpatches  % Normalize and plot weights by weight index
      weightspatch{j} = weightsdata{size(weightsdata,1)}.values{1}(:,:,:,j);
      weightspatch{j} = weightspatch{j}-wmin;
      weightspatch{j} = weightspatch{j}*255/(wmax-wmin);
      weightspatch{j} = uint8(permute(weightspatch{j},[2 1 3]));
   end
   numpatches_x = ceil(sqrt(weightsnumpatches));
   numpatches_y = ceil(weightsnumpatches/numpatches_x);
   nxp = size(weightsdata{size(weightsdata,1)}.values{1})(1);
   nyp = size(weightsdata{size(weightsdata,1)}.values{1})(2);
   bordersize = 2;
   dicsize_x = numpatches_x * nxp + bordersize * (numpatches_x + 1);
   dicsize_y = numpatches_y * nyp + bordersize * (numpatches_y + 1);
   dictionary = uint8(zeros(dicsize_y,dicsize_x,numcolors));
   count = 1;
   for j = 1:numpatches_y
      for k = 1:numpatches_x
         if (count > weightsnumpatches)
            break;
         end
         dictionary((j-1)*nyp + (j*bordersize) + 1 : j*(nyp+bordersize), ...
         (k-1)*nxp + (k*bordersize) + 1 : k*(nxp+bordersize), :) = weightspatch{count};
         count++;
      end
   end


   outFile = ['Analysis/' weightspvps{i}(endPrefix+1:startSuffix-1) '_WeightsByFeatureIndex_' sprintf('%.08d',t) '.png']
   imwrite(dictionary,outFile);
   if ((sparse_flag) && !(isempty(sparsemeanfeaturevals{i})))
      [dontcare sortedindex] = sort(sparsemeanfeaturevals{i});
      sortedindex = fliplr(sortedindex);
      dictionary = uint8(zeros(dicsize_y,dicsize_x,numcolors));
      count = 1;
      for j = 1:numpatches_y  % Plot weights by activity (Top left = most active)
         for k = 1:numpatches_x
            if (count > weightsnumpatches)
               break;
            end
            dictionary((j-1)*nyp + (j*bordersize) + 1 : j*(nyp+bordersize), ...
            (k-1)*nxp + (k*bordersize) + 1 : k*(nxp+bordersize), :) = weightspatch{sortedindex(count)};
            count++;
         end
      end
      
      if (t == t_sparsemeanfeaturevals{i})
         outFile = ['Analysis/' weightspvps{i}(endPrefix+1:startSuffix-1) '_WeightsByActivity_' sprintf('%.08d',t) '.png']
         imwrite(dictionary,outFile);
      else  % If last sparse pvp write time and last weights pvp write time are not the same, specifies both
         outFile = ['Analysis/' weightspvps{i}(endPrefix+1:startSuffix-1) '_Weights_' sprintf('%.08d',t) 'ByActivity@_' sprintf('%.08d',t_sparsemeanfeaturevals{i}) '.png']
         imwrite(dictionary,outFile);
      end
   end
end
