%%  analyzeRunDict.m - An analysis tool for viewing run statistics
%%  -Wesley Chavez 3/18/15
%%  -Adapted by Sheng Lundquist
addpath('/home/ec2-user/workspace/PetaVision/mlab/util');

numImagesToWrite = 5;  % The number of inputs/recons to write
maxPatches = 512;
%Output PV directory
outDir = '/home/ec2-user/output/dictTrain/binoc_512_white/';

sparsepvps = {[outDir, '/V1.pvp']}
errpvps = {[outDir, '/LeftError.pvp'], [outDir, '/RightError.pvp']}
inputpvps = {[outDir, 'LeftRescale.pvp'], [outDir, 'RightRescale.pvp']}
reconpvps = {[outDir, 'LeftImage.pvp'], [outDir, 'RightImage.pvp'], [outDir, 'LeftRecon.pvp'], [outDir, 'RightRecon.pvp']}

%Find checkpoint directory
chkptdirs = dir([outDir, '/Checkpoints/']);
%Remove . and ..
chkptdirs(1) = [];
chkptdirs(1) = [];
S = [chkptdirs(:).datenum].';
[S, S] = sort(S);
chkptDir = {chkptdirs(S).name}{1}

weightspvps = {
   [outDir, '/Checkpoints/', chkptDir, '/V1ToLeftError_W.pvp']; ...
   [outDir, '/Checkpoints/', chkptDir, '/V1ToRightError_W.pvp']; ...
}

prefix='([^\/]*)$'
suffix='.pvp';

mkdir([outDir, 'Analysis']);
mkdir([outDir, 'Analysis/Input']);
mkdir([outDir, 'Analysis/Recons']);
mkdir([outDir, 'Analysis/Errors']);
mkdir([outDir, 'Analysis/Sparsity']);
mkdir([outDir, 'Analysis/Weights']);

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
            outFile = [outDir, '/Analysis/Input/' uniqueinputpvps{i}(startPrefix:startSuffix-1) '_' sprintf('%.08d',t) '.png']
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
            outFile = [outDir, '/Analysis/Input/' uniqueinputpvps{i}(startPrefix:startSuffix-1) '_' sprintf('%.08d',t) '.png']
            imwrite(p,outFile);
         end
      end
   else
      display(['Not writing ' uniqueinputpvps{i}(startPrefix:startSuffix-1) ' layer; nf needs to be 1 or 3 for visualization.']);
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
         outFile = [outDir, '/Analysis/Recons/' reconpvps{i}(startPrefix:startSuffix-1) '_' sprintf('%.08d',t) '.png']
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
            p_nf = (p_nf-minp)/(maxp-minp);
            p_nf = p_nf*255;
            p_nf = permute(p_nf,[2 1 3]);
            p_nf = uint8(p_nf);

            if (reconheader.nf == 1)
               outFile = [outDir, '/Analysis/Recons/' reconpvps{i}(startPrefix:startSuffix-1) '_' sprintf('%.08d',t) '.png']
            else
               outFile = [outDir, '/Analysis/Recons/' reconpvps{i}(startPrefix:startSuffix-1) '_Feature_' sprintf('%.03d',k) '_'  sprintf('%.08d',t) '.png']
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
         for j = 1:size(t_input{i},2)  % If PetaVision implementation is still running, errdata might contain more frames, even if synced with input, since errpvp is read after inputpvp. 
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
      for j = 1:size(t_input{i},2)
         t_err{i}(j) = errdata{j}.time;
         err{i}(j) = std(errdata{j}.values(:))/inputstd{i}(j);
      end
      h_err = figure;
      plot(t_err{i},err{i});
      outFile = [outDir, '/Analysis/Errors/' errpvps{i}(startPrefix:startSuffix-1) '_RMS_' sprintf('%.08d',t_err{i}(length(t_err{i}))) '.png']
      print(h_err,outFile);
   else
      for j = 1:size(errdata,1)
         t_err{i}(j) = errdata{j}.time;
         err{i}(j) = std(errdata{j}.values(:));
      end
      h_err = figure;
      plot(t_err{i},err{i});
      outFile = [outDir, '/Analysis/Errors/' errpvps{i}(startPrefix:startSuffix-1) '_Std_' sprintf('%.08d',t_err{i}(length(t_err{i}))) '.png']
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
   outFile = [outDir, '/Analysis/Sparsity/' uniquesparsepvps{i}(startPrefix:startSuffix-1) '_Sparsity_' sprintf('%.08d',unique_t_sparse{i}(length(unique_t_sparse{i}))) '.png']
   print(h_sparse,outFile);

   h_sparsefeaturevals = figure;
   bar(unique_sparsemeanfeaturevals{i});
   outFile = [outDir, '/Analysis/Sparsity/' uniquesparsepvps{i}(startPrefix:startSuffix-1) '_MeanFeatureValues_' sprintf('%.08d',unique_t_sparse{i}(length(unique_t_sparse{i}))) '.png']
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

%%%% Weights
%%%%
%%%%
disp('------Analyzing Weights------');
for i = 1:size(weightspvps,2)
   if (isempty(weightspvps{i}))
      continue;
   end
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
  % if !((numcolors == 1) || (numcolors == 3))
  %    display(['Not writing ' weightspvps{i}(endPrefix+1:startSuffix-1) ' weights; numColors currently needs to be 1 or 3 for visualization.']);
  %    continue;
  % end
   t = weightsdata{size(weightsdata,1)}.time;
   weightsnumpatches = size(weightsdata{size(weightsdata,1)}.values{1})(4)
   subplot_x = ceil(sqrt(weightsnumpatches));
   subplot_y = ceil(weightsnumpatches/subplot_x);
   patch_x = size(weightsdata{1}.values{1}, 2);
   patch_y = size(weightsdata{1}.values{1}, 1);
   patch_f = size(weightsdata{1}.values{1}, 3);
   %h_weightsbyindex = figure;

   clear weightspatch;

   weight_patch_array = ...
      zeros(subplot_y*patch_y, subplot_x*patch_x, patch_f);

   for j = 1:weightsnumpatches  % Normalize and plot weights by weight index
      weightspatch{j} = weightsdata{size(weightsdata,1)}.values{1}(:,:,:,j);
      weightspatch{j} = weightspatch{j}-min(weightspatch{j}(:));
      weightspatch{j} = weightspatch{j}*255/max(weightspatch{j}(:));
      weightspatch{j} = uint8(permute(weightspatch{j},[2 1 3]));

      col_ndx = 1 + mod(j-1, subplot_x);
      row_ndx = 1 + floor((j-1) / subplot_y);

      weight_patch_array(((row_ndx-1)*patch_y+1):row_ndx*patch_y, ...
           ((col_ndx-1)*patch_x+1):col_ndx*patch_x,:) = weightspatch{j};

      %subplot(subplot_y,subplot_x,j);
      %imshow(weightspatch{j});
   end
   outFile = [outDir, '/Analysis/Weights/' weightspvps{i}(startPrefix:startSuffix-1) '_WeightsByFeatureIndex_' sprintf('%.08d',t) '.png']
   imwrite((uint8)(weight_patch_array), outFile);
   %print(h_weightsbyindex,outFile);

   if ((sparse_flag) && !(isempty(sparsemeanfeaturevals{i})))
      [dontcare sortedindex] = sort(sparsemeanfeaturevals{i});
      sortedindex = fliplr(sortedindex);
      %h_weightsbyactivity = figure;
      weight_patch_array = ...
         zeros(subplot_y*patch_y, subplot_x*patch_x, patch_f);

      for j = 1:weightsnumpatches  % Plot weights by activity (Top left = most active)
         col_ndx = 1 + mod(j-1, subplot_x);
         row_ndx = 1 + floor((j-1) / subplot_y);
         weight_patch_array(((row_ndx-1)*patch_y+1):row_ndx*patch_y, ...
           ((col_ndx-1)*patch_x+1):col_ndx*patch_x,:) = weightspatch{sortedindex(j)};

         %subplot(subplot_y,subplot_x,j);
         %imshow(weightspatch{sortedindex(j)});
      end

      outFile = [outDir, '/Analysis/Weights/' weightspvps{i}(startPrefix:startSuffix-1) '_WeightsByActivity_' sprintf('%.08d',t) '.png']
      imwrite((uint8)(weight_patch_array), outFile);

      %print(h_weightsbyactivity,outFile);
   end
end
