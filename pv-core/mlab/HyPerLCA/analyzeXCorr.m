%  A function to calculate the xCorr of data on feature space over all positions and time frames
%  iidx, jidx, and finalCorr are a cell array the length of xCorr_list. All 3 vectors are sorted based on correlation,
%  excluding self correlation. iidx and ijdx define which 2 dictionary elements the finalCorr value is for.
%  Note that iidx and jidx are not mirrored, so the correlation pair 1,2 may exist in iidx and jidx respectively, but not
%  in jidx and iidx respectively
function [...
   iidx,             ... % Cell array, length of xCorr_list, contains vectors of point a
   jidx,             ... % Cell array, length of xCorr_list, contains vectors of point b
   finalCorr         ... % Cell array, length of xCorr_list, contains the corr value between a and b, ranked from most to least corr not including self
] = analyzeXCorr(...
   xCorr_list,       ... % A list of activity files to calculate the correlation of 
   output_dir,       ... % The parent output directory of pv
   plot_corr,        ... % Flag that determines if a plot should be made
   frames_calc,      ... % The number of frames to calculate. Will calculate the lastest frame_calc frames. 0 is all frames
   numprocs          ... % The number of processes to use for parallization
   )

   global isTest = false;
   if nargin == 4
      numprocs = nproc
   elseif nargin < 4
      disp(['analyzeXCorr needs 4 or 5 arguments']);
      keyboard;
   end
   
   %setup
   corr_dir= [output_dir, filesep, "xCorr"];
   mkdir(corr_dir);
   iidx = cell(length(xCorr_list), 1);
   jidx = cell(length(xCorr_list), 1);
   finalCorr = cell(length(xCorr_list), 1);
   for iList = 1:length(xCorr_list)
      pvpfile = [output_dir, filesep, xCorr_list{iList}, '.pvp'];
      disp(['Calculating xcorr of ',pvpfile])
      if(isTest)
         [data,hdr] = readpvpfile(pvpfile, 0, 20, 1);
      else
         [data,hdr] = readpvpfile(pvpfile);
      end
      if(frames_calc > 0)
         if(frames_calc < length(data))
            data = data(end-frames_calc:end);
         end%if
      end%if
      if(hdr.filetype ~= 2 && hdr.filetype ~=6)
         disp(['File type not supported, only sparse with activity pvp files accepted']);
         keyboard;
      end%if

      %data = data(1:20);

      %Split by number of processes
      inCellData = cell(numprocs, 1);
      %If there's less frames than number of processors, just feed it into parcellfun
      if (length(data) <= numprocs)
         inCellData = data;
      %If the length of data fits perfectly to numprocs
      else
         procSize = floor(length(data) / numprocs);
         for iProc = 1:numprocs
            beginIdx = (iProc - 1) * procSize + 1;
            endIdx = iProc * procSize;
            if(iProc == numprocs)
               inCellData{iProc} = data(beginIdx:end);
            else
               inCellData{iProc} = data(beginIdx:endIdx);
            end
         end%for
      end%if

      [sumAiAj, sumAi, sumAj, sumsqAi, sumsqAj] = parcellfun(numprocs, @corrOverFrame, inCellData, {hdr}, 'UniformOutput', false);
      %Sum over all cells of results
      try
         sumAiAj = sum(cat(2, sumAiAj{:}), 2);
         sumAi   = sum(cat(2, sumAi{:}), 2);
         sumAj   = sum(cat(2, sumAj{:}), 2);
         sumsqAi = sum(cat(2, sumsqAi{:}), 2);
         sumsqAj = sum(cat(2, sumsqAj{:}), 2);
      catch
         disp('Error in cat')
         keyboard
      end


      %Calculate xCorr
      N = length(data) * hdr.nx * hdr.ny;
      numer = (sumAiAj ./ N) - ((sumAi ./ N) .* (sumAj ./ N));
      denom = sqrt(((sumsqAi ./ N) - ((sumAi .* sumAi)./(N .* N))) .* ((sumsqAj./N) - ((sumAj .* sumAj)./(N.*N))));
      finalCorr{iList} = numer ./ denom;

      %Calculate indicies
      iidx{iList}    = zeros((hdr.nf+1)*hdr.nf/2, 1);
      jidx{iList}    = zeros((hdr.nf+1)*hdr.nf/2, 1);
      for iShift = 0:floor(hdr.nf/2)
         i = 1:hdr.nf;
         i = i';
         j = circshift(i, iShift);
         %Calculate what position it should be put in
         beginIdx = iShift * hdr.nf + 1;
         endIdx   = (iShift+1) * hdr.nf;

         iidx{iList}   (beginIdx:endIdx)     = i;
         jidx{iList}   (beginIdx:endIdx)     = j;
      end%for
      %If nf is even, there are nf/2 extra pairwise multiplications, so throw away last nf/2 elements in matricies
      if(mod(hdr.nf, 2) == 0)
         extra = hdr.nf/2;
         iidx{iList}    = iidx{iList}(1:end-extra, :);
         jidx{iList}    = jidx{iList}(1:end-extra, :);
      end%if

      %Rank 3 vectors by correlation excluding self correlations
      tempCorr = finalCorr{iList};
      %Set self corrilations to zero
      tempCorr(find(iidx{iList} == jidx{iList})) = 0;
      [sortedCorr, corrIdx] = sort(tempCorr, 'descend');
      %Sort all 3 output matricies by corr rank
      finalCorr{iList} = finalCorr{iList}(corrIdx);
      iidx{iList} = iidx{iList}(corrIdx);
      jidx{iList} = jidx{iList}(corrIdx);

      if(plot_corr)
         outImg = full(sparse(iidx{iList}, jidx{iList}, finalCorr{iList}));
         %Fill out the rest of the image
         outImg = (outImg + outImg')/2;
         imhd = imagesc(outImg);
         colorbar;
         saveas(imhd, ...
                [corr_dir, filesep, xCorr_list{iList}, ".png"]);
         drawnow
      end%if
   end%for pvp_file
end%function
%A function to calculate all required sums of all possible pairs over position for calculation of correlation in feature space
%The input parameter data is a single frame
function [sumAiAj, sumAi, sumAj, sumsqAi, sumsqAj] = corrOverPos(data, hdr)
   global isTest;
   N = hdr.nx * hdr.ny * hdr.nf;
   if(~isTest)
      active_ndx   = data.values(:,1);
      if hdr.filetype == 6
         active_vals  = data.values(:,2);
         vec_mat      = full(sparse(active_ndx+1,1,active_vals,N,1,N)); %%Column vector. PetaVision increments in order: nf, nx, ny
      else
         vec_mat      = full(sparse(active_ndx+1,1,1,N,1,N)); %%Column vector. PetaVision increments in order: nf, nx, ny
      end
      rs_mat       = reshape(vec_mat,hdr.nf,hdr.nx*hdr.ny); %rs_mat is now nf by position
   else
      assert(hdr.nf > 60);
      rs_mat = rand(hdr.nf, hdr.nx*hdr.ny);
      rs_mat(40:50, :) = rs_mat(10:20, :);
   end%if isTest
   %circular shift along feature axis to multiply by every possible pair
   %Will do shifts nf/2 times to guarentee every pair gets multiplied with circshift
   sumAiAj = zeros((hdr.nf+1)*hdr.nf./2, hdr.nx.*hdr.ny);
   sumAi   = zeros((hdr.nf+1)*hdr.nf./2, hdr.nx.*hdr.ny);
   sumAj   = zeros((hdr.nf+1)*hdr.nf./2, hdr.nx.*hdr.ny);
   sumsqAi = zeros((hdr.nf+1)*hdr.nf./2, hdr.nx.*hdr.ny);
   sumsqAj = zeros((hdr.nf+1)*hdr.nf./2, hdr.nx.*hdr.ny);
   for iShift = 0:floor(hdr.nf/2)
      %rs_mat is i, shift_mat is j
      shift_mat = circshift(rs_mat, iShift);
      %Calculate what position it should be put in
      beginIdx = iShift * hdr.nf + 1;
      endIdx   = (iShift+1) * hdr.nf;
      %Store value
      sumAiAj(beginIdx:endIdx, :)  = rs_mat .* shift_mat;
      sumAi  (beginIdx:endIdx, :)  = rs_mat;
      sumsqAi(beginIdx:endIdx, :)  = rs_mat .* rs_mat;
      sumAj  (beginIdx:endIdx, :)  = shift_mat;
      sumsqAj(beginIdx:endIdx, :)  = shift_mat .* shift_mat;
   end%for
   %If nf is even, there are nf/2 extra pairwise multiplications, so throw away last nf/2 elements in matricies
   if(mod(hdr.nf, 2) == 0)
      extra = hdr.nf/2;
      sumAiAj = sumAiAj(1:end-extra, :);
      sumAi   = sumAi(1:end-extra, :);
      sumsqAi = sumsqAi(1:end-extra, :);
      sumAj   = sumAj(1:end-extra, :);
      sumsqAj = sumsqAj(1:end-extra, :);
   end%if
   %sum over positions
   sumAiAj = sum(sumAiAj, 2);
   sumAi = sum(sumAi, 2);
   sumsqAi = sum(sumsqAi, 2);
   sumAj = sum(sumAj, 2);
   sumsqAj = sum(sumsqAj, 2);
end%function

%A function to calculate all required sums of all possible pairs over frames for calculation of correlation in feature space
%The input parameter data is a cell array of frames
function [sumAiAj, sumAi, sumAj, sumsqAi, sumsqAj] = corrOverFrame(data, hdr)
   if(~iscell(data))
      data = {data};
   end%if
   [sumAiAj, sumAi, sumAj, sumsqAi, sumsqAj] = cellfun(@corrOverPos, data, {hdr}, 'UniformOutput', false);
   %Sum over all cells of results
   sumAiAj = sum(cat(2, sumAiAj{:}), 2);
   sumAi   = sum(cat(2, sumAi{:}), 2);
   sumAj   = sum(cat(2, sumAj{:}), 2);
   sumsqAi = sum(cat(2, sumsqAi{:}), 2);
   sumsqAj = sum(cat(2, sumsqAj{:}), 2);
end%function
