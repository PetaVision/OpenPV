function [scores, imageEstClass, confus, conf, images, imageClass] = VL_CIFAR()
  %   PV_CIFAR10 adapted from PHOW_CALTECH101
  %  uses PetaVision representations instead of SIFT + kmeans
  more off
  addpath("/Applications/vlfeat-0.9.18/toolbox");
  vl_setup;

  MAX_PROCS = 4;
  conf.dataDirParent = '/Users/garkenyon/workspace/HyPerHLCA/CIFAR_C1X2_task/' ;
  conf.dataDirChild = 'data_batch_all4' ;
  conf.dataDir = [conf.dataDirParent conf.dataDirChild]
  [STATUS, MSG, MSGID] = mkdir(conf.dataDirParent)
  [STATUS, MSG, MSGID] = mkdir(conf.dataDirParent, conf.dataDirChild)
  conf.numTrain = 42000; %% out of 10,000 total images in each data_batch_*, 1000 in each class
  conf.numTest =   8000;
  conf.numClasses = 10; %102 ;
  conf.svm.C = 0.05; %%0.25; %%1; %%5; %%10 ;  %% lower values of C reduce performance on training set but may improve generalization to test set

  conf.svm.solver = 'sdca' ;
  %conf.svm.solver = 'sgd' ;
  %conf.svm.solver = 'liblinear' ;

  conf.svm.biasMultiplier = 1 ;

  conf.clobber = true; %false ;
  conf.tinyProblem = false; %%true ;
  conf.prefix = 'baseline' ;
  conf.randSeed = time ;

  if conf.tinyProblem
    conf.prefix = 'tiny' ;
    conf.numTrain = 40;
    conf.numTest = 40;
    conf.numClasses = 10;
    conf.numSpatialX = 1 ;
    conf.numSpatialY = 1 ;
  endif

  conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']) ;
  conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
  conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']) ;
  mkdir(conf.resultPath);

  randn('state',conf.randSeed) ;
  rand('state',conf.randSeed) ;
  vl_twister('state',conf.randSeed) ;


  % --------------------------------------------------------------------
  %                                                           Setup data
  % --------------------------------------------------------------------
  %keyboard;
  oracle_file = [conf.dataDir, "/timestamps/Image.txt"]
  [oracle_fid, oracle_msg] = fopen(oracle_file, 'rt');
  [oracle_frame, oracle_time, oracle_imagepath, oracle_count, oracle_errmsg] = fscanf (oracle_fid, "%i,%i,%s", "C");
  oracle_classID = oracle_imagepath(regexp(oracle_imagepath, '/\d/')+1);
  [oracle_frame_tmp, oracle_time_tmp, oracle_imagepath_tmp, oracle_count, oracle_errmsg] = fscanf (oracle_fid, "%i,%i,%s", "C");
  while ~feof(oracle_fid)
    oracle_frame = [oracle_frame; oracle_frame_tmp];
    oracle_time = [oracle_time; oracle_time_tmp];
    oracle_imagepath = [oracle_imagepath; oracle_imagepath_tmp];
    oracle_classID_tmp_ndx = strfind(oracle_imagepath_tmp, 'data_batch_');
    %%oracle_classID_tmp_ndx = strfind(oracle_imagepath_tmp, 'test_batch');
    oracle_classID_tmp = oracle_imagepath_tmp(oracle_classID_tmp_ndx+length('data_batch_')+2);
    %%oracle_classID_tmp = oracle_imagepath_tmp(oracle_classID_tmp_ndx+length('test_batch')+1);
    oracle_classID = [oracle_classID; oracle_classID_tmp];
    %keyboard;
    %disp(["oracle_frame_tmp = ", num2str(oracle_frame_tmp)]);
    %disp(["oracle_time_tmp = ", num2str(oracle_time_tmp)]);
    %disp(["oracle_imagepath_tmp = ", oracle_imagepath_tmp]);
    %disp(["oracle_classID_tmp = ", oracle_classID_tmp]);
    [oracle_frame_tmp, oracle_time_tmp, oracle_imagepath_tmp, oracle_count, oracle_errmsg] = fscanf (oracle_fid, "%i,%i,%s", "C");
  %if mod(oracle_frame, 10) == 0
  %endif
  endwhile
  fclose(oracle_fid);

  num_oracle = length(oracle_frame);
  classes = num2str([0:conf.numClasses-1]');
  imageClass_organized = cell(1,conf.numClasses);
  parfor i_class = 1 : conf.numClasses
  imageClass_organized{i_class} = find(oracle_classID == num2str(i_class-1))';
  endparfor

  start_train_offset = num_oracle-conf.numTrain-conf.numTest+1-10; %2; % %% subtract 10 to leave some slack at the end
  start_test_offset = start_train_offset + conf.numTrain; % num_oracle-conf.numTest+1
  selTrain = [start_train_offset:start_train_offset+conf.numTrain-1] ;
  selTest = [start_test_offset:start_test_offset+conf.numTest-1];
  mistakes = intersect(selTrain,selTest);
  if ~isempty(mistakes)
    error("train and test sets contain overlapping elements")
  endif
  imageClass = cat(2, imageClass_organized{:}) ;

  model.classes = classes ;
  %%model.numSpatialX = conf.numSpatialX ;
  %%model.numSpatialY = conf.numSpatialY ;
  model.w = [] ;
  model.b = [] ;
  model.classify = @classify ;

  % --------------------------------------------------------------------
  %                                           Compute spatial histograms
  % --------------------------------------------------------------------

  if ~exist(conf.histPath) || conf.clobber
    hists = [];

    addpath("~/workspace/PetaVision/mlab/util");
    pvp_file_list =  {[conf.dataDir], ["/a2_S1.pvp"]; [conf.dataDir], ["/a6_C1.pvp"]; [conf.dataDir], ["/a10_S2.pvp"]};
    conf.numSpatialX = [16, 8, 1]; %%  %%this appears to be for partitioning the image into "halves", "quadrants", etc
    conf.numSpatialY = [16, 8, 1];

    num_pvp = size(pvp_file_list,1);
    for i_pvp = 2 : 2 %%num_pvp
      hists_tmp = {} ;
      pvp_file = [pvp_file_list{i_pvp,1}, pvp_file_list{i_pvp,2}]
      if exist(pvp_file, "file") ~= 2
	error(["pvp_file does not exist: ", pvp_file]);
      endif
      progress_step = (conf.numTest+conf.numTrain) / 10;
      %% subtract 4 from pvp_start_frame to make sure we capture the oracle training and test frames
      pvp_start_frame = max(1, start_train_offset - 4); %num_oracle-(conf.numTest+conf.numTrain);
      pvp_last_frame = pvp_start_frame + (conf.numTest+conf.numTrain) + 4;
      [pvp_struct, pvp_hdr] = ...
      readpvpfile(pvp_file, progress_step, pvp_last_frame, pvp_start_frame, 1);
      num_pvp_frames = size(pvp_struct,1);
      if num_pvp_frames < conf.numTest+conf.numTrain
	error("not enough PV data for training and testing")
      endif

      nf = pvp_hdr.nf;
      nx = pvp_hdr.nx;
      ny = pvp_hdr.ny;
      n_Sparse = nx * ny * nf;
      size_values = size(pvp_struct{1}.values);
      pvp_hist_edges = [0:1:nf]+0.5;
      i_pvp_frame = selTrain(1) - pvp_start_frame + 1;
      parfor i_oracle_frame = [selTrain, selTest]
      %% find greatest pvp_time < oracle_time OF THE NEXT FRAME
      pvp_time = pvp_struct{i_pvp_frame}.time;
      while pvp_time > oracle_time(i_oracle_frame+1)
	i_pvp_frame = i_pvp_frame - 1;
	if i_pvp_frame < 1
	  keyboard;
	endif
	pvp_time = pvp_struct{i_pvp_frame}.time;
      endwhile
      next_pvp_time = pvp_struct{i_pvp_frame+1}.time;
      while next_pvp_time < oracle_time(i_oracle_frame + 1)
	i_pvp_frame = i_pvp_frame + 1;
	pvp_time = next_pvp_time;
	if i_pvp_frame + 1 > num_pvp_frames
	  keyboard;
	endif
	next_pvp_time = pvp_struct{i_pvp_frame+1}.time;
      endwhile

      pvp_values = squeeze(pvp_struct{i_pvp_frame}.values);
      if ndims(size_values) <= 2 %% convert sparse to full
	pvp_active_ndx = pvp_values(:,1);
	if columns(pvp_values) == 2
	  pvp_active_vals = pvp_values(:,2);
	else
	  pvp_active_vals = ones(size(pvp_active_ndx));
	endif
	pvp_values = sparse(pvp_active_ndx+1,1,pvp_active_vals,n_Sparse,1,n_Sparse);
	pvp_values = full(pvp_values);
      endif
      pvp_values = reshape(pvp_values, [nf, nx, ny]);
      pvp_values = permute(pvp_values, [3, 2, 1]);

      %% grab bounding box info here if relevant
      
      %% pool over spatial scales
      pvp_hist_values = ...
      reshape(pvp_values, ...
	      [nx/conf.numSpatialX(i_pvp), conf.numSpatialX(i_pvp), ny/conf.numSpatialY(i_pvp), conf.numSpatialY(i_pvp), nf]);
      pvp_hist_values = sum(pvp_hist_values, 3);
      pvp_hist_values = sum(pvp_hist_values, 1);
      pvp_hist_values = squeeze(pvp_hist_values);
      pvp_hist_values = pvp_hist_values(:);
      hists_tmp{i_oracle_frame} = sparse(pvp_hist_values);

      endparfor


      hists_tmp = cat(2, hists_tmp{:}) ;
      size(hists_tmp)
      size(hists)
      %keyboard;
      if ~isempty(hists)
	hists = [hists; hists_tmp];
      else
	hists = hists_tmp;
      endif

    endfor %% i_pvp

    
    save(conf.histPath, 'hists') ;
  else
    load(conf.histPath) ;
  end

  %keyboard;
  %%%%%%%%%%%%%%%%%%%%%%%%
  % k-means would be done here
  %%%%%%%%%%%%%%%%%%%%%%%%


      
  % --------------------------------------------------------------------
  %                                                  Compute feature map
  % --------------------------------------------------------------------

  %hists = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;


  libSVM_flag = true; %%true;
%%libsvm_options:
%%-s svm_type : set type of SVM (default 0)
%%	0 -- C-SVC		(multi-class classification)
%%	1 -- nu-SVC		(multi-class classification)
%%	2 -- one-class SVM
%%	3 -- epsilon-SVR	(regression)
%%	4 -- nu-SVR		(regression)
%%-t kernel_type : set type of kernel function (default 2)
%%	0 -- linear: u'*v
%%	1 -- polynomial: (gamma*u'*v + coef0)^degree
%%	2 -- radial basis function: exp(-gamma*|u-v|^2)
%%	3 -- sigmoid: tanh(gamma*u'*v + coef0)
%%	4 -- precomputed kernel (kernel values in training_instance_matrix)
%%-d degree : set degree in kernel function (default 3)
%%-g gamma : set gamma in kernel function (default 1/num_features)
%%-r coef0 : set coef0 in kernel function (default 0)
%%-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%%-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
%%-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
%%-m cachesize : set cache memory size in MB (default 100)
%%-e epsilon : set tolerance of termination criterion (default 0.001)
%%-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
%%-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
%%-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
%%-v n : n-fold cross validation mode
%%-q : quiet mode (no outputs)
  if libSVM_flag
     hists = sparse(hists);
     addpath("~/Desktop/libsvm-3.18/matlab");
     perm = 1:length(selTrain); %randperm(length(selTrain)) ;
     training_label_vector = str2num(oracle_classID(selTrain))+1  ;
     training_instance_matrix =  (hists(:, selTrain(perm)-selTrain(1)+1))';     
     model = svmtrain(training_label_vector, training_instance_matrix, "-s 0 -t 0 -h 0"); %% [, 'libsvm_options']);
     testing_label_vector = str2num(oracle_classID(selTest))+1  ;
     testing_instance_matrix =  (hists(:, selTest(:)-selTrain(1)+1))';          
     [predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector, testing_instance_matrix, model); %% [, 'libsvm_options']);
     confus = zeros(length(classes));
     for oracle_classID = 1 : length(classes)
	 num_classID = sum(testing_label_vector == oracle_classID);
	 for predicted_classID = 1 : length(classes)
	   confus(oracle_classID, predicted_classID) = ...
	   sum(testing_label_vector == oracle_classID & predicted_label == predicted_classID) / (num_classID + (num_classID==0));
         endfor
     endfor
     fh_confus = figure;
     subplot(1,2,1);
     imagesc(confus);
     axis off;
     axis image
     title(["test = ", num2str(accuracy(1))]);

     [predicted_label_train, accuracy_train, decision_values_train] = svmpredict(training_label_vector, training_instance_matrix, model); %% [, 'libsvm_options']);
     confus_train = zeros(length(classes));
     for oracle_classID = 1 : length(classes)
	 num_classID = sum(testing_label_vector == oracle_classID);
	 for predicted_classID = 1 : length(classes)
	   confus_train(oracle_classID, predicted_classID) = ...
	   sum(training_label_vector == oracle_classID & predicted_label_train == predicted_classID) / (num_classID + (num_classID==0));
         endfor
     endfor
     subplot(1,2,2);
     imagesc(confus_train);
     axis off;
     axis image
     title(["train = ", num2str(accuracy_train(1))]);
     
     keyboard;
  else

  % --------------------------------------------------------------------
  %                                                            Train SVM
  % --------------------------------------------------------------------

  if ~exist(conf.modelPath) || conf.clobber
    switch conf.svm.solver
      case {'sgd', 'sdca'}
	lambda = 1 / (conf.svm.C *  length(selTrain)) ;
	w = [] ;
	for ci = 1:length(classes)
          perm = 1:length(selTrain); %randperm(length(selTrain)) ;
          fprintf("Training model for class %s\n", classes(ci)); %%classes{ci}) ;
          %%y = 2 * (imageClass(selTrain) == ci) - 1 ;
          y = 2 * (str2num(oracle_classID(selTrain))+1 == ci) - 1 ;
          [w(:,ci) b(ci) info] = vl_svmtrain(hists(:, selTrain(perm)-selTrain(1)+1), y(perm), lambda, ...
					     'Solver', conf.svm.solver, ...
					     'MaxNumIterations', 50/lambda, ...
					     'BiasMultiplier', conf.svm.biasMultiplier, ...
					     'Epsilon', 1e-3);
	endfor

      case 'liblinear'
	svm = train(str2num(oracle_classID(selTrain))+1, ... %% imageClass(selTrain)', ...
                    sparse(double(hists(:,selTrain-selTrain(1)+1))),  ...
                    sprintf(' -s 3 -B %f -c %f', ...
                            conf.svm.biasMultiplier, conf.svm.C), ...
                    'col') ;
	w = svm.w(:,1:end-1)' ;
	b =  svm.w(:,end)' ;
    endswitch

    model.b = conf.svm.biasMultiplier * b ;
    model.w = w ;
    
    save(conf.modelPath, 'model') ;
  else
    load(conf.modelPath) ;
  endif

% --------------------------------------------------------------------
%                                                Test SVM and evaluate
% --------------------------------------------------------------------

% Estimate the class of the test images
%scores = model.w' * hists + model.b' * ones(1,size(hists,2)) ;
% scores_diag stores the scores for the training and test examples for the svm trained on that classID--these should be mostly greater then zero
%keyboard;
for ci = 1:length(classes)
  sel_train_ndx = selTrain(find(str2num(oracle_classID(selTrain))+1 == ci));
  sel_test_ndx = selTest(find(str2num(oracle_classID(selTest))+1 == ci));
  scores_train_tmp = squeeze(model.w(:,ci))' * squeeze(hists(:,sel_train_ndx-selTrain(1)+1)) + model.b(ci) * ones(1,length(sel_train_ndx)) ;
  scores_test_tmp = squeeze(model.w(:,ci))' * squeeze(hists(:,sel_test_ndx-selTrain(1)+1)) + model.b(ci) * ones(1,length(sel_test_ndx)) ;
  scores_diag_train{ci,1} = scores_train_tmp;
  scores_diag_test{ci,1} = scores_test_tmp; 
endfor
%keyboard;
% scores_confus stores the scores for all training and test examples for all svms
scores_confus_train = zeros(length(classes),length(selTrain));
scores_confus_test = zeros(length(classes),length(selTest));
[sorted_classIDs_train, sorted_classID_train_ndx] = sort(oracle_classID(selTrain(:)));
[sorted_classIDs_test, sorted_classID_test_ndx] = sort(oracle_classID(selTest(:)));
for ci = 1:length(classes)
  scores_train_tmp = squeeze(model.w(:,ci))' * squeeze(hists(:,selTrain-selTrain(1)+1)) + model.b(ci) * ones(1,length(selTrain)) ;
  scores_test_tmp = squeeze(model.w(:,ci))' * squeeze(hists(:,selTest-selTrain(1)+1)) + model.b(ci) * ones(1,length(selTest)) ;
  scores_confus_train(ci,:) = scores_train_tmp(sorted_classID_train_ndx);
  scores_confus_test(ci,:) = scores_test_tmp(sorted_classID_test_ndx); 
endfor

scores = cat(2, scores_confus_train, scores_confus_test);
[drop, imageEstClass] = max(scores, [], 1) ;
[drop, imageEstClass_test] = max(scores_confus_test, [], 1) ;
[drop, imageEstClass_train] = max(scores_confus_train, [], 1) ;

%keyboard;
% Compute the confusion matrix
idx = sub2ind([length(classes), length(classes)], ...
              (str2num(sorted_classIDs_test(:))+1)', imageEstClass_test) ;
confus = zeros(length(classes)) ;
confus = vl_binsum(confus, ones(size(idx)), idx) ;

idx_train = sub2ind([length(classes), length(classes)], ...
		    (str2num(sorted_classIDs_train(:))+1)', imageEstClass_train) ;
confus_train = zeros(length(classes)) ;
confus_train = vl_binsum(confus_train, ones(size(idx_train)), idx_train) ;

% Plots
%keyboard;
results_fig = figure(1) ; clf(results_fig);
set(results_fig, 'name', 'results' );
sub_hndl = subplot(1,3,1) ;
imagesc(scores);
axis off
title('Scores');
%%set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
subplot(1,3,2) ;
axis off
axis image
imagesc(confus_train) ;
percent_accuracy_train = length(find((str2num(sorted_classIDs_train(:))+1)' == imageEstClass_train)) / length(imageEstClass_train);
title(sprintf("Train (%.2f)", ...
              100 * percent_accuracy_train )) ;
subplot(1,3,3) ;
axis off
axis image
imagesc(confus) ;
percent_accuracy = length(find((str2num(sorted_classIDs_test(:))+1)' == imageEstClass_test)) / length(imageEstClass_test);
title(sprintf("Test (%.2f)", ...
              100 * percent_accuracy )) ;
saveas(results_fig, [conf.resultPath, filesep, "C1", "_", "scores.png"])
disp(["percent_accuracy_test = ", num2str(100*percent_accuracy), "%"])
disp(["percent_accuracy_train = ", num2str(100*percent_accuracy_train), "%"])

print('-dpng', [conf.resultPath '.png']) ;
save([conf.resultPath '.mat'], 'confus', 'conf', 'scores', 'images', 'imageEstClass', 'imageClass') ;
endif  %% libSVM_flag
endfunction

% -------------------------------------------------------------------------
%function im = standarizeImageCIFAR(im)
% -------------------------------------------------------------------------

%im = im2single(im) ;
%if size(im,1) > 32, im = imresize(im, [32 NaN]) ; end

% -------------------------------------------------------------------------
function [className, score] = classify(model, im)
  % -------------------------------------------------------------------------

  hist = getImageDescriptorCIFAR(model, im) ;
  hists = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) ;
  scores = model.w' * hists + model.b' ;
  [score, best] = max(scores) ;
  className = model.classes{best} ;
endfunction
