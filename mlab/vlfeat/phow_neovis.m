function phow_neovis
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SIFT adapted from vlfeat - phow_caltech101.m
    %%    A. Vedaldi, B. Fulkerson. VLFEAT: An Open and Portable Library of Computer Vision Algorithms. 2008. http://www.vlfeat.org/
    %% PHOG adapted from code released by Anna Bosch and Andrew Zisserman - http://www.robots.ox.ac.uk/~vgg/research/caltech/phog.html
    %%
    %% Test phow+sift on NeoVision dataset
    %%  Compute performance and independence of the two algorithms, and their combination
    %%
    %% Sizes are the scales to be observed - corresponds to the bin size
    %%
    %% Discriptor is 4x4 feature detectors with 8 orientations in each feature detectors. Each of the 16 feature detectors
    %%  are binxbin pixels in the image.
    %%
    %% Dylan Paiton
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    close all ;
    clear all ;

    addpath('~/Documents/workspace/PetaVision/mlab/util/');
    addpath('./otherFiles/');

    conf.datasetTrainDir    = './data/NeoVis/training/Tiles/' ;
    conf.datasetTestDir     = './data/NeoVis/challenge/Tiles/' ;
    conf.dataDir            = 'data/' ;
    conf.ext                = 'png' ;
    conf.numTrain           = 21000 ;   % Number of training target images - This is the num train per category
    conf.numTest            = 200000 ;  % 25,134 target & 295,609 distractors
    conf.numClasses         = 2 ;
    conf.numSpatialX        = [2 4];
    conf.numSpatialY        = [2 4];
    conf.quantizer          = 'kdtree' ;
    conf.homkerGamma        = 1 ;
    conf.svm.C              = 10 ;
    conf.svm.solver         = 'pegasos' ;
    conf.svm.biasMultiplier = 1 ;
    conf.kmeans.numTrain    = 21000 ; % Size of training subset used to define cluster centers
    conf.kmeans.numWords    = 60 ;    % Bosch uses 300 word vocabulary to do 101 categories; Brumby uses 31 words for 2 categories
    conf.phowOpts           = {'Verbose',false,'Sizes',[4 8 12 16],'Fast',true,'Step',10} ;
    conf.phowOpts_color     = 'Gray' ;
    conf.phowOpts           = {conf.phowOpts{1:8},'Color',conf.phowOpts_color} ;
    conf.phogOpts           = {'Bin',40,'phogAngle',360,'L',3} ;
    conf.activityPath       = '~/Google Drive/Code/PHOODD Analysis/vlfeat-0.9.16/apps/data/PetaVision_Activity/Training/' ;
    conf.phooddFileList     = ['a3.pvp';'a4.pvp';'a5.pvp';'a6.pvp';'a7.pvp'];
    conf.phooddUseSiteInfo  = true;
    conf.trainTargetCSV     = [conf.datasetTrainDir,'Car/targets.csv'];
    conf.trainDistractorCSV = [conf.datasetTrainDir,'distractors/distractors.csv'];
    conf.testTargetCSV      = [conf.datasetTestDir,'Car/targets.csv'];
    conf.testDistractorCSV  = [conf.datasetTestDir,'distractors/distractors.csv'];

    conf.clobber            = false;
    conf.prefix             = 'neoVis'  ;
    conf.randSeed           = 123456789 ;

    conf.siftVocabPath      = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_vocab.mat']) ;
    conf.siftHistPath       = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_hists.mat']) ;
    conf.siftModelPath      = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_model.mat']) ;
    conf.siftResultPath     = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_result']) ;
    conf.siftTestPath       = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_testResult.mat']);
    conf.phogHistPath       = fullfile(conf.dataDir, [conf.prefix,'-PHOG_hists.mat']) ;
    conf.phogModelPath      = fullfile(conf.dataDir, [conf.prefix,'-PHOG_model.mat']) ;
    conf.phogResultPath     = fullfile(conf.dataDir, [conf.prefix,'-PHOG_result']) ;
    conf.phogTestPath       = fullfile(conf.dataDir, [conf.prefix,'-PHOG_testResult.mat']);
    conf.phooddVocabPath    = fullfile(conf.dataDir, [conf.prefix,'-PHOODD_vocab.mat']);
    conf.phooddHistPath     = fullfile(conf.dataDir, [conf.prefix,'-PHOODD_hists.mat']);
    conf.phooddModelPath    = fullfile(conf.dataDir, [conf.prefix,'-PHOODD_model.mat']);
    conf.phooddResultPath   = fullfile(conf.dataDir, [conf.prefix,'-PHOODD_result.mat']);
    conf.phooddTestPath     = fullfile(conf.dataDir, [conf.prefix,'-PHOODD_testResult.mat']);
    conf.papvTestPath       = fullfile(conf.dataDir, [conf.prefix,'-PAPV_testResult.mat']);

    randn('state',conf.randSeed) ;
    rand('state',conf.randSeed) ;
    vl_twister('state',conf.randSeed) ;

    % --------------------------------------------------------------------
    %                                                           Setup data
    % --------------------------------------------------------------------
    trainClasses = dir(conf.datasetTrainDir) ;
    trainClasses = trainClasses([trainClasses.isdir]) ;
    trainClasses = {trainClasses(3:conf.numClasses+2).name} ; %Remove '.' and '..' from dirs

    testClasses = dir(conf.datasetTestDir) ;
    testClasses = testClasses([testClasses.isdir]) ;
    testClasses = {testClasses(3:conf.numClasses+2).name} ; %Remove '.' and '..' from dirs

    if ne(length(trainClasses),length(testClasses))
        error('phow_neovis: ERROR: Number of training classes does not equal the number of test classes')
    else
        classes = trainClasses ;
    end

    % Same number of images per category, as long as numTrain and numTest is < the smallest category
    disp('phow_neovis: Setting up training set.')
    trainImages = {} ;
    trainImageClass = {} ;
    for ci = 1:length(classes)
      ims = dir(fullfile(conf.datasetTrainDir, classes{ci}, ['*.',conf.ext]))' ;
      ims = vl_colsubset(ims, conf.numTrain) ;
      ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
      trainImages = {trainImages{:}, ims{:}} ;
      trainImageClass{end+1} = ci * ones(1,length(ims)) ;
    end
    selTrain = find(mod(0:length(trainImages)-1, conf.numTrain) < conf.numTrain) ;
    trainImageClass = cat(2, trainImageClass{:}) ;

    disp('phow_neovis: Setting up test set.')
    testImages = {} ;
    testImageClass = {} ;
    for ci = 1:length(classes)
      ims = dir(fullfile(conf.datasetTestDir, classes{ci}, ['*.',conf.ext]))' ;
      ims = vl_colsubset(ims, conf.numTest) ;
      ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
      testImages = {testImages{:}, ims{:}} ;
      testImageClass{end+1} = ci * ones(1,length(ims)) ;
    end
    selTest = find(mod(0:length(testImages)-1, conf.numTest) < conf.numTest) ;
    testImageClass = cat(2, testImageClass{:}) ;

    cSift_model.classes     = classes ;
    cSift_model.phowOpts    = conf.phowOpts ;
    cSift_model.numSpatialX = conf.numSpatialX ;
    cSift_model.numSpatialY = conf.numSpatialY ;
    cSift_model.quantizer   = conf.quantizer ;
    cSift_model.vocab       = [] ;
    cSift_model.w           = [] ;
    cSift_model.b           = [] ;
    cSift_model.classify    = @classify ;
    phog_model              = cSift_model ;
    gSift_model             = cSift_model ;

   %% --------------------------------------------------------------------
   %%                                                           Color SIFT 
   %% --------------------------------------------------------------------
   %% --------------------------------------------------------------------
   %%                                                     Train vocabulary
   %% --------------------------------------------------------------------
   %disp('phow_neovis: Training Color SIFT Vocabulary.')
   %if ~exist(conf.siftVocabPath) || conf.clobber
   %  % Get some PHOW descriptors to train the dictionary
   %  selTrainFeats = vl_colsubset(selTrain, conf.kmeans.numTrain) ;
   %  descrs = {} ;
   %  %for ii = 1:length(selTrainFeats)
   %  parfor ii = 1:length(selTrainFeats)
   %    try
   %        im = imread(fullfile(conf.datasetTrainDir, trainImages{selTrainFeats(ii)})) ;
   %    catch ME
   %        disp('Image ',fullfile(conf.datasetTrainDir,trainImages{selTrainFeats(ii)}),' failed.')
   %    end
   %    im = standarizeImage(im) ;
   %    [drop, descrs{ii}] = vl_phow(im, cSift_model.phowOpts{:}) ;
   %  end

   %  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
   %  descrs = single(descrs) ;

   %  % Quantize the descriptors to get the visual words
   %  cSift_vocab = vl_kmeans(descrs, conf.kmeans.numWords, 'algorithm', 'elkan') ;
   %  save(conf.siftVocabPath, 'cSift_vocab') ;
   %else
   %  load(conf.siftVocabPath) ;
   %end

   %cSift_model.vocab = cSift_vocab ;

   %if strcmp(cSift_model.quantizer, 'kdtree')
   %  cSift_model.kdtree = vl_kdtreebuild(cSift_model.vocab) ;
   %end

   %% --------------------------------------------------------------------
   %%                                    Compute C-SIFT spatial histograms
   %% --------------------------------------------------------------------
   %if ~exist(conf.siftHistPath) || conf.clobber
   %  cSift_hist = cell(1,length(trainImages)) ;
   %  % for ii = 1:length(trainImages)
   %  parfor ii = 1:length(trainImages)
   %    fprintf('PHOW: Processing %s (%.2f %%)\r', trainImages{ii}, 100 * ii / length(trainImages)) ;
   %    im = imread(fullfile(conf.datasetTrainDir, trainImages{ii})) ;
   %    cSift_hist{ii} = getImageDescriptor(cSift_model, im);
   %  end

   %  cSift_hist = cat(2, cSift_hist{:}) ;
   %  save(conf.siftHistPath, 'cSift_hist') ;
   %else
   %  load(conf.siftHistPath) ;
   %end

   %% --------------------------------------------------------------------
   %%                                           Compute C-SIFT feature map
   %% --------------------------------------------------------------------

   %sift_psix = vl_homkermap(cSift_hist, 1, 'kchi2', 'gamma', conf.homkerGamma) ; %Speeds up SVM but does not affect performance

   %% --------------------------------------------------------------------
   %%                                                     Train SVM C-SIFT
   %% --------------------------------------------------------------------

   %if ~exist(conf.siftModelPath) || conf.clobber
   %  switch conf.svm.solver
   %    case 'pegasos'
   %      lambda = 1 / (conf.svm.C *  length(selTrain)) ;
   %      w = [] ;
   %      % for ci = 1:length(classes)
   %      parfor ci = 1:length(classes)
   %        perm = randperm(length(selTrain)) ;
   %        fprintf('Training SIFT model for class %s\n', classes{ci}) ;
   %        y = 2 * (trainImageClass(selTrain) == ci) - 1 ;
   %        data = vl_maketrainingset(sift_psix(:,selTrain(perm)), int8(y(perm))) ;
   %        [w(:,ci) b(ci)] = vl_svmpegasos(data, lambda, ...
   %                                        'MaxIterations', 50/lambda, ...
   %                                        'BiasMultiplier', conf.svm.biasMultiplier) ;
   %      end
   %    case 'liblinear'
   %      svm = train(trainImageClass(selTrain)', ...
   %                  sparse(double(sift_psix(:,selTrain))),  ...
   %                  sprintf(' -s 3 -B %f -c %f', ...
   %                          conf.svm.biasMultiplier, conf.svm.C), ...
   %                  'col') ;
   %      w = svm.w' ;
   %  end

   %  cSift_model.b = conf.svm.biasMultiplier * b ;
   %  cSift_model.w = w ;

   %  save(conf.siftModelPath, 'cSift_model') ;
   %else
   %  load(conf.siftModelPath) ;
   %end

   %% --------------------------------------------------------------------
   %%                                   Test C-SIFT SVM on Independent Set
   %% --------------------------------------------------------------------

   %if ~exist(conf.siftTestPath) || conf.clobber
   %    cSiftStruct.tp         = zeros([1 length(selTest)]) ;
   %    cSiftStruct.fp         = zeros([1 length(selTest)]) ;
   %    cSiftStruct.tn         = zeros([1 length(selTest)]) ;
   %    cSiftStruct.fn         = zeros([1 length(selTest)]) ;
   %    cSiftStruct.cars       = zeros([1 length(selTest)]) ;
   %    cSiftStruct.dist       = zeros([1 length(selTest)]) ;
   %    cSiftStruct.labels     = zeros([1 length(selTest)]) ;
   %    cSiftStruct.roc_scores = zeros([1 length(selTest)]) ;
   %    cSiftStruct.roc_labels = zeros([1 length(selTest)]) ;
   %    for ii = 1:length(selTest)
   %        fprintf('PHOW: Testing %s (%.2f %%)\n', testImages{ii}, 100 * ii / length(selTest)) ;
   %        try
   %            im = imread(fullfile(conf.datasetTestDir, testImages{selTest(ii)})) ;
   %            splitDir = regexp(testImages{selTest(ii)},'/','split') ;
   %            category = splitDir(1) ;
   %        catch ME
   %            disp('Image ',fullfile(conf.datasetTestDir,trainImages{selTest(ii)}),' failed.')
   %        end

   %        [className, score, indi_scores] = classify(cSift_model, im, conf.homkerGamma) ;
   %        cSiftStruct.roc_scores(ii) = indi_scores(1)-indi_scores(2);

   %        if strcmp(category{:},'Car') 
   %            cSiftStruct.roc_labels(ii)  = 1 ;
   %            if strcmp(className,'Car')
   %                cSiftStruct.tp(ii)      = 1 ;
   %                cSiftStruct.cars(ii)    = 1 ;
   %            elseif strcmp(className,'distractors')
   %                cSiftStruct.fn(ii)      = 1 ;
   %                cSiftStruct.dist(ii)    = 1 ;
   %            else
   %                error(['phow_neovis: ',className,' is not one of the allowed classes.'])
   %            end
   %        elseif strcmp(category{:},'distractors')
   %            cSiftStruct.roc_labels(ii)  = -1 ;
   %            if strcmp(className,'distractors')
   %                cSiftStruct.tn(ii)      = 1 ;
   %                cSiftStruct.dist(ii)    = 1 ;
   %            elseif strcmp(className,'Car')
   %                cSiftStruct.fp(ii)      = 1 ;
   %                cSiftStruct.cars(ii)    = 1 ;
   %            else
   %                error(['phow_neovis: ',className,' is not one of the allowed classes.'])
   %            end
   %        else
   %            error(['phow_neovis: ',category{:},' is not one of the allowed categories.'])
   %        end
   %    end
   %    save(conf.siftTestPath,'cSiftStruct');
   %else
   %    load(conf.siftTestPath);
   %end
   %cSiftStruct.SIFT_TP      = length(find(cSiftStruct.tp)) ;
   %cSiftStruct.SIFT_TN      = length(find(cSiftStruct.tn)) ;
   %cSiftStruct.SIFT_numCar  = length(find(cSiftStruct.tp+cSiftStruct.fn)) ;
   %cSiftStruct.SIFT_numDist = length(find(cSiftStruct.tn+cSiftStruct.fp)) ;

   %% --------------------------------------------------------------------
   %%                                               Plot C-SIFT ROC Curves
   %% --------------------------------------------------------------------
   %figure ; clf ;
   %[cSIFT_TPR,cSIFT_TNR,cSIFT_INFO] = vl_roc(cSiftStruct.roc_labels,cSiftStruct.roc_scores,'plot','fptp') ;
   %title('SIFT ROC')
   %cSIFT_FPR   = 1 - cSIFT_TNR ;
   %cSIFT_FP    = length(find(cSiftStruct.fp)) ;
   %cSIFT_TP_RT = cSiftStruct.SIFT_TP/cSiftStruct.SIFT_numCar ;
   %cSIFT_FP_RT = cSIFT_FP/cSiftStruct.SIFT_numDist ;
   %cSIFT_AUC   = trapz([0 cSIFT_FP_RT 1],[0 cSIFT_TP_RT 1]) ;

   %% --------------------------------------------------------------------
   %%                                                            Gray SIFT 
   %% --------------------------------------------------------------------
   %conf.phowOpts_color     = 'GRAY' ;
   %conf.phowOpts           = {conf.phowOpts{1:8},'Color',conf.phowOpts_color} ;
   %conf.siftVocabPath  = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_vocab.mat']) ;
   %conf.siftHistPath   = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_hists.mat']) ;
   %conf.siftModelPath  = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_model.mat']) ;
   %conf.siftResultPath = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_result']) ;
   %conf.siftTestPath   = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_testResult.mat']);
   %conf.siftTestPath2  = fullfile(conf.dataDir, [conf.prefix,'-',conf.phowOpts_color,'_SIFT_testResult2.mat']);

   %% --------------------------------------------------------------------
   %%                                              Train G-SIFT vocabulary
   %% --------------------------------------------------------------------
   %disp('phow_neovis: Training Gray SIFT Vocabulary.')
   %if ~exist(conf.siftVocabPath) || conf.clobber
   %  % Get some PHOW descriptors to train the dictionary
   %  selTrainFeats = vl_colsubset(selTrain, conf.kmeans.numTrain) ;
   %  descrs = {} ;
   %  %for ii = 1:length(selTrainFeats)
   %  parfor ii = 1:length(selTrainFeats)
   %    try
   %        im = imread(fullfile(conf.datasetTrainDir, trainImages{selTrainFeats(ii)})) ;
   %    catch ME
   %        disp('Image ',fullfile(conf.datasetTrainDir,trainImages{selTrainFeats(ii)}),' failed.')
   %    end
   %    im = standarizeImage(im) ;
   %    [drop, descrs{ii}] = vl_phow(im, gSift_model.phowOpts{:}) ;
   %  end

   %  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
   %  descrs = single(descrs) ;

   %  % Quantize the descriptors to get the visual words
   %  gSift_vocab = vl_kmeans(descrs, conf.kmeans.numWords, 'algorithm', 'elkan') ;
   %  save(conf.siftVocabPath, 'gSift_vocab') ;
   %else
   %  load(conf.siftVocabPath) ;
   %end

   %gSift_model.vocab = gSift_vocab ;

   %if strcmp(gSift_model.quantizer, 'kdtree')
   %  gSift_model.kdtree = vl_kdtreebuild(gSift_model.vocab) ;
   %end

   %% --------------------------------------------------------------------
   %%                                    Compute G-SIFT spatial histograms
   %% --------------------------------------------------------------------
   %if ~exist(conf.siftHistPath) || conf.clobber
   %  gSift_hist = cell(1,length(trainImages));
   %  % for ii = 1:length(trainImages)
   %  parfor ii = 1:length(trainImages)
   %    fprintf('PHOW: Processing %s (%.2f %%)\r', trainImages{ii}, 100 * ii / length(trainImages)) ;
   %    im = imread(fullfile(conf.datasetTrainDir, trainImages{ii})) ;
   %    gSift_hist{ii} = getImageDescriptor(gSift_model, im);
   %  end

   %  gSift_hist = cat(2, gSift_hist{:}) ;
   %  save(conf.siftHistPath, 'gSift_hist') ;
   %else
   %  load(conf.siftHistPath) ;
   %end

   %% --------------------------------------------------------------------
   %%                                           Compute G-SIFT feature map
   %% --------------------------------------------------------------------

   %sift_psix = vl_homkermap(gSift_hist, 1, 'kchi2', 'gamma', conf.homkerGamma) ; %Speeds up SVM but does not affect performance

   %% --------------------------------------------------------------------
   %%                                                     Train SVM G-SIFT
   %% --------------------------------------------------------------------

   %if ~exist(conf.siftModelPath) || conf.clobber
   %  switch conf.svm.solver
   %    case 'pegasos'
   %      lambda = 1 / (conf.svm.C *  length(selTrain)) ;
   %      w = [] ;
   %      % for ci = 1:length(classes)
   %      parfor ci = 1:length(classes)
   %        perm = randperm(length(selTrain)) ;
   %        fprintf('Training SIFT model for class %s\n', classes{ci}) ;
   %        y = 2 * (trainImageClass(selTrain) == ci) - 1 ;
   %        data = vl_maketrainingset(sift_psix(:,selTrain(perm)), int8(y(perm))) ;
   %        [w(:,ci) b(ci)] = vl_svmpegasos(data, lambda, ...
   %                                        'MaxIterations', 50/lambda, ...
   %                                        'BiasMultiplier', conf.svm.biasMultiplier) ;
   %      end
   %    case 'liblinear'
   %      svm = train(trainImageClass(selTrain)', ...
   %                  sparse(double(sift_psix(:,selTrain))),  ...
   %                  sprintf(' -s 3 -B %f -c %f', ...
   %                          conf.svm.biasMultiplier, conf.svm.C), ...
   %                  'col') ;
   %      w = svm.w' ;
   %  end

   %  gSift_model.b = conf.svm.biasMultiplier * b ;
   %  gSift_model.w = w ;

   %  save(conf.siftModelPath, 'gSift_model') ;
   %else
   %  load(conf.siftModelPath) ;
   %end

   %% --------------------------------------------------------------------
   %%                                   Test G-SIFT SVM on Independent Set
   %% --------------------------------------------------------------------

   %if ~exist(conf.siftTestPath) || conf.clobber
   %    gSiftStruct.tp         = zeros([1 length(selTest)]) ;
   %    gSiftStruct.fp         = zeros([1 length(selTest)]) ;
   %    gSiftStruct.tn         = zeros([1 length(selTest)]) ;
   %    gSiftStruct.fn         = zeros([1 length(selTest)]) ;
   %    gSiftStruct.cars       = zeros([1 length(selTest)]) ;
   %    gSiftStruct.dist       = zeros([1 length(selTest)]) ;
   %    gSiftStruct.labels     = zeros([1 length(selTest)]) ;
   %    gSiftStruct.roc_scores = zeros([1 length(selTest)]) ;
   %    gSiftStruct.roc_labels = zeros([1 length(selTest)]) ;
   %    for ii = 1:length(selTest)
   %        fprintf('PHOW: Testing %s (%.2f %%)\n', testImages{ii}, 100 * ii / length(selTest)) ;
   %        try
   %            im = imread(fullfile(conf.datasetTestDir, testImages{selTest(ii)})) ;
   %            splitDir = regexp(testImages{selTest(ii)},'/','split') ;
   %            category = splitDir(1) ;
   %        catch ME
   %            disp('Image ',fullfile(conf.datasetTestDir,trainImages{selTest(ii)}),' failed.')
   %        end

   %        [className, score, indi_scores] = classify(gSift_model, im, conf.homkerGamma) ;
   %        gSiftStruct.roc_scores(ii) = indi_scores(1)-indi_scores(2);

   %        if strcmp(category{:},'Car') 
   %            gSiftStruct.roc_labels(ii)  = 1 ;
   %            if strcmp(className,'Car')
   %                gSiftStruct.tp(ii)      = 1 ;
   %                gSiftStruct.cars(ii)    = 1 ;
   %            elseif strcmp(className,'distractors')
   %                gSiftStruct.fn(ii)      = 1 ;
   %                gSiftStruct.dist(ii)    = 1 ;
   %            else
   %                error(['phow_neovis: ',className,' is not one of the allowed classes.'])
   %            end
   %        elseif strcmp(category{:},'distractors')
   %            gSiftStruct.roc_labels(ii)  = -1 ;
   %            if strcmp(className,'distractors')
   %                gSiftStruct.tn(ii)      = 1 ;
   %                gSiftStruct.dist(ii)    = 1 ;
   %            elseif strcmp(className,'Car')
   %                gSiftStruct.fp(ii)      = 1 ;
   %                gSiftStruct.cars(ii)    = 1 ;
   %            else
   %                error(['phow_neovis: ',className,' is not one of the allowed classes.'])
   %            end
   %        else
   %            error(['phow_neovis: ',category{:},' is not one of the allowed categories.'])
   %        end
   %    end
   %    save(conf.siftTestPath,'gSiftStruct');
   %else
   %    load(conf.siftTestPath);
   %end
   %gSiftStruct.SIFT_TP      = length(find(gSiftStruct.tp)) ;
   %gSiftStruct.SIFT_TN      = length(find(gSiftStruct.tn)) ;
   %gSiftStruct.SIFT_numCar  = length(find(gSiftStruct.tp+gSiftStruct.fn)) ;
   %gSiftStruct.SIFT_numDist = length(find(gSiftStruct.tn+gSiftStruct.fp)) ;

   %% --------------------------------------------------------------------
   %%                                               Plot G-SIFT ROC Curves
   %% --------------------------------------------------------------------
   %figure ; clf ;
   %[gSIFT_TPR,gSIFT_TNR,gSIFT_INFO] = vl_roc(gSiftStruct.roc_labels,gSiftStruct.roc_scores,'plot','fptp') ;
   %title('GRAY SIFT ROC')
   %gSIFT_FPR   = 1 - gSIFT_TNR ;
   %gSIFT_FP    = length(find(gSiftStruct.fp)) ;
   %gSIFT_TP_RT = gSiftStruct.SIFT_TP/gSiftStruct.SIFT_numCar ;
   %gSIFT_FP_RT = gSIFT_FP/gSiftStruct.SIFT_numDist ;
   %gSIFT_AUC   = trapz([0 gSIFT_FP_RT 1],[0 gSIFT_TP_RT 1]) ;

   %% --------------------------------------------------------------------
   %%                                                                 PHOG
   %% --------------------------------------------------------------------
   %if ~exist(conf.phogHistPath) || conf.clobber
   %    phog_hist   = {} ;
   %    bin       = conf.phogOpts{2} ;
   %    phogAngle = conf.phogOpts{4} ;
   %    L         = conf.phogOpts{6} ;
   %    %for ii = 1:length(selTrain)
   %    parfor ii = 1:length(selTrain)
   %        fprintf('PHOG: Processing %s (%.2f %%)\n', trainImages{ii}, 100 * ii / length(trainImages)) ;
   %        I = fullfile(conf.datasetTrainDir, trainImages{selTrain(ii)}) ;
   %        try
   %            img = imread(I) ;
   %        catch ME
   %            disp('Image ',I,' failed.')
   %        end

   %        [height width ibins] = size(img) ; 
   %        roi = [1;height;1;width] ;% roi - Region Of Interest (ytop,ybottom,xleft,xright)
   %        phog_hist{ii} = anna_phog(I,img,bin,phogAngle,L,roi) ;
   %    end

   %    phog_hist = cat(2, phog_hist{:}) ;
   %    save(conf.phogHistPath, 'phog_hist') ;
   %else
   %    load(conf.phogHistPath) ;
   %end

   %% --------------------------------------------------------------------
   %%                                             Compute PHOG feature map
   %% --------------------------------------------------------------------
   %phog_psix = vl_homkermap(phog_hist, 1, 'kchi2', 'gamma', conf.homkerGamma) ;

   %% --------------------------------------------------------------------
   %%                                                       Train SVM PHOG
   %% --------------------------------------------------------------------
   %if ~exist(conf.phogModelPath) || conf.clobber
   %  switch conf.svm.solver
   %    case 'pegasos'
   %      lambda = 1 / (conf.svm.C *  length(selTrain)) ;
   %      w = [] ;
   %      %for ci = 1:length(classes)
   %      parfor ci = 1:length(classes)
   %        fprintf('Training PHOG model for class %s\n', classes{ci}) ;
   %        perm = randperm(length(selTrain)) ;
   %        y = 2 * (trainImageClass(selTrain) == ci) - 1 ;
   %        data = vl_maketrainingset(phog_psix(:,selTrain(perm)), int8(y(perm))) ;
   %        [w(:,ci) b(ci)] = vl_svmpegasos(data, lambda, ...
   %                                        'MaxIterations', 50/lambda, ...
   %                                        'BiasMultiplier', conf.svm.biasMultiplier) ;
   %      end
   %    case 'liblinear'
   %      svm = train(trainImageClass(selTrain)', ...
   %                  sparse(double(phog_psix(:,selTrain))),  ...
   %                  sprintf(' -s 3 -B %f -c %f', ...
   %                          conf.svm.biasMultiplier, conf.svm.C), ...
   %                  'col') ;
   %      w = svm.w' ;
   %  end

   %  phog_model.b = conf.svm.biasMultiplier * b ;
   %  phog_model.w = w ;

   %  save(conf.phogModelPath, 'phog_model') ;
   %else
   %  load(conf.phogModelPath) ;
   %end

   %% --------------------------------------------------------------------
   %%                                     Test PHOG SVM on Independent Set
   %% --------------------------------------------------------------------

   %if ~exist(conf.phogTestPath) || conf.clobber
   %    phogStruct.tp         = zeros([1 length(selTest)]) ;
   %    phogStruct.fp         = zeros([1 length(selTest)]) ;
   %    phogStruct.tn         = zeros([1 length(selTest)]) ;
   %    phogStruct.fn         = zeros([1 length(selTest)]) ;
   %    phogStruct.cars       = zeros([1 length(selTest)]) ;
   %    phogStruct.dist       = zeros([1 length(selTest)]) ;
   %    phogStruct.roc_labels = zeros([1 length(selTest)]) ;
   %    phogStruct.roc_scores = zeros([1 length(selTest)]) ;
   %    bin          = conf.phogOpts{2} ;
   %    phogAngle    = conf.phogOpts{4} ;
   %    L            = conf.phogOpts{6} ;
   %    for ii = 1:length(selTest)
   %        fprintf('PHOG: Testing %s (%.2f %%)\n', testImages{ii}, 100 * ii / length(selTest)) ;
   %        I  = fullfile(conf.datasetTestDir, testImages{selTest(ii)}) ;
   %        try
   %            im = imread(I) ;
   %            splitDir = regexp(testImages{selTest(ii)},'/','split') ;
   %            category = splitDir(1) ; % Should return a single string
   %        catch ME
   %            disp('Image ',fullfile(conf.datasetTestDir,testImages{selTest(ii)}),' failed.')
   %        end

   %        [height width ibins] = size(im) ; 
   %        roi = [1;height;1;width] ;% roi - Region Of Interest (ytop,ybottom,xleft,xright)

   %        phog_hist = anna_phog(I,im,bin,phogAngle,L,roi) ;
   %        phog_hist = phog_hist/sum(phog_hist) ;

   %        phog_psix = vl_homkermap(phog_hist, 1, 'kchi2', 'gamma', conf.homkerGamma) ;

   %        indi_scores   = phog_model.w' * phog_psix + phog_model.b' ;
   %        [score, best] = max(indi_scores) ; % Distance from zero represents confidence. pos is cat selection
   %        className = phog_model.classes{best} ;

   %        phogStruct.roc_scores(ii) = indi_scores(1)-indi_scores(2);

   %        if strcmp(category{:},'Car') 
   %            phogStruct.roc_labels(ii)  = 1 ;
   %            if strcmp(className,'Car')
   %                phogStruct.tp(ii) = 1 ;
   %                phogStruct.cars(ii)    = 1 ;
   %            elseif strcmp(className,'distractors')
   %                phogStruct.fn(ii) = 1 ;
   %                phogStruct.dist(ii)    = 1 ;
   %            else
   %                error(['phog_neovis: ',className,' is not one of the allowed class names.'])
   %            end
   %        elseif strcmp(category{:},'distractors')
   %            phogStruct.roc_labels(ii)  = -1 ;
   %            if strcmp(className,'distractors')
   %                phogStruct.tn(ii) = 1 ;
   %                phogStruct.dist(ii)    = 1 ;
   %            elseif strcmp(className,'Car')
   %                phogStruct.fp(ii) = 1 ;
   %                phogStruct.cars(ii)    = 1 ;
   %            else
   %                error(['phog_neovis: ',className,' is not one of the allowed class names.'])
   %            end
   %        else
   %            error(['phog_neovis: ',category{:},' is not one of the allowed categories.'])
   %        end
   %    end
   %  save(conf.phogTestPath, 'phogStruct') ;
   %else
   %  load(conf.phogTestPath) ;
   %end

   %PHOG_TP      = length(find(phogStruct.tp)) ;
   %PHOG_TN      = length(find(phogStruct.tn)) ;
   %PHOG_numCar  = length(find(phogStruct.tp+phogStruct.fn)) ;
   %PHOG_numDist = length(find(phogStruct.tn+phogStruct.fp)) ;

   %% --------------------------------------------------------------------
   %%                                                 Plot PHOG ROC Curves
   %% --------------------------------------------------------------------
   %figure ; clf ;
   %[PHOG_TPR,PHOG_TNR,PHOG_INFO] = vl_roc(phogStruct.roc_labels,phogStruct.roc_scores,'plot','fptp') ;
   %title('PHOG ROC')
   %PHOG_FPR   = 1 - PHOG_TNR ;
   %PHOG_FP    = length(find(phogStruct.fp)) ;
   %phog_tp_rt = PHOG_TP/PHOG_numCar ;
   %phog_fp_rt = PHOG_FP/PHOG_numDist ;
   %phog_AUC   = trapz([0 phog_fp_rt 1],[0 phog_tp_rt 1]) ;


    % --------------------------------------------------------------------
    %                                                               PHOODD
    % --------------------------------------------------------------------
    if ~exist(conf.phooddHistPath,'file') || conf.clobber
        phoodd_hist      = cell(length(conf.phooddFileList),length(selTrain)) ;
        numEdges         = cell(length(conf.phooddFileList)) ;
        times            = cell(length(conf.phooddFileList)) ;
        indi_phoodd_hist = cell(length(conf.phooddFileList));
        for ii = 1:length(selTrain)
        %parfor ii = 1:length(selTrain) %TODO: FIX THIS
           for jj = 1:length(conf.phooddFileList)
               if trainImageClass(ii) == 1
                 [numEdges{jj}, times{jj}] = generatePvpHists([conf.activityPath,conf.phooddFileList(jj,:)],conf.trainTargetCSV,'Car',conf.phooddUseSiteInfo);
              else
                 [numEdges{jj}, times{jj}] = generatePvpHists([conf.activityPath,conf.phooddFileList(jj,:)],conf.trainDistractorCSV,'distractor',conf.phooddUseSiteInfo);
              end
              numBins            = length(numEdges(1,:));
              phoodd_hist{jj,ii} = hist(numEdges{jj}(ii,:),numBins);
           end
        end

        for jj = 1:length(conf.phooddFileList)
           indi_phoodd_hist{jj} = cat(2, phoodd_hist{jj,:}) ;
        end
        save(conf.phooddHistPath, 'indi_phoodd_hist') ;
    else
        load(conf.phooddHistPath) ;
    end

    % --------------------------------------------------------------------
    %                                           Compute PHOODD feature map
    % --------------------------------------------------------------------
    for jj = 1:length(conf.phooddFileList)
       phoodd_psix{jj} = vl_homkermap(indi_phoodd_hist{jj}, 1, 'kchi2', 'gamma', conf.homkerGamma) ;
    end

    % --------------------------------------------------------------------
    %                                                     Train SVM PHOODD
    % --------------------------------------------------------------------
    if ~exist(conf.phooddModelPath) || conf.clobber
       for jj = 1:length(conf.phooddFileList)
          switch conf.svm.solver
             case 'pegasos'
                lambda = 1 / (conf.svm.C *  length(selTrain)) ;
                w = [] ;
                %for ci = 1:length(classes)
                parfor ci = 1:length(classes)
                fprintf('Training PHOODD model for class %s\n', classes{ci}) ;
                perm = randperm(length(selTrain)) ;
                y = 2 * (trainImageClass(selTrain) == ci) - 1 ;
                data = vl_maketrainingset(phoodd_psix{jj}(:,selTrain(perm)), int8(y(perm))) ;
                [w(:,ci) b(ci)] = vl_svmpegasos(data, lambda, ...
                   'MaxIterations', 50/lambda, ...
                   'BiasMultiplier', conf.svm.biasMultiplier) ;
             end
             case 'liblinear'
                svm = train(trainImageClass(selTrain)', ...
                   sparse(double(phoodd_psix{jj}(:,selTrain))),  ...
                   sprintf(' -s 3 -B %f -c %f', ...
                   conf.svm.biasMultiplier, conf.svm.C), ...
                   'col') ;
                w = svm.w' ;
             end

          phoodd_model{jj}.b = conf.svm.biasMultiplier * b ;
          phoodd_model{jj}.w = w ;
       end 

      save(conf.phooddModelPath, 'phoodd_model') ;
    else
      load(conf.phooddModelPath) ;
    end

   %% --------------------------------------------------------------------
   %%                                     Test PHOODD SVM on Independent Set
   %% --------------------------------------------------------------------

   %if ~exist(conf.phooddTestPath) || conf.clobber
   %    phooddStruct.tp         = zeros([1 length(selTest)]) ;
   %    phooddStruct.fp         = zeros([1 length(selTest)]) ;
   %    phooddStruct.tn         = zeros([1 length(selTest)]) ;
   %    phooddStruct.fn         = zeros([1 length(selTest)]) ;
   %    phooddStruct.cars       = zeros([1 length(selTest)]) ;
   %    phooddStruct.dist       = zeros([1 length(selTest)]) ;
   %    phooddStruct.roc_labels = zeros([1 length(selTest)]) ;
   %    phooddStruct.roc_scores = zeros([1 length(selTest)]) ;
   %    for ii = 1:length(selTest)
   %        fprintf('PHOODD: Testing %s (%.2f %%)\n', testImages{ii}, 100 * ii / length(selTest)) ;
   %        I  = fullfile(conf.datasetTestDir, testImages{selTest(ii)}) ;
   %        try
   %            im = imread(I) ;
   %            splitDir = regexp(testImages{selTest(ii)},'/','split') ;
   %            category = splitDir(1) ; % Should return a single string
   %        catch ME
   %            disp('Image ',fullfile(conf.datasetTestDir,testImages{selTest(ii)}),' failed.')
   %        end

   %        [height width ibins] = size(im) ; 
   %        roi = [1;height;1;width] ;% roi - Region Of Interest (ytop,ybottom,xleft,xright)

   %        phoodd_hist = anna_phoodd(I,im,bin,phooddAngle,L,roi) ;
   %        phoodd_hist = phoodd_hist/sum(phoodd_hist) ;

   %        phoodd_psix = vl_homkermap(phoodd_hist, 1, 'kchi2', 'gamma', conf.homkerGamma) ;

   %        indi_scores   = phoodd_model.w' * phoodd_psix + phoodd_model.b' ;
   %        [score, best] = max(indi_scores) ; % Distance from zero represents confidence. pos is cat selection
   %        className = phoodd_model.classes{best} ;

   %        phooddStruct.roc_scores(ii) = indi_scores(1)-indi_scores(2);

   %        if strcmp(category{:},'Car') 
   %            phooddStruct.roc_labels(ii)  = 1 ;
   %            if strcmp(className,'Car')
   %                phooddStruct.tp(ii) = 1 ;
   %                phooddStruct.cars(ii)    = 1 ;
   %            elseif strcmp(className,'distractors')
   %                phooddStruct.fn(ii) = 1 ;
   %                phooddStruct.dist(ii)    = 1 ;
   %            else
   %                error(['phoodd_neovis: ',className,' is not one of the allowed class names.'])
   %            end
   %        elseif strcmp(category{:},'distractors')
   %            phooddStruct.roc_labels(ii)  = -1 ;
   %            if strcmp(className,'distractors')
   %                phooddStruct.tn(ii) = 1 ;
   %                phooddStruct.dist(ii)    = 1 ;
   %            elseif strcmp(className,'Car')
   %                phooddStruct.fp(ii) = 1 ;
   %                phooddStruct.cars(ii)    = 1 ;
   %            else
   %                error(['phoodd_neovis: ',className,' is not one of the allowed class names.'])
   %            end
   %        else
   %            error(['phoodd_neovis: ',category{:},' is not one of the allowed categories.'])
   %        end
   %    end
   %  save(conf.phooddTestPath, 'phooddStruct') ;
   %else
   %  load(conf.phooddTestPath) ;
   %end

   %PHOODD_TP      = length(find(phooddStruct.tp)) ;
   %PHOODD_TN      = length(find(phooddStruct.tn)) ;
   %PHOODD_numCar  = length(find(phooddStruct.tp+phooddStruct.fn)) ;
   %PHOODD_numDist = length(find(phooddStruct.tn+phooddStruct.fp)) ;

   %% --------------------------------------------------------------------
   %%                                                 Plot PHOODD ROC Curves
   %% --------------------------------------------------------------------
   %figure ; clf ;
   %[PHOODD_TPR,PHOODD_TNR,PHOODD_INFO] = vl_roc(phooddStruct.roc_labels,phooddStruct.roc_scores,'plot','fptp') ;
   %title('PHOODD ROC')
   %PHOODD_FPR = 1 - PHOODD_TNR ;
   %PHOODD_FP  = length(find(phooddStruct.fp)) ;
   %phoodd_tp_rt = PHOODD_TP/PHOODD_numCar ;
   %phoodd_fp_rt = PHOODD_FP/PHOODD_numDist ;
   %phoodd_AUC   = trapz([0 phoodd_fp_rt 1],[0 phoodd_tp_rt 1]) ;
    
    
   %% --------------------------------------------------------------------
   %%                              Test PANN/PetaVision on Independent Set
   %% --------------------------------------------------------------------
   %if ~exist(conf.papvTestPath) || conf.clobber
   %    [papvStruct.pv_stats papvStruct.pa_stats papvStruct.roc_labels] = papv_vl_roc(testImages,selTest);
   %    save(conf.papvTestPath, 'papvStruct') ;
   %else
   %    load(conf.papvTestPath) ;
   %end

   %% --------------------------------------------------------------------
   %%                                                 Plot PAPV ROC Curves
   %% --------------------------------------------------------------------
   %pos_idx      = find(papvStruct.roc_labels>0) ;
   %neg_idx      = find(papvStruct.roc_labels<0) ;
   %PAPV_numCar  = length(find(papvStruct.roc_labels==1));
   %PAPV_numDist = length(find(papvStruct.roc_labels==-1));

   %pa_roc_scores = papvStruct.pa_stats.tp+papvStruct.pa_stats.tn ; %Latter half of tp is 0s and former half of tn is 0s
   %[PANN_TPR,PANN_TNR,PANN_INFO] = vl_roc(papvStruct.roc_labels,pa_roc_scores) ;
   %PANN_FPR = 1 - PANN_TNR;
   %PANN_FP  = length(find(papvStruct.pa_stats.fp)) ;
   %PANN_TP = length(find(papvStruct.roc_labels(pos_idx).*pa_roc_scores(pos_idx) > 0)) ;
   %PANN_TN = length(find(papvStruct.roc_labels(neg_idx).*pa_roc_scores(neg_idx) > 0)) ;
   %pann_tp_rt = PANN_TP/PAPV_numCar ;
   %pann_fp_rt = PANN_FP/PAPV_numDist ;
   %pann_AUC   = trapz([0 pann_fp_rt 1],[0 pann_tp_rt 1]) ;

   %pv_roc_scores = papvStruct.pv_stats.tp+papvStruct.pv_stats.tn ; %Latter half of tp is 0s and former half of tn is 0s
   %[PETA_TPR,PETA_TNR,PETA_INFO] = vl_roc(papvStruct.roc_labels,pv_roc_scores) ;
   %PETA_FPR = 1 - PETA_TNR;
   %PETA_FP  = length(find(papvStruct.pv_stats.fp)) ;
   %PETA_TP  = length(find(papvStruct.roc_labels(pos_idx).*pv_roc_scores(pos_idx) > 0)) ;
   %PETA_TN  = length(find(papvStruct.roc_labels(neg_idx).*pv_roc_scores(neg_idx) > 0)) ;
   %peta_tp_rt = PETA_TP/PAPV_numCar ;
   %peta_fp_rt = PETA_FP/PAPV_numDist ;
   %peta_AUC   = trapz([0 peta_fp_rt 1],[0 peta_tp_rt 1]) ;


   %% --------------------------------------------------------------------
   %%                                                        Print Results
   %% --------------------------------------------------------------------
   %disp(['--------------------------------'])
   %disp(['Color SIFT tp      = ',num2str(cSiftStruct.SIFT_TP)])
   %disp(['Color SIFT tn      = ',num2str(cSiftStruct.SIFT_TN)])
   %disp(['Color SIFT fp      = ',num2str(cSIFT_FP)])
   %disp(['Color SIFT fn      = ',num2str(length(find(cSiftStruct.fn)))])
   %disp(['Color SIFT numCars = ',num2str(cSiftStruct.SIFT_numCar)])
   %disp(['Color SIFT numDist = ',num2str(cSiftStruct.SIFT_numDist)])
   %disp(['Color SIFT AUC     = ',num2str(cSIFT_INFO.auc)])
   %disp(['Color SIFT eer     = ',num2str(cSIFT_INFO.eer)])
   %disp(['Color SIFT tp perc = ',num2str(100*(cSiftStruct.SIFT_TP/cSiftStruct.SIFT_numCar))])
   %disp(['Color SIFT tn perc = ',num2str(100*(cSiftStruct.SIFT_TN/cSiftStruct.SIFT_numDist))])
   %disp(['--------------------------------'])
   %disp(['Gray SIFT tp      = ',num2str(gSiftStruct.SIFT_TP)])
   %disp(['Gray SIFT tn      = ',num2str(gSiftStruct.SIFT_TN)])
   %disp(['Gray SIFT fp      = ',num2str(cSIFT_FP)])
   %disp(['Gray SIFT fn      = ',num2str(length(find(gSiftStruct.fn)))])
   %disp(['Gray SIFT numCars = ',num2str(gSiftStruct.SIFT_numCar)])
   %disp(['Gray SIFT numDist = ',num2str(gSiftStruct.SIFT_numDist)])
   %disp(['Gray SIFT AUC     = ',num2str(cSIFT_INFO.auc)])
   %disp(['Gray SIFT eer     = ',num2str(cSIFT_INFO.eer)])
   %disp(['Gray SIFT tp perc = ',num2str(100*(gSiftStruct.SIFT_TP/gSiftStruct.SIFT_numCar))])
   %disp(['Gray SIFT tn perc = ',num2str(100*(gSiftStruct.SIFT_TN/gSiftStruct.SIFT_numDist))])
   %disp(['--------------------------------'])
   %disp(['PHOG tp            = ',num2str(PHOG_TP)])
   %disp(['PHOG tn            = ',num2str(PHOG_TN)])
   %disp(['PHOG fp            = ',num2str(PHOG_FP)])
   %disp(['PHOG fn            = ',num2str(length(find(phogStruct.fn)))])
   %disp(['PHOG numCars       = ',num2str(PHOG_numCar)])
   %disp(['PHOG numDist       = ',num2str(PHOG_numDist)])
   %disp(['PHOG AUC           = ',num2str(PHOG_INFO.auc)])
   %disp(['PHOG eer           = ',num2str(PHOG_INFO.eer)])
   %disp(['PHOG tp perc       = ',num2str(100*(PHOG_TP/PHOG_numCar))])
   %disp(['PHOG tn perc       = ',num2str(100*(PHOG_TN/PHOG_numDist))])
   %disp(['--------------------------------'])
   %disp(['PANN tp            = ',num2str(PANN_TP)])
   %disp(['PANN tn            = ',num2str(PANN_TN)])
   %disp(['PANN fp            = ',num2str(length(find(papvStruct.pa_stats.fp)))])
   %disp(['PANN fn            = ',num2str(length(find(papvStruct.pa_stats.fn)))])
   %disp(['PANN numCars       = ',num2str(PAPV_numCar)])
   %disp(['PANN numDist       = ',num2str(PAPV_numDist)])
   %disp(['PANN AUC           = ',num2str(PANN_INFO.auc)])
   %disp(['PANN eer           = ',num2str(PANN_INFO.eer)])
   %disp(['PANN tp perc       = ',num2str(100*(PANN_TP/PAPV_numCar))])
   %disp(['PANN tn perc       = ',num2str(100*(PANN_TN/PAPV_numDist))])
   %disp(['--------------------------------'])
   %disp(['PETA tp            = ',num2str(PETA_TP)])
   %disp(['PETA tn            = ',num2str(PETA_TN)])
   %disp(['PETA fp            = ',num2str(length(find(papvStruct.pv_stats.fp)))])
   %disp(['PETA fn            = ',num2str(length(find(papvStruct.pv_stats.fn)))])
   %disp(['PETA numCars       = ',num2str(PAPV_numCar)])
   %disp(['PETA numDist       = ',num2str(PAPV_numDist)])
   %disp(['PETA AUC           = ',num2str(PETA_INFO.auc)])
   %disp(['PETA eer           = ',num2str(PETA_INFO.eer)])
   %disp(['PETA tp perc       = ',num2str(100*(PETA_TP/PAPV_numCar))])
   %disp(['PETA tn perc       = ',num2str(100*(PETA_TN/PAPV_numDist))])
   %disp(['--------------------------------'])

   %if (logical(papvStruct.roc_labels+1) == logical(phogStruct.roc_labels+1))
   %    gt_mask = logical(papvStruct.roc_labels+1) ;
   %else
   %    disp('phow_neovis: ERROR: PA/PV and PHOG ground truths are not the same.')
   %    keyboard
   %end

   %% -------------------------------------------------------------------------
   %%                           Compute co-dependence statistics for cSIFT/PHOG
   %% -------------------------------------------------------------------------
   %if ne(size(phogStruct.roc_labels),size(cSiftStruct.roc_labels))
   %    disp('phow_neovis: ERROR: PHOG and cSIFT struct have different sized ground-truth vectors.')
   %    keyboard
   %end
   %if ~all(phogStruct.roc_labels == cSiftStruct.roc_labels)
   %    disp('phow_neovis: ERROR: PHOG and cSIFT do not have same ground-truth labels.')
   %    keyboard
   %end

   %disp(' ')
   %disp('--------------------------------')
   %disp(['cSIFT/PHOG: y1 = cSIFT, y2 = PHOG: '])
   %[LR_HR, LR_FA, codep] = compute_codep(cSiftStruct.cars,phogStruct.cars,gt_mask) ;

   %figure ; clf ;
   %hold on
   %    %sift_hroc = plot(cSIFT_FPR, cSIFT_TPR, 'g', 'linewidth', 2) ;
   %    %sift_err  = plot(cSIFT_INFO.eer, 1-cSIFT_INFO.eer, 'k*', 'linewidth', 1) ;

   %    %phog_hroc = plot(PHOG_FPR, PHOG_TPR, 'b', 'linewidth', 2) ;
   %    %phog_err  = plot(PHOG_INFO.eer, 1-PHOG_INFO.eer, 'k*', 'linewidth', 1) ;

   %    sift_roc   = plot([0 cSIFT_FP_RT 1], [0 cSIFT_TP_RT 1], 'g-*', 'linewidth', 2) ;
   %    phog_roc   = plot([0 phog_fp_rt 1], [0 phog_tp_rt 1], 'b-*', 'linewidth', 2) ;

   %    lr_AUC    = trapz(LR_FA,LR_HR) ;
   %    lr_hroc   = plot(LR_FA, LR_HR, 'k-*', 'linewidth', 2) ;

   %    hrand     = plot([0 1], [0 1], 'r--', 'linewidth', 2) ;
   %    hopt      = plot([1 0], [0 1], 'k--', 'linewidth', 1) ;
   %hold off
   %xlabel('false positve rate') ;
   %ylabel('true positve rate (recall)') ;
   %loc = 'se' ;
   %title('cSIFT, PHOG, LR ROC')
   %grid on ;
   %xlim([0 1]) ;
   %ylim([0 1]) ;
   %axis square ;
   %legend([sift_roc phog_roc lr_hroc hrand], ['cSIFT AUC   = ',num2str(cSIFT_AUC)], ['PHOG AUC = ',num2str(phog_AUC)], ['LR AUC   = ',num2str(lr_AUC)], 'ROC rand.', 'location', loc) ;

   %disp(['cSIFT/PHOG: Combined Hit rates:',char(10),char(9),mat2str(LR_HR,3)])
   %disp(['cSIFT/PHOG: Combined False Alarm rates:',char(10),char(9),mat2str(LR_FA,3)])
   %disp(['cSIFT/PHOG: The covariance under H1:  ',num2str(codep.covariance_HR)])
   %disp(['cSIFT/PHOG: The covariance under H0:  ',num2str(codep.covariance_FA)])
   %disp(['cSIFT/PHOG: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H1: ',num2str(codep.h1_mutual_info)])
   %disp(['cSIFT/PHOG: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H0: ',num2str(codep.h0_mutual_info)])

   %% -------------------------------------------------------------------------
   %%                          Compute co-dependence statistics for cSIFT/gSIFT
   %% -------------------------------------------------------------------------
   %if ne(size(gSiftStruct.roc_labels),size(cSiftStruct.roc_labels))
   %    disp('phow_neovis: ERROR: gSIFT and cSIFT struct have different sized ground-truth vectors.')
   %    keyboard
   %end
   %if ~all(gSiftStruct.roc_labels == cSiftStruct.roc_labels)
   %    disp('phow_neovis: ERROR: gSIFT and cSIFT do not have same ground-truth labels.')
   %    keyboard
   %end

   %disp(' ')
   %disp('--------------------------------')
   %disp(['cSIFT/gSIFT: y1 = cSIFT, y2 = gSIFT: '])
   %[LR_HR, LR_FA, codep] = compute_codep(cSiftStruct.cars,gSiftStruct.cars,gt_mask) ;

   %figure ; clf ;
   %hold on
   %    %c_sift_hroc = plot(cSIFT_FPR, cSIFT_TPR, 'g', 'linewidth', 2) ;
   %    %c_sift_err  = plot(cSIFT_INFO.eer, 1-cSIFT_INFO.eer, 'k*', 'linewidth', 1) ;

   %    %g_sift_hroc = plot(gSIFT_FPR, gSIFT_TPR, 'g', 'linewidth', 2) ;
   %    %g_sift_err  = plot(gSIFT_INFO.eer, 1-gSIFT_INFO.eer, 'k*', 'linewidth', 1) ;

   %    c_sift_roc   = plot([0 cSIFT_FP_RT 1], [0 cSIFT_TP_RT 1], 'g-*', 'linewidth', 2) ;
   %    g_sift_roc   = plot([0 gSIFT_FP_RT 1], [0 gSIFT_TP_RT 1], 'b-*', 'linewidth', 2) ;

   %    lr_AUC    = trapz(LR_FA,LR_HR) ;
   %    lr_hroc   = plot(LR_FA, LR_HR, 'k-*', 'linewidth', 2) ;

   %    hrand     = plot([0 1], [0 1], 'r--', 'linewidth', 2) ;
   %    hopt      = plot([1 0], [0 1], 'k--', 'linewidth', 1) ;
   %hold off
   %xlabel('false positve rate') ;
   %ylabel('true positve rate (recall)') ;
   %loc = 'se' ;
   %title('cSIFT, gSIFT, LR ROC')
   %grid on ;
   %xlim([0 1]) ;
   %ylim([0 1]) ;
   %axis square ;
   %legend([c_sift_roc g_sift_roc lr_hroc hrand], ['cSIFT AUC   = ',num2str(cSIFT_AUC)], ['gSIFT AUC = ',num2str(gSIFT_AUC)], ['LR AUC   = ',num2str(lr_AUC)], 'ROC rand.', 'location', loc) ;

   %disp(['cSIFT/gSIFT: Combined Hit rates:',char(10),char(9),mat2str(LR_HR,3)])
   %disp(['cSIFT/gSIFT: Combined False Alarm rates:',char(10),char(9),mat2str(LR_FA,3)])
   %disp(['cSIFT/gSIFT: The covariance under H1:  ',num2str(codep.covariance_HR)])
   %disp(['cSIFT/gSIFT: The covariance under H0:  ',num2str(codep.covariance_FA)])
   %disp(['cSIFT/gSIFT: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H1: ',num2str(codep.h1_mutual_info)])
   %disp(['cSIFT/gSIFT: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H0: ',num2str(codep.h0_mutual_info)])

   %% -------------------------------------------------------------------------
   %%                           Compute co-dependence statistics for cSIFT/PANN
   %% -------------------------------------------------------------------------
   %disp(' ')
   %disp('--------------------------------')
   %disp(['cSIFT/PANN: y1 = PANN, y2 = cSIFT: '])

   %[LR_HR, LR_FA, codep] = compute_codep(papvStruct.pa_stats.cars,cSiftStruct.cars,gt_mask) ;

   %figure ; clf ;
   %hold on
   %    pann_tp_rt = PANN_TP/PAPV_numCar ;
   %    pann_fp_rt = PANN_FP/PAPV_numDist ;
   %    pann_AUC   = trapz([0 pann_fp_rt 1],[0 pann_tp_rt 1]) ;
   %    pann_roc   = plot([0 pann_fp_rt 1], [0 pann_tp_rt 1], 'g-*', 'linewidth', 2) ;

   %    sift_roc   = plot([0 cSIFT_FP_RT 1], [0 cSIFT_TP_RT 1], 'b-*', 'linewidth', 2) ;

   %    lr_AUC    = trapz(LR_FA,LR_HR) ;
   %    lr_hroc   = plot(LR_FA,LR_HR,'k-*', 'linewidth', 2) ;

   %    hrand     = plot([0 1], [0 1], 'r--', 'linewidth', 2) ;
   %    hopt      = plot([1 0], [0 1], 'k--', 'linewidth', 1) ;
   %hold off

   %xlabel('false positve rate') ;
   %ylabel('true positve rate (recall)') ;
   %loc = 'se' ;
   %title('PANN, cSIFT, LR ROC')

   %grid on ;
   %xlim([0 1]) ;
   %ylim([0 1]) ;
   %axis square ;
   %legend([pann_roc sift_roc lr_hroc hrand], ['PANN AUC = ',num2str(pann_AUC)], ['cSIFT AUC = ',num2str(cSIFT_AUC)], ['LR AUC = ',num2str(lr_AUC)], 'ROC rand.', 'location', loc) ;

   %disp(['cSIFT/PANN: Combined Hit rates:',char(10),char(9),mat2str(LR_HR,3)])
   %disp(['cSIFT/PANN: Combined False Alarm rates:',char(10),char(9),mat2str(LR_FA,3)])
   %disp(['cSIFT/PANN: The covariance under H1:  ',num2str(codep.covariance_HR)])
   %disp(['cSIFT/PANN: The covariance under H0:  ',num2str(codep.covariance_FA)])
   %disp(['cSIFT/PANN: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H1: ',num2str(codep.h1_mutual_info)])
   %disp(['cSIFT/PANN: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H0: ',num2str(codep.h0_mutual_info)])

   %% -------------------------------------------------------------------------
   %%                           Compute co-dependence statistics for cSIFT/PETA
   %% -------------------------------------------------------------------------
   %disp(' ')
   %disp('--------------------------------')
   %disp(['cSIFT/PETA: y1 = PETA, y2 = cSIFT: '])

   %[LR_HR, LR_FA, codep] = compute_codep(papvStruct.pv_stats.cars,cSiftStruct.cars,gt_mask) ;

   %figure ; clf ;
   %hold on
   %    peta_tp_rt = PETA_TP/PAPV_numCar ;
   %    peta_fp_rt = PETA_FP/PAPV_numDist ;
   %    peta_AUC   = trapz([0 peta_fp_rt 1],[0 peta_tp_rt 1]) ;
   %    peta_roc   = plot([0 peta_fp_rt 1], [0 peta_tp_rt 1], 'g-*', 'linewidth', 2) ;

   %    sift_roc   = plot([0 cSIFT_FP_RT 1], [0 cSIFT_TP_RT 1], 'b-*', 'linewidth', 2) ;

   %    lr_AUC    = trapz(LR_FA,LR_HR) ;
   %    lr_hroc   = plot(LR_FA,LR_HR,'k-*', 'linewidth', 2) ;

   %    hrand     = plot([0 1], [0 1], 'r--', 'linewidth', 2) ;
   %    hopt      = plot([1 0], [0 1], 'k--', 'linewidth', 1) ;
   %hold off

   %xlabel('false positve rate') ;
   %ylabel('true positve rate (recall)') ;
   %loc = 'se' ;
   %title('PETA, cSIFT, LR ROC')

   %grid on ;
   %xlim([0 1]) ;
   %ylim([0 1]) ;
   %axis square ;
   %legend([peta_roc sift_roc lr_hroc hrand], ['PETA AUC = ',num2str(pann_AUC)], ['cSIFT AUC = ',num2str(cSIFT_AUC)], ['LR AUC = ',num2str(lr_AUC)], 'ROC rand.', 'location', loc) ;

   %disp(['cSIFT/PETA: Combined Hit rates:',char(10),char(9),mat2str(LR_HR,3)])
   %disp(['cSIFT/PETA: Combined False Alarm rates:',char(10),char(9),mat2str(LR_FA,3)])
   %disp(['cSIFT/PETA: The covariance under H1:  ',num2str(codep.covariance_HR)])
   %disp(['cSIFT/PETA: The covariance under H0:  ',num2str(codep.covariance_FA)])
   %disp(['cSIFT/PETA: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H1: ',num2str(codep.h1_mutual_info)])
   %disp(['cSIFT/PETA: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H0: ',num2str(codep.h0_mutual_info)])

   %% -------------------------------------------------------------------------
   %%                            Compute co-dependence statistics for PANN/PETA
   %% -------------------------------------------------------------------------
   %disp(' ')
   %disp('--------------------------------')
   %disp(['PANN/PETA: y1 = PANN, y2 = PETA: '])
   %[LR_HR, LR_FA, codep] = compute_codep(papvStruct.pa_stats.cars,papvStruct.pv_stats.cars,gt_mask) ;

   %figure ; clf ;
   %hold on
   %    %pann_hroc = plot(PANN_FPR, PANN_TPR, 'g', 'linewidth', 2) ;
   %    %pann_err  = plot(PANN_INFO.eer, 1-PANN_INFO.eer, 'k*', 'linewidth', 1) ;

   %    %peta_hroc = plot(PETA_FPR, PETA_TPR, 'b', 'linewidth', 2) ;
   %    %peta_err  = plot(PETA_INFO.eer, 1-PETA_INFO.eer, 'k*', 'linewidth', 1) ;

   %    pann_roc   = plot([0 pann_fp_rt 1], [0 pann_tp_rt 1], 'g-*', 'linewidth', 2) ;
   %    peta_roc   = plot([0 peta_fp_rt 1], [0 peta_tp_rt 1], 'b-*', 'linewidth', 2) ;

   %    lr_AUC    = trapz(LR_FA,LR_HR) ;
   %    lr_hroc   = plot(LR_FA,LR_HR,'k-*', 'linewidth', 2) ;

   %    hrand     = plot([0 1], [0 1], 'r--', 'linewidth', 2) ;
   %    hopt      = plot([1 0], [0 1], 'k--', 'linewidth', 1) ;
   %hold off
   %xlabel('false positve rate') ;
   %ylabel('true positve rate (recall)') ;
   %loc = 'se' ;
   %title('PANN, PETA, LR ROC')
   %grid on ;
   %xlim([0 1]) ;
   %ylim([0 1]) ;
   %axis square ;
   %legend([pann_roc peta_roc lr_hroc hrand], ['PANN AUC = ',num2str(pann_AUC)], ['PETA AUC = ',num2str(peta_AUC)], ['LR AUC = ',num2str(lr_AUC)], 'ROC rand.', 'location', loc) ;

   %disp(['PANN/PETA: Combined Hit rates:',char(10),char(9),mat2str(LR_HR,3)])
   %disp(['PANN/PETA: Combined False Alarm rates:',char(10),char(9),mat2str(LR_FA,3)])
   %disp(['PANN/PETA: The covariance under H1:  ',num2str(codep.covariance_HR)])
   %disp(['PANN/PETA: The covariance under H0:  ',num2str(codep.covariance_FA)])
   %disp(['PANN/PETA: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H1: ',num2str(codep.h1_mutual_info)])
   %disp(['PANN/PETA: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H0: ',num2str(codep.h0_mutual_info)])

   %% -------------------------------------------------------------------------
   %%                            Compute co-dependence statistics for PANN/PHOG
   %% -------------------------------------------------------------------------
   %disp(' ')
   %disp('--------------------------------')
   %disp(['PANN/PHOG: y1 = PANN, y2 = PHOG: '])

   %[LR_HR, LR_FA, codep] = compute_codep(papvStruct.pa_stats.cars,phogStruct.cars,gt_mask) ;

   %figure ; clf ;
   %hold on
   %    pann_roc   = plot([0 pann_fp_rt 1], [0 pann_tp_rt 1], 'g-*', 'linewidth', 2) ;
   %    phog_roc   = plot([0 phog_fp_rt 1], [0 phog_tp_rt 1], 'b-*', 'linewidth', 2) ;

   %    lr_AUC    = trapz(LR_FA,LR_HR) ;
   %    lr_hroc   = plot(LR_FA,LR_HR,'k-*', 'linewidth', 2) ;

   %    hrand     = plot([0 1], [0 1], 'r--', 'linewidth', 2) ;
   %    hopt      = plot([1 0], [0 1], 'k--', 'linewidth', 1) ;
   %hold off
   %xlabel('false positve rate') ;
   %ylabel('true positve rate (recall)') ;
   %loc = 'se' ;
   %title('PANN, PHOG, LR ROC')
   %grid on ;
   %xlim([0 1]) ;
   %ylim([0 1]) ;
   %axis square ;
   %legend([pann_roc phog_roc lr_hroc hrand], ['PANN AUC = ',num2str(pann_AUC)], ['PHOG AUC = ',num2str(phog_AUC)], ['LR AUC = ',num2str(lr_AUC)], 'ROC rand.', 'location', loc) ;

   %disp(['PANN/PHOG: Combined Hit rates:',char(10),char(9),mat2str(LR_HR,3)])
   %disp(['PANN/PHOG: Combined False Alarm rates:',char(10),char(9),mat2str(LR_FA,3)])
   %disp(['PANN/PHOG: The covariance under H1:  ',num2str(codep.covariance_HR)])
   %disp(['PANN/PHOG: The covariance under H0:  ',num2str(codep.covariance_FA)])
   %disp(['PANN/PHOG: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H1: ',num2str(codep.h1_mutual_info)])
   %disp(['PANN/PHOG: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H0: ',num2str(codep.h0_mutual_info)])

   %% -------------------------------------------------------------------------
   %%                            Compute co-dependence statistics for PETA/PHOG
   %% -------------------------------------------------------------------------
   %disp(' ')
   %disp('--------------------------------')
   %disp(['PETA/PHOG: y1 = PETA, y2 = PHOG: '])

   %[LR_HR, LR_FA, codep] = compute_codep(papvStruct.pv_stats.cars,phogStruct.cars,gt_mask) ;

   %figure ; clf ;
   %hold on
   %    peta_roc   = plot([0 peta_fp_rt 1], [0 peta_tp_rt 1], 'g-*', 'linewidth', 2) ;
   %    phog_roc   = plot([0 phog_fp_rt 1], [0 phog_tp_rt 1], 'b-*', 'linewidth', 2) ;

   %    lr_AUC    = trapz(LR_FA,LR_HR) ;
   %    lr_hroc   = plot(LR_FA,LR_HR,'k-*', 'linewidth', 2) ;

   %    hrand     = plot([0 1], [0 1], 'r--', 'linewidth', 2) ;
   %    hopt      = plot([1 0], [0 1], 'k--', 'linewidth', 1) ;
   %hold off
   %xlabel('false positve rate') ;
   %ylabel('true positve rate (recall)') ;
   %loc = 'se' ;
   %title('PETA, PHOG, LR ROC')
   %grid on ;
   %xlim([0 1]) ;
   %ylim([0 1]) ;
   %axis square ;
   %legend([peta_roc phog_roc lr_hroc hrand], ['PETA AUC = ',num2str(peta_AUC)], ['PHOG AUC = ',num2str(phog_AUC)], ['LR AUC = ',num2str(lr_AUC)], 'ROC rand.', 'location', loc) ;

   %disp(['PETA/PHOG: Combined Hit rates:',char(10),char(9),mat2str(LR_HR,3)])
   %disp(['PETA/PHOG: Combined False Alarm rates:',char(10),char(9),mat2str(LR_FA,3)])
   %disp(['PETA/PHOG: The covariance under H1:  ',num2str(codep.covariance_HR)])
   %disp(['PETA/PHOG: The covariance under H0:  ',num2str(codep.covariance_FA)])
   %disp(['PETA/PHOG: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H1: ',num2str(codep.h1_mutual_info)])
   %disp(['PETA/PHOG: The mutual information (derived from the Kullback-Leibler distance)',...
   %    ' between the classifiers under H0: ',num2str(codep.h0_mutual_info)])

    keyboard
end
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
function [LR_HR, LR_FA, codep] = compute_codep(y1,y2,gt_mask)
    num_hits_gt  = length(find(gt_mask)) ;
    num_hits_gti = length(find(~gt_mask));

    y1i = logical(~y1) ;
    y2i = logical(~y2) ;

    y1_HR = [0 length(find(y1.*gt_mask))/num_hits_gt 1] ;
    y2_HR = [0 length(find(y2.*gt_mask))/num_hits_gt 1] ;

    y1_FA = [0 1-length(find(y1i.*~gt_mask))/num_hits_gti 1] ;
    y2_FA = [0 1-length(find(y2i.*~gt_mask))/num_hits_gti 1] ;

    comb_00 = y1i .* y2i ; 
    comb_01 = y1i .* y2  ;
    comb_10 = y1  .* y2i ;  
    comb_11 = y1  .* y2  ; 
    masks{1} = comb_00 ;
    masks{2} = comb_01 ;     
    masks{3} = comb_10 ; 
    masks{4} = comb_11 ;

    avg_HR   = zeros([1 4]) ;
    avg_FA   = zeros([1 4]) ;
    for i_test = 1:4
        co_mask = masks{i_test} ;

        ind12  = find(co_mask.*gt_mask) ;
        ind12i = find(~co_mask.*~gt_mask) ;

        %hit rate
        avg_HR(i_test)     = length(ind12) / (num_hits_gt+0.0000001) ;

        %false alarm rate
        avg_FA(i_test)     = 1 - (length(ind12i) / (num_hits_gti+0.0000001)) ;
    end

    %% Verify that marginals = independent values
    y1_marg = round(1000*(avg_HR(3)+avg_HR(4)))/1000;
    y1_ind  = round(1000*y1_HR(2))/1000;
    y2_marg = round(1000*(avg_HR(2)+avg_HR(4)))/1000;
    y2_ind  = round(1000*y2_HR(2))/1000;
    if ne(y1_marg,y1_ind) | ne(y2_marg,y2_ind)
        disp(['phow_neovis: Marginal values are not correct.',char(10),char(9),...
            'Marginal y1    = ',num2str(avg_HR(3)+avg_HR(4)),char(10),char(9),...
            'Independent y1 = ',num2str(y1_HR(2))])
            'Marginal y2    = ',num2str(avg_HR(2)+avg_HR(4)),char(10),char(9),...
            'Independent y2 = ',num2str(y2_HR(2)),char(10),char(9),...
        keyboard
    end

    h1_joint_prob = avg_HR(4);
    h0_joint_prob = avg_FA(4);

    ux_HR = avg_HR(3)+avg_HR(4); %y1 H1 marginal
    uy_HR = avg_HR(2)+avg_HR(4); %y2 H1 marginal
    codep.covariance_HR = ((0-ux_HR)*(0-uy_HR)*avg_HR(1))+... %00
        ((0-ux_HR)*(1-uy_HR)*avg_HR(2))+...                   %01
        ((1-ux_HR)*(0-uy_HR)*avg_HR(3))+...                   %10
        ((1-ux_HR)*(1-uy_HR)*avg_HR(4));                      %11

    ux_FA = avg_FA(3)+avg_FA(4); %y1 H0 marginal
    uy_FA = avg_FA(2)+avg_FA(4); %y2 H0 marginal
    codep.covariance_FA = ((0-ux_FA)*(0-uy_FA)*avg_FA(1))+... %00
        ((0-ux_FA)*(1-uy_FA)*avg_FA(2))+...                   %01
        ((1-ux_FA)*(0-uy_FA)*avg_FA(3))+...                   %10
        ((1-ux_FA)*(1-uy_FA)*avg_FA(4));                      %11

    %KL divergence between two random variables can be computed as the mutual information
    %  I(X;Y) = sum(p(x,y)*log_2(p(x,y)/(p(x)*p(y))))
    %
    %  I(X;Y) should be 0 if the random variables are stistically independent
    %
    % Derived from: http://www.snl.salk.edu/~shlens/kl.pdf
    %
    codep.h1_mutual_info = h1_joint_prob*log2(h1_joint_prob/(y1_HR(2)*y2_HR(2)));
    codep.h0_mutual_info = h0_joint_prob*log2(h0_joint_prob/(y1_FA(2)*y2_FA(2)));

    %% Determine optimal LR-ROC
    likelihood_ratios = avg_HR./avg_FA ;

    [sorted_likelihood sorted_idx] = sort(likelihood_ratios,'ascend') ;

    bool_operator = zeros([1 3]) ;
    for i_tau = 1:3
        switch(i_tau)
            case 1
                Pr = [sorted_idx(2) sorted_idx(3) sorted_idx(4)];
                if     find(Pr==2)&find(Pr==3)&find(Pr==4)
                    bool_operator(i_tau) = 8;
                elseif find(Pr==1)&find(Pr==3)&find(Pr==4)
                    bool_operator(i_tau) = 12;
                elseif find(Pr==1)&find(Pr==2)&find(Pr==4)
                    bool_operator(i_tau) = 14;
                elseif find(Pr==1)&find(Pr==2)&find(Pr==3)
                    bool_operator(i_tau) = 15;
                else
                    error('Couldn''t find optimal boolean combination')
                end
            case 2
                Pr = [sorted_idx(3) sorted_idx(4)];
                if     find(Pr==3)&find(Pr==4)
                    bool_operator(i_tau) = 4;
                elseif find(Pr==2)&find(Pr==4)
                    bool_operator(i_tau) = 6;
                elseif find(Pr==2)&find(Pr==3)
                    bool_operator(i_tau) = 7;
                elseif find(Pr==1)&find(Pr==4)
                    bool_operator(i_tau) = 10;
                elseif find(Pr==1)&find(Pr==3)
                    bool_operator(i_tau) = 11;
                elseif find(Pr==1)&find(Pr==2)
                    bool_operator(i_tau) = 13;
                else
                    error('Couldn''t find optimal boolean combination')
                end
            case 3
                Pr = [sorted_idx(4)];
                if     find(Pr==4)
                    bool_operator(i_tau) = 2;
                elseif find(Pr==3)
                    bool_operator(i_tau) = 3;
                elseif find(Pr==2)
                    bool_operator(i_tau) = 5;
                elseif find(Pr==1)
                    bool_operator(i_tau) = 9;
                else
                    error('Couldn''t find optimal boolean combination')
                end
            otherwise
                error('Impossible switch error.')
        end
    end

    bool_algebra_list = {'0';'y1 AND y2';'y1 AND ~y2';'y1';'~y1 AND y2';'y2';'y1 XOR y2';'y1 OR y2';...
        '~y1 OR ~y2';'~(y1 XOR y2)';'~y2';'~(~y1 AND y2)';'~y1';'~(y1 AND ~y2)';'~(y1 OR y2)';'1'} ;

    disp('')
    disp('--------------------------------')
    disp(['Boolean rules: ',char(10),char(9),bool_algebra_list{bool_operator(3),:},char(10),...
        char(9),bool_algebra_list{bool_operator(2),:},char(10),char(9),bool_algebra_list{bool_operator(1),:}])

    %% 16 possible boolean combinations
    if (length(y1)==length(y2) && length(y1)==length(gt_mask))
        maskLength = length(y1);
    else
        disp('phow_neovis: ERROR: Input masks do not have the same sized masks')
        keyboard
    end
    bool = zeros([16 maskLength]);
    bool(1,:)  = zeros([1 maskLength]);         %0
    bool(2,:)  = y1.*y2;                        %y1 AND y2
    bool(3,:)  = y1.*y2i;                       %y1 AND ~y2
    bool(4,:)  = y1;                            %y1
    bool(5,:)  = y1i.*y2;                       %~y1 AND y2
    bool(6,:)  = y2;                            %y2
    bool(7,:)  = ~logical((~y1.*~y2)+(y1.*y2)); %y1 XOR y2
    bool(8,:)  = logical(y1+y2);                %y1 OR y2
    bool(9,:)  = logical(y1i+y2i);              %~y1 OR ~y2
    bool(10,:) = logical(~bool(7,:));           %~(y1 XOR y2)
    bool(11,:) = y2i;                           %~y2
    bool(12,:) = logical(~bool(5,:));           %~(~y1 AND y2)
    bool(13,:) = y1i;                           %~y1
    bool(14,:) = logical(~bool(3,:));           %~(y1 AND ~y2)
    bool(15,:) = logical(~bool(8,:));           %~(y1 OR y2)
    bool(16,:) = ones([1 maskLength]);          %1
    
    comb_HR = zeros([1 3]);
    comb_FA = zeros([1 3]);
    for i_tau = 1:3
        comb_mask      = bool(bool_operator(i_tau),:);
        comb_HR(i_tau) = length(find(comb_mask.*gt_mask))/(num_hits_gt+0.00000001);
        comb_FA(i_tau) = 1-length(find(~comb_mask.*~gt_mask))/(num_hits_gti+0.00000001);
    end

    LR_HR = [0 comb_HR(3) comb_HR(2) comb_HR(1) 1];
    LR_FA = [0 comb_FA(3) comb_FA(2) comb_FA(1) 1];

end

% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------
    im = im2single(im) ;
    %if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end %%I don't know why we would want this
end

% -------------------------------------------------------------------------
function out_hist = getImageDescriptor(model, im)
% -------------------------------------------------------------------------
    im = standarizeImage(im) ;
    width = size(im,2) ;
    height = size(im,1) ;
    numWords = size(model.vocab, 2) ;

    % get PHOW (dSIFT + IMSMOOTH) features
    [frames, descrs] = vl_phow(im, model.phowOpts{:}) ;

    % quantize appearance
    switch model.quantizer
      case 'vq'
        [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
      case 'kdtree'
        binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                      single(descrs), ...
                                      'MaxComparisons', 15)) ;
    end

    hists = {} ;
    for i = 1:length(model.numSpatialX)
      binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
      binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

      % combined quantization
      bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                     binsy,binsx,binsa) ;
      out_hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
      out_hist = vl_binsum(out_hist, ones(size(bins)), bins) ;
      hists{i} = single(out_hist / sum(out_hist)) ; %normalize
    end
    out_hist = cat(1,hists{:}) ;
    out_hist = out_hist / sum(out_hist) ;
end

% -------------------------------------------------------------------------
function [className, score, indi_scores] = classify(model, im, homkerGamma)
% -------------------------------------------------------------------------
    testHist = getImageDescriptor(model, im) ;
    psix = vl_homkermap(testHist, 1, 'kchi2', 'gamma', homkerGamma) ;
    indi_scores = model.w' * psix + model.b' ;
    [score, best] = max(indi_scores) ;
    className = model.classes{best} ;
end
