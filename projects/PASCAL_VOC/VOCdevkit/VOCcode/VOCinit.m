clear VOCopts

% use VOC2006 or VOC2007 data

VOC2006=false; % set true to use VOC2006 data

% dataset

if ~exist('VOC_dataset', 'var') || isempty(VOC_dataset)
  if VOC2006 && ~imageNet_synset_flag
      VOCopts.dataset='VOC2006';
  elseif ~imageNet_synset_flag
      VOCopts.dataset='VOC2007';
  else
      VOCopts.dataset= 'imageNet';
  end
else
  VOCopts.dataset=VOC_dataset;
endif


% get current directory with forward slashes

%cwd=cd;
%cwd = '/Users/garkenyon/workspace/PASCAL_VOC';
if ~exist('VOCdevkit_path') || ~exist(VOCdevkit_path) 
  cwd = pwd;
else
  current_dir = pwd;
  chdir(VOCdevkit_path);
  cd ..
  cwd = pwd;
endif

cwd(cwd=='\')='/';

% change this path to point to your copy of the PASCAL VOC data
VOCopts.datadir=[cwd '/' 'VOCdevkit' '/'];

% change this path to a writable directory for your results
VOCopts.resdir=[cwd '/results/' VOCopts.dataset '/'];

% change this path to a writable local directory for the example code
VOCopts.localdir=[cwd '/local/' VOCopts.dataset '/'];

% initialize the test set

VOCopts.testset='val'; % use validation data for development test set
% VOCopts.testset='test'; % use test set for final challenge

% initialize main challenge paths

VOCopts.annopath=[VOCopts.datadir VOCopts.dataset '/' 'Annotations/%s.xml'];
VOCopts.imgpath=[VOCopts.datadir VOCopts.dataset '/' 'JPEGImages/%s.jpg'];
VOCopts.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Main/%s.txt'];
VOCopts.clsimgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Main/%s_%s.txt'];
VOCopts.clsrespath=[VOCopts.resdir 'Main/%s_cls_' VOCopts.testset '_%s.txt'];
VOCopts.detrespath=[VOCopts.resdir 'Main/%s_det_' VOCopts.testset '_%s.txt'];

% initialize segmentation task paths

VOCopts.seg.clsimgpath=[VOCopts.datadir VOCopts.dataset '/SegmentationClass/%s.png'];
VOCopts.seg.instimgpath=[VOCopts.datadir VOCopts.dataset '/SegmentationObject/%s.png'];

VOCopts.seg.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Segmentation/%s.txt'];

VOCopts.seg.clsresdir=[VOCopts.resdir 'Segmentation/%s_%s_cls'];
VOCopts.seg.instresdir=[VOCopts.resdir 'Segmentation/%s_%s_inst'];
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
VOCopts.seg.instrespath=[VOCopts.seg.instresdir '/%s.png'];

% initialize layout task paths

VOCopts.layout.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Layout/%s.txt'];
VOCopts.layout.respath=[VOCopts.resdir 'Layout/%s_layout_' VOCopts.testset '_%s.xml'];

% initialize the VOC challenge options

if VOC2006 && ~imageNet_synset_flag
    
    % VOC2006 classes
    
    VOCopts.classes={...
        'bicycle'
        'bus'
        'car'
        'cat'
        'cow'
        'dog'
        'horse'
        'motorbike'
        'person'
        'sheep'};
elseif ~imageNet_synset_flag

    % VOC2007 classes
    
    VOCopts.classes={...
        'aeroplane'
        'bicycle'
        'bird'
        'boat'
        'bottle'
        'bus'
        'car'
        'cat'
        'chair'
        'cow'
        'diningtable'
        'dog'
        'horse'
        'motorbike'
        'person'
        'pottedplant'
        'sheep'
        'sofa'
        'train'
        'tvmonitor'};
        
else
  if exist('imageNet_synset_classes', 'var') && ~isempty(imageNet_synset_classes)
    VOCopts.classes=imageNet_synset_classes;
  else
    error(['not defined imageNet_synset_classes']);
  endif
        
end

VOCopts.nclasses=length(VOCopts.classes);	

VOCopts.poses={...
    'Unspecified'
    'SideFaceLeft'
    'SideFaceRight'
    'Frontal'
    'Rear'};

VOCopts.nposes=length(VOCopts.poses);

VOCopts.parts={...
    'head'
    'hand'
    'foot'};    

VOCopts.maxparts=[1 2 2];   % max of each of above parts

VOCopts.nparts=length(VOCopts.parts);

VOCopts.minoverlap=0.5;

% initialize example options

VOCopts.exannocachepath=[VOCopts.localdir '%s_anno.mat'];

VOCopts.exfdpath=[VOCopts.localdir '%s_fd.mat'];
