%% wrapper for calling chipPASCAL in a loop over imageNet categories
%% chipPASCAL assumes that all images are stored in a folder called JPEGImages 
%% and all annotations are stored in a folder Annotations, both located
%% in a parent folder denoted by the variable "imageNet_synset_name", 
%% which should point to a folder stored in VOCdevkit

clear all
addpath('./');
setImagePaths;

addpath('/Users/garkenyon/workspace/PASCAL_VOC/mlab/ImageNetToolboxV0.3')
imageNet_dir = "/Users/garkenyon/workspace/PASCAL_VOC/imageNet"; %% folder containing imageNet tar.gz fliles, 1 tar file per object category
imageNet_annotation_path = [imageNet_dir, '/', 'Annotations'];  %% folder containing xml annotaion files packed as tar.gz files, 1 per object category
imageNet_list = glob([imageNet_dir, '/', '*.tar']);
num_imageNet = length(imageNet_list);
confirm_recursive_rmdir(0);

imageNet_synset_classes = cell;
for i_imageNet = 1 : num_imageNet
  imageNet_tarpath = imageNet_list{i_imageNet};
  [imageNet_tardir, imageNet_wnid, imageNet_tarext, ~] = fileparts(imageNet_tarpath);
  imageNet_synset_classes{i_imageNet} = imageNet_wnid;
endfor

for i_imageNet = num_imageNet : num_imageNet
  imageNet_tarpath = imageNet_list{i_imageNet};
  [imageNet_tardir, imageNet_wnid, imageNet_tarext, ~] = fileparts(imageNet_tarpath);
  annotations_parent_path_tmp = [VOCdevkit_path, '/', 'imageNet'];
  annotations_path_tmp = [annotations_parent_path_tmp, '/', 'Annotations'];
  if exist(annotations_path_tmp, 'dir')
    rmdir(annotations_path_tmp, 's');
  endif
  mkdir(annotations_parent_path_tmp);
  mkdir(annotations_parent_path_tmp, 'Annotations');
  imageNet_annotation_zipfile = [imageNet_annotation_path, '/', imageNet_wnid, '.tar.gz'];
  if ~exist(imageNet_annotation_zipfile, 'file')
    warning(['does not exist: imageNet_annotation_zipfile: ', imageNet_annotation_zipfile])
  else
    gunzip(imageNet_annotation_zipfile, annotations_path_tmp);
    movefile([annotations_path_tmp, '/', 'Annotation', '/', imageNet_wnid, '/', '*.xml'], annotations_path_tmp);
  endif
  %%wnidToDefinition(imageNet_annotation_path, [imageNet_wnid, '.xml']);  %% requires matlab
  JPEGImages_tmp = [annotations_parent_path_tmp, '/', 'JPEGImages'];
  rmdir(JPEGImages_tmp, 's');
  mkdir(JPEGImages_tmp);
  gunzip(imageNet_tarpath, JPEGImages_tmp);
  imageNet_synset_flag =  true;
  imageNet_synset_name = imageNet_wnid;
  chipPASCAL;

endfor