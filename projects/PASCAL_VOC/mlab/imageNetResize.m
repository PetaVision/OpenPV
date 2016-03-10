%% wrapper for calling chipPASCAL in a loop over imageNet categories
%% chipPASCAL assumes that all images are stored in a folder called JPEGImages 
%% and all annotations are stored in a folder Annotations, both located
%% in a parent folder denoted by the variable "imageNet_synset_name", 
%% which should point to a folder stored in VOCdevkit

clear all
if exist("/nh/compneuro/Data", "dir")
  machine_path = "/nh/compneuro/Data";
elseif exist("/home/ec2-user/mountData", "dir")
  machine_path = "/home/ec2-user/mountData";
elseif exist("/Users/gkenyon")
  machine_path = "/Users/gkenyon";
elseif exist("/home/gkenyon")
  machine_path = "/home/gkenyon";
endif
setImagePaths;

imageNet_dir = "/home/ec2-user/mountData/openpv/projects/PASCAL_VOC/imageNet"; %% folder containing imageNet tar.gz fliles, 1 tar file per object category
imageNet_annotation_path = [imageNet_dir, '/', 'Annotations'];  %% folder containing xml annotaion files packed as tar.gz files, 1 per object category
imageNet_list = glob([imageNet_dir, '/', '*.tar']);
%%imageNet_list = glob([imageNet_dir, '/', 'n03790512.tar']);
num_imageNet = length(imageNet_list);
confirm_recursive_rmdir(0);

%% basic data structure for holding sparse ground truth
num_annotated = 0;  %% total number of resized images has to be computed on the fly 
num_resized_failed = 0;
num_non_RGB = 0;
classID_data = cell(); %% size of cell is total number of resized images, which we don't know yet
resized_filepathnames = cell();


if exist([imageNet_dir, filesep, 'glossary.mat'],'file')
  load([imageNet_dir, filesep, 'glossary.mat'], '-text');
else
  num_glossary = 0;
  glossary_file = [imageNet_dir, filesep, 'glossary.txt'];
  glossary_fid = fopen(glossary_file);
  glossary_line = fgets(glossary_fid);
  glossary_wnid = cell;
  glossary_word = cell;
  while (glossary_line ~= -1)
   num_glossary = num_glossary + 1;
   glossary_wnid{num_glossary,1} = glossary_line(1:9);
   glossary_word{num_glossary,1} = glossary_line(11:end-1);
   glossary_line = fgets(glossary_fid);
  endwhile
  fclose(glossary_fid);
  save('-text', [imageNet_dir, filesep, 'glossary.mat'], "glossary_wnid" ,"glossary_word")
endif

imageNet_synset_classes = cell;
for i_imageNet = 1 : num_imageNet
  imageNet_tarpath = imageNet_list{i_imageNet};
  [imageNet_tardir, imageNet_wnid, imageNet_tarext, ~] = fileparts(imageNet_tarpath);
  glossary_ndx = strmatch(imageNet_wnid, glossary_wnid);
  if isempty(glossary_ndx)
    warning(["no glossary entry for: ", imageNet_wnid]);
    imageNet_synset_classes{i_imageNet} = imageNet_wnid;
    continue
  endif
  class_name_ndx = strfind(glossary_word{glossary_ndx,1}, ',');
  if isempty(class_name_ndx)
    class_name_ndx = length(glossary_word{glossary_ndx,1})+1;
  endif
  imageNet_synset_classes{i_imageNet} = [imageNet_wnid, '_', glossary_word{glossary_ndx,1}(1:class_name_ndx-1)]; 
endfor

for i_imageNet = 1 : num_imageNet
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
  %%imageNet_synset_name = imageNet_wnid;
  imageNet_synset_name = imageNet_synset_classes{i_imageNet}
  chipPASCAL;
endfor

%% shuffle 
[~, imageNet_shuffle_ndx] = sort(rand(num_annotated,1));
if length(classID_data) ~= num_annotated
  warning(["length(classID_data) = ", num2str(length(classID_data)), " ~= num_annotated =", num2str(num_annotated)])
  keyboard
endif
resized_filepathnames = resized_filepathnames(imageNet_shuffle_ndx(:));
classID_data = classID_data(imageNet_shuffle_ndx);
resized_list = [VOC_dataset_path, filesep, VOC_dataset, "_", orientation_type, "_list.txt"];
resized_fid = fopen(resized_list, "w", "native");
for i_resized = 1 : num_annotated
  fputs(resized_fid, [resized_filepathnames{i_resized}, "\n"]);
endfor
fclose(resized_fid)
classID_file = [resized_list(1:strfind(resized_list,"_list")-1), ".pvp"]; %%
writepvpsparseactivityfile(classID_file, classID_data, resized_width, resized_height, num_classes);
