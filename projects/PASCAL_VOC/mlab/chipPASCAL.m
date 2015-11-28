%% script for resizing PASCAL VOC and imageNet images.
%% sparse ground truth representations, CSV annotation file, visualization masks 
%% and individual chips corresponding to each bounding box are generated as well.
%% the user specifies one of 3 orientation types, "landscape", "portriat" or "square",
%% or else specifies "any",
%% along with the new dimensions for
%% all resized images whose original dimensions are consistent with the
%% specified orientation_type.
%% if a specific orientation type is specified,
%% images are resized so that the "shorest"
%% resized dimension, measured as a fraction of the target ratio,
%% exactly matches the specified dimension, with the other
%% "longer" dimension cropped symmetrically so as to match the
%% specified size.
%% If the orientation type is "any", the shortest image dimension is set
%% to a standard size and the longest dimension is set so as to maintain the
%% original aspect ratio.
%% Optional padding can be added around all borders of the resized image using mirror BCs.
%% If a specific orientation type is specified, all resized images have the same final size.
%% Uses VOCdevkit
%% Also generates CSV file giving corners of each bounding box in
%% resized dimensions, using NeoVision2 format (see init CSV file fbelow)
%% Also generates a mask for each image with the class ID of each
%% pixel encoded by the activating of the corresponding bit
%% also generates a list of full paths to each resized image
%%
%% author: Garrett T. Kenyon, garkenyon@gmail.com ~2014
%%

%%clear all
%%close all
more off
%setenv("GNUTERM","X11")

%%  the line below allows chipPASCAL to be applied to a single imageNet category
if ~exist('imageNet_synset_flag', 'var') || isempty(imageNet_synset_flag)
  imageNet_synset_flag =  false;
endif

%% set up paths (edit this script to change paths to customize implementations
if ~exist("workspace_path", 'var') || ~exist(workspace_path, 'dir') || isempty(workspace_path)
  setImagePaths;
endif

%% specify target image type and dimensions
if ~exist('orientation_type', 'var') || isempty(orientation_type)
  orientation_type = 'landscape'; %%"any"; %%""square"; %%  "portrait"; %%
  disp(["orientation_type 1= ", orientation_type]);
endif
target_resize_length = 256;
target_resize_ratio = 3/4;
if strcmp(orientation_type, "landscape")
  target_resized_height = round(target_resize_length * target_resize_ratio); 
  target_resized_width = target_resize_length; 
elseif strcmp(orientation_type, "portrait")
  target_resized_height = target_resize_length; 
  target_resized_width = round(target_resize_length * target_resize_ratio); 
elseif strcmp(orientation_type, "square")
  target_resized_height = target_resize_length; 
  target_resized_width = target_resize_length; 
elseif strcmp(orientation_type, "any")  %% leave target size undefined until later, set smallest image dimension to target_resize_length and keep aspect ratio unchanged
  target_resized_height = [];
  target_resized_width = []; 
endif
border_padding = 0; %%8;

chip_flag = false; %%true;  %% if true, write resized chips to corresponding folders
mask_flag = false;   %% if true, make mask images for visualization purposes

%% diagnostic flags, causes figures to be displayed to screen rather
%% than only written to image files
plot_bbox = false; %%true; %%
if plot_bbox 
  bbox_fig = figure;
  set(bbox_fig, 'name', 'bbox');
  mask_fig = figure;
  set(mask_fig, 'name', 'mask');
  chip_fig = figure;
  set(chip_fig, 'name', 'chip');
endif

%% format to save corresponding images as (png, jpg, tiff, etc)
resized_ext = '.png';
mask_ext = '.png';
chip_ext = '.png';

%% set results paths
if ~imageNet_synset_flag
  VOC_dataset = "VOC2007";
else
  VOC_dataset = "imageNet"; 
endif
VOC_dataset_path = fullfile(PASCAL_VOC_path, VOC_dataset);
mkdir(VOC_dataset_path);
train_dir = "padded"; %%"train"
train_path = fullfile(PASCAL_VOC_path, VOC_dataset, train_dir);
mkdir(train_path);
if mask_flag
  mask_dir = "mask"
  mask_path = fullfile(PASCAL_VOC_path, VOC_dataset, mask_dir);
  mkdir(mask_path);
endif
if chip_flag
  chip_dir = "chip"
  chip_path = fullfile(PASCAL_VOC_path, VOC_dataset, chip_dir);
  mkdir(chip_path);
endif

% initialize VOC options
VOCinit;

%% working annotation path
[annotation_path, ~, annotation_ext] = fileparts(VOCopts.annopath);

%% minimus size that bounding box must be along either dimension in order to add to ground truth
min_bbox_size = 32;

%% colormap for category masks, 1 color for each class_ID
log2_max_class_ndx = 24;
max_class_ndx = 2^log2_max_class_ndx;
color_sep = floor(max_class_ndx / VOCopts.nclasses); %% separation
class_ID_list = (1:VOCopts.nclasses) * color_sep;

%% save labels as sparse PVP file with nf = # object classess
num_classes = VOCopts.nclasses;
if ~exist("classID_data") || isempty(classID_data)
  classID_data = cell(); 
  resized_filepathnames = cell();
  %% size of cell is number of resized images, which
  %% we don't know yet ...so grow classID_data dynamically
endif

%% load list of raw images
training_path = fullfile(VOCdevkit_path, VOCopts.dataset, "JPEGImages");
if ~exist(training_path, "dir")
  error(["training_path does not exist: ", training_path]);
end
if ~imageNet_synset_flag
  raw_ext = "jpg";
else
  raw_ext = "JPEG";
endif
raw_list = glob([training_path, filesep, "*." raw_ext]);
num_raw = length(raw_list)

%% open text files for storing lists of resized images and class ID masks
if border_padding > 0
  if ~imageNet_synset_flag
    resized_list = [VOC_dataset_path, filesep, VOC_dataset, "_padded", ...
		  num2str(border_padding), "_", orientation_type, "_list.txt"];
  else
    resized_list = [VOC_dataset_path, filesep, VOC_dataset, "_", imageNet_synset_name, "_", "_padded", ...
		  num2str(border_padding), '_',  orientation_type, "_list.txt"];
  endif
else
  if ~imageNet_synset_flag
    resized_list = [VOC_dataset_path, filesep, VOC_dataset, "_", orientation_type, "_list.txt"];
  else
    resized_list = [VOC_dataset_path, filesep, VOC_dataset, "_", imageNet_synset_name, '_',  orientation_type, "_list.txt"];
  endif
endif
resized_fid = fopen(resized_list, "w", "native");
if mask_flag
  mask_list = [resized_list(1:strfind(resized_list,"_list")-1),"_mask_list.txt"]; 
  mask_fid = fopen(mask_list, "w", "native");
endif

%% initialize CSV file (DARPA Neovision2 format)
CSV_file = [resized_list(1:strfind(resized_list,"_list")-1),".csv"]; 
CSV_fid = fopen(CSV_file, "wt", "native");
CSV_header = ["Frame	"];
CSV_header = [CSV_header, "BoundingBox_X1	"];
CSV_header = [CSV_header, "BoundingBox_Y1	"];
CSV_header = [CSV_header, "BoundingBox_X2	"];	
CSV_header = [CSV_header, "BoundingBox_Y2	"];	
CSV_header = [CSV_header, "BoundingBox_X3	"];	
CSV_header = [CSV_header, "BoundingBox_Y3	"];	
CSV_header = [CSV_header, "BoundingBox_X4	"];	
CSV_header = [CSV_header, "BoundingBox_Y4	"];	
CSV_header = [CSV_header, "ObjectType	"];	%
CSV_header = [CSV_header, "Occlusion	"];
CSV_header = [CSV_header, "Ambiguous	"];	
CSV_header = [CSV_header, "Confidence	"];	%% 1;
CSV_header = [CSV_header, "SiteInfo	"];	%% "";
CSV_header = [CSV_header, "Version	"]; 
CSV_header = [CSV_header, "Filename	"];   %% custom field
CSV_header = [CSV_header, "\n"]; 
fputs(CSV_fid, CSV_header);

%% fixed default values ...
CSV_Occlusion_str = "0";
CSV_Ambiguous_str = "0";
CSV_Confidence_str = "0";
CSV_SiteInfo_str = "0";
CSV_Version_str = "0";
CSV_filler_str = [CSV_Occlusion_str, ", ", CSV_Ambiguous_str, ", ", ...
		  CSV_Confidence_str, ", ", CSV_SiteInfo_str, ", ", ...
		  CSV_Version_str];


%% begin main loop over images
target_W2H_ratio = [];
if ~isempty(target_resized_width) && ~isempty(target_resized_height)
  target_W2H_ratio = target_resized_width / target_resized_height;
endif
if ~exist("num_annotated", "var") || isempty(num_annotated)
  num_annotated = 0;
  num_resized_failed = 0;
  num_non_RGB = 0;
endif
for i_raw = 1 : num_raw
  if mod(i_raw, ceil(num_raw/10)) == 0
    disp(["i_raw = ", num2str(i_raw)]);
  endif 
  raw_fullfile = raw_list{i_raw,:};
  [raw_dir, raw_name, raw_ext] = fileparts(raw_fullfile);
  raw_info = imfinfo(raw_fullfile);
  raw_image = imread(raw_fullfile);
  raw_height = raw_info.Height;
  raw_width = raw_info.Width;
  %% discard non-color or non 8 bit depth images
  if ~isrgb(raw_image) || raw_info.BitDepth ~= 8
    num_non_RGB = num_non_RGB + 1;
    continue
  endif
  
  %% remove artificial borders along rows
  while raw_height > 0 && ((all(raw_image(raw_height,:,:)==0) || all(raw_image(raw_height,:,:)==255)))
    raw_height = raw_height - 1;
  endwhile
  top_border_crop = raw_info.Height - raw_height;
  raw_image = raw_image(1:raw_height,:,:);
  raw_first_row = 1;
  while raw_first_row < raw_height && ((all(raw_image(raw_first_row,:,:)==0) || all(raw_image(raw_first_row,:,:)==255)))
    raw_first_row = raw_first_row + 1;
  endwhile
  bottom_border_crop = raw_first_row - 1;
  raw_image = raw_image(raw_first_row:raw_height,:,:);
  raw_height = size(raw_image, 1);
  %% now remove artificial border along the columns
  while raw_width > 0 && ((all(raw_image(:, raw_width,:)==0) || all(raw_image(:, raw_width,:)==255)))
    raw_width = raw_width - 1;
  endwhile
  right_border_crop = raw_info.Width - raw_width;
  raw_image = raw_image(:,1:raw_width,:);
  raw_first_col = 1;
  while raw_first_col < raw_width && ((all(raw_image(:,raw_first_col)==0) || all(raw_image(raw_first_col,:)==255)))
    raw_first_col = raw_first_col + 1;
  endwhile
  left_border_crop = raw_first_col - 1;
  raw_image = raw_image(:,raw_first_col:raw_width,:);
  raw_width = size(raw_image, 2);
  
  
  
  %% keep only images with correct orientation_type
  raw_W2H_ratio = raw_width / raw_height;
  if strcmp(orientation_type, "landscape") && raw_height >= raw_width
    continue
  elseif strcmp(orientation_type, "portrait") && raw_height <= raw_width
    continue
  elseif strcmp(orientation_type, "square") && raw_height ~= raw_width
    continue
  endif

  %% check if orientation_type == 'any'
  if strcmp(orientation_type, "any")
    target_W2H_ratio = raw_W2H_ratio;
    if raw_height <= raw_width
       target_resized_height = target_resize_length;
       target_resized_width = round(target_resize_length * target_W2H_ratio);
    else
       target_resized_width = target_resize_length;
       target_resized_height = round(target_resize_length / target_W2H_ratio);
    endif  
  endif
  resized_width = target_resized_width;
  resized_height = target_resized_height;
  resized_W2H_ratio = target_W2H_ratio;


  %% resize shortest image dimension relative to target ratio, crop other dimension taking center portion
  resize_scale = 1.0;
  if raw_W2H_ratio <= resized_W2H_ratio %% raw image is too skinny (or just right), fix width and crop height
    resize_scale = resized_width / raw_width;
    resized_image_tmp = imresize(raw_image, resize_scale);
    if size(resized_image_tmp, 2) == resized_width  %% bingo
       ;
    elseif size(resized_image_tmp, 2) == resized_width + 1  %% resize can overshoot by 1 pixel so just crop
      resized_image_tmp = resized_image_tmp(:,1:resized_width,:);
    elseif size(resized_image_tmp, 2) == resized_width - 1  %% if resize undershoots by 1 pixel mirror the last column
      resized_image_tmp = [resized_image_tmp; resized_image_tmp(:,resized_width-1,:)];
    else %% give up
      num_resized_failed = num_resized_failed + 1;
      disp("resize failed")
      disp(["i_raw = ", num2str(i_raw)])
      disp(["raw_name = ", raw_name])
      disp(["raw_height = ", num2str(raw_height), ", raw_width = ", num2str(raw_width)]);
      continue;
    endif
    resized_row_excess = size(resized_image_tmp, 1) - resized_height;
    resized_row_start = floor(resized_row_excess/2);
    resized_row_stop = size(resized_image_tmp, 1) - ceil(resized_row_excess/2);
    resized_col_start = 1;
    resized_col_stop = size(resized_image_tmp, 2);
    resized_image = resized_image_tmp(resized_row_start+1:resized_row_stop,:,:);
  elseif raw_W2H_ratio > resized_W2H_ratio %% raw image is too short, fix height and crop width
    resize_scale = resized_height / raw_height;
    resized_image_tmp = imresize(raw_image, resize_scale);
    if size(resized_image_tmp, 1) == resized_height   %% bingo
       ; 
    elseif size(resized_image_tmp, 1) == resized_height + 1  %% resize can overshoot by 1 pixel so just crop
      resized_image_tmp = resized_image_tmp(1:resized_height,:,:);
    elseif size(resized_image_tmp, 2) == resized_width - 1  %% if resize undershoots by 1 pixel mirror the last column
      resized_image_tmp = [resized_image_tmp; resized_image_tmp(resized_height-1,:,:)];
    else %% give up
      num_resized_failed = num_resized_failed + 1;
      disp("resize failed")
      disp(["i_raw = ", num2str(i_raw)])
      disp(["raw_name = ", raw_name])
      disp(["raw_height = ", num2str(raw_height), ", raw_width = ", num2str(raw_width)]);
      continue;
    endif
    resized_col_excess = size(resized_image_tmp, 2) - resized_width;
    resized_col_start = floor(resized_col_excess/2);
    resized_col_stop = size(resized_image_tmp, 2) - ceil(resized_col_excess/2);
    resized_row_start = 1;
    resized_row_stop = size(resized_image_tmp, 1);
    resized_image = resized_image_tmp(:, resized_col_start+1:resized_col_stop,:);
  endif
  if size(resized_image,1) ~= resized_height || size(resized_image,2) ~= resized_width
    disp("resize failed")
    disp(["i_raw = ", num2str(i_raw)])
    disp(["raw_name = ", raw_name])
    disp(["size(resized_image,1) = ", num2str(size(resized_image,1)), ", size(resized_image,2) = ", num2str(size(resized_image,2))]);
    keyboard;
  %error("resize failed");
  endif

%%  [padded_image] = addMirrorBCRGB(resized_image, border_padding);
  [padded_image] = addMirrorBC2(resized_image, border_padding);

  %% fix bounding boxes
  % read annotation
  annotation_file = fullfile(annotation_path,[raw_name, annotation_ext]);
  if ~exist(annotation_file, "file")
    continue;
  endif
  num_annotated = num_annotated + 1;
  %% write padded image to file, since this will be cropped to size(resized_image)
  resized_image_path = fullfile(train_path, [raw_name, resized_ext]);
  imwrite(uint8(padded_image), resized_image_path);
  fputs(resized_fid, [resized_image_path, "\n"]);
  resized_filepathnames{num_annotated} = resized_image_path;
  
  rec=PASreadrecord(annotation_file);
  num_bbox = length(rec.objects);
  %% xmin, ymin, xmax, ymax
  %% 0,0 at top left corner
  bbox_xmin = 1;
  bbox_xmax = 3;
  bbox_ymin = 2;
  bbox_ymax = 4;
  if plot_bbox
    bbox_image = resized_image;
  endif
  if mask_flag
    mask_image = zeros(size(resized_image));;
  endif
  classID_struct = struct;
  classID_struct.time = double(num_annotated);
  classID_struct.values = [];
  num_active = 0;
  for i_bbox = 1 : num_bbox
    if ~imageNet_synset_flag
      class_name = rec.objects(i_bbox).class;
    else
      bbox_wnid = rec.objects(i_bbox).class;
      glossary_ndx = strmatch(bbox_wnid, glossary_wnid);
      class_name_ndx = strfind(glossary_word{glossary_ndx,1}, ',');
      if isempty(class_name_ndx)
       class_name_ndx = length(glossary_word{glossary_ndx,1})+1;
      endif
      class_name = [bbox_wnid, '_', glossary_word{glossary_ndx,1}(1:class_name_ndx-1)]; 
    endif
    raw_bbox = rec.objects(i_bbox).bbox;
    resized_bbox = zeros(size(raw_bbox));
    class_ndx = strmatch(class_name,VOCopts.classes);
    class_color = zeros(1,1,3);
    class_color(:) = getClassColor(class_ID_list(class_ndx));
    %% subtract border crop
    resized_bbox(bbox_xmin) = raw_bbox(bbox_xmin) - left_border_crop; 
    resized_bbox(bbox_xmax) = raw_bbox(bbox_xmax) - left_border_crop;  
    resized_bbox(bbox_ymin) = raw_bbox(bbox_ymin) - top_border_crop; 
    resized_bbox(bbox_ymax) = raw_bbox(bbox_ymax) - top_border_crop;  
    %% apply resize scale
    %%resized_bbox = resized_bbox .* resize_scale;
    resized_bbox(bbox_xmin) = floor(0.5+(resized_bbox(bbox_xmin)-0.5) .* resize_scale);
    resized_bbox(bbox_xmax) = ceil(0.5+(resized_bbox(bbox_xmax)-0.5) .* resize_scale);
    resized_bbox(bbox_ymin) = floor(0.5+(resized_bbox(bbox_ymin)-0.5) .* resize_scale);
    resized_bbox(bbox_ymax) = ceil(0.5+(resized_bbox(bbox_ymax)-0.5) .* resize_scale);
    %% apply final crop
    resized_bbox(bbox_xmin) = resized_bbox(bbox_xmin) - resized_col_start + 1; 
    resized_bbox(bbox_xmax) = resized_bbox(bbox_xmax) - resized_col_start + 1;  
    resized_bbox(bbox_ymin) = resized_bbox(bbox_ymin) - resized_row_start + 1; 
    resized_bbox(bbox_ymax) = resized_bbox(bbox_ymax) - resized_row_start + 1;  
    resized_bbox(resized_bbox < 1) = 1;
    if resized_bbox(bbox_xmax) > resized_width 
      resized_bbox(bbox_xmax) = resized_width;
    endif
    if resized_bbox(bbox_ymax) > resized_height 
      resized_bbox(bbox_ymax) = resized_height;
    endif
    if (resized_bbox(bbox_ymax) <= (resized_bbox(bbox_ymin) + min_bbox_size)) || (resized_bbox(bbox_xmax) <= (resized_bbox(bbox_xmin) + min_bbox_size))
      continue
    endif
    CSV_bbox_str = [num2str(num_annotated), ", "];
    CSV_bbox_str = [CSV_bbox_str, num2str(resized_bbox(bbox_xmin)), ", ", num2str(resized_bbox(bbox_ymin)), ", "];
    CSV_bbox_str = [CSV_bbox_str, num2str(resized_bbox(bbox_xmin)), ", ", num2str(resized_bbox(bbox_ymax)), ", "];
    CSV_bbox_str = [CSV_bbox_str, num2str(resized_bbox(bbox_xmax)), ", ", num2str(resized_bbox(bbox_ymax)), ", "];
    CSV_bbox_str = [CSV_bbox_str, num2str(resized_bbox(bbox_xmax)), ", ", num2str(resized_bbox(bbox_ymin)), ", "];
    CSV_bbox_str = [CSV_bbox_str, class_name];
    CSV_bbox_str = [CSV_bbox_str, ", ", CSV_filler_str];
    CSV_bbox_str = [CSV_bbox_str, ", ", raw_name];
    CSV_bbox_str = [CSV_bbox_str, "\n"];
    fputs(CSV_fid, CSV_bbox_str);
    if plot_bbox
      bbox_image(resized_bbox(bbox_ymin):resized_bbox(bbox_ymax), resized_bbox(bbox_xmin),1) = class_color(1);
      bbox_image(resized_bbox(bbox_ymin):resized_bbox(bbox_ymax), resized_bbox(bbox_xmax),1) = class_color(1);
      bbox_image(resized_bbox(bbox_ymin), resized_bbox(bbox_xmin):resized_bbox(bbox_xmax),1) = class_color(1);
      bbox_image(resized_bbox(bbox_ymax), resized_bbox(bbox_xmin):resized_bbox(bbox_xmax),1) = class_color(1);
      bbox_image(resized_bbox(bbox_ymin):resized_bbox(bbox_ymax), resized_bbox(bbox_xmin),2) = class_color(2);
      bbox_image(resized_bbox(bbox_ymin):resized_bbox(bbox_ymax), resized_bbox(bbox_xmax),2) = class_color(2);
      bbox_image(resized_bbox(bbox_ymin), resized_bbox(bbox_xmin):resized_bbox(bbox_xmax),2) = class_color(2);
      bbox_image(resized_bbox(bbox_ymax), resized_bbox(bbox_xmin):resized_bbox(bbox_xmax),2) = class_color(2);
      bbox_image(resized_bbox(bbox_ymin):resized_bbox(bbox_ymax), resized_bbox(bbox_xmin),3) = class_color(3);
      bbox_image(resized_bbox(bbox_ymin):resized_bbox(bbox_ymax), resized_bbox(bbox_xmax),3) = class_color(3);
      bbox_image(resized_bbox(bbox_ymin), resized_bbox(bbox_xmin):resized_bbox(bbox_xmax),3) = class_color(3);
      bbox_image(resized_bbox(bbox_ymax), resized_bbox(bbox_xmin):resized_bbox(bbox_xmax),3) = class_color(3);
    endif
    mask_bbox = zeros(size(resized_image));
    mask_bbox_x = [resized_bbox(bbox_xmin), resized_bbox(bbox_xmin), resized_bbox(bbox_xmax), resized_bbox(bbox_xmax), resized_bbox(bbox_xmin)];
    mask_bbox_y = [resized_bbox(bbox_ymin), resized_bbox(bbox_ymax), resized_bbox(bbox_ymax), resized_bbox(bbox_ymin), resized_bbox(bbox_ymin)];
    mask_bbox = poly2mask(mask_bbox_x, mask_bbox_y, size(resized_image,1), size(resized_image,2));
    if mask_flag
      mask_bbox_color = repmat(mask_bbox, [1, 1, 3]) .* repmat(class_color, [size(resized_image,1), size(resized_image,2), 1]); 
      mask_image(mask_bbox_color>0) = mask_bbox_color(mask_bbox_color>0);
    endif

    %% build sparse activity vector
    num_active_bbox = nnz(mask_bbox);
    num_active = num_active + num_active_bbox;
    mask_bbox_ndx2D = find(mask_bbox(:));
    [mask_bbox_row, mask_bbox_col] = ...
    ind2sub([resized_height, ...
	     resized_width], ...
	    mask_bbox_ndx2D);
    mask_bbox_ndx3D = sub2ind([num_classes, ...
			       resized_width, ...
			       resized_height], ...
			      repmat(class_ndx, num_active_bbox, 1), ...
			      mask_bbox_col(:), ...
			      mask_bbox_row(:));
    classID_struct.values = [classID_struct.values(:); ...
			     uint32(mask_bbox_ndx3D(:)-1)];
    if numel(classID_struct.values) ~= num_active
      keyboard;
    endif

    %% write chip
    if ~chip_flag
       continue;
    endif
    chip_classID_path = fullfile(chip_path, class_name);
    mkdir(chip_classID_path);
    chip_image_path = fullfile(chip_classID_path, [raw_name, class_name, "_", num2str(i_bbox), chip_ext]);
    chip_dimensions_rows = [resized_bbox(bbox_ymin):resized_bbox(bbox_ymax)];
    chip_dimensions_cols = [resized_bbox(bbox_xmin):resized_bbox(bbox_xmax)];
    chip_image = resized_image(chip_dimensions_rows, chip_dimensions_cols, :);
    if plot_bbox
      figure(chip_fig);
      imagesc(chip_image);
      axis off; axis image; box off;
      keyboard
    endif
    imwrite(uint8(chip_image), chip_image_path);


  endfor %% i_bbox
  classID_data{num_annotated,1} = classID_struct;
  
  if plot_bbox
    figure(bbox_fig)
    imagesc(bbox_image);
    axis off; axis image; box off;
    figure(mask_fig);
    imagesc(mask_image);
    axis off; axis image; box off;
    keyboard;
  endif
  if mask_flag
    mask_image_path = fullfile(mask_path, [raw_name, mask_ext]);
    imwrite(uint8(mask_image), mask_image_path);
    fputs(mask_fid, [mask_image_path, "\n"]);  
  endif
endfor %% i_raw
fclose(resized_fid)
if mask_flag
  fclose(mask_fid)
endif
disp(["num_raw = ", num2str(num_raw)]);
disp(["num_non_RGB = ", num2str(num_non_RGB)]);
disp(["num_annotated = ", num2str(num_annotated)]);
disp(["num_resized_failed = ", num2str(num_resized_failed)]);


if ~imageNet_synset_flag && num_annotated > 0
  classID_file = [resized_list(1:strfind(resized_list,"_list")-1), ".pvp"]; %%
  writepvpsparseactivityfile(classID_file, classID_data, resized_width, resized_height, num_classes);
endif

