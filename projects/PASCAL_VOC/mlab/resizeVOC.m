%% script for resizing PASCAL VOC images
%% user specifies one of 3 orientation types, landscape, portriat or square,
%% along with the new dimensions for
%% all images whose original dimensions are consistent with the
%% specified orientation_type.
%% Images are resized so that the "shorest"
%% resized dimension, measured as a fraction of the target ratio,
%% exactly matches the specified dimension, with the other
%% "longer" dimension cropped symmetrically so as to match the
%% specified size.
%% Optional padding can be added around all borders using mirror BCs.
%% All resized images have the same final size.
%% uses VOCdevkit
%% also generates CSV file giving corners of each bounding box in
%% resized dimensions, using NeoVision2 format (see init CSV file fbelow)
%% also generates a mask for each image with the class ID of each
%% pixel encoded by the activating of the corresponding bit
%% also generates a list of full paths to each resized image
%%
%% author: Garrett T. Kenyon, garkenyon@gmail.com ~2014
%%

clear all
close all
more off
setenv("GNUTERM","X11")

addpath("~/workspace/PetaVision/mlab/imgProc");

%% specify target image type and dimensions
orientation_type = "landscape"; %%"square"; %%  "portrait"; %%
disp(["orientation_type 1= ", orientation_type]);
if strcmp(orientation_type, "landscape")
  resized_height = 192; %%360;
  resized_width = 256; %%480;
elseif strcmp(orientation_type, "portrait")
  resized_height = 256; %%480;
  resized_width = 192; %%360;
else
  resized_height = 256; %%420;
  resized_width = 256; %%420;
endif
border_padding = 0; %%8;

%% diagnostic flags, causes figures to be displayed to screen rather
%% than only written to image files
plot_bbox = false;
if plot_bbox 
  bbox_fig = figure;
  set(bbox_fig, 'name', 'bbox');
  mask_fig = figure;
  set(mask_fig, 'name', 'mask');
endif

%% format to save corresponding images as (png, jpg, tiff, etc)
resized_ext = '.png';
mask_ext = '.png';

%% set results paths
workspace_path = "~/workspace";
PASCAL_VOC_path = [workspace_path, filesep, "PASCAL_VOC"];
mkdir(PASCAL_VOC_path);
VOC_dataset = "VOC2007";
VOC_dataset_path = fullfile(PASCAL_VOC_path, VOC_dataset);
mkdir(PASCAL_VOC_path, VOC_dataset);
train_dir = "padded"; %%"train"
train_path = fullfile(PASCAL_VOC_path, VOC_dataset, train_dir);
mkdir(train_path);
mask_dir = "mask"
mask_path = fullfile(PASCAL_VOC_path, VOC_dataset, mask_dir);
mkdir(mask_path);
addpath([PASCAL_VOC_path, filesep, "mlab"]);

%% collect PASCAL_VOC specific stuff here...
%% set VOCdev paths
VOCdevkit_path = fullfile(PASCAL_VOC_path, "VOCdevkit");
if ~exist(VOCdevkit_path, "dir")
  error("VOCdevkit_path does not exist: ", VOCdevkit_path);
endif
addpath(VOCdevkit_path);
VOCcode_path = fullfile(VOCdevkit_path, "VOCcode");
addpath(VOCcode_path);

% initialize VOC options
VOCinit;

%%ids=textread(sprintf(VOCopts.imgsetpath,'train'),'%s');
[annotation_path, ~, annotation_ext] = fileparts(VOCopts.annopath);
%% end PASCAL_VOC specific stuff

%% colormap for category masks, 1 color for each class_ID
log2_max_class_ndx = 24;
max_class_ndx = 2^log2_max_class_ndx;
color_sep = floor(max_class_ndx / VOCopts.nclasses); %% separation
class_ID_list = (1:VOCopts.nclasses) * color_sep;

%% save labels as sparse PVP file with nf = # object classess
num_classes = VOCopts.nclasses;
util_path = [workspace_path, filesep, "PetaVision", filesep, "mlab", filesep, "util"];
addpath(util_path);
classID_data = cell(); %% size of cell is number of resized images, which
%% we don't know yet...so grow classID_data dynamically

%% load list of raw images
training_path = fullfile(VOCdevkit_path, VOC_dataset, "JPEGImages");
if ~exist(training_path, "dir")
  error(["training_path does not exist: ", training_path]);
end
raw_list = glob([training_path, filesep, "*.jpg"]);
num_raw = length(raw_list)

%% open text files for storing lists of resized images and class ID masks
resized_list = [VOC_dataset_path, filesep, VOC_dataset, "_padded", ...x
		num2str(border_padding), "_", orientation_type, "_list.txt"];
resized_fid = fopen(resized_list, "w", "native");
mask_list = [resized_list(1:strfind(resized_list,"_list")-1),"_mask_list.txt"]; %%[VOC_dataset_path, filesep, VOC_dataset, "_mask_", orientation_type, "_list.txt"];
mask_fid = fopen(mask_list, "w", "native");


%% initialize CSV file (DARPA Neovision2 format)
CSV_file = [resized_list(1:strfind(resized_list,"_list")-1),".csv"]; %%[VOC_dataset_path, filesep, VOC_dataset, "_padded_", orientation_type, ".csv"];
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
resized_W2H_ratio = resized_width / resized_height;
num_resized = 0;
num_resized_failed = 0;
num_non_RGB = 0;
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
  %% keep only images with correct orientation_type
  if strcmp(orientation_type, "landscape") && raw_height >= raw_width
    continue
  elseif strcmp(orientation_type, "portrait") && raw_height <= raw_width
    continue
  elseif strcmp(orientation_type, "square") && raw_height ~= raw_width
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

  %% resize shortest image dimension relative to target ratio, crop other dimension taking center portion
  raw_W2H_ratio = raw_width / raw_height;
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

  %% write padded image to file, since this will be cropped to size(resized_image)
  resized_image_path = fullfile(train_path, [raw_name, resized_ext]);
  imwrite(uint8(padded_image), resized_image_path);
  fputs(resized_fid, [resized_image_path, "\n"]);
  num_resized = num_resized + 1;
  
  %% fix bounding boxes
  % read annotation
  rec=PASreadrecord(fullfile(annotation_path,[raw_name, annotation_ext]));
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
  mask_image = zeros(size(resized_image));;
  classID_struct = struct;
  classID_struct.time = double(num_resized);
  classID_struct.values = [];
  num_active = 0;
  for i_bbox = 1 : num_bbox
    mask_bbox = zeros(size(resized_image));
    class_name = rec.objects(i_bbox).class;
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
    CSV_bbox_str = [num2str(num_resized), ", "];
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
    mask_bbox_x = [resized_bbox(bbox_xmin), resized_bbox(bbox_xmin), resized_bbox(bbox_xmax), resized_bbox(bbox_xmax), resized_bbox(bbox_xmin)];
    mask_bbox_y = [resized_bbox(bbox_ymin), resized_bbox(bbox_ymax), resized_bbox(bbox_ymax), resized_bbox(bbox_ymin), resized_bbox(bbox_ymin)];
    mask_bbox = poly2mask(mask_bbox_x, mask_bbox_y, size(resized_image,1), size(resized_image,2));
    mask_bbox_color = repmat(mask_bbox, [1, 1, 3]) .* repmat(class_color, [size(resized_image,1), size(resized_image,2), 1]); 
    mask_image(mask_bbox_color>0) = mask_bbox_color(mask_bbox_color>0);

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
  endfor %% i_bbox
  classID_data{num_resized,1} = classID_struct;
  
  if plot_bbox
    figure(bbox_fig)
    imagesc(bbox_image);
    axis off; axis image; box off;
    figure(mask_fig);
    imagesc(mask_image);
    axis off; axis image; box off;
    keyboard;
  endif
  mask_image_path = fullfile(mask_path, [raw_name, mask_ext]);
  imwrite(uint8(mask_image), mask_image_path);
  fputs(mask_fid, [mask_image_path, "\n"]);  
endfor %% i_raw
fclose(resized_fid)
fclose(mask_fid)
disp(["num_raw = ", num2str(num_raw)]);
disp(["num_non_RGB = ", num2str(num_non_RGB)]);
disp(["num_resized = ", num2str(num_resized)]);
disp(["num_resized_failed = ", num2str(num_resized_failed)]);

classID_file = [resized_list(1:strfind(resized_list,"_list")-1), "_classID.pvp"]; %%
writepvpsparseactivityfile(classID_file, classID_data, resized_width, resized_height, num_classes);

