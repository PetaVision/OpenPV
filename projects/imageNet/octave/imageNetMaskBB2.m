function [mask_image, scale_x, scale_y] = ...
      imageNetMaskBB2(original_image, original_pathname, original_info, xml_pathname)

  global VERBOSE_FLAG
  global GRABCUT_FLAG
  global MASKS_DIR
  global TMP_DIR
  global TMP_MASK_DIR
  
  if ~exist ("VERBOSE_FLAG") || isempty (VERBOSE_FLAG)
    VERBOSE_FLAG = 0;
  endif
  
  if ~exist ("GRABCUT_FLAG") || isempty (GRABCUT_FLAG)
    GRABCUT_FLAG = 1;
  endif

  if ~exist ("MASKS_DIR") || isempty (MASKS_DIR)
    MASKS_DIR = "/Users/dylanpaiton/Documents/Work/LANL/Image_Net/Database/img/masks/";
    mkdir (MASKS_DIR);
  endif

  if ~exist ("TMP_DIR") || isempty (TMP_DIR)
    TMP_DIR = "/Users/dylanpaiton/Documents/Work/LANL/tmp";
    mkdir (TMP_DIR);
  endif

  if ~exist ("TMP_MASK_DIR") || isempty (TMP_MASK_DIR)
    TMP_MASK_DIR = "/Users/dylanpaiton/Documents/Work/LANL/tmp_masks";
    mkdir (TMP_MASK_DIR);
  endif

  mask_image = [];
  %%BB_mask = zeros(size(original_image));
  BB_mask = [];
  scale_x = 1;
  scale_y = 1;
  
  original_height = original_info.Height;
  original_width = original_info.Width;
  fid = fopen(xml_pathname, "r");
  if fid < 0
    keyboard
  endif
  xml_str  = ...
      fscanf(fid, "%s", Inf);
  fclose(fid);
  
  width_ndx1 = strfind(xml_str, "<width>");
  width_ndx2 = strfind(xml_str, "</width>");
  annotation_width = str2num(xml_str(width_ndx1+7:width_ndx2-1));
  
  height_ndx1 = strfind(xml_str, "<height>");
  height_ndx2 = strfind(xml_str, "</height>");
  annotation_height = str2num(xml_str(height_ndx1+8:height_ndx2-1));
  
  scale_x = original_width / annotation_width;
  scale_y = original_height / annotation_height;
  
  object_ndx = strfind(xml_str, "<object>");
  bndbox_ndx = strfind(xml_str, "<bndbox>");
  num_object = length(object_ndx);
  if num_object > 1
    disp(["num_object = ", num2str(num_object)]);
  endif
  num_bndbox = length(bndbox_ndx);
  if num_bndbox > 1
    disp(["num_bndbox = ", num2str(num_bndbox)]);
    %%keyboard;
  endif
  
  xmin_ndx1 = strfind(xml_str, "<xmin>");
  xmin_ndx2 = strfind(xml_str, "</xmin>");
  
  xmax_ndx1 = strfind(xml_str, "<xmax>");
  xmax_ndx2 = strfind(xml_str, "</xmax>");
  
  ymin_ndx1 = strfind(xml_str, "<ymin>");
  ymin_ndx2 = strfind(xml_str, "</ymin>");
  
  ymax_ndx1 = strfind(xml_str, "<ymax>");
  ymax_ndx2 = strfind(xml_str, "</ymax>");
  
  num_BB = 0;
  BB_ndx = zeros(num_bndbox,1);
  BB_list = zeros(num_bndbox,4);
  for i_bndbox = 1 : num_bndbox
    BB_xmin = str2num(xml_str(xmin_ndx1(i_bndbox)+6:xmin_ndx2(i_bndbox)-1));
    BB_xmin = BB_xmin + 1; %% BB indices are 0 based
    BB_xmax = str2num(xml_str(xmax_ndx1(i_bndbox)+6:xmax_ndx2(i_bndbox)-1));
    BB_xmax = BB_xmax + 1; %% BB indices are 0 based
    BB_ymin = str2num(xml_str(ymin_ndx1(i_bndbox)+6:ymin_ndx2(i_bndbox)-1));
    BB_ymin = BB_ymin + 1; %% BB indices are 0 based
    BB_ymax = str2num(xml_str(ymax_ndx1(i_bndbox)+6:ymax_ndx2(i_bndbox)-1));
    BB_ymax = BB_ymax + 1; %% BB indices are 0 based
    
    if scale_x ~= 1.0
      BB_xmin = round(BB_xmin * scale_x);
      BB_xmax = round(BB_xmax * scale_x);
    endif
    if scale_y ~= 1.0
      BB_ymin = round(BB_ymin * scale_y);
      BB_ymax = round(BB_ymax * scale_y);
    endif
    
    if BB_xmin < 1
      continue;
    endif
    if BB_xmax > size(original_image, 2)
      continue;
    endif
    if BB_ymin < 1
      continue;
    endif
    if BB_ymax > size(original_image, 1)
      continue;
    endif
    
    num_BB = num_BB + 1;
    BB_ndx = num_BB;
    BB_list(BB_ndx,:) = [BB_xmin, BB_xmax, BB_ymin, BB_ymax];

    annotation_area = annotation_width * annotation_height;
    BB_SIZE_FLAG = 0;
    for i_BB = 1 : num_BB
        BB_width = BB_list(i_BB,2) - BB_list(i_BB,1);
        BB_height = BB_list(i_BB,4) - BB_list(i_BB,3);
        BB_area = BB_width * BB_height;
        if BB_area/annotation_area > 0.95
            BB_SIZE_FLAG = 1;
        endif
    endfor

    if GRABCUT_FLAG && ~BB_SIZE_FLAG
      num_grabcut_iterations = 4;
      original_filename = strFolderFromPath(original_pathname);
      tmp_original_pathname =  [TMP_DIR, original_filename];
      xml_filename = strFolderFromPath(xml_pathname);
      base_filename = strRemoveExtension(xml_filename);
      mask_filename = strcat(base_filename, ".jpg");
      mask_pathname = strcat(MASKS_DIR, mask_filename);
      tmp_mask_pathname = [TMP_MASK_DIR, mask_filename];
      copyfile(original_pathname, tmp_original_pathname);
      %% grabcut is typically in either /opt/local/bin or /usr/local/bin
      grabcut_cmd = ...
	  sprintf("../grabcut/grabcut %i %s %s %i %i %i %i", ...
		  num_grabcut_iterations, ...
		  tmp_original_pathname, ...
		  tmp_mask_pathname, ...
		  BB_xmin, BB_ymin, BB_xmax, BB_ymax);
      if VERBOSE_FLAG
	    disp(["grabcut_cmd = ", grabcut_cmd]);
      endif
      system(grabcut_cmd);
      if exist(tmp_mask_pathname, "file")
	   BB_mask_tmp = imread(tmp_mask_pathname);
	   delete(tmp_mask_pathname);
	   if isempty(BB_mask_tmp)	  
	     BB_mask_tmp = zeros(size(original_image));
	     BB_mask_tmp(BB_ymin:BB_ymax, BB_xmin:BB_xmax,:) = 255;
	   endif
      else
	    BB_mask_tmp = zeros(size(original_image));
	    BB_mask_tmp(BB_ymin:BB_ymax, BB_xmin:BB_xmax,:) = 255;
      endif
      if exist(tmp_original_pathname, "file")
	    delete(tmp_original_pathname);
      endif
    else %% ~GRABCUT_FLAG
	     BB_mask_tmp = zeros(size(original_image,1),size(original_image,2));
         BB_mask_tmp(BB_ymin:BB_ymax, BB_xmin:BB_xmax,:) = 255;
    endif
    BB_mask_tmp = squeeze(sum(BB_mask_tmp(:,:,1:end),3));
    if isempty(BB_mask)
      BB_mask = 255 * (BB_mask_tmp > 0);
    else
      %%BB_mask = squeeze(sum(BB_mask(:,:,1:end),3));
      BB_mask = 255 * ((BB_mask_tmp > 0) | (BB_mask > 0));
    endif
  endfor  %% i_bndbox
  if num_BB > 0
    mask_image = ...
	original_image .* ...
	repmat((BB_mask > 0), [1,1,size(original_image,3)]);
    mask_image = uint8(mask_image);
    BB_mask = uint8(BB_mask);
  endif
  
endfunction %% imageNetMaskBB
