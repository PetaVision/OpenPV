function [mask_image, BB_mask, scale_x, scale_y] = ...
      imageNetMaskBB(original_image, original_info, xml_file)

  global NUM_FIGS

  mask_image = [];
  BB_mask_tmp = zeros(size(original_image));
  scale_x = 1;
  scale_y = 1;
  
  original_height = original_info.Height;
  original_width = original_info.Width;
  fid = fopen(xml_file, "r");
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
    
    BB_mask_tmp(BB_ymin:BB_ymax, BB_xmin:BB_xmax,:) = 255;
    
  endfor  %% i_bndbox
  if num_BB > 0
    mask_image = original_image .* (BB_mask_tmp > 0);
    mask_image = uint8(mask_image);
    BB_mask = uint8(BB_mask_tmp);
  else
    return;
  endif
  if num_BB > 1 && NUM_FIGS < 11  
    imageNetDrawBB(original_image, BB_list);
  endif
  
endfunction %% imageNetMaskBB