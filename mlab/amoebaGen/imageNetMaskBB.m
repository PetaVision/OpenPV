function [mask_image, BB_mask, scale_x, scale_y] = ...
      imageNetMaskBB(original_image, original_info, xml_file)

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

  xmin_ndx1 = strfind(xml_str, "<xmin>");
  xmin_ndx2 = strfind(xml_str, "</xmin>");
  BB_xmin = str2num(xml_str(xmin_ndx1+6:xmin_ndx2-1));
  BB_xmin = BB_xmin + 1; %% BB indices are 0 based

  xmax_ndx1 = strfind(xml_str, "<xmax>");
  xmax_ndx2 = strfind(xml_str, "</xmax>");
  BB_xmax = str2num(xml_str(xmax_ndx1+6:xmax_ndx2-1));
  BB_xmax = BB_xmax + 1; %% BB indices are 0 based

  ymin_ndx1 = strfind(xml_str, "<ymin>");
  ymin_ndx2 = strfind(xml_str, "</ymin>");
  BB_ymin = str2num(xml_str(ymin_ndx1+6:ymin_ndx2-1));
  BB_ymin = BB_ymin + 1; %% BB indices are 0 based

  ymax_ndx1 = strfind(xml_str, "<ymax>");
  ymax_ndx2 = strfind(xml_str, "</ymax>");
  BB_ymax = str2num(xml_str(ymax_ndx1+6:ymax_ndx2-1));
  BB_ymax = BB_ymax + 1; %% BB indices are 0 based

  scale_x = original_width / annotation_width;
  scale_y = original_height / annotation_height;

  if scale_x ~= 1.0
    BB_xmin = round(BB_xmin * scale_x);
    BB_xmax = round(BB_xmax * scale_x);
  endif
  if scale_y ~= 1.0
    BB_ymin = round(BB_ymin * scale_y);
    BB_ymax = round(BB_ymax * scale_y);
  endif

  BB_mask = zeros(size(original_image));
  BB_mask(BB_ymin:BB_ymax, BB_xmin:BB_xmax,:) = 1;

  mask_image = original_image .* (BB_mask > 0);
  mask_image = uint8(mask_image);
  BB_mask = uint8(BB_mask);
  

endfunction %% imageNetMaskBB