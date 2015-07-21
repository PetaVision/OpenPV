
close all
clear all
setenv('GNUTERM', 'x11');
%%fh = figure;
%%close(fh);

image_dir = "~/Pictures/";
image_type = "png";

target_dir = ...
    [image_dir, "ImageNet/truck/"];  %%
target_subdir = ...
    [ target_dir, "standard/" ];
target_path = ...
    [target_subdir, '*.', image_type];

distractor_flag = 0;
if distractor_flag
  distractor_dir = ...
      [image_dir, "ImageNet/automobile/"]; 
  distractor_subdir = ...
      [ distractor_dir, 'original/' ];
  distractor_path = ...
      [distractor_subdir, '*.', image_type];
endif

sigma_canny = 1.0;%0.5;%
if sigma_canny == 1
  canny_name = 'canny_sigma_1_0/';
elseif sigma_canny == 0.5
  canny_name = 'canny_sigma_0_5/';
endif

canny_target_dir = ...
    [ target_dir, canny_name ];
if ~exist( canny_target_dir, 'dir')
  [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', canny_target_dir); 
  if SUCCESS ~= 1
    error(MESSAGEID, MESSAGE);
  endif
endif

if distractor_flag
  canny_distractor_dir = ...
      [ distractor_dir, canny_name ];
  if ~exist( canny_distractor_dir, 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', canny_distractor_dir); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif
endif

DoG_flag = 1;
if DoG_flag
  amp_center = 1;
  sigma_center = sigma_canny;
  amp_surround = 1;
  sigma_surround = 2 * sigma_canny;
  
  DoG_target_dir = ...
      [ target_dir, 'DoG/' ];
  if ~exist( DoG_target_dir, 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', DoG_target_dir); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif
  
  if distractor_flag
    DoG_distractor_dir = ...
	[ distractor_dir, 'DoG/' ];
    if ~exist( DoG_distractor_dir, 'dir')
      [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', DoG_distractor_dir); 
      if SUCCESS ~= 1
	error(MESSAGEID, MESSAGE);
      endif
    endif
  endif

endif

mask_flag = 0;
if mask_flag
  mask_dir = ...
      [ target_dir, 'masks/' ];
  
  mask_canny_target_dir = ...
      [ target_dir, 'mask_canny/' ];
  if ~exist( mask_canny_target_dir, 'dir')
    [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', mask_canny_target_dir); 
    if SUCCESS ~= 1
      error(MESSAGEID, MESSAGE);
    endif
  endif
  
  if distractor_flag
    mask_canny_distractor_dir = ...
	[ distractor_dir, 'mask_canny/' ];
    if ~exist( 'mask_canny_distractor_dir', 'dir')
      [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', mask_canny_distractor_dir); 
      if SUCCESS ~= 1
	error(MESSAGEID, MESSAGE);
      endif
    endif
  endif

  if DoG_flag
    
    mask_DoG_target_dir = ...
	[ target_dir, 'mask_DoG/' ];
    if ~exist( 'mask_DoG_target_dir', 'dir')
      [SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', mask_DoG_target_dir); 
      if SUCCESS ~= 1
	error(MESSAGEID, MESSAGE);
      endif
    endif
    
    if distractor_flag
      mask_DoG_distractor_dir = ...
	  [ target_dir, 'mask_DoG/' ];
      if ~exist( 'mask_DoG_distractor_dir', 'dir')
	[SUCCESS,MESSAGE,MESSAGEID] = feval( 'mkdir', mask_DoG_distractor_dir); 
	if SUCCESS ~= 1
	  error(MESSAGEID, MESSAGE);
	endif
      endif
    endif
    
  endif
  
endif

image_struct = cell(2,1);
num_images = zeros(2,1);
for target_flag = 1 : distractor_flag + 1
  if target_flag == 1
    [image_struct{1}] = dir(target_path);
    image_subdir = target_subdir;
  else
    [image_struct{2}] = dir(distractor_path);
    image_subdir = distractor_subdir;
  endif
  num_images(target_flag) = size(image_struct{target_flag},1);
  disp(['num_images = ', num2str(num_images(target_flag))]);
  for i_image = 1 : num_images(target_flag)
    image_name = image_struct{target_flag}(i_image).name;
    base_name = image_name(1:strfind(image_name, [".", image_type])-1);
    image_name = [image_subdir, image_name];
    %%    [image_color, image_map, image_alpha] = ...
    image_color = ...
	imread(image_name);
    image_gray = col2gray(image_color);
    image_margin = 8;
    extended_image_gray = addMirrorBC(image_gray, image_margin); 
    [extended_image_canny, image_orient] = ...
	canny(extended_image_gray, sigma_canny);
    %% [image_canny] = grayBorder(image_canny, canny_margin_width, canny_gray_val);
    image_canny = ...
	extended_image_canny(image_margin+1:end-image_margin, ...
			     image_margin+1:end-image_margin);
    if target_flag == 1
      canny_filename = ...
	  [canny_target_dir, base_name];
    else
      canny_filename = ...
	  [canny_distractor_dir, base_name];
    endif
    savefile2(canny_filename, image_canny);

    if DoG_flag == 1
      [extended_image_DoG, DoG_gray_val] = ...
	  DoG(extended_image_gray, ...
	      amp_center, ...
	      sigma_center, ...
	      amp_surround, ...
	      sigma_surround);
      image_DoG = ...
	  extended_image_DoG(image_margin+1:end-image_margin, ...
			     image_margin+1:end-image_margin);
      %%[image_DoG] = grayBorder(image_DoG, DoG_margin_width, DoG_gray_val);
      if target_flag == 1
	DoG_filename = ...
	    [DoG_target_dir, base_name];
      else
	DoG_filename = ...
	    [DoG_distractor_dir, base_name];
      endif
      savefile2(DoG_filename, image_DoG);
    endif

    if mask_flag
      if target_flag == 1
	mask_base_name = base_name;
	mask_canny_image_dir = mask_canny_target_dir;
	if DoG_flag
	  mask_DoG_image_dir = mask_DoG_target_dir;
	endif
      else
	i_image_rand = ceil(rand()*num_images(1));
	mask_image_name = image_struct{1}(i_image_rand).name;
	mask_base_name = mask_image_name(1:strfind(mask_image_name, ...
						   ['.', image_type])-1);
	mask_canny_image_dir = mask_canny_distractor_dir;
	if DoG_flag
	  mask_DoG_image_dir = mask_DoG_distractor_dir;
	endif
      endif
      mask_base_name = [mask_base_name, "_mask"];
      mask_image_name = [mask_dir, mask_base_name, ".", image_type];    
      if exist( mask_image_name, "file")
	mask_image_color = ...
	    imread(mask_image_name);
	mask_image_gray = col2gray(mask_image_color);

	canny_gray_val = 0;
	mask_image_canny = ...
	    image_canny .* (mask_image_gray == 0) + ...
	    canny_gray_val .* (mask_image_gray > 0);
	mask_canny_filename = ...
	    [mask_canny_image_dir, base_name, "_mask"];
	savefile2(mask_canny_filename, mask_image_canny);
	
	if DoG_flag
	  mask_image_DoG = ...
	      image_DoG .* (mask_image_gray == 0) + ...
	      DoG_gray_val .* (mask_image_gray > 0);
	  mask_DoG_filename = ...
	      [mask_DoG_image_dir, base_name, "_mask"];
	  savefile2(mask_DoG_filename, mask_image_DoG);
	endif
      else
	disp(["~exist( 'mask_image_name', ""file"") ", mask_image_name]);
      endif %% exist( 'mask_image_name', "file")
    endif  %% mask_flag
    
    if mod( i_image, ceil(num_images/10) ) == 1
      disp(['i_image = ', num2str(i_image)]);
      fh = figure('name', 'Original');
      imagesc(image_gray); colormap(gray);
      fh = figure('name', 'canny');
      imagesc(image_canny); colormap(gray);
      if DoG_flag == 1
	fh = figure('name', 'DoG');
	imagesc(image_DoG); colormap(gray);
      endif
      if mask_flag
	if exist( mask_image_name, "file")
	  fh = figure('name', 'mask_canny');
	  imagesc(mask_image_canny); colormap(gray);
	  if DoG_flag == 1
	    fh = figure('name', 'mask_DoG');
	    imagesc(mask_image_DoG); colormap(gray);
	  endif
	endif
      endif

    endif
    
  endfor %% i_image

endfor  %% target_flag

