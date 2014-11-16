
%% quick and dirty visualization harness for ground truth pvp files
%% each classID is assigned a different color
%% where bounding boxes overlapp, the color is a mixture
%% use this script to visualize ground truth sparse pvp files and for comparison with 
%% original images to verify that the bounding box annotations are reasonable
%% hit any key to advance to the next image
setenv("GNUTERM","X11")
addpath("~/workspace/PetaVision/mlab/imgProc");
addpath("~/workspace/PetaVision/mlab/util");
classID_file = fullfile("~/workspace/PASCAL_VOC/VOC2007", "VOC2007_padded0_square_classID.pvp"); 
[data,hdr] = readpvpfile(classID_file); 
close all
figure;
num_neurons = hdr.nf * hdr.nx * hdr.ny;
num_frames = length(data);
num_colors = 2^24;
for i_frame = 1 : num_frames
    num_active = length(data{i_frame}.values);
    active_ndx = data{i_frame}.values+1;
    active_sparse = sparse(active_ndx,1,1,num_neurons,1,num_active);
    classID_cube = full(active_sparse);
    classID_cube = reshape(classID_cube, [hdr.nf, hdr.nx, hdr.ny]);
    classID_cube = permute(classID_cube, [3,2,1]);
    classID_heatmap = zeros(hdr.ny, hdr.nx, 3);
    for i_classID = 1 : hdr.nf
	if ~any(classID_cube(:,:,i_classID))
	   continue;
	endif
	class_color_code = i_classID * num_colors / hdr.nf;
	class_color = getClassColor(class_color_code);
	classID_band = repmat(classID_cube(:,:,i_classID), [1,1,3]);
	classID_band(:,:,1) = classID_band(:,:,1) * class_color(1);
	classID_band(:,:,2) = classID_band(:,:,2) * class_color(2);
	classID_band(:,:,3) = classID_band(:,:,3) * class_color(3);
	classID_heatmap = classID_heatmap + classID_band;
    endfor
    classID_heatmap = mod(classID_heatmap, 255);
    image(uint8(classID_heatmap)); axis off; axis image, box off;
    pause;
endfor
