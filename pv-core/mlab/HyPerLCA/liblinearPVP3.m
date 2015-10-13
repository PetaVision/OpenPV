
%% Quick and dirty visualization harness for ground truth pvp files
%% each classID is assigned a different color
%% where bounding boxes overlapp, the color is a mixture
%% use this script to visualize ground truth sparse pvp files and for comparison with 
%% original images to verify that the bounding box annotations are reasonable
%% hit any key to advance to the next image
close all
more off
pkg load all
setenv("GNUTERM","X11")
%%setenv("GNUTERM","aqua")
addpath("~/openpv/pv-core/mlab/imgProc");
addpath("~/openpv/pv-core/mlab/util");
addpath("~/openpv/pv-core/mlab/HyPerLCA");
%%addpath("/shared/liblinear-2.1/matlab");
addpath("~/Desktop/liblinear-2.01/matlab");

plot_flag = true;
%%run_type = "ICA";
%%run_type = "ICAX4"
run_type = "ICAX16"
%%run_type = "S1S2"
%%run_type = "DCA";
%%run_type = "scene"
if strcmp(run_type, "ICA")
  output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_1536_ICA/VOC2007_landscape17";
elseif strcmp(run_type, "ICAX4")
  output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1X4_1536_ICA/VOC2007_landscape11";
elseif strcmp(run_type, "ICAX16")
  output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1X16_1536_ICA/VOC2007_landscape6";
elseif strcmp(run_type, "S1S2")
  output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_96_S2_1536/VOC2007_landscape29";
elseif strcmp(run_type, "DCA")
  output_dir = "/home/gkenyon/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DCA/VOC2007_landscape7";
elseif strcmp(run_type, "scene")
  output_dir = "/Volumes/mountData/scene/scene_S1X4_1536_ICA/mount_rushmore1";
  %%output_dir = "/Volumes/mountData/scene/scene_S1X4_1536_ICA/half_dome_yosemite1";
  %%output_dir = "/Volumes/mountData/scene/scene_S1X4_1536_ICA/el_capitan_yosemite1";
endif


%%draw reconstructed image
Recon_flag = false; %%true;
if Recon_flag
DoG_weights = [];
if strcmp(run_type, "ICA")
  Recon_list = {[""],  ["Image"]};
elseif strcmp(run_type, "ICAX4")
  Recon_list = {[""],  ["Image"]};
elseif strcmp(run_type, "ICAX16")
  Recon_list = {[""],  ["Image"]};
elseif strcmp(run_type, "S1S2")
  Recon_list = {[""],  ["Image"]};
elseif strcmp(run_type, "DCA")
  Recon_list = {[""],  ["Image"]; [""], ["ImageDeconS1"]; [""], ["ImageDeconS2"]; [""], ["ImageDeconS3"]; [""], ["ImageDecon"]; [""], ["ImageDeconError"] };
elseif strcmp(run_type, "scene")
  Recon_list = {["a3_"],  ["Image"]; ["a0_"],  ["ImageReconS1"]};
endif
%% list of layers to unwhiten
num_Recon_list = size(Recon_list,1);
Recon_unwhiten_list = zeros(num_Recon_list,1);
%% list of layers to use as a normalization reference for unwhitening
Recon_normalize_list = 1:num_Recon_list;
%% list of (previous) layers to sum with current layer
Recon_sum_list = cell(num_Recon_list,1);
num_Recon_frames_per_layer = 1;
Recon_LIFO_flag = true;
[Recon_hdr, ...
 Recon_fig, ...
 Recon_fig_name, ...
 Recon_vals,  ...
 Recon_time, ...
 Recon_mean,  ...
 Recon_std] = ...
analyzeUnwhitenedReconPVP(Recon_list, ...
			  num_Recon_frames_per_layer, ...
			  output_dir, plot_flag, ...
			  Recon_sum_list, ...
			  Recon_LIFO_flag);
drawnow;
endif %% Recon_flag

%% GT activity
GT_flag = true;
num_GT_images = 7958;
displayPeriod = 1200;
nx_GT = 1;
ny_GT = 1;
if strcmp(run_type, "ICA")
  GT_list ={[""], ["GroundTruth"]};
elseif strcmp(run_type, "ICAX4")
  GT_list ={[""], ["GroundTruth"]};
elseif strcmp(run_type, "ICAX16")
  GT_list ={[""], ["GroundTruth"]};
elseif strcmp(run_type, "S1S2")
  GT_list ={[""], ["GroundTruth"]};
elseif strcmp(run_type, "DCA")
  GT_list ={[""], ["GroundTruth"]};
endif
GT_file = [output_dir, filesep, GT_list{1,1}, GT_list{1,2}, ".pvp"];
GT_fid = fopen(GT_file);
GT_hdr = readpvpheader(GT_fid);
fclose(GT_file);
nx_GT = GT_hdr.nx;
ny_GT = GT_hdr.ny;
num_GT_frames = GT_hdr.nbands;
SVM_flag = true;
fraction_GT_frames_read = 1;
num_slack = 24;
GT_slack = num_slack;
Sparse_slack = GT_slack;
num_GT_images = min(num_GT_images, num_GT_frames - GT_slack);
numEpochs = floor(num_GT_frames/num_GT_images);
min_GT_skip = max(1, num_GT_frames - num_GT_images*numEpochs - GT_slack);
min_Sparse_skip = min_GT_skip;
fraction_GT_progress = 10;
num_GT_epochs = 1;
num_GT_procs = 1;
num_epochs = num_GT_epochs;
num_procs = num_GT_procs;
GT_frames_list = [];
if GT_flag
  load_GT_flag = false;
  [GT_hdr, ...
   GT_hist_rank_array, ...
   GT_times_array, ...
   GT_percent_active_array, ...
   GT_percent_change_array, ...
   GT_std_array, ...
   GT_struct_array, ...
   GT_max_val_array, ...
   GT_min_val_array, ...
   GT_mean_val_array, ...
   GT_std_val_array, ...
   GT_median_val_array] = ...
  analyzeSparseEpochsPVP3(GT_list, ...
			  output_dir, ...
			  load_GT_flag, ...
			  plot_flag, ...
			  fraction_GT_frames_read, ...
			  min_GT_skip, ...
			  fraction_GT_progress, ...
			  GT_frames_list, ...
			  num_procs, ...
			  num_epochs);
  drawnow;

  %% GT Recon Error
  if strcmp(run_type, "ICA")
    nonSparse_list = {[""], ["GroundTruthReconS1Error"]};
  elseif strcmp(run_type, "ICAX4")
    nonSparse_list = {[""], ["GroundTruthReconS1Error"]};
  elseif strcmp(run_type, "ICAX16")
    nonSparse_list = {[""], ["GroundTruthReconS1Error"]};
  elseif strcmp(run_type, "S1S2")
    nonSparse_list = {[""], ["GroundTruthReconS1Error"]; [""], ["GroundTruthReconS2Error"]; [""], ["GroundTruthReconS1S2Error"]};
  elseif strcmp(run_type, "DCA")
    nonSparse_list = {[""], ["GroundTruthReconS1Error"]; [""], ["GroundTruthReconS2Error"]; [""], ["GroundTruthReconS3Error"]};
  elseif strcmp(run_type, "scene")
    nonSparse_list = {["a0_"], ["ImageReconS1Error"]};
  endif
  num_nonSparse_list = size(nonSparse_list,1);
  nonSparse_skip = repmat(1, num_nonSparse_list, 1);
  nonSparse_norm_strength = ones(num_nonSparse_list,1);
  GT_std_ndx = ones(num_nonSparse_list,1);
  if strcmp(run_type, "ICA")
    nonSparse_norm_list = {[""], ["GroundTruth"]};
  elseif strcmp(run_type, "ICAX4")
    nonSparse_norm_list = {[""], ["GroundTruth"]};
  elseif strcmp(run_type, "ICAX16")
    nonSparse_norm_list = {[""], ["GroundTruth"]};
  elseif strcmp(run_type, "S1S2") || strcmp(run_type, "DCA")
    nonSparse_norm_list = {[""], ["GroundTruth"]; [""], ["GroundTruth"]; [""], ["GroundTruth"]};
  endif
  fraction_nonSparse_frames_read = 1;
  min_nonSparse_skip = min_GT_skip;
  fraction_nonSparse_progress = 10;
  [nonSparse_times_array, ...
   nonSparse_RMS_array, ...
   nonSparse_norm_RMS_array, ...
   nonSparse_RMS_fig] = ...
  analyzeNonSparsePVP(nonSparse_list, ...
		      nonSparse_skip, ...
		      nonSparse_norm_list, ...
		      nonSparse_norm_strength, ...
		      GT_times_array, ...
		      GT_std_array, ...
		      GT_std_ndx, ...
		      output_dir, plot_flag, fraction_nonSparse_frames_read, min_nonSparse_skip, fraction_nonSparse_progress);
  for i_nonSparse = 1 : num_nonSparse_list
    figure(nonSparse_RMS_fig(i_nonSparse));
    grid on
    set(gca, 'linewidth', 2.0, 'color', [1 1 1]);
  endfor
  drawnow;
endif %% GT_flag


%% Sparse activity
load_SparseHistPool_flag = false
if strcmp(run_type, "ICA")
  Sparse_list ={[""], ["S1"]};
elseif strcmp(run_type, "ICAX4")
  Sparse_list ={[""], ["S1"]};
elseif strcmp(run_type, "ICAX16")
  Sparse_list ={[""], ["S1"]};
elseif strcmp(run_type, "S1S2")
  Sparse_list ={[""], ["S1"]; [""], ["S2"]};
elseif strcmp(run_type, "DCA")
  Sparse_list ={[""], ["S1"]; [""], ["S2"]; [""], ["S3"]};
elseif strcmp(run_type, "scene")
  Sparse_list ={["a2_"], ["S1"]};
endif
fraction_Sparse_frames_read = 1;
min_Sparse_skip = min_GT_skip;
fraction_Sparse_progress = 10;
num_epochs = num_GT_epochs;
num_procs = num_GT_procs;
Sparse_frames_list = [];
load_Sparse_flag = false;
if ~load_SparseHistPool_flag
[Sparse_hdr, ...
 Sparse_hist_rank_array, ...
 Sparse_times_array, ...
 Sparse_percent_active_array, ...
 Sparse_percent_change_array, ...
 Sparse_std_array, ...
 Sparse_struct_array, ...
 Sparse_max_val_array, ...
 Sparse_min_val_array, ...
 Sparse_mean_val_array, ...
 Sparse_std_val_arra, ...
 Sparse_median_val_array] = ...
analyzeSparseEpochsPVP3(Sparse_list, ...
			output_dir, ...
			load_Sparse_flag, ...
			plot_flag, ...
			fraction_Sparse_frames_read, ...
			min_Sparse_skip, ...
			fraction_Sparse_progress, ...
			Sparse_frames_list, ...
			num_procs, ...
			num_epochs);
drawnow;
num_Sparse_hist_pool_bins = 4+3; %%6+3; %%8+3;
save_Sparse_hist_pool_flag = false;
[Sparse_hist_pool_hdr, ...
 Sparse_hist_pool_array, ...
 Sparse_hist_pool_times_array, ...
 Sparse_max_pool_array, ...
 Sparse_mean_pool_array] = ...
analyzeSparseHistPoolEpochsPVP2(Sparse_list, ...
			       output_dir, ...
			       Sparse_hist_rank_array, ...
			       load_Sparse_flag, ...
			       plot_flag, ...
			       fraction_Sparse_frames_read, ...
			       min_Sparse_skip, ...
			       fraction_Sparse_progress, ...
			       Sparse_min_val_array, ...
			       Sparse_max_val_array, ...
			       Sparse_mean_val_array, ...
			       Sparse_std_val_arra, ...
			       Sparse_median_val_array, ...
			       nx_GT, ny_GT, ...
			       num_Sparse_hist_pool_bins, ...
			       save_Sparse_hist_pool_flag, ...
			       num_procs, ...
			       num_epochs);

				%pause;
else
  Sparse_dir = [output_dir, filesep, "Sparse"];
  for i_Sparse = 1 : length(Sparse_list)
    
  endfor %% i_Sparse
endif %% load_SparseHistPool_flag

GT_flag = true;
if GT_flag
  VOC_classes={'background'; 'aeroplane'; 'bicycle'; 'bird'; 'boat'; 'bottle'; 'bus'; 'car'; 'cat'; 'chair'; 'cow'; 'diningtable'; 'dog'; 'horse'; 'motorbike'; 'person'; 'pottedplant'; 'sheep'; 'sofa'; 'train'; 'tvmonitor'};
  num_VOC_classes = length(VOC_classes);

  target_class_indices = [0 1 2 3 6 7 8 12 13 14 15 19]+1; %%[2:numel(VOC_classes)]; %%
  target_classes = VOC_classes(target_class_indices)
  num_target_classes = length(target_classes);
  SVM_flag = true;
  if SVM_flag
    svm_dir = [output_dir, filesep, "svm"];
    mkdir(svm_dir);
    
    %% training label vector
    GT_progress_step = ceil(num_GT_frames / fraction_GT_progress);
    [GT_data, GT_hdr] = readpvpfile(GT_file, GT_progress_step, num_GT_frames, min_GT_skip, 1);
    num_GT_frames = length(GT_data);
    if ~exist("num_GT_images") || isempty(num_GT_images) || num_GT_images <= 0
      num_GT_images = num_GT_frames;
    endif
    last_GT_frame = num_GT_frames - floor(GT_slack/2);  
    first_GT_frame = last_GT_frame - num_GT_images + 1;
    i_GT_image = 0;
    training_label_vector_array = zeros(ny_GT, nx_GT, num_GT_images, num_target_classes);
    for i_GT_frame = first_GT_frame : last_GT_frame
      i_GT_image = i_GT_image + 1;
      active_indices = GT_data{i_GT_frame}.values(:,1);
      [active_kf, active_kx, active_ky] = ind2sub([num_VOC_classes, nx_GT, ny_GT], active_indices+1);
      active_ndx = sub2ind([ny_GT, nx_GT, num_VOC_classes], active_ky, active_kx, active_kf);
      GT_frame = zeros(ny_GT, nx_GT, num_VOC_classes);
      GT_frame(active_ndx) = 1;
      GT_frame = reshape(GT_frame, [ny_GT, nx_GT, num_VOC_classes]);
      %% gotta do 1 vs all since bounding boxes overlapp
      for i_target_class = 1 : num_target_classes
	i_VOC_class = target_class_indices(i_target_class);
	training_label_vector_array(:,:,i_GT_image, i_target_class) = GT_frame(:,:,i_VOC_class);
      endfor %% class_ndx
    endfor %% i_frame
    training_label_pos = cell(num_target_classes,1);
    training_label_neg = cell(num_target_classes,1);
    pos_labels_ndx = cell(num_target_classes,1);
    neg_labels_ndx = cell(num_target_classes,1);
    neg_labels_rank = cell(num_target_classes,1);
    for i_target_class = 1 : num_target_classes
      training_label_vector = ...
      reshape(training_label_vector_array(:,:,:,i_target_class), [ny_GT * nx_GT * num_GT_images, 1]);
      pos_labels_ndx{i_target_class} = find(training_label_vector);
      neg_labels_ndx{i_target_class} = find(~training_label_vector);
      neg_pos_ratio = length(neg_labels_ndx{i_target_class})/length(pos_labels_ndx{i_target_class})
      pos_labels_ndx{i_target_class} = repmat(pos_labels_ndx{i_target_class}, ceil(neg_pos_ratio), 1);;
      pos_labels_ndx{i_target_class} = pos_labels_ndx{i_target_class}(1:length(neg_labels_ndx{i_target_class}));
      %%[neg_labels_sorted, neg_labels_rank{i_target_class}] = sort(rand(length(neg_labels_ndx{i_target_class}),1));
      training_label_pos{i_target_class} = repmat(training_label_vector(pos_labels_ndx{i_target_class}), ceil(neg_pos_ratio), 1);
      training_label_pos{i_target_class} = training_label_pos{i_target_class}(1:length(neg_labels_ndx{i_target_class}));
      %%training_label_neg{i_target_class} = training_label_vector(neg_labels_ndx{i_target_class}(neg_labels_rank{i_target_class}(1:neg_pos_ratio*length(pos_labels_ndx{i_target_class}))));
      training_label_neg{i_target_class} = training_label_vector(neg_labels_ndx{i_target_class});
    endfor %% class_ndx
    
    %%traing_instance_matrix
    liblinear_options_str = ['-s 0 -C -B 1'];
    num_Sparse_list = size(Sparse_list,1);
    training_hist_pool_matrix_array = cell(num_Sparse_list + (num_Sparse_list>1),1);
    training_max_pool_matrix_array = cell(num_Sparse_list + (num_Sparse_list>1),1);
    training_mean_pool_matrix_array = cell(num_Sparse_list + (num_Sparse_list>1),1);
    %%training_combo_pool_matrix_array = cell(num_Sparse_list + (num_Sparse_list>1),1);
    nf_Sparse_array = zeros(num_Sparse_list + (num_Sparse_list > 1),1);
    first_Sparse_frame_array = zeros(num_Sparse_list,1);
    last_Sparse_frame_array = zeros(num_Sparse_list,1);
    num_Sparse_frames_array = zeros(num_Sparse_list,1);
    xval_model_hist_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
    xval_model_max_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
    xval_model_mean_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
    %%xval_model_combo_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
    for i_Sparse = 1 : (num_Sparse_list + (num_Sparse_list > 1))
      if i_Sparse <= num_Sparse_list
	nf_Sparse_array(i_Sparse) = Sparse_hist_pool_hdr{i_Sparse}.nf;
	num_Sparse_frames_array(i_Sparse) = size(Sparse_hist_pool_array{i_Sparse},1);
	last_Sparse_frame_array(i_Sparse) = num_Sparse_frames_array(i_Sparse);
	first_Sparse_frame_array(i_Sparse) = num_Sparse_frames_array(i_Sparse) - num_GT_images + 1;
	last_Sparse_time = Sparse_hist_pool_times_array{i_Sparse}(last_Sparse_frame_array(i_Sparse));
	while last_Sparse_time > GT_data{last_GT_frame}.time
	  last_Sparse_frame_array(i_Sparse) = last_Sparse_frame_array(i_Sparse) - 1;
	  first_Sparse_frame_array(i_Sparse) = first_Sparse_frame_array(i_Sparse) - 1;
	  if first_Sparse_frame_array(i_Sparse) < 1
	    error(["first_Sparse_frame_array(", num2str(i_Sparse), ")", "=", num2str(first_Sparse_frame_array(i_Sparse)), " < 1"])
	  endif
	  last_Sparse_time = Sparse_hist_pool_times_array{i_Sparse}(last_Sparse_frame_array(i_Sparse));
	endwhile
	while last_Sparse_time < GT_data{last_GT_frame}.time
	  last_Sparse_frame_array(i_Sparse) = last_Sparse_frame_array(i_Sparse) + 1;
	  first_Sparse_frame_array(i_Sparse) = first_Sparse_frame_array(i_Sparse) + 1;
	  if last_Sparse_frame_array(i_Sparse) > num_Sparse_frames_array(i_Sparse)
	    error(["last_Sparse_frame_array(", num2str(i_Sparse), ")", "=", num2str(last_Sparse_frame_array(i_Sparse)), " > ", "num_Sparse_frames = ", num2str(num_Sparse_frames_array(i_Sparse))])
	  endif
	  last_Sparse_time = Sparse_hist_pool_times_array{i_Sparse}(last_Sparse_frame_array(i_Sparse));
	endwhile
      else
	nf_Sparse_array(num_Sparse_list+1) = prod(nf_Sparse_array(:));
	Sparse_list{num_Sparse_list+1,1} = [];
	Sparse_list{num_Sparse_list+1,2} = Sparse_list{1,2};
	for j_Sparse = 2 : num_Sparse_list
	  Sparse_list{num_Sparse_list+1,2} = [Sparse_list{num_Sparse_list+1,2},Sparse_list{j_Sparse,2}];
	endfor %% j_Sparse
      endif %% i_Sparse > 1
      %%keyboard;
      for i_target_class = 1 : num_target_classes
	disp(["training model for target class = ", target_classes{i_target_class}, " using ", Sparse_list{i_Sparse,2}]);
	if i_Sparse <= num_Sparse_list	  
	  training_hist_pool_matrix = zeros(num_Sparse_hist_pool_bins, nf_Sparse_array(i_Sparse), ny_GT, nx_GT, num_GT_images);
	  training_max_pool_matrix = zeros(nf_Sparse_array(i_Sparse), ny_GT, nx_GT, num_GT_images);
	  training_mean_pool_matrix = zeros(nf_Sparse_array(i_Sparse), ny_GT, nx_GT, num_GT_images);
	  %%training_combo_pool_matrix = zeros(num_Sparse_hist_pool_bins+1, nf_Sparse_array(i_Sparse), ny_GT, nx_GT, num_GT_images);
	  for i_Sparse_frame = first_Sparse_frame_array(i_Sparse) : last_Sparse_frame_array(i_Sparse)
	    j_Sparse_frame = i_Sparse_frame - first_Sparse_frame_array(i_Sparse) + 1;
	    training_hist_pool_matrix(:, :, :, :, j_Sparse_frame) = ...
	    Sparse_hist_pool_array{i_Sparse}{i_Sparse_frame};
	    training_max_pool_matrix(:, :, :, j_Sparse_frame) = ...
	    Sparse_max_pool_array{i_Sparse}{i_Sparse_frame};
	    training_mean_pool_matrix(:, :, :, j_Sparse_frame) = ...
	    Sparse_mean_pool_array{i_Sparse}{i_Sparse_frame};
	    %%training_combo_pool_matrix(1:num_Sparse_hist_pool_bins, :, :, :, j_Sparse_frame) = ...
	    %%Sparse_hist_pool_array{i_Sparse}{i_Sparse_frame};
	    %%training_combo_pool_matrix(num_Sparse_hist_pool_bins+1, :, :, :, j_Sparse_frame) = ...
	    %%Sparse_max_pool_array{i_Sparse}{i_Sparse_frame};
	  endfor %% i_Sparse_frame
	  training_hist_pool_matrix = ...
	  sparse(reshape(training_hist_pool_matrix, ...
			 [num_Sparse_hist_pool_bins *  nf_Sparse_array(i_Sparse), ny_GT * nx_GT * num_GT_images]));
	  training_max_pool_matrix = ...
	  sparse(reshape(training_max_pool_matrix, ...
			 [nf_Sparse_array(i_Sparse), ny_GT * nx_GT * num_GT_images]));
	  training_mean_pool_matrix = ...
	  sparse(reshape(training_mean_pool_matrix, ...
			 [nf_Sparse_array(i_Sparse), ny_GT * nx_GT * num_GT_images]));
	%%training_combo_pool_matrix = ...
	%%sparse(reshape(training_combo_pool_matrix, ...
	%%		 [(num_Sparse_hist_pool_bins+1) *  nf_Sparse_array(i_Sparse), ny_GT * nx_GT * num_GT_images]));
	  training_hist_pool_pos = training_hist_pool_matrix(:,pos_labels_ndx{i_target_class});
	  %%training_hist_pool_neg = training_hist_pool_matrix(:, neg_labels_ndx{i_target_class}(neg_labels_rank{i_target_class}(1:neg_pos_ratio*length(pos_labels_ndx{i_target_class}))));
	  training_hist_pool_neg = training_hist_pool_matrix(:,neg_labels_ndx{i_target_class});
	  training_max_pool_pos = training_max_pool_matrix(:,pos_labels_ndx{i_target_class});
	  %%training_max_pool_neg = training_max_pool_matrix(:, neg_labels_ndx{i_target_class}(neg_labels_rank{i_target_class}(1:neg_pos_ratio*length(pos_labels_ndx{i_target_class}))));
	  training_max_pool_neg = training_max_pool_matrix(:, neg_labels_ndx{i_target_class});
	  training_mean_pool_pos = training_mean_pool_matrix(:,pos_labels_ndx{i_target_class});
	  %%training_mean_pool_neg = training_mean_pool_matrix(:, neg_labels_ndx{i_target_class}(neg_labels_rank{i_target_class}(1:neg_pos_ratio*length(pos_labels_ndx{i_target_class}))));
	  training_mean_pool_neg = training_mean_pool_matrix(:, neg_labels_ndx{i_target_class});
	  %%training_combo_pool_pos = training_combo_pool_matrix(:,pos_labels_ndx{i_target_class});
	  %%training_combo_pool_neg = training_combo_pool_matrix(:, neg_labels_ndx{i_target_class}(neg_labels_rank{i_target_class}(1:neg_pos_ratio*length(pos_labels_ndx{i_target_class}))));
	  %%training_combo_pool_neg = training_combo_pool_matrix(:, neg_labels_ndx{i_target_class});
	  
	  xval_model_hist_pool_array{i_Sparse, i_target_class} = ...
	  train([training_label_pos{i_target_class}; training_label_neg{i_target_class}], ...
		[training_hist_pool_pos, training_hist_pool_neg], ...
		liblinear_options_str, 'col');
	  xval_model_max_pool_array{i_Sparse, i_target_class} = ...
	  train([training_label_pos{i_target_class}; training_label_neg{i_target_class}], ...
		[training_max_pool_pos, training_max_pool_neg], ...
		liblinear_options_str, 'col');
	  xval_model_mean_pool_array{i_Sparse, i_target_class} = ...
	  train([training_label_pos{i_target_class}; training_label_neg{i_target_class}], ...
		[training_mean_pool_pos, training_mean_pool_neg], ...
		liblinear_options_str, 'col');
	%%xval_model_combo_pool_array{i_Sparse, i_target_class} = ...
	%%train([training_label_pos{i_target_class}; training_label_neg{i_target_class}], ...
	%%	[training_combo_pool_pos, training_combo_pool_neg], ...
	  %%	liblinear_options_str, 'col');

	  
	  
 	  if num_Sparse_list>1
	    if i_Sparse == 1
	      training_hist_pool_matrix_array{num_Sparse_list+1, i_target_class} = ...
	      [training_hist_pool_pos, training_hist_pool_neg];
	      training_max_pool_matrix_array{num_Sparse_list+1, i_target_class} = ...
	      [training_max_pool_pos, training_max_pool_neg];
	      training_mean_pool_matrix_array{num_Sparse_list+1, i_target_class} = ...
	      [training_mean_pool_pos, training_mean_pool_neg];
	      %%training_combo_pool_matrix_array{num_Sparse_list+1, i_target_class} = ...
	      %%[training_combo_pool_pos, training_combo_pool_neg];
	    else
	      training_hist_pool_matrix_array{num_Sparse_list+1, i_target_class} = ...
	      [training_hist_pool_matrix_array{num_Sparse_list+1, i_target_class}; ...
	       [training_hist_pool_pos, training_hist_pool_neg]];
	      training_max_pool_matrix_array{num_Sparse_list+1, i_target_class} = ...
	      [training_max_pool_matrix_array{num_Sparse_list+1, i_target_class}; ...
	       [training_max_pool_pos, training_max_pool_neg]];
	      training_mean_pool_matrix_array{num_Sparse_list+1, i_target_class} = ...
	      [training_mean_pool_matrix_array{num_Sparse_list+1, i_target_class}; ...
	       [training_mean_pool_pos, training_mean_pool_neg]];
	      %%training_combo_pool_matrix_array{num_Sparse_list+1, i_target_class} = ...
	      %%[training_combo_pool_matrix_array{num_Sparse_list+1, i_target_class}; ...
	      %% [training_combo_pool_pos, training_combo_pool_neg]];
	    endif
	  endif
	else
	  xval_model_hist_pool_array{num_Sparse_list+1, i_target_class} = ...
	  train([training_label_pos{i_target_class}; training_label_neg{i_target_class}], ...
		training_hist_pool_matrix_array{num_Sparse_list+1, i_target_class}, ...
		liblinear_options_str, 'col');
	  xval_model_max_pool_array{num_Sparse_list+1, i_target_class} = ...
	  train([training_label_pos{i_target_class}; training_label_neg{i_target_class}], ...
		training_max_pool_matrix_array{num_Sparse_list+1, i_target_class}, ...
		liblinear_options_str, 'col');
	  xval_model_mean_pool_array{num_Sparse_list+1, i_target_class} = ...
	  train([training_label_pos{i_target_class}; training_label_neg{i_target_class}], ...
		training_mean_pool_matrix_array{num_Sparse_list+1, i_target_class}, ...
		liblinear_options_str, 'col');
	%%xval_model_combo_pool_array{num_Sparse_list+1, i_target_class} = ...
	%%train([training_label_pos{i_target_class}; training_label_neg{i_target_class}], ...
	%%	training_combo_pool_matrix_array{num_Sparse_list+1, i_target_class}, ...
	%%	liblinear_options_str, 'col');
	endif %% num_Sparse_list > 1
      endfor %% class_ndx
    endfor %% i_Sparse
    
    save([svm_dir, filesep, "svm.txt"], "target_classes", "xval_model_hist_pool_array", "xval_model_max_pool_array", "xval_model_combo_pool_array");
    if plot_flag
      num_pool = 3;
      pool_types = {"hist"; "max"; "mean"};
      %%pool_types = {"hist"; "max"; "combo"};
      xval_model_array = zeros(num_target_classes, num_Sparse_list + (num_Sparse_list>1), num_pool);
      for i_pool = 1:num_pool
	for i_target_class = 1 : num_target_classes
	  for i_Sparse = 1 : (num_Sparse_list + (num_Sparse_list > 1))
	    if strcmp(pool_types{i_pool},"hist")
	      xval_model_array(i_target_class, i_Sparse, i_pool) = ...
	      xval_model_hist_pool_array{i_Sparse, i_target_class}(2);
	    elseif strcmp(pool_types{i_pool},"max")
	      xval_model_array(i_target_class, i_Sparse, i_pool) = ...
	      xval_model_max_pool_array{i_Sparse, i_target_class}(2);
	    elseif strcmp(pool_types{i_pool},"mean")
	      xval_model_array(i_target_class, i_Sparse, i_pool) = ...
	      xval_model_mean_pool_array{i_Sparse, i_target_class}(2);
	    %%elseif strcmp(pool_types{i_pool},"combo")
	    %%  xval_model_array(i_target_class, i_Sparse, i_pool) = ...
	    %%  xval_model_combo_pool_array{i_Sparse, i_target_class}(2);
	    endif %% pool_types
	  endfor %% i_Sparse
	endfor  %% i_target_class
      endfor %% i_pool
      max_model = max(xval_model_array(:));
      num_xval_model_rows = max(1,floor(sqrt(num_target_classes)));
      num_xval_model_cols = ceil(num_target_classes/num_xval_model_rows);
      xval_model_fig = figure;
      target_axis = zeros(num_target_classes,1);
      xval_model_fig_name = ["linearSVM", Sparse_list{num_Sparse_list+(num_Sparse_list>1),2}];
      set(xval_model_fig, 'name', xval_model_fig_name);
      for i_target_class = 1 : num_target_classes
	i_xval_model_row = mod(i_target_class-1, num_xval_model_cols) + 1;
	j_xval_model_col = ceil(i_target_class / num_xval_model_cols);
	taget_axis(i_target_class) = subplot(num_xval_model_rows, num_xval_model_cols, i_target_class);
	xval_model_handle = bar(taget_axis(i_target_class), squeeze(xval_model_array(i_target_class, :, :))');
	xval_model_colormap = colormap(prism(length(xval_model_handle)));
	colormap(xval_model_colormap);
	title(taget_axis(i_target_class), target_classes{i_target_class});
	if num_Sparse_list > 1
	  axis(taget_axis(i_target_class), [0.5 (num_Sparse_list+(num_Sparse_list>1)+0.5) 0.5 min(max_model*(1.1),1)]);
	else
	  axis(taget_axis(i_target_class), [0.5 (num_pool+0.5) 0.5 min(max_model*(1.1),1)]);
	endif
	set(gca, 'xticklabel', pool_types);
	if i_target_class == num_target_classes
	  [legend_handle, legend_object, legend_plot, legend_labels] = legend(xval_model_handle, Sparse_list(:,2), 'location', 'northeast');
	endif
      endfor %% i_target_class
      saveas(xval_model_fig, [svm_dir, filesep, xval_model_fig_name, ".png"]);
    endif %% plot_flag
  endif  %% svm_flag





  SLP_flag = true 
  if SLP_flag
    SLP_dir = [output_dir, filesep, "SLP"];
    mkdir(SLP_dir);
    for i_scale = 1 : 1 +  2*(strcmp(run_type, "S1S2") || strcmp(run_type, "DCA"))
      if strcmp(run_type, "ICA") || strcmp(run_type, "ICAX4") || strcmp(run_type, "ICAX16")
	gt_classID_file = fullfile([output_dir, filesep, "GroundTruth.pvp"])
      elseif strcmp(run_type, "S1S2") || strcmp(run_type, "DCA")
	gt_classID_file = fullfile([output_dir, filesep, "GroundTruth.pvp"])
      endif
      if i_scale == 1
	if strcmp(run_type, "ICA") || strcmp(run_type, "ICAX4") || strcmp(run_type, "ICAX16")
	  pred_classID_file = fullfile([output_dir, filesep, "GroundTruthReconS1.pvp"])
	elseif strcmp(run_type, "S1S2") || strcmp(run_type, "DCA")      
	  pred_classID_file = fullfile([output_dir, filesep, "GroundTruthReconS1.pvp"])
	endif
      elseif i_scale == 2
	if strcmp(run_type, "S1S2")  || strcmp(run_type, "DCA")
	  pred_classID_file = fullfile([output_dir, filesep, "GroundTruthReconS2.pvp"])
	endif
      elseif i_scale == 3
	if strcmp(run_type, "S1S2")      
	  pred_classID_file = fullfile([output_dir, filesep, "GroundTruthReconS1S2.pvp"]);
	else
	  pred_classID_file = fullfile([output_dir, filesep, "GroundTruthReconS3.pvp"]);											
	endif    
      endif
      pred_classID_fid = fopen(pred_classID_file);
      pred_classID_hdr = readpvpheader(pred_classID_fid);
      fclose(pred_classID_fid);
      tot_pred_classID_frames = pred_classID_hdr.nbands;
      [pred_data,pred_hdr] = readpvpfile(pred_classID_file, ceil(tot_pred_classID_frames/1), tot_pred_classID_frames, 1, 1); 
      gt_classID_fid = fopen(gt_classID_file);
      gt_classID_hdr = readpvpheader(gt_classID_fid);
      fclose(gt_classID_fid);
      tot_gt_classID_frames = gt_classID_hdr.nbands;
      [gt_data,gt_hdr] = readpvpfile(gt_classID_file, ceil(tot_gt_classID_frames/1), tot_gt_classID_frames, 1, 1); 
      %%[imageRecon_data,imageRecon_hdr] = readpvpfile(imageRecon_file); 
      %true_num_neurons = true_hdr.nf * true_hdr.nx * true_hdr.ny;
      %true_num_frames = length(true_data);
      pred_num_neurons = pred_hdr.nf * pred_hdr.nx * pred_hdr.ny;
      pred_num_frames = length(pred_data);
      gt_num_neurons = gt_hdr.nf * gt_hdr.nx * gt_hdr.ny;
      gt_num_frames = length(gt_data);
      %%imageRecon_num_neurons = imageRecon_hdr.nf * imageRecon_hdr.nx * imageRecon_hdr.ny;
      %%imageRecon_num_frames = length(imageRecon_data);
      classID_hist_bins = -0.25:0.01:2.0;
      num_classID_bins = length(classID_hist_bins);
      pred_classID_hist = zeros(num_classID_bins, length(target_class_indices),2);
      %%pred_classID_sum = zeros(length(target_class_indices), 1);
      %%pred_classID_sum2 = zeros(length(target_class_indices), 1);
      classID_colormap = prism(length(target_class_indices)+0); %%hot(gt_hdr.nf+1); %%rainbow(length(target_class_indices)); %%prism(length(target_class_indices));
      use_false_positive_thresh = false; %%true; %%
      false_positive_thresh = .99;
      display_frame = min(pred_num_frames, gt_num_frames);
      for i_frame = 1 : display_frame
	
	%% ground truth layer is sparse
	if mod(i_frame, ceil(gt_num_frames/10)) == 0
	  display(["i_frame = ", num2str(i_frame)])
	endif
	gt_time = gt_data{i_frame}.time;
	gt_num_active = length(gt_data{i_frame}.values);
	gt_active_ndx = gt_data{i_frame}.values+1;
	gt_active_sparse = sparse(gt_active_ndx,1,1,gt_num_neurons,1,gt_num_active);
	gt_classID_cube = full(gt_active_sparse);
	gt_classID_cube = reshape(gt_classID_cube, [gt_hdr.nf, gt_hdr.nx, gt_hdr.ny]);
	gt_classID_cube = permute(gt_classID_cube, [3,2,1]);
	
	%% only display predictions for these frames
	if i_frame == display_frame 
	  display(["i_frame = ", num2str(i_frame)]);
	  display(["gt_time = ", num2str(gt_time)]);
	  
	  [gt_classID_val, gt_classID_ndx] = max(gt_classID_cube, [], 3);
	  min_gt_classID = min(gt_classID_val(:))
	  %%gt_classID_cube = gt_classID_cube .* (gt_classID_cube >= min_gt_classID);
	  gt_classID_heatmap = zeros(gt_hdr.ny, gt_hdr.nx, 3);
	  i_target_classID = 0;
	  for i_classID = target_class_indices %%1 : gt_hdr.nf
	    i_target_classID = i_target_classID + 1;
	    if ~any(any(gt_classID_cube(:,:,i_classID)))
	      continue;
	    endif
	    gt_class_color = classID_colormap(i_target_classID, :); %%getClassColor(gt_class_color_code);
	    gt_classID_band = repmat(gt_classID_cube(:,:,i_classID), [1,1,3]);
	    gt_classID_band(:,:,1) = gt_classID_band(:,:,1) * gt_class_color(1)*255;
	    gt_classID_band(:,:,2) = gt_classID_band(:,:,2) * gt_class_color(2)*255;
	    gt_classID_band(:,:,3) = gt_classID_band(:,:,3) * gt_class_color(3)*255;
	    gt_classID_heatmap = gt_classID_heatmap + gt_classID_band .* repmat(squeeze(sum(gt_classID_heatmap,3)==0),[1,1,3]);
	  endfor
	  gt_classID_heatmap = floor(mod(gt_classID_heatmap, 256));
	  if plot_flag
	    gt_fig = figure("name", ["Ground Truth: ", num2str(gt_time, "%i")]);
	    image(uint8(gt_classID_heatmap)); axis off; axis image, box off;
	    drawnow
	  endif
	  imwrite(uint8(gt_classID_heatmap), [SLP_dir, filesep, "gt_", num2str(gt_time, "%i"), "_", num2str(i_scale), '.png'], 'png');
	endif %% display_frame == i_frame

	%% recon layer is not sparse
	pred_time = pred_data{i_frame}.time;
	pred_classID_cube = pred_data{i_frame}.values;
	pred_classID_cube = permute(pred_classID_cube, [2,1,3]);
	target_classID_cube = pred_classID_cube(:,:,target_class_indices);
	i_target_classID = 0;
	for i_classID = target_class_indices %%1 : pred_hdr.nf
	  i_target_classID = i_target_classID + 1;
	  pred_classID_tmp = squeeze(pred_classID_cube(:,:,i_classID));
	  gt_classID_tmp = squeeze(gt_classID_cube(:,:,i_classID));
	  pos_pred_tmp = pred_classID_tmp(gt_classID_tmp(:)~=0);
	  neg_pred_tmp = pred_classID_tmp(gt_classID_tmp(:)==0);
	  if any(pos_pred_tmp)
	    pred_classID_hist(:,i_target_classID,1) = squeeze(pred_classID_hist(:,i_target_classID,1)) + hist(pos_pred_tmp(:), classID_hist_bins)';
	  endif
	  if any(neg_pred_tmp)
	    pred_classID_hist(:,i_target_classID,2) = squeeze(pred_classID_hist(:,i_target_classID,2)) + hist(neg_pred_tmp(:), classID_hist_bins)';
	  endif
	endfor
	if i_frame == display_frame
	  display(["pred_time = ", num2str(pred_time)]);
	  pred_classID_cumsum = squeeze(cumsum(pred_classID_hist, 1));
	  pred_classID_sum = squeeze(sum(pred_classID_hist, 1));
	  pred_classID_norm = repmat(reshape(pred_classID_sum, [1,length(target_class_indices),2]), [num_classID_bins,1,1]);
	  pred_classID_cumprob = pred_classID_cumsum ./ (pred_classID_norm + (pred_classID_norm==0));
	  pred_classID_thresh_bin = zeros(length(target_class_indices),1);
	  pred_classID_thresh = zeros(length(target_class_indices),1);
	  pred_classID_true_pos = zeros(length(target_class_indices),1);
	  pred_classID_false_pos = zeros(length(target_class_indices),1);
	  pred_classID_accuracy = zeros(length(target_class_indices),1);
	  i_target_classID = 0;
	  for i_classID = target_class_indices %%1 : pred_hdr.nf
	    i_target_classID = i_target_classID + 1;
	    pos_hist_tmp = squeeze(pred_classID_hist(:,i_target_classID,1)) ./ squeeze(pred_classID_norm(:,i_target_classID,1));
	    neg_hist_tmp = squeeze(pred_classID_hist(:,i_target_classID,2)) ./ squeeze(pred_classID_norm(:,i_target_classID,2));
	    if use_false_positive_thresh
	      pred_classID_bin_tmp = find( pred_classID_cumprob(:,i_target_classID,2)>false_positive_thresh, 1, "first");
	    else
	      pos_hist = pred_classID_hist(:,i_target_classID,1); %%/pred_classID_sum(i_target_classID,1);
	      neg_hist = pred_classID_hist(:,i_target_classID,2); %%/(pred_classID_sum(i_target_classID,2));
	      diff_hist = pos_hist - neg_hist;
	      pred_classID_bin_tmp = find((diff_hist>0).*(classID_hist_bins(:)>0), 1, "first");
	    endif
	    if ~isempty(pred_classID_bin_tmp)
	      pred_classID_thresh_bin(i_target_classID) = pred_classID_bin_tmp;
	      pred_classID_thresh(i_target_classID) = classID_hist_bins(pred_classID_bin_tmp);
	      pred_classID_true_pos(i_target_classID) = (1 - pred_classID_cumprob(pred_classID_bin_tmp,i_target_classID,1));
	      pred_classID_false_pos(i_target_classID) = (pred_classID_cumprob(pred_classID_bin_tmp,i_target_classID,2));
	      pred_classID_accuracy(i_target_classID) = (pred_classID_true_pos(i_target_classID) + pred_classID_false_pos(i_target_classID)) / 2;
	    else
	      pred_classID_thresh(i_target_classID) = classID_hist_bins(end);
	      pred_classID_true_pos(i_target_classID) = 0.0;
	      pred_classID_false_pos(i_target_classID) = 0.0;
	      pred_classID_accuracy(i_target_classID) = 0.0;
	    endif
	  endfor
	  [pred_classID_val, pred_classID_ndx] = max(pred_classID_cube, [], 3);
	  min_pred_classID = min(pred_classID_val(:))
	  [max_pred_classID, max_pred_classID_ndx] = max(pred_classID_val(:))
	  disp(VOC_classes{pred_classID_ndx(max_pred_classID_ndx)});
	  mean_pred_classID = mean(pred_classID_val(:))
	  std_pred_classID = std(pred_classID_val(:))
	  pred_classID_thresh = reshape(pred_classID_thresh, [1,1, length(target_class_indices)]);
	  pred_classID_mask = double(target_classID_cube >= repmat(pred_classID_thresh, [pred_hdr.ny, pred_hdr.nx, 1]));
	  pred_classID_confidences = cell(length(target_class_indices), 1);
	  pred_classID_max_confidence = squeeze(max(squeeze(max(target_classID_cube, [], 2)), [], 1));
	  %% confidence is measured as a percentage relative to threshold
	  %%   scaled by the residual hit rate relative to the false alarm rate
	  pred_classID_max_percent_confidence = (pred_classID_max_confidence(:) - pred_classID_thresh(:)) ./ (pred_classID_thresh(:) + (pred_classID_thresh(:)==0));
	  pred_classID_relative_accuracy = ((pred_classID_true_pos(:) - (1-false_positive_thresh)) ./ (1-pred_classID_false_pos(:)));
	  pred_classID_max_confidence = pred_classID_max_percent_confidence;
	  %%pred_classID_max_confidence = ((pred_classID_true_pos(:) - (1-pred_classID_false_pos(:))) ./ (1-pred_classID_false_pos(:))) .* (pred_classID_max_confidence(:) - pred_classID_thresh(:)) ./ (pred_classID_max_confidence(:) + pred_classID_thresh(:));
	  [pred_classID_sorted_confidence, pred_classID_sorted_ndx] = sort(pred_classID_max_confidence, 'descend');
	  target_confidences = cell(length(target_class_indices),1);
	  i_target_classID = 0;
	  for i_classID = target_class_indices %%1 : pred_hdr.nf
	    i_target_classID = i_target_classID + 1;
	    target_confidences{i_target_classID, 1} = [target_classes{pred_classID_sorted_ndx(i_target_classID)}, ...
						       ', ', num2str(pred_classID_sorted_confidence(i_target_classID)), ...
						       ', ', num2str(pred_classID_thresh(pred_classID_sorted_ndx(i_target_classID))), ...
						       ', ', num2str(pred_classID_accuracy(pred_classID_sorted_ndx(i_target_classID))), ...
						       ', ', num2str(pred_classID_true_pos(pred_classID_sorted_ndx(i_target_classID))), ...
						       ', ', num2str(pred_classID_false_pos(pred_classID_sorted_ndx(i_target_classID)))];
	  endfor
	  if plot_flag
	    pred_classID_heatmap = zeros(pred_hdr.ny, pred_hdr.nx, 3);
	    pred_fig = figure("name", ["Predict: ", num2str(pred_time, "%i")]);
	    image(uint8(pred_classID_heatmap)); axis off; axis image, box off;
	    hold on
	    i_target_classID = 0;
	    for i_classID = target_class_indices %% 1 : pred_hdr.nf
	      i_target_classID = i_target_classID + 1;
	      if ~any(pred_classID_mask(:,:,i_target_classID))
		continue;
	      endif
	      if i_target_classID ~= pred_classID_sorted_ndx(1)
		%%continue;
	      endif
	      pred_class_color = classID_colormap(i_target_classID, :); 
	      pred_classID_band = repmat(pred_classID_mask(:,:,i_target_classID), [1,1,3]);
	      pred_classID_band(:,:,1) = pred_classID_band(:,:,1) * pred_class_color(1)*255;
	      pred_classID_band(:,:,2) = pred_classID_band(:,:,2) * pred_class_color(2)*255;
	      pred_classID_band(:,:,3) = pred_classID_band(:,:,3) * pred_class_color(3)*255;
	      pred_classID_heatmap = pred_classID_heatmap + pred_classID_band .* (pred_classID_heatmap < pred_classID_band);
	      th = text(3, ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)), target_classes{i_target_classID});
	      pred_classID_heatmap(ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)):ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)), 1:2, 1) = pred_class_color(1)*255;
	      pred_classID_heatmap(ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)):ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)), 1:2, 2) = pred_class_color(2)*255;
	      pred_classID_heatmap(ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)):ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)), 1:2, 3) = pred_class_color(3)*255;
	      set(th, 'color', pred_class_color(:));
				%keyboard;
	    endfor
	    pred_classID_heatmap = mod(pred_classID_heatmap, 256);
	    image(uint8(pred_classID_heatmap)); axis off; axis image, box off;
	    drawnow
	    %%get(th)
	  endif %% plot_flag
	  imwrite(uint8(pred_classID_heatmap), [SLP_dir, filesep, "pred_", num2str(pred_time, "%i"), "_", num2str(i_scale), '.png'], 'png');
	  disp(target_confidences)
		%disp([pred_classID_true_pos; pred_classID_false_pos])
		%keyboard
	endif  %% pred_time == Recon_time

	if plot_flag && i_frame == min(pred_num_frames, gt_num_frames)
	  hist_fig = figure("name", ["hist_positive: ", num2str(i_scale), "_", num2str(pred_time, "%i")]);
	  num_subplot_rows = ceil(sqrt(numel(target_class_indices)));
	  num_subplot_cols = ceil(numel(target_class_indices) / num_subplot_rows);
	  i_subplot = 0;
	  i_target_classID = 0;
	  for i_classID  = target_class_indices %% 1 : pred_hdr.nf
	    i_target_classID = i_target_classID + 1;
	    i_subplot = i_subplot + 1;
	    subplot(num_subplot_rows,num_subplot_cols,i_subplot, 'color', [0 0 0])
	    %%
	    pos_hist = squeeze(pred_classID_hist(:,i_target_classID,1)) ./ squeeze(pred_classID_norm(:,i_target_classID,1));
	    hist_width_tmp = round(num_classID_bins/1);
	    bins_tmp = [pred_classID_thresh_bin(i_target_classID)-hist_width_tmp:pred_classID_thresh_bin(i_target_classID)+hist_width_tmp];
	    bins_tmp_fixed = bins_tmp(find(bins_tmp>0,1,"first"):find(bins_tmp<num_classID_bins,1,"last"));
	    bh_pos = bar(classID_hist_bins(bins_tmp_fixed), pos_hist(bins_tmp_fixed), "stacked", "facecolor", "g", "edgecolor", "g");
	    %%hist_fig = figure("name", ["hist_negative: ", num2str(pred_time, "%i")]);
	    axis off
	    box off
	    hold on
	    neg_hist = squeeze(pred_classID_hist(:,i_target_classID,2)) ./ squeeze(pred_classID_norm(:,i_target_classID,2));
	    bh_neg = bar(classID_hist_bins(bins_tmp_fixed), neg_hist(bins_tmp_fixed), "stacked", "facecolor", "r", "edgecolor", "r");
	    max_pos_hist = max(pos_hist(:));
	    max_neg_hist = max(neg_hist(:));
	    lh = line([pred_classID_thresh(i_target_classID) pred_classID_thresh(i_target_classID)], [0 max(max_pos_hist,max_neg_hist)]);
	    set(lh, 'color', [0 0 1])
	    set(lh, 'linewidth', 1.0)
	    title(target_classes{i_target_classID});
	  endfor
	  saveas(hist_fig, [SLP_dir, filesep, "hist_", num2str(i_scale), "_", num2str(pred_time, "%i"), ".png"], "png");
	endif
	
      endfor  %% i_frame
      save([SLP_dir, filesep, "hist_", num2str(pred_time, "%i"), ".mat"], "classID_hist_bins", "pred_classID_hist", "pred_classID_norm", "pred_classID_cumprob", "pred_classID_cumsum")
    endfor  %% i_scale
  endif %% SLP_flag
endif %% GT_flag
