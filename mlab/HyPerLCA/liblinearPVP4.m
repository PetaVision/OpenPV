
%% applies liblinear to sparse coded layers, assumes ground truth is provided as a sparse PVP file (ground truth could also be full, I think)
%% breaks the sparse output into the same number of tiles as are specified in the ground truth pvp file
%% if more than one sparse layer is specified, liblinear is also applied to the concatenation of all sparse layers
%% once a linear SVM has been trained and applied to the sparse representations of all images
%% a ROC like analysis is performed to determine the optimal threshold, or point at which false positives == true_postives
%% if a perceptron is included, output from that is analyzed as well
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
%%run_type = "ICAX2_S1X4";
%%run_type = "ICA_S1X4"
%%run_type = "ICA_S1X16"
run_type = "S1S2"
%%run_type = "DCA";
%%run_type = "scene"
if strcmp(run_type, "ICAX2_S1X4")
  output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1X4_1536_ICAX2/VOC2007_landscape1";
elseif strcmp(run_type, "ICA_S1X4")
  output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1X4_1536_ICA/VOC2007_landscape11";
elseif strcmp(run_type, "ICA_S1X16")
  output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1X16_1536_ICA/VOC2007_landscape6";
elseif strcmp(run_type, "S1S2")
  output_dir = "/Volumes/mountData/PASCAL_VOC/PASCALX3_S1_96_S2_1536/VOC2007_landscape31";
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
  Recon_list = {[""],  ["Image"]};
  if strcmp(run_type, "DCA")
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
GT_flag = true;  %% not currently used
SVM_flag = true;  %% not currently used
hist_pool_flag = false; %%true;
max_pool_flag = true;
mean_pool_flag = false; %%true;
train_long_flag = true; %%false;  %% determines whether all out-of-class examples are used (in-class examples are replicated to match number of out of class examples)

num_GT_images = 7958/1;
displayPeriod = 1200;

nx_GT = 1;
ny_GT = 1;
GT_list ={[""], ["GroundTruth"]};
GT_file = [output_dir, filesep, GT_list{1,1}, GT_list{1,2}, ".pvp"];
GT_fid = fopen(GT_file);
GT_hdr = readpvpheader(GT_fid);
fclose(GT_file);
nx_GT = GT_hdr.nx;
ny_GT = GT_hdr.ny;
num_GT_frames = GT_hdr.nbands;
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
nonSparse_list = {[""], ["GroundTruthReconS1Error"]};
if strcmp(run_type, "S1S2")
  nonSparse_list = {}; %%{[""], ["GroundTruthReconS1Error"]; [""], ["GroundTruthReconS2Error"]; [""], ["GroundTruthReconS1S2Error"]};
elseif strcmp(run_type, "DCA")
  nonSparse_list = {[""], ["GroundTruthReconS1Error"]; [""], ["GroundTruthReconS2Error"]; [""], ["GroundTruthReconS3Error"]};
elseif strcmp(run_type, "scene")
  nonSparse_list = {["a0_"], ["ImageReconS1Error"]};
endif
num_nonSparse_list = size(nonSparse_list,1);
nonSparse_skip = repmat(1, num_nonSparse_list, 1);
nonSparse_norm_strength = ones(num_nonSparse_list,1);
GT_std_ndx = ones(num_nonSparse_list,1);
nonSparse_norm_list = {[""], ["GroundTruth"]};
if strcmp(run_type, "S1S2") || strcmp(run_type, "DCA")
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


%% Sparse activity
load_SparseHistPool_flag = false
Sparse_list ={[""], ["S1"]};
if strcmp(run_type, "S1S2")
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
  [Sparse_hdr_array, ...
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
				  hist_pool_flag, ...
				  max_pool_flag, ...
				  mean_pool_flag, ...
				  num_procs, ...
				  num_epochs);

				%pause;
else
  Sparse_dir = [output_dir, filesep, "Sparse"];
  for i_Sparse = 1 : length(Sparse_list)
    
  endfor %% i_Sparse
endif %% load_SparseHistPool_flag

VOC_classes={'background'; 'aeroplane'; 'bicycle'; 'bird'; 'boat'; 'bottle'; 'bus'; 'car'; 'cat'; 'chair'; 'cow'; 'diningtable'; 'dog'; 'horse'; 'motorbike'; 'person'; 'pottedplant'; 'sheep'; 'sofa'; 'train'; 'tvmonitor'};
num_VOC_classes = length(VOC_classes);

target_class_indices = [0 2 6 7 14 15]+1; %%[0 1 2 3 6 7 8 12 13 14 15 19]+1; 
target_classes = VOC_classes(target_class_indices)
num_target_classes = length(target_classes);
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
  if numel(size(GT_data{i_GT_frame}.values)) <= 2
    active_indices = GT_data{i_GT_frame}.values(:,1);
    [active_kf, active_kx, active_ky] = ind2sub([num_VOC_classes, nx_GT, ny_GT], active_indices+1);
    active_ndx = sub2ind([ny_GT, nx_GT, num_VOC_classes], active_ky, active_kx, active_kf);
    GT_frame = zeros(ny_GT, nx_GT, num_VOC_classes);
    GT_frame(active_ndx) = 1;
    GT_frame = reshape(GT_frame, [ny_GT, nx_GT, num_VOC_classes]);
  else
    GT_frame = permute(GT_data{i_GT_frame}.values, [2 1 3]);
  endif  %% numel(size(values)) <= 2
  %% gotta do 1 vs all since bounding boxes overlapp
  for i_target_classID = 1 : num_target_classes
    i_VOC_class = target_class_indices(i_target_classID);
    training_label_vector_array(:,:,i_GT_image, i_target_classID) = GT_frame(:,:,i_VOC_class);
  endfor %% class_ndx
endfor %% i_frame
training_label_vector = cell(num_target_classes,1);
training_label_pos = cell(num_target_classes,1);
training_label_neg = cell(num_target_classes,1);
training_label_pos_long = cell(num_target_classes,1);
training_label_neg_long = cell(num_target_classes,1);
pos_labels_ndx = cell(num_target_classes,1);
neg_labels_ndx = cell(num_target_classes,1);
pos_labels_ndx_long = cell(num_target_classes,1);
neg_labels_ndx_long = cell(num_target_classes,1);
%%pos_labels_rank = cell(num_target_classes,1);
%%neg_labels_rank = cell(num_target_classes,1);
neg_pos_ratio = ones(num_target_classes,1);
for i_target_classID = 1 : num_target_classes
  training_label_vector{i_target_classID} = ...
  reshape(training_label_vector_array(:,:,:,i_target_classID), [ny_GT * nx_GT * num_GT_images, 1]);
  pos_labels_ndx{i_target_classID} = find(training_label_vector{i_target_classID});
  neg_labels_ndx_long{i_target_classID} = find(~training_label_vector{i_target_classID});
  neg_pos_ratio(i_target_classID) = length(neg_labels_ndx_long{i_target_classID})/length(pos_labels_ndx{i_target_classID});
  %% we can either select a random subset of the negative labels or else replicate the positive labels
  %% the latter is likely more accurate but more computationally expensive
  %%repmat_pos_labels_flag = false;
  %%if repmat_pos_labels_flag %% use every true negative output tile, repmat true positive tiles to match number of true negatives (very computationally expensive!!!)_
  pos_labels_ndx_long{i_target_classID} = repmat(pos_labels_ndx{i_target_classID}, ceil(neg_pos_ratio(i_target_classID)), 1);;
  pos_labels_ndx_long{i_target_classID} = pos_labels_ndx_long{i_target_classID}(1:length(neg_labels_ndx_long{i_target_classID}));
  %%else %% select a random number of true negative tiles to match number of true postive tiles (much faster, maybe not as accurate)
  [neg_labels_sorted, neg_labels_rank] = sort(rand(length(neg_labels_ndx_long{i_target_classID}),1));
  neg_labels_ndx{i_target_classID} = neg_labels_ndx_long{i_target_classID}(neg_labels_rank(1:length(pos_labels_ndx{i_target_classID})));
  %%endif %% repmat_pos_labels_flag
  training_label_pos{i_target_classID} = training_label_vector{i_target_classID}(pos_labels_ndx{i_target_classID});
  training_label_neg{i_target_classID} = training_label_vector{i_target_classID}(neg_labels_ndx{i_target_classID});
  training_label_pos_long{i_target_classID} = training_label_vector{i_target_classID}(pos_labels_ndx_long{i_target_classID});
  training_label_neg_long{i_target_classID} = training_label_vector{i_target_classID}(neg_labels_ndx_long{i_target_classID});
endfor %% class_ndx

%%traing_instance_matrix
liblinear_xval_options_str = ['-s 0 -C -B 1'];

%% set up data structures for storing liblinear svm results
num_Sparse_list = size(Sparse_list,1);
training_hist_pool_matrix_array = cell(num_Sparse_list + (num_Sparse_list>1),1);
training_max_pool_matrix_array = cell(num_Sparse_list + (num_Sparse_list>1),1);
training_mean_pool_matrix_array = cell(num_Sparse_list + (num_Sparse_list>1),1);
%%training_combo_pool_matrix_array = cell(num_Sparse_list + (num_Sparse_list>1),1);
training_hist_pool_matrix_array_long = cell(num_Sparse_list + (num_Sparse_list>1),1);
training_max_pool_matrix_array_long = cell(num_Sparse_list + (num_Sparse_list>1),1);
training_mean_pool_matrix_array_long = cell(num_Sparse_list + (num_Sparse_list>1),1);
%%training_combo_pool_matrix_array_long = cell(num_Sparse_list + (num_Sparse_list>1),1);
training_hist_pool_matrix_array_predict = cell(num_Sparse_list + (num_Sparse_list>1),1);
training_max_pool_matrix_array_predict = cell(num_Sparse_list + (num_Sparse_list>1),1);
training_mean_pool_matrix_array_predict = cell(num_Sparse_list + (num_Sparse_list>1),1);
%%training_combo_pool_matrix_array_predict = cell(num_Sparse_list + (num_Sparse_list>1),1);
nf_Sparse_array = zeros(num_Sparse_list + (num_Sparse_list > 1),1);
first_Sparse_frame_array = zeros(num_Sparse_list,1);
last_Sparse_frame_array = zeros(num_Sparse_list,1);
num_Sparse_frames_array = zeros(num_Sparse_list,1);
xval_model_hist_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
xval_model_max_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
xval_model_mean_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
%%xval_model_combo_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
model_hist_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
model_max_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
model_mean_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
%%model_combo_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
predicted_label_hist_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
predicted_label_max_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
predicted_label_mean_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
%%predicted_label_combo_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
accuracy_hist_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
accuracy_max_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
accuracy_mean_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
%%accuracy_combo_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
prob_values_hist_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
prob_values_max_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
prob_values_mean_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);
%%prob_values_combo_pool_array = cell(num_Sparse_list + (num_Sparse_list > 1), num_target_classes);

%% outer loop over sparse layers: one extra layer is used for the concatenation of all sparse layers
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
  else %% to save memory, Sparse_[hist/max/mean/combo]_pool_array is only used to store concatenated activity
    nf_Sparse_array(num_Sparse_list+1) = prod(nf_Sparse_array(:));
    Sparse_list{num_Sparse_list+1,1} = [];
    Sparse_list{num_Sparse_list+1,2} = Sparse_list{1,2};
    for j_Sparse = 2 : num_Sparse_list
      Sparse_list{num_Sparse_list+1,2} = [Sparse_list{num_Sparse_list+1,2},Sparse_list{j_Sparse,2}];
    endfor %% j_Sparse
  endif %% i_Sparse > 1
  for i_target_classID = 1 : num_target_classes
    disp(["training model for target class = ", target_classes{i_target_classID}, " using ", Sparse_list{i_Sparse,2}, ": neg_pos_ratio = ", num2str(neg_pos_ratio(i_target_classID))]);
    if i_Sparse <= num_Sparse_list
      if hist_pool_flag
	training_hist_pool_matrix = zeros(num_Sparse_hist_pool_bins, nf_Sparse_array(i_Sparse), ny_GT, nx_GT, num_GT_images);
      endif
      if max_pool_flag
	training_max_pool_matrix = zeros(nf_Sparse_array(i_Sparse), ny_GT, nx_GT, num_GT_images);
      endif
      if mean_pool_flag
	training_mean_pool_matrix = zeros(nf_Sparse_array(i_Sparse), ny_GT, nx_GT, num_GT_images);
      endif
      %%training_combo_pool_matrix = zeros(num_Sparse_hist_pool_bins+1, nf_Sparse_array(i_Sparse), ny_GT, nx_GT, num_GT_images);
      for i_Sparse_frame = first_Sparse_frame_array(i_Sparse) : last_Sparse_frame_array(i_Sparse)
	j_Sparse_frame = i_Sparse_frame - first_Sparse_frame_array(i_Sparse) + 1;
	if hist_pool_flag
	  training_hist_pool_matrix(:, :, :, :, j_Sparse_frame) = ...
	  Sparse_hist_pool_array{i_Sparse}{i_Sparse_frame};
	endif
	if max_pool_flag
	  training_max_pool_matrix(:, :, :, j_Sparse_frame) = ...
	  Sparse_max_pool_array{i_Sparse}{i_Sparse_frame};
	endif
	if mean_pool_flag
	  training_mean_pool_matrix(:, :, :, j_Sparse_frame) = ...
	  Sparse_mean_pool_array{i_Sparse}{i_Sparse_frame};
	endif
	%%training_combo_pool_matrix(1:num_Sparse_hist_pool_bins, :, :, :, j_Sparse_frame) = ...
	%%Sparse_hist_pool_array{i_Sparse}{i_Sparse_frame};
	%%training_combo_pool_matrix(num_Sparse_hist_pool_bins+1, :, :, :, j_Sparse_frame) = ...
	%%Sparse_max_pool_array{i_Sparse}{i_Sparse_frame};
      endfor %% i_Sparse_frame
      if hist_pool_flag
	training_hist_pool_matrix = ...
	sparse(reshape(training_hist_pool_matrix, ...
		       [num_Sparse_hist_pool_bins *  nf_Sparse_array(i_Sparse), ny_GT * nx_GT * num_GT_images]));
      endif
      if max_pool_flag
	training_max_pool_matrix = ...
	sparse(reshape(training_max_pool_matrix, ...
		       [nf_Sparse_array(i_Sparse), ny_GT * nx_GT * num_GT_images]));
      endif
      if mean_pool_flag
	training_mean_pool_matrix = ...
	sparse(reshape(training_mean_pool_matrix, ...
		       [nf_Sparse_array(i_Sparse), ny_GT * nx_GT * num_GT_images]));
      endif
      %%training_combo_pool_matrix = ...
      %%sparse(reshape(training_combo_pool_matrix, ...
      %%		 [(num_Sparse_hist_pool_bins+1) *  nf_Sparse_array(i_Sparse), ny_GT * nx_GT * num_GT_images]));
      if hist_pool_flag
	training_hist_pool_pos = training_hist_pool_matrix(:,pos_labels_ndx{i_target_classID});
	training_hist_pool_neg = training_hist_pool_matrix(:, neg_labels_ndx{i_target_classID});
      endif
      if max_pool_flag
	training_max_pool_pos = training_max_pool_matrix(:,pos_labels_ndx{i_target_classID});
	training_max_pool_neg = training_max_pool_matrix(:, neg_labels_ndx{i_target_classID});
      endif
      if mean_pool_flag
	training_mean_pool_pos = training_mean_pool_matrix(:,pos_labels_ndx{i_target_classID});
	training_mean_pool_neg = training_mean_pool_matrix(:, neg_labels_ndx{i_target_classID});
      endif
      %%training_combo_pool_pos = training_combo_pool_matrix(:,pos_labels_ndx{i_target_classID});
      %%training_combo_pool_neg = training_combo_pool_matrix(:, neg_labels_ndx{i_target_classID});
      if hist_pool_flag
	training_hist_pool_pos_long = training_hist_pool_matrix(:,pos_labels_ndx_long{i_target_classID});
	training_hist_pool_neg_long = training_hist_pool_matrix(:, neg_labels_ndx_long{i_target_classID});
      endif
      if max_pool_flag
	training_max_pool_pos_long = training_max_pool_matrix(:,pos_labels_ndx_long{i_target_classID});
	training_max_pool_neg_long = training_max_pool_matrix(:, neg_labels_ndx_long{i_target_classID});
      endif
      if mean_pool_flag
	training_mean_pool_pos_long = training_mean_pool_matrix(:,pos_labels_ndx_long{i_target_classID});
	training_mean_pool_neg_long = training_mean_pool_matrix(:, neg_labels_ndx_long{i_target_classID});
      endif
      %%training_combo_pool_pos_long = training_combo_pool_matrix(:,pos_labels_ndx_long{i_target_classID});
      %%training_combo_pool_neg_long = training_combo_pool_matrix(:, neg_labels_ndx_long{i_target_classID});

      if train_long_flag
	if hist_pool_flag
	  xval_model_hist_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		[training_hist_pool_pos_long, training_hist_pool_neg_long], ...
		liblinear_xval_options_str, 'col');
	endif
	if max_pool_flag
	  xval_model_max_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		[training_max_pool_pos_long, training_max_pool_neg_long], ...
	      liblinear_xval_options_str, 'col');
	endif
	if mean_pool_flag
	  xval_model_mean_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		[training_mean_pool_pos_long, training_mean_pool_neg_long], ...
		liblinear_xval_options_str, 'col');
	endif
	%%xval_model_combo_pool_array{i_Sparse, i_target_classID} = ...
	%%train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
	%%	[training_combo_pool_pos_long, training_combo_pool_neg_long], ...
	%%	liblinear_xval_options_str, 'col');
      else
	if hist_pool_flag
	  xval_model_hist_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		[training_hist_pool_pos, training_hist_pool_neg], ...
		liblinear_xval_options_str, 'col');
	endif
	if max_pool_flag
	  xval_model_max_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		[training_max_pool_pos, training_max_pool_neg], ...
	      liblinear_xval_options_str, 'col');
	endif
	if mean_pool_flag
	  xval_model_mean_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		[training_mean_pool_pos, training_mean_pool_neg], ...
		liblinear_xval_options_str, 'col');
	endif
	%%xval_model_combo_pool_array{i_Sparse, i_target_classID} = ...
	%%train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
	%%	[training_combo_pool_pos, training_combo_pool_neg], ...
	%%	liblinear_xval_options_str, 'col');
      endif  %% train_long_flag
      
      if train_long_flag
	if hist_pool_flag
	  model_hist_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		[training_hist_pool_pos_long, training_hist_pool_neg_long], ...
		['-s 0', '-c ', num2str(xval_model_hist_pool_array{i_Sparse, i_target_classID}(1)), '-B 1'], 'col');
	endif
	if max_pool_flag
	  model_max_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		[training_max_pool_pos_long, training_max_pool_neg_long], ...
		['-s 0', '-c ', num2str(xval_model_max_pool_array{i_Sparse, i_target_classID}(1)), '-B 1'], 'col');
	endif
	if mean_pool_flag
	  model_mean_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		[training_mean_pool_pos_long, training_mean_pool_neg_long], ...
		['-s 0', '-c ', num2str(xval_model_mean_pool_array{i_Sparse, i_target_classID}(1)), '-B 1'], 'col');
	endif
	%%model_combo_pool_array{i_Sparse, i_target_classID} = ...
	%%train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
	%%	[training_combo_pool_pos_long, training_combo_pool_neg_long], ...
	%%	['-s 0', '-c ', num2str(xval_model_combo_pool_array{i_Sparse, i_target_classID}(1)), '-B 1'], 'col');
      else
	if hist_pool_flag
	  model_hist_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		[training_hist_pool_pos, training_hist_pool_neg], ...
		['-s 0', '-c ', num2str(xval_model_hist_pool_array{i_Sparse, i_target_classID}(1)), '-B 1'], 'col');
	endif
	if max_pool_flag
	  model_max_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		[training_max_pool_pos, training_max_pool_neg], ...
		['-s 0', '-c ', num2str(xval_model_max_pool_array{i_Sparse, i_target_classID}(1)), '-B 1'], 'col');
	endif
	if mean_pool_flag
	  model_mean_pool_array{i_Sparse, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		[training_mean_pool_pos, training_mean_pool_neg], ...
		['-s 0', '-c ', num2str(xval_model_mean_pool_array{i_Sparse, i_target_classID}(1)), '-B 1'], 'col');
	endif
	%%model_combo_pool_array{i_Sparse, i_target_classID} = ...
	%%train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
	%%	[training_combo_pool_pos, training_combo_pool_neg], ...
	%%	['-s 0', '-c ', num2str(xval_model_combo_pool_array{i_Sparse, i_target_classID}(1)), '-B 1'], 'col');
      endif  %% train_long_flag

      if hist_pool_flag
	[predicted_label_hist_pool_array{i_Sparse, i_target_classID}, ...
	 accuracy_hist_pool_array{i_Sparse, i_target_classID}, ...
	 prob_values_hist_pool_array{i_Sparse, i_target_classID}] = ...
	predict(training_label_vector{i_target_classID}, ...
		training_hist_pool_matrix, ...
		model_hist_pool_array{i_Sparse, i_target_classID}, ...
		['-b 1'], 'col');
      endif
      if max_pool_flag
	[predicted_label_max_pool_array{i_Sparse, i_target_classID}, ...
	 accuracy_max_pool_array{i_Sparse, i_target_classID}, ...
	 prob_values_max_pool_array{i_Sparse, i_target_classID}] = ...
	predict(training_label_vector{i_target_classID}, ...
		training_max_pool_matrix, ...
		model_max_pool_array{i_Sparse, i_target_classID}, ...
		['-b 1'], 'col');
      endif
      if mean_pool_flag
	[predicted_label_mean_pool_array{i_Sparse, i_target_classID}, ...
	 accuracy_mean_pool_array{i_Sparse, i_target_classID}, ...
	 prob_values_mean_pool_array{i_Sparse, i_target_classID}] = ...
	predict(training_label_vector{i_target_classID}, ...
		training_mean_pool_matrix, ...
		model_mean_pool_array{i_Sparse, i_target_classID}, ...
		['-b 1'], 'col');
      endif
      
      if num_Sparse_list>1
	if i_Sparse == 1
	  if hist_pool_flag
	    training_hist_pool_matrix_array{num_Sparse_list+1, i_target_classID} = ...
	    [training_hist_pool_pos, training_hist_pool_neg];
	  endif
	  if max_pool_flag
	    training_max_pool_matrix_array{num_Sparse_list+1, i_target_classID} = ...
	    [training_max_pool_pos, training_max_pool_neg];
	  endif
	  if mean_pool_flag
	    training_mean_pool_matrix_array{num_Sparse_list+1, i_target_classID} = ...
	    [training_mean_pool_pos, training_mean_pool_neg];
	  endif
	  %%training_combo_pool_matrix_array{num_Sparse_list+1, i_target_classID} = ...
	  %%[training_combo_pool_pos, training_combo_pool_neg];
	  
	  if hist_pool_flag
	    training_hist_pool_matrix_array_long{num_Sparse_list+1, i_target_classID} = ...
	    [training_hist_pool_pos_long, training_hist_pool_neg_long];
	  endif
	  if max_pool_flag
	    training_max_pool_matrix_array_long{num_Sparse_list+1, i_target_classID} = ...
	    [training_max_pool_pos_long, training_max_pool_neg_long];
	  endif
	  if mean_pool_flag
	    training_mean_pool_matrix_array_long{num_Sparse_list+1, i_target_classID} = ...
	    [training_mean_pool_pos_long, training_mean_pool_neg_long];
	  endif
	  %%training_combo_pool_matrix_array_long{num_Sparse_list+1, i_target_classID} = ...
	  %%[training_combo_pool_pos_long, training_combo_pool_neg_long];
	  
	  if hist_pool_flag
	    training_hist_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID} = ...
	    training_hist_pool_matrix;
	  endif
	  if max_pool_flag
	    training_max_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID} = ...
	    training_max_pool_matrix;
	  endif
	  if mean_pool_flag
	    training_mean_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID} = ...
	    training_mean_pool_matrix;
	  endif
	  %%training_combo_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID} = ...
	  %%training_combo_pool_matrix;
	else
	  if hist_pool_flag
	    training_hist_pool_matrix_array{num_Sparse_list+1, i_target_classID} = ...
	    [training_hist_pool_matrix_array{num_Sparse_list+1, i_target_classID}; ...
	     [training_hist_pool_pos, training_hist_pool_neg]];
	  endif
	  if max_pool_flag
	    training_max_pool_matrix_array{num_Sparse_list+1, i_target_classID} = ...
	    [training_max_pool_matrix_array{num_Sparse_list+1, i_target_classID}; ...
	     [training_max_pool_pos, training_max_pool_neg]];
	  endif
	  if mean_pool_flag
	    training_mean_pool_matrix_array{num_Sparse_list+1, i_target_classID} = ...
	    [training_mean_pool_matrix_array{num_Sparse_list+1, i_target_classID}; ...
	     [training_mean_pool_pos, training_mean_pool_neg]];
	  endif
	  %%training_combo_pool_matrix_array{num_Sparse_list+1, i_target_classID} = ...
	  %%[training_combo_pool_matrix_array{num_Sparse_list+1, i_target_classID}; ...
	  %% [training_combo_pool_pos, training_combo_pool_neg]];
	  
	  if hist_pool_flag
	    training_hist_pool_matrix_array_long{num_Sparse_list+1, i_target_classID} = ...
	    [training_hist_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}; ...
	     [training_hist_pool_pos_long, training_hist_pool_neg_long]];
	  endif
	  if max_pool_flag
	    training_max_pool_matrix_array_long{num_Sparse_list+1, i_target_classID} = ...
	    [training_max_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}; ...
	     [training_max_pool_pos_long, training_max_pool_neg_long]];
	  endif
	  if mean_pool_flag
	    training_mean_pool_matrix_array_long{num_Sparse_list+1, i_target_classID} = ...
	    [training_mean_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}; ...
	     [training_mean_pool_pos_long, training_mean_pool_neg_long]];
	  endif
	  %%training_combo_pool_matrix_array_long{num_Sparse_list+1, i_target_classID} = ...
	  %%[training_combo_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}; ...
	  %% [training_combo_pool_pos_long, training_combo_pool_neg_long]];
	  
	  if hist_pool_flag
	    training_hist_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID} = ...
	    [training_hist_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID}; ...
	     training_hist_pool_matrix];
	  endif
	  if max_pool_flag
	    training_max_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID} = ...
	    [training_max_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID}; ...
	     training_max_pool_matrix];
	  endif
	  if mean_pool_flag
	    training_mean_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID} = ...
	    [training_mean_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID}; ...
	     training_mean_pool_matrix];
	  endif
	  %%training_combo_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID} = ...
	  %%[training_combo_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID}; ...
	  %% training_combo_pool_matrix];
	endif
      endif
    else
      if train_long_flag
	if hist_pool_flag
	  xval_model_hist_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		training_hist_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}, ...
		liblinear_xval_options_str, 'col');
	endif
	if max_pool_flag
	  xval_model_max_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		training_max_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}, ...
		liblinear_xval_options_str, 'col');
	endif
	if mean_pool_flag
	  xval_model_mean_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		training_mean_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}, ...
		liblinear_xval_options_str, 'col');
	endif
	%%xval_model_combo_pool_array{num_Sparse_list+1, i_target_classID} = ...
	%%train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
	%%	training_combo_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}, ...
	%%	liblinear_xval_options_str, 'col');
      else
	if hist_pool_flag
	  xval_model_hist_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		training_hist_pool_matrix_array{num_Sparse_list+1, i_target_classID}, ...
		liblinear_xval_options_str, 'col');
	endif
	if max_pool_flag
	  xval_model_max_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		training_max_pool_matrix_array{num_Sparse_list+1, i_target_classID}, ...
		liblinear_xval_options_str, 'col');
	endif
	if mean_pool_flag
	  xval_model_mean_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		training_mean_pool_matrix_array{num_Sparse_list+1, i_target_classID}, ...
		liblinear_xval_options_str, 'col');
	endif
	%%xval_model_combo_pool_array{num_Sparse_list+1, i_target_classID} = ...
	%%train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
	%%	training_combo_pool_matrix_array{num_Sparse_list+1, i_target_classID}, ...
	%%	liblinear_xval_options_str, 'col');
      endif %% train_long_flag
      if train_long_flag
	if hist_pool_flag
	  model_hist_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		training_hist_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}, ...
		['-s 0', '-c ', num2str(xval_model_hist_pool_array{num_Sparse_list+1, i_target_classID}(1)), '-B 1'], 'col');
	endif
	if max_pool_flag
	  model_max_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		training_max_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}, ...
		['-s 0', '-c ', num2str(xval_model_max_pool_array{num_Sparse_list+1, i_target_classID}(1)), '-B 1'], 'col');
	endif
	if mean_pool_flag
	  model_mean_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos_long{i_target_classID}; training_label_neg_long{i_target_classID}], ...
		training_mean_pool_matrix_array_long{num_Sparse_list+1, i_target_classID}, ...
		['-s 0', '-c ', num2str(xval_model_mean_pool_array{num_Sparse_list+1, i_target_classID}(1)), '-B 1'], 'col');
	endif
      else
	if hist_pool_flag
	  model_hist_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		training_hist_pool_matrix_array{num_Sparse_list+1, i_target_classID}, ...
		['-s 0', '-c ', num2str(xval_model_hist_pool_array{num_Sparse_list+1, i_target_classID}(1)), '-B 1'], 'col');
	endif
	if max_pool_flag
	  model_max_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		training_max_pool_matrix_array{num_Sparse_list+1, i_target_classID}, ...
		['-s 0', '-c ', num2str(xval_model_max_pool_array{num_Sparse_list+1, i_target_classID}(1)), '-B 1'], 'col');
	endif
	if mean_pool_flag
	  model_mean_pool_array{num_Sparse_list+1, i_target_classID} = ...
	  train([training_label_pos{i_target_classID}; training_label_neg{i_target_classID}], ...
		training_mean_pool_matrix_array{num_Sparse_list+1, i_target_classID}, ...
		['-s 0', '-c ', num2str(xval_model_mean_pool_array{num_Sparse_list+1, i_target_classID}(1)), '-B 1'], 'col');
	endif
      endif %% train_long_flag

      if hist_pool_flag
	[predicted_label_hist_pool_array{i_Sparse, i_target_classID}, ...
	 accuracy_hist_pool_array{i_Sparse, i_target_classID}, ...
	 prob_values_hist_pool_array{i_Sparse, i_target_classID}] = ...
	predict(training_label_vector{i_target_classID}, ...
		training_hist_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID}, ...
		model_hist_pool_array{i_Sparse, i_target_classID}, ...
		['-b 1'], 'col');
      endif
      if max_pool_flag
	[predicted_label_max_pool_array{i_Sparse, i_target_classID}, ...
	 accuracy_max_pool_array{i_Sparse, i_target_classID}, ...
	 prob_values_max_pool_array{i_Sparse, i_target_classID}] = ...
	predict(training_label_vector{i_target_classID}, ...
		training_max_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID}, ...
		model_max_pool_array{i_Sparse, i_target_classID}, ...
		['-b 1'], 'col');
      endif
      if mean_pool_flag
	[predicted_label_mean_pool_array{i_Sparse, i_target_classID}, ...
	 accuracy_mean_pool_array{i_Sparse, i_target_classID}, ...
	 prob_values_mean_pool_array{i_Sparse, i_target_classID}] = ...
	predict(training_label_vector{i_target_classID}, ...
		training_mean_pool_matrix_array_predict{num_Sparse_list+1, i_target_classID}, ...
		model_mean_pool_array{i_Sparse, i_target_classID}, ...
		['-b 1'], 'col');
      endif
      
    endif %% num_Sparse_list > 1
  endfor %% class_ndx
endfor %% i_Sparse

save([svm_dir, filesep, "svm.txt"], "target_classes", "xval_model_hist_pool_array", "xval_model_max_pool_array", "xval_model_mean_pool_array", "model_hist_pool_array", "model_max_pool_array", "model_mean_pool_array", "predicted_label_hist_pool_array", "predicted_label_max_pool_array", "predicted_label_mean_pool_array", "accuracy_hist_pool_array", "accuracy_max_pool_array", "accuracy_mean_pool_array", "prob_values_hist_pool_array", "prob_values_max_pool_array", "prob_values_mean_pool_array");
if plot_flag
  num_pool = hist_pool_flag + max_pool_flag + mean_pool_flag;
  i_pool = 0;
  if hist_pool_flag
    i_pool = i_pool + 1;
    pool_types{i_pool} = "hist";
  endif
  if max_pool_flag
    i_pool = i_pool + 1;
    pool_types{i_pool} = "max";
  endif
  if mean_pool_flag
    i_pool = i_pool + 1;
    pool_types{i_pool} = "mean";
  endif
  xval_model_array = zeros(num_target_classes, num_Sparse_list + (num_Sparse_list>1), num_pool);
  for i_pool = 1:num_pool
    for i_target_classID = 1 : num_target_classes
      for i_Sparse = 1 : (num_Sparse_list + (num_Sparse_list > 1))
	if strcmp(pool_types{i_pool},"hist") && hist_pool_flag
	  xval_model_array(i_target_classID, i_Sparse, i_pool) = ...
	  xval_model_hist_pool_array{i_Sparse, i_target_classID}(2);
	elseif strcmp(pool_types{i_pool},"max") && max_pool_flag
	  xval_model_array(i_target_classID, i_Sparse, i_pool) = ...
	  xval_model_max_pool_array{i_Sparse, i_target_classID}(2);
	elseif strcmp(pool_types{i_pool},"mean") && mean_pool_flag
	  xval_model_array(i_target_classID, i_Sparse, i_pool) = ...
	  xval_model_mean_pool_array{i_Sparse, i_target_classID}(2);
	  %%elseif strcmp(pool_types{i_pool},"combo")
	  %%  xval_model_array(i_target_classID, i_Sparse, i_pool) = ...
	  %%  xval_model_combo_pool_array{i_Sparse, i_target_classID}(2);
	endif %% pool_types
      endfor %% i_Sparse
    endfor  %% i_target_classID
  endfor %% i_pool
  max_model = max(xval_model_array(:));
  num_xval_model_rows = max(1,floor(sqrt(num_target_classes)));
  num_xval_model_cols = ceil(num_target_classes/num_xval_model_rows);
  xval_model_fig = figure;
  target_axis = zeros(num_target_classes,1);
  xval_model_fig_name = ["linearSVM", Sparse_list{num_Sparse_list+(num_Sparse_list>1),2}];
  set(xval_model_fig, 'name', xval_model_fig_name);
  for i_target_classID = 1 : num_target_classes
    i_xval_model_row = mod(i_target_classID-1, num_xval_model_cols) + 1;
    j_xval_model_col = ceil(i_target_classID / num_xval_model_cols);
    target_axis(i_target_classID) = subplot(num_xval_model_rows, num_xval_model_cols, i_target_classID);
    xval_model_handle = bar(target_axis(i_target_classID), squeeze(xval_model_array(i_target_classID, :, :))');
    xval_model_colormap = colormap(prism(length(xval_model_handle)));
    colormap(xval_model_colormap);
    title(target_axis(i_target_classID), target_classes{i_target_classID});
    if num_Sparse_list > 1
      axis(target_axis(i_target_classID), [0.5 (num_Sparse_list+(num_Sparse_list>1)+0.5) 0.5 min(max_model*(1.1),1)]);
    else
      axis(target_axis(i_target_classID), [0.5 (num_pool+0.5) 0.5 min(max_model*(1.1),1)]);
    endif
    set(gca, 'xticklabel', pool_types);
    if i_target_classID == num_target_classes
      [legend_handle, legend_object, legend_plot, legend_labels] = legend(target_axis(i_target_classID), Sparse_list(:,2), 'location', 'northeast');
    endif
  endfor %% i_target_classID
  saveas(xval_model_fig, [svm_dir, filesep, xval_model_fig_name, ".png"]);
endif %% plot_flag


for i_Sparse = 1 : (num_Sparse_list + (num_Sparse_list > 1))
  disp(["i_Sparse = ", num2str(i_Sparse)])
  pred_classID_file = fullfile([output_dir, filesep, "GroundTruthRecon", Sparse_list{i_Sparse,2}, ".pvp"])
  pred_data = [];
  SLP_flag = false && exist(pred_classID_file);
  if SLP_flag
    pool_types{num_pool+1} = "SLP"
  endif
  for i_pool = 1 : num_pool + SLP_flag  %% loop over SLP result, if present
    disp(["pool_type = ", pool_types{i_pool}]);
    if i_pool <= num_pool  %% use SVM ground truth prediction
      pred_hdr = GT_hdr;
    elseif i_pool > num_pool  %% use SLP ground truth prediction
      pred_classID_fid = fopen(pred_classID_file);
      pred_classID_hdr = readpvpheader(pred_classID_fid);
      fclose(pred_classID_fid);
      tot_pred_classID_frames = pred_classID_hdr.nbands;
      [pred_data,pred_hdr] = readpvpfile(pred_classID_file, ceil(tot_pred_classID_frames/1), tot_pred_classID_frames, 1, 1);
      pred_num_neurons = pred_hdr.nf * pred_hdr.nx * pred_hdr.ny;
      pred_num_frames = length(pred_data);
    endif %% i_pool
    
    gt_num_neurons = GT_hdr.nf * GT_hdr.nx * GT_hdr.ny;
    gt_num_frames = num_GT_images; %%length(gt_data);

    if i_pool > num_pool
      classID_hist_bins = -0.25:0.01:2.0;
    else
      %%min_prob = min(prob_values_hist_pool_array{i_Sparse, i_target_classID}(:));
      %%max_prob = max(prob_values_hist_pool_array{i_Sparse, i_target_classID}(:));
      classID_hist_bins = 0:0.01:1.0;
    endif
    num_classID_bins = length(classID_hist_bins);
    pred_classID_hist = zeros(num_classID_bins, length(target_class_indices),2);
    classID_colormap = prism(length(target_class_indices)+0); 
    use_false_positive_thresh = false; %%true; %%
    false_positive_thresh = .99;
    display_frame = num_GT_images;

    for i_frame = 1 : num_GT_images

      if mod(i_frame, ceil(gt_num_frames/10)) == 0
	display(["i_frame = ", num2str(i_frame)])
      endif
      gt_time = GT_data{i_frame+first_GT_frame-1}.time;
      gt_classID_cube = squeeze(training_label_vector_array(:, :, i_frame, :));
      
      %% display Ground Truth only for display frame only once (same for all sparse layers, classifiers)
      if i_frame == display_frame && i_Sparse == 1 && i_pool == 1  
	display(["i_frame = ", num2str(i_frame)]);
	display(["gt_time = ", num2str(gt_time)]);
	
	[gt_classID_val, gt_classID_ndx] = max(gt_classID_cube, [], 3);
	min_gt_classID = min(gt_classID_val(:))
	gt_classID_heatmap = zeros(GT_hdr.ny, GT_hdr.nx, 3);
	for i_target_classID = 1 : num_target_classes 
	  if ~any(any(gt_classID_cube(:,:,i_target_classID)))
	    continue;
	  else
	    disp(['ground truth includes: ', target_classes(i_target_classID)]);
	  endif
	  gt_class_color = classID_colormap(i_target_classID, :); %%getClassColor(gt_class_color_code);
	  gt_classID_band = repmat(gt_classID_cube(:,:,i_target_classID), [1,1,3]);
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
	imwrite(uint8(gt_classID_heatmap), [svm_dir, filesep, "gt_", num2str(gt_time, "%i"), ".png"], "png");
      endif %% display_frame == i_frame  && i_Sparse == 1
      
      if i_pool > num_pool
	%% gt recon layer is not sparse
	pred_time = pred_data{i_frame+first_GT_frame-1}.time;
	pred_classID_cube = pred_data{i_frame+first_GT_frame-1}.values;
	pred_classID_cube = permute(pred_classID_cube, [2,1,3]);
	target_classID_cube = pred_classID_cube(:,:,target_class_indices);
      else
	pred_time = gt_time;
	for i_target_classID = 1 : num_target_classes
	  if strcmp(pool_types{i_pool},"hist")
	    target_classID_tmp = prob_values_hist_pool_array{i_Sparse, i_target_classID}(1+(i_frame-1) * GT_hdr.nx * GT_hdr.ny : i_frame * GT_hdr.nx * GT_hdr.ny);
	    target_classID_cube(:,:,i_target_classID) = reshape(target_classID_tmp, GT_hdr.ny, GT_hdr.nx );
	  elseif strcmp(pool_types{i_pool},"max")
	    target_classID_tmp = prob_values_max_pool_array{i_Sparse, i_target_classID}(1+(i_frame-1) * GT_hdr.nx * GT_hdr.ny : i_frame * GT_hdr.nx * GT_hdr.ny);
	    target_classID_cube(:,:,i_target_classID) = reshape(target_classID_tmp, GT_hdr.ny, GT_hdr.nx );
	  elseif strcmp(pool_types{i_pool},"mean")
	    target_classID_tmp = prob_values_mean_pool_array{i_Sparse, i_target_classID}(1+(i_frame-1) * GT_hdr.nx * GT_hdr.ny : i_frame * GT_hdr.nx * GT_hdr.ny);
	    target_classID_cube(:,:,i_target_classID) = reshape(target_classID_tmp, GT_hdr.ny, GT_hdr.nx );
	  elseif strcmp(pool_types{i_pool},"combo")
	    target_classID_tmp = prob_values_combo_pool_array{i_Sparse, i_target_classID}(1+(i_frame-1) * GT_hdr.nx * GT_hdr.ny : i_frame * GT_hdr.nx * GT_hdr.ny);
	    target_classID_cube(:,:,i_target_classID) = reshape(target_classID_tmp, GT_hdr.ny, GT_hdr.nx );
	  endif %% pool_types
	endfor %% i_target_classID
      endif

      
      for i_target_classID = 1 : num_target_classes 
	pred_classID_tmp = squeeze(target_classID_cube(:,:,i_target_classID));
	gt_classID_tmp = squeeze(gt_classID_cube(:,:,i_target_classID));
	pos_pred_tmp = pred_classID_tmp(gt_classID_tmp(:)~=0);
	neg_pred_tmp = pred_classID_tmp(gt_classID_tmp(:)==0);
	if any(pos_pred_tmp)
	  pred_classID_hist(:,i_target_classID,1) = squeeze(pred_classID_hist(:,i_target_classID,1)) + hist(pos_pred_tmp(:), classID_hist_bins)';
	endif
	if any(neg_pred_tmp)
	  pred_classID_hist(:,i_target_classID,2) = squeeze(pred_classID_hist(:,i_target_classID,2)) + hist(neg_pred_tmp(:), classID_hist_bins)';
	endif
      endfor %% i_target_classID
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
	for i_target_classID = 1 : num_target_classes
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
	endfor %% i_target_classID
	[pred_classID_val, pred_classID_ndx] = max(target_classID_cube, [], 3);
	min_pred_classID = min(pred_classID_val(:))
	[max_pred_classID, max_pred_classID_ndx] = max(pred_classID_val(:))
	disp(target_classes{pred_classID_ndx(max_pred_classID_ndx)});
	mean_pred_classID = mean(pred_classID_val(:))
	std_pred_classID = std(pred_classID_val(:))
	pred_classID_thresh = reshape(pred_classID_thresh, [1,1, length(target_class_indices)]);
	pred_classID_mask = double(target_classID_cube >= repmat(pred_classID_thresh, [pred_hdr.ny, pred_hdr.nx, 1]));
	pred_classID_confidences = cell(length(target_class_indices), 1);
	
	classID_bin_width = (classID_hist_bins(end) - classID_hist_bins(1)) / (length(classID_hist_bins)-1);
	pred_classID_max_confidence = squeeze(max(squeeze(max(target_classID_cube, [], 2)), [], 1));
	pred_classID_max_confidence_bin = 1+floor((pred_classID_max_confidence - classID_hist_bins(1)) ./ classID_bin_width);
	pred_classID_max_confidence_bin(find(pred_classID_max_confidence_bin(:) > num_classID_bins)) = num_classID_bins;
	pred_classID_max_confidence_bin(pred_classID_max_confidence_bin(:) < 1) =  1;
	
	%%pred_classID_max_percent_confidence = (pred_classID_max_confidence(:) - pred_classID_thresh(:)) ./ (pred_classID_thresh(:) + (pred_classID_thresh(:)==0));
	%%pred_classID_relative_accuracy = ((pred_classID_true_pos(:) - (1-false_positive_thresh)) ./ (1-pred_classID_false_pos(:)));
	%%pred_classID_max_confidence = pred_classID_max_percent_confidence;
	
	[pred_classID_sorted_confidence, pred_classID_sorted_ndx] = sort(pred_classID_max_confidence, 'descend');
	target_confidences = cell(length(target_class_indices),1);
	for i_target_classID = 1 : num_target_classes
	  pred_classID_max_confidence(i_target_classID) = ...
	  (1 - pred_classID_cumprob(pred_classID_max_confidence_bin(i_target_classID),i_target_classID,1)) ./ ...
	  ((1 - pred_classID_cumprob(pred_classID_max_confidence_bin(i_target_classID),i_target_classID,1)) + ...
	   (1 - pred_classID_cumprob(pred_classID_max_confidence_bin(i_target_classID),i_target_classID,2)));
	  target_confidences{i_target_classID, 1} = [target_classes{pred_classID_sorted_ndx(i_target_classID)}, ...
						     ', ', 'confidence = ', num2str(pred_classID_sorted_confidence(i_target_classID)), ...
						     ', ', 'thresh = ', num2str(pred_classID_thresh(pred_classID_sorted_ndx(i_target_classID))), ...
						     ', ', 'accuracy = ', num2str(pred_classID_accuracy(pred_classID_sorted_ndx(i_target_classID))), ...
						     ', ', 'true_pos = ', num2str(pred_classID_true_pos(pred_classID_sorted_ndx(i_target_classID))), ...
						     ', ', 'true_neg = ', num2str(pred_classID_false_pos(pred_classID_sorted_ndx(i_target_classID)))];
	endfor
	if plot_flag
	  pred_classID_heatmap = zeros(pred_hdr.ny, pred_hdr.nx, 3);
	  pred_fig = figure("name", ["Predict: ", num2str(pred_time, "%i"), "_", Sparse_list{i_Sparse,2}, "_", pool_types{i_pool}]);
	  image(uint8(pred_classID_heatmap)); axis off; axis image, box off;
	  hold on
	  for i_target_classID = 1 : num_target_classes
	    if ~any(pred_classID_mask(:,:,i_target_classID))
	      continue;
	    else
	      disp(['prediction includes: ', target_classes(i_target_classID)]);
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
	    %%th = text(3, ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)), target_classes{i_target_classID});
	    pred_classID_heatmap(ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)):ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)), 1:2, 1) = pred_class_color(1)*255;
	    pred_classID_heatmap(ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)):ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)), 1:2, 2) = pred_class_color(2)*255;
	    pred_classID_heatmap(ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)):ceil(i_target_classID*pred_hdr.ny/length(target_class_indices)), 1:2, 3) = pred_class_color(3)*255;
	    %%set(th, 'color', pred_class_color(:));
	  endfor
	  pred_classID_heatmap = mod(pred_classID_heatmap, 256);
	  image(uint8(pred_classID_heatmap)); axis off; axis image, box off;
	  drawnow
	  %%get(th)
	endif %% plot_flag
	imwrite(uint8(pred_classID_heatmap), [svm_dir, filesep, "pred_", num2str(pred_time, "%i"), "_", Sparse_list{i_Sparse,2}, "_", pool_types{i_pool}, ".png"], "png");
	disp(target_confidences)
		%disp([pred_classID_true_pos; pred_classID_false_pos])
		%keyboard
      endif  %% pred_time == Recon_time
      
      if plot_flag && i_frame == display_frame
	hist_fig = figure("name", ["ROC", "_", num2str(pred_time, "%i"), "_", Sparse_list{i_Sparse,2}, "_", pool_types{i_pool}]);
	num_subplot_rows = ceil(sqrt(numel(target_class_indices)));
	num_subplot_cols = ceil(numel(target_class_indices) / num_subplot_rows);
	i_subplot = 0;
	for i_target_classID  = 1 : num_target_classes
	  i_subplot = i_subplot + 1;
	  subplot(num_subplot_rows,num_subplot_cols,i_subplot, 'color', [0 0 0])
	  %%
	  pos_hist = squeeze(pred_classID_hist(:,i_target_classID,1)) ./ squeeze(pred_classID_norm(:,i_target_classID,1));
	  hist_width_tmp = round(num_classID_bins/2);
	  bins_tmp = [pred_classID_thresh_bin(i_target_classID)-hist_width_tmp:pred_classID_thresh_bin(i_target_classID)+hist_width_tmp];
	  if i_pool > num_pool
	    bins_tmp_fixed = bins_tmp(find(bins_tmp>0,1,"first"):find(bins_tmp<num_classID_bins,1,"last"));
	  else
	    bins_tmp_fixed = 1:num_classID_bins;
	  endif
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
	saveas(hist_fig, [svm_dir, filesep, "hist_", "_", num2str(pred_time, "%i"), "_", Sparse_list{i_Sparse,2}, "_", pool_types{i_pool}, ".png"], "png");
      endif
      
    endfor  %% i_frame
    save([svm_dir, filesep, "ROC_", num2str(pred_time, "%i"), "_", Sparse_list{i_Sparse,2}, "_", pool_types{i_pool}, ".mat"], "classID_hist_bins", "pred_classID_hist", "pred_classID_norm", "pred_classID_cumprob", "pred_classID_cumsum")
  endfor  %% i_pool
endfor %% i_Sparse
