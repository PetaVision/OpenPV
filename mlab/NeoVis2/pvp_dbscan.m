function [hit_list] = pvp_dbscan(pvp_activity) % gjk 11/15/11 %% revised: gtk 12/8/11

  global NFEATURES NCOLS NROWS N
  global pvp_patch_size
  global pvp_std_patch_size
  global pvp_density_thresh 

  hit_list = []; %% cell(1);
  num_hits = 0;
  num_patch_rows = ceil(2 * NROWS / pvp_patch_size(1)) - 1;
  num_patch_cols = ceil(2 * NCOLS / pvp_patch_size(2)) - 1;
  %%miss_list = zeros(num_patch_rows, num_patch_cols);

  if nnz(pvp_activity) == 0
    return;
  endif

  pvp_activity3D = reshape(full(pvp_activity), [NFEATURES NCOLS NROWS]);
  pvp_activity3D = permute(pvp_activity3D, [3,2,1]);
  %%size(pvp_activity3D);
  pvp_activity2D = squeeze(sum(pvp_activity3D, 3));
  num_active = nnz(pvp_activity2D);
  disp(["num_active = ", num2str(num_active)]);

  pvp_activity1D= zeros(num_active,2);
  [pvp_activity1D(:,2),pvp_activity1D(:,1)]=find(pvp_activity2D);

  %% maximum distance between points in same cluster
  cluster_radius = min(pvp_patch_size - pvp_std_patch_size) / 2;
  if cluster_radius < min(pvp_patch_size)/4
    cluster_radius = min(pvp_patch_size)/4;
  endif
  disp(["cluster_radius = ", num2str(cluster_radius)]);
  min_cluster_count = ceil(cluster_radius.^2 * 4 * num_active / numel(pvp_activity));
  disp(["min_cluster_count = ", num2str(min_cluster_count)]);
  [class_vector,type_vector] = dbscan(pvp_activity1D, min_cluster_count, cluster_radius);
  
  max_class_vector = max(class_vector);
  disp(["max_class_vector = ", num2str(max_class_vector)]);
  if max_class_vector < 1 return; endif
  pvp_class_count = zeros(max_class_vector,1);
  pvp_class_mean = zeros(max_class_vector,2);  %% 
  pvp_class_min = repmat([NROWS, NCOLS], [max_class_vector,1]);  %% 
  pvp_class_max = ones(max_class_vector,2);  %% 
  %% count and feature score ( 0 is all in one feature, 1 is evenly distributed)
  pvp_features  = zeros(max_class_vector, NFEATURES);  %% 8 features
  
  
  for i=1:num_active  % loop over all points and get the cluster centroids
    if class_vector(i) > 0
      pvp_class_count(class_vector(i),1) = pvp_class_count(class_vector(i),1) + 1; %% count 
      pvp_class_mean(class_vector(i),1) = pvp_class_mean(class_vector(i),1) + pvp_activity1D(i,1);  
      pvp_class_mean(class_vector(i),2) = pvp_class_mean(class_vector(i),2) + pvp_activity1D(i,2);  
      pvp_class_min(class_vector(i),1) = min(pvp_class_min(class_vector(i),1), pvp_activity1D(i,1)); 
      pvp_class_max(class_vector(i),1) = max(pvp_class_max(class_vector(i),1), pvp_activity1D(i,1)); 
      pvp_class_min(class_vector(i),2) = min(pvp_class_min(class_vector(i),2), pvp_activity1D(i,2)); 
      pvp_class_max(class_vector(i),2) = max(pvp_class_max(class_vector(i),2), pvp_activity1D(i,2)); 
      pvp_features(class_vector(i),:) = ...
	  pvp_features(class_vector(i),:) + squeeze(pvp_activity3D(pvp_activity1D(i,2),pvp_activity1D(i,1),:))';
    endif
  endfor 
  disp([repmat("class_count = ", max_class_vector, 1), num2str(pvp_class_count(:))]);

  pvp_class_mean(:,1) = ...
      pvp_class_mean(:,1) ./ ...
      (pvp_class_count(:,1) + (pvp_class_count(:,1)==0)); 
  pvp_class_mean(:,2) = ...
      pvp_class_mean(:,2) ...
      ./ (pvp_class_count(:,1) + (pvp_class_count(:,1)==0));

  feature_std = std(repmat(1:NFEATURES, max_class_vector, 1) .* pvp_features, 1, 2)./(sum(pvp_features,2)); %% weigthed average over orientations for each class
  feature_scores = 2 * feature_std(:) / NFEATURES;
  disp([repmat("feature_scores = ", max_class_vector, 1), num2str(feature_scores(:))]);
  %% zero means all in one feature, greater than one means more evenly distributed.

  pvp_class_confidence = ...
      pvp_class_count(:,1) / ...
      (max(pvp_class_count(:,1)) + (max(pvp_class_count(:,1))==0)); %normalize

  %%BB_mask = zeros(NROWS, NCOLS);

  num_hits = 0;
  max_hits = max_class_vector;
  for i_hit=1:max_hits    
    
    cluster_x_min = fix(pvp_class_mean(i_hit,1)-pvp_patch_size(2)/2); 
    cluster_x_max = fix(pvp_class_mean(i_hit,1)+pvp_patch_size(2)/2);
    cluster_y_min = fix(pvp_class_mean(i_hit,2)-pvp_patch_size(1)/2);
    cluster_y_max = fix(pvp_class_mean(i_hit,2)+pvp_patch_size(1)/2);

    cluster_x_min = max(cluster_x_min,1);  % resize to frame
    cluster_x_min = min(cluster_x_min,NCOLS);
    cluster_x_max = max(cluster_x_max,1);  % resize to frame
    cluster_x_max = min(cluster_x_max,NCOLS);
    cluster_y_min = max(cluster_y_min,1);  % resize to frame
    cluster_y_min = min(cluster_y_min,NROWS);
    cluster_y_max = max(cluster_y_max,1);  % resize to frame
    cluster_y_max = min(cluster_y_max,NROWS);

    patch_x_min = pvp_class_min(i_hit,1);
    patch_x_max = pvp_class_max(i_hit,1);
    patch_y_min = pvp_class_min(i_hit,2);
    patch_y_max = pvp_class_max(i_hit,2);
    disp(["patch_x_min = ", num2str(patch_x_min)]);
    disp(["patch_y_min = ", num2str(patch_y_min)]);
    disp(["patch_x_max = ", num2str(patch_x_max)]);
    disp(["patch_y_max = ", num2str(patch_y_max)]);
    
    num_hits = num_hits + 1;
    
    hit_list{num_hits} = struct;
    hit_list{num_hits}.hit_density = feature_scores(i_hit);
    hit_list{num_hits}.patch_X1 = patch_x_min;
    hit_list{num_hits}.patch_Y1 = patch_y_min;
    hit_list{num_hits}.patch_X2 = patch_x_max;
    hit_list{num_hits}.patch_Y2 = patch_y_min;
    hit_list{num_hits}.patch_X3 = patch_x_max;
    hit_list{num_hits}.patch_Y3 = patch_y_max;
    hit_list{num_hits}.patch_X4 = patch_x_min;
    hit_list{num_hits}.patch_Y4 = patch_y_max;
    hit_list{num_hits}.Confidence = pvp_class_confidence(i_hit,1);
    hit_list{num_hits}.BoundingBox_X1 = cluster_x_min;
    hit_list{num_hits}.BoundingBox_Y1 = cluster_y_min;
    hit_list{num_hits}.BoundingBox_X2 = cluster_x_max;
    hit_list{num_hits}.BoundingBox_Y2 = cluster_y_min;
    hit_list{num_hits}.BoundingBox_X3 = cluster_x_max;
    hit_list{num_hits}.BoundingBox_Y3 = cluster_y_max;
    hit_list{num_hits}.BoundingBox_X4 = cluster_x_min;
    hit_list{num_hits}.BoundingBox_Y4 = cluster_y_max;
    hit_list{num_hits}.edge_count = pvp_class_count(i_hit);
    hit_list{num_hits}.feature_score = feature_scores(i_hit);

    %%BB_mask(patch_y_min:patch_y_max, patch_x_min:patch_x_max) = 1;
  endfor% max_class_vector loop that contains all the clusters
  
endfunction %% pvp_testFeatures; gjk 11/11/11, revised: gtk 12/8/11
