function [hit_list, miss_list] = pvp_testFeaturesB(pvp_activity) % gjk 11/15/11

  global NFEATURES NCOLS NROWS N
  global pvp_patch_size
  global pvp_density_thresh 

  hit_list = cell(1);
  num_hits = 0;
  num_patch_rows = ceil(2 * NROWS / pvp_patch_size(1)) - 1;
  num_patch_cols = ceil(2 * NCOLS / pvp_patch_size(2)) - 1;
  miss_list = zeros(num_patch_rows, num_patch_cols);

  if nnz(pvp_activity) == 0
    return;
  endif

   pvp_activity3D = reshape(full(pvp_activity), [NFEATURES NCOLS NROWS]);
   pvp_activity3D = permute(pvp_activity3D, [3,2,1]);
   size(pvp_activity3D)
   pvp_activity2D = squeeze(sum(pvp_activity3D, 3));
   nactive = nnz(pvp_activity2D);

   pvp_activity1D= zeros(nactive,2);
   [pvp_activity1D(:,2),pvp_activity1D(:,1)]=find(pvp_activity2D);
   [class_vector,type_vector] = dbscan(pvp_activity1D,10,50);
      
   max_class_vector = max(class_vector);
   if max_class_vector < 1 return; endif
%%   pvp_centroids = zeros(max_class_vector,8);  %% x,y and normalized
   pvp_centroids = repmat([0 0 0 0 1920 0 1080 0],max_class_vector,1);
   %% count and feature score ( 0 is all in one feature, 1 is evenly
   %% distributed) plus xmin xmax ymin and ymax

   size(pvp_centroids)

   pvp_features  = zeros(max_class_vector,8);  %% 8 features
   
   
   for i=1:nactive  % loop over all points and get the cluster centroids
    if class_vector(i) > 0
     pvp_centroids(class_vector(i),1)=pvp_centroids(class_vector(i),1)+pvp_activity1D(i,1);
     pvp_centroids(class_vector(i),5)=min(pvp_activity1D(i,1),pvp_centroids(class_vector(i),5));
     pvp_centroids(class_vector(i),6)=max(pvp_activity1D(i,1),pvp_centroids(class_vector(i),6));
	 %% add x
     pvp_centroids(class_vector(i),2)=pvp_centroids(class_vector(i),2)+pvp_activity1D(i,2);  %% add y
     pvp_centroids(class_vector(i),7)=min(pvp_activity1D(i,2),pvp_centroids(class_vector(i),7));
     pvp_centroids(class_vector(i),8)=max(pvp_activity1D(i,2),pvp_centroids(class_vector(i),8));
     pvp_centroids(class_vector(i),3)=pvp_centroids(class_vector(i),3)+1; \
	 %% increment for later mean
     pvp_features(class_vector(i),1)=pvp_features(class_vector(i),1)+pvp_activity3D(pvp_activity1D(i,2),pvp_activity1D(i,1),1);
     pvp_features(class_vector(i),2)=pvp_features(class_vector(i),2)+pvp_activity3D(pvp_activity1D(i,2),pvp_activity1D(i,1),2);
     pvp_features(class_vector(i),3)=pvp_features(class_vector(i),3)+pvp_activity3D(pvp_activity1D(i,2),pvp_activity1D(i,1),3);
     pvp_features(class_vector(i),4)=pvp_features(class_vector(i),4)+pvp_activity3D(pvp_activity1D(i,2),pvp_activity1D(i,1),4);
     pvp_features(class_vector(i),5)=pvp_features(class_vector(i),5)+pvp_activity3D(pvp_activity1D(i,2),pvp_activity1D(i,1),5);
     pvp_features(class_vector(i),6)=pvp_features(class_vector(i),6)+pvp_activity3D(pvp_activity1D(i,2),pvp_activity1D(i,1),6);
     pvp_features(class_vector(i),7)=pvp_features(class_vector(i),7)+pvp_activity3D(pvp_activity1D(i,2),pvp_activity1D(i,1),7);
     pvp_features(class_vector(i),8)=pvp_features(class_vector(i),8)+pvp_activity3D(pvp_activity1D(i,2),pvp_activity1D(i,1),8);
    endif
   endfor 

     pvp_features

     pvp_centroids(:,1)=pvp_centroids(:,1)./(pvp_centroids(:,3) + (pvp_centroids(:,3)==0)); % get mean
     pvp_centroids(:,2)=pvp_centroids(:,2)./(pvp_centroids(:,3) + \
					     (pvp_centroids(:,3)==0));
     feature_averages = zeros(max_class_vector);
     feature_scores   = zeros(max_class_vector);
     feature_averages = pvp_centroids(:,3)/8; %% average over 8 kernel orientations
     feature_scores   = \
	 (pvp_features(:,1)-feature_averages).^2+(pvp_features(:,2)-feature_averages).^2+(pvp_features(:,3)-feature_averages).^2+(pvp_features(:,4)-feature_averages).^2+(pvp_features(:,5)-feature_averages).^2+(pvp_features(:,6)-feature_averages).^2+(pvp_features(:,7)-feature_averages).^2+(pvp_features(:,8)-feature_averages).^2;

     feature_scores =  feature_scores./(feature_averages.^2+(feature_averages==0))./56.;
     feature_scores =  1.0 - feature_scores; %% zero means all in one
     %% feature, one means evenly distributed.

     pvp_centroids(:,3)=pvp_centroids(:,3)/(max(pvp_centroids(:,3)) + (max(pvp_centroids(:,3))==0)); %normalize
     pvp_centroids(:,4)=feature_scores;

     pvp_centroids
  
     num_hits = 0;

   for i_num_hits=1:max_class_vector

     if (feature_scores(i_num_hits) > 0.1) &&       (feature_scores(i_num_hits) < 0.9) && 	   (pvp_centroids(i_num_hits,3) > 0.1)  % threshold for feature distribution
     
                 
%       cluster_col_min = fix(pvp_centroids(i_num_hits,1)-pvp_patch_size(2)/2); 
%       cluster_col_max = fix(pvp_centroids(i_num_hits,1)+pvp_patch_size(2)/2);
%       cluster_row_min = fix(pvp_centroids(i_num_hits,2)-pvp_patch_size(1)/2);
%       cluster_row_max = fix(pvp_centroids(i_num_hits,2)+pvp_patch_size(1)/2);

                
        cluster_col_min = fix(pvp_centroids(i_num_hits,5)); 
        cluster_col_max = fix(pvp_centroids(i_num_hits,6));
        cluster_row_min = fix(pvp_centroids(i_num_hits,7));
        cluster_row_max = fix(pvp_centroids(i_num_hits,8));

      cluster_col_min = max(cluster_col_min,1);  % resize to frame
      cluster_col_min = min(cluster_col_min,NCOLS);
      cluster_col_max = max(cluster_col_max,1);  % resize to frame
      cluster_col_max = min(cluster_col_max,NCOLS);
      cluster_row_min = max(cluster_row_min,1);  % resize to frame
      cluster_row_min = min(cluster_row_min,NROWS);
      cluster_row_max = max(cluster_row_max,1);  % resize to frame
      cluster_row_max = min(cluster_row_max,NROWS);
     
        num_hits = num_hits + 1;
   
	hit_list{num_hits} = struct;
	hit_list{num_hits}.hit_density = feature_scores(i_num_hits);
	hit_list{num_hits}.patch_X1 = 1; % cluster_col_min;
	hit_list{num_hits}.patch_Y1 = 1; % cluster_row_min;
	hit_list{num_hits}.patch_X2 = 1; % cluster_col_max;
	hit_list{num_hits}.patch_Y2 = 1; % cluster_row_min;
	hit_list{num_hits}.patch_X3 = 1; % cluster_col_max;
	hit_list{num_hits}.patch_Y3 = 1; % cluster_row_max;
	hit_list{num_hits}.patch_X4 = 1; % cluster_col_min;
	hit_list{num_hits}.patch_Y4 = 1; % cluster_row_max;
	hit_list{num_hits}.Confidence = pvp_centroids(i_num_hits,3);
	hit_list{num_hits}.BoundingBox_X1 = cluster_col_min;
	hit_list{num_hits}.BoundingBox_Y1 = cluster_row_min;
	hit_list{num_hits}.BoundingBox_X2 = cluster_col_max;
	hit_list{num_hits}.BoundingBox_Y2 = cluster_row_min;
	hit_list{num_hits}.BoundingBox_X3 = cluster_col_max;
	hit_list{num_hits}.BoundingBox_Y3 = cluster_row_max;
	hit_list{num_hits}.BoundingBox_X4 = cluster_col_min;
	hit_list{num_hits}.BoundingBox_Y4 = cluster_row_max;
        endif % of feature threshold
      % TODO miss list
    endfor% max_class_vector loop that contains all the clusters
    
endfunction %% pvp_testClusters gjk 11/11/11
