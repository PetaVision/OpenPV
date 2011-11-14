function [hit_list, miss_list] = pvp_testClusters(pvp_activity) % gjk 11/11/11

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
   pvp_activity2D = squeeze(sum(pvp_activity3D, 3));
   nactive = nnz(pvp_activity2D);

   pvp_activity1D= zeros(nactive,2);
   [pvp_activity1D(:,2),pvp_activity1D(:,1)]=find(pvp_activity2D);
   [class_vector,type_vector] = dbscan(pvp_activity1D,10,[]);
      
   max_class_vector = max(class_vector);
   pvp_centroids = zeros(max_class_vector,3);  %% x,y and count

   for i=1:nactive  % loop over all points and get the cluster centroids
    if class_vector(i) > 0
     pvp_centroids(class_vector(i),1)=pvp_centroids(class_vector(i),1)+pvp_activity1D(i,1);  %% add x
     pvp_centroids(class_vector(i),2)=pvp_centroids(class_vector(i),2)+pvp_activity1D(i,2);  %% add y
     pvp_centroids(class_vector(i),3)=pvp_centroids(class_vector(i),3)+1;  %% increment for later mean
    endif
   endfor 

   pvp_centroids(:,1)=pvp_centroids(:,1)./pvp_centroids(:,3); % get mean
   pvp_centroids(:,2)=pvp_centroids(:,2)./pvp_centroids(:,3);
   pvp_centroids(:,3)=pvp_centroids(:,3)/max(pvp_centroids(:,3)); %normalize
   %%pvp_centroids;
 
   for num_hits=1:max_class_vector
     
                      % patch_size(2) is bigger for cars, assume it is col
      cluster_col_min = fix(pvp_centroids(num_hits,1)-pvp_patch_size(2)/2); 
      cluster_col_max = fix(pvp_centroids(num_hits,1)+pvp_patch_size(2)/2);
      cluster_row_min = fix(pvp_centroids(num_hits,2)-pvp_patch_size(1)/2);
      cluster_row_max = fix(pvp_centroids(num_hits,2)+pvp_patch_size(1)/2);

      cluster_col_min = max(cluster_col_min,1);  % resize to frame
      cluster_col_min = min(cluster_col_min,NCOLS);
      cluster_col_max = max(cluster_col_max,1);  % resize to frame
      cluster_col_max = min(cluster_col_max,NCOLS);
      cluster_row_min = max(cluster_row_min,1);  % resize to frame
      cluster_row_min = min(cluster_row_min,NROWS);
      cluster_row_max = max(cluster_row_max,1);  % resize to frame
      cluster_row_max = min(cluster_row_max,NROWS);
     
   
	hit_list{num_hits} = struct;
	hit_list{num_hits}.hit_density = pvp_centroids(num_hits,3);
	hit_list{num_hits}.patch_X1 = 1; % cluster_col_min;
	hit_list{num_hits}.patch_Y1 = 1; % cluster_row_min;
	hit_list{num_hits}.patch_X2 = 1; % cluster_col_max;
	hit_list{num_hits}.patch_Y2 = 1; % cluster_row_min;
	hit_list{num_hits}.patch_X3 = 1; % cluster_col_max;
	hit_list{num_hits}.patch_Y3 = 1; % cluster_row_max;
	hit_list{num_hits}.patch_X4 = 1; % cluster_col_min;
	hit_list{num_hits}.patch_Y4 = 1; % cluster_row_max;
	hit_list{num_hits}.Confidence = pvp_centroids(num_hits,3);
	hit_list{num_hits}.BoundingBox_X1 = cluster_col_min;
	hit_list{num_hits}.BoundingBox_Y1 = cluster_row_min;
	hit_list{num_hits}.BoundingBox_X2 = cluster_col_max;
	hit_list{num_hits}.BoundingBox_Y2 = cluster_row_min;
	hit_list{num_hits}.BoundingBox_X3 = cluster_col_max;
	hit_list{num_hits}.BoundingBox_Y3 = cluster_row_max;
	hit_list{num_hits}.BoundingBox_X4 = cluster_col_min;
	hit_list{num_hits}.BoundingBox_Y4 = cluster_row_max;
      % TODO miss list
    endfor% max_class_vector loop that contains all the clusters
    
endfunction %% pvp_testClusters gjk 11/11/11
