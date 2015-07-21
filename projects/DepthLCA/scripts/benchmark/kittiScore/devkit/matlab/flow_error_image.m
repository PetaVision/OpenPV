function F_err = flow_error_image (F_gt,F_est,dilate_radius)

if nargin==2
  dilate_radius = 1;
end

[E,F_val] = flow_error_map (F_gt,F_est);

cols = flow_error_colormap();

F_err = zeros(size(F_gt));
for i=1:size(cols,1)
  [v,u] = find(F_val > 0 & E >= cols(i,1) & E <= cols(i,2));
  F_err(sub2ind(size(F_err),v,u,1*ones(length(v),1))) = cols(i,3);
  F_err(sub2ind(size(F_err),v,u,2*ones(length(v),1))) = cols(i,4);
  F_err(sub2ind(size(F_err),v,u,3*ones(length(v),1))) = cols(i,5);
end

F_err = imdilate(F_err,strel('disk',dilate_radius));
