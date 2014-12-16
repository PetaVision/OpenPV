function flow_error_histogram (F_gt,F_est)

[E,F_val] = flow_error_map (F_gt,F_est);

cols = flow_error_colormap();

hist_err = zeros(1,size(cols,1));
hist_lbl = [];
for i=1:size(cols,1)
  hist_err(i) = length(find(F_val > 0 & E >= cols(i,1) & E <= cols(i,2)));
  if i<3
    hist_lbl{i} = sprintf('%.2f',cols(i,2));
  else
    hist_lbl{i} = sprintf('%.1f',cols(i,2));
  end
end
hist_err = 100*hist_err/sum(hist_err);

h_bar = bar(hist_err,'BarWidth',1);
f_col = repmat(1:numel(hist_err),5,1);
f_col = [f_col(:);nan];
set(get(h_bar,'children'),'facevertexcdata',f_col);
colormap(cols(:,3:5));

axis tight; fs = 14;
xlabel('Maximum End-point Error (Px)','FontSize',fs);
ylabel('Number of Pixels (%)','FontSize',fs);
set(gca,'FontSize',fs);
set(gca,'XTickLabel',hist_lbl);
set(gca,'Position',[0.1 0.13 0.88 0.85]);
