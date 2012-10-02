%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Image Printing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function printImage(mat, activityTimeIndex, arborId, outDir, scaleFlag, figTitle)
   global VIEW_FIGS;
   global GRAY_SC;
   global WRITE_FIGS;

   assert(~isempty(find(mat)), 'printImage: Empty Matrix');
   assert(isempty(find(isnan(mat))), 'printImage: NaN in Matrix');
   
   if(VIEW_FIGS)
      figure;
   else
      figure('Visible', 'off');
   end
   if (scaleFlag < 0)
      scaleMax = max(max(mat(:)), abs(min(mat(:))));
      scale = [-scaleMax, scaleMax];
   else
      scale = [-scaleFlag, scaleFlag]
   end
   imagesc(mat, scale);
   %Find max/min of mat, and set scale equal to that
   if(GRAY_SC)
      colormap(gray);
   else
      colormap(cm());
   end
   colorbar;

   title([figTitle, ' - time: ', num2str(activityTimeIndex - 1), ' arbor: ', num2str(arborId)]);
   if(WRITE_FIGS)
      print_filename = [outDir, figTitle, '_', num2str(activityTimeIndex - 1), '_', num2str(arborId), '.jpg'];
      print(print_filename);
   end
end
