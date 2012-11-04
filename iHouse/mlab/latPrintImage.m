%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Image Printing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function latPrintImage(mat, time, arborId, outDir, scaleFlag, figTitle)
   global VIEW_FIGS;
   global GRAY_SC;
   global WRITE_FIGS;

   if(isempty(find(mat)))
      scaleFlag = 1;
   end
   assert(isempty(find(isnan(mat))), 'printImage: NaN in Matrix');
   
   if(VIEW_FIGS)
      figure;
   else
      figure('Visible', 'off');
   end
   if (scaleFlag < 0)
      scaleMax = max(mat(:));
      scale = [0, scaleMax];
   else
      scale = [0, scaleFlag];
   end
   imagesc(mat, scale);
   %Find max/min of mat, and set scale equal to that
   if(GRAY_SC)
      colormap(gray);
   else
      colormap(cm());
   end
   colorbar;

   title([figTitle, ' - time: ', num2str(time), ' arbor: ', num2str(arborId)]);
   if(WRITE_FIGS)
      print_filename = [outDir, figTitle, '_', num2str(double(time), '%.0f'), '_', num2str(arborId), '.jpg'];
      print(print_filename);
   end
end
