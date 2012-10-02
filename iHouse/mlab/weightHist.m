function weightHist(weightValues, activityTimeIndex, numbins, outDir, figTitle)
   global VIEW_FIGS;
   global GRAY_SC;
   global WRITE_FIGS;
   [procsX procsY numArbors] = size(weightValues);
   outVec = [];
   for i = 1:(procsX * procsY * numArbors)
      outVec = [outVec; weightValues{i}(:)];
   end

   inc = 1/numbins;
   scVec = [0:inc:5];

   if(VIEW_FIGS)
      figure;
   else
      figure('Visible', 'off');
   end
   hist(outVec, scVec);
   if(GRAY_SC)
      colormap(gray);
   end
   title([figTitle, ' - time: ', num2str(activityTimeIndex - 1)]);
   if(WRITE_FIGS)
      print_filename = [outDir, figTitle, '_', num2str(activityTimeIndex - 1), '.jpg'];
      print(print_filename);
   end
end
   
