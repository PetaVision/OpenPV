function combinebatches(directory, layer_name, num_batch, batch_method, batch_width, total_frames)

   result_index = 0; 
   total_found = 0;
   if strcmp(batch_method, "byFile") == 1 % Really, Octave?
      per_batch = num_batch / batch_width;
      i = batch_width - 1;
      while i >= 0
         result_index = i * per_batch + 1;
         fname = [directory, "/batchsweep_", num2str(i, "%02d"), "/", layer_name, "_", num2str(i), ".pvp"];
         printf("Reading %s... ", fname);
         fflush(stdout);
         source_data = readpvpfile(fname);
         num_frames = size(source_data)(1);
         total_found += num_frames;
         printf("%d frames.\n", num_frames);
         fflush(stdout);
         source_index = 1;
         per_batch_left = per_batch;
         for f = 1:num_frames
            result{result_index}.values = source_data{source_index}.values;
            result{result_index}.time   = source_data{source_index}.time;
            source_index++;
            per_batch_left--;
            if per_batch_left > 0
               result_index++;
            else
               per_batch_left = per_batch;
               result_index += num_batch - 1;
            endif
         endfor
         i--;
      endwhile

   elseif strcmp(batch_method, "byList") == 1
      i = batch_width - 1;
      while i >= 0
         result_index = total_frames / batch_width * i + 1;
         fname = [directory, "/batchsweep_", num2str(i, "%02d"), "/", layer_name, "_", num2str(i), ".pvp"];
         printf("Reading %s... ", fname);
         fflush(stdout);
         source_data = readpvpfile(fname);
         num_frames = size(source_data)(1);
         total_found += num_frames;
         printf("%d frames.\n", num_frames);
         fflush(stdout);
         source_index = 1;
         for f = 1:num_frames
            result{result_index}.values = source_data{source_index}.values;
            result{result_index}.time   = source_data{source_index}.time;
            result_index++;
            source_index++;
         endfor
         i--;
      endwhile

   else
      disp("Supported batch methods are byFile or byList. Exiting.");
      return;
   end

   if total_found != total_frames
      printf("Warning: Found %d frames, expected %d.\n", total_found, total_frames);
      fflush(stdout);
   endif

   writepvpactivityfile([layer_name, ".pvp"], result);
   printf("Finished assembling %s\n", [layer_name, ".pvp"]);
   fflush(stdout);
endfunction
