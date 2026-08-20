function combinebatches(directory, layer_name, num_batch, batch_method, batch_width, total_frames)
   isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; % for compatibility

   file_type = -1;
   nx = 0;
   ny = 0;
   nf = 0;
   result_index = 0; 
   total_found = 0;
   if strcmp(batch_method, "byFile") == 1 % Really, Octave?
      per_batch = num_batch / batch_width;
      i = batch_width - 1;
      while i >= 0
         result_index = i * per_batch + 1;
         fname = [directory, "/batchsweep_", num2str(i, "%02d"), "/", layer_name, "_", num2str(i), ".pvp"];
         fprintf("Reading %s... ", fname);
         if isOctave, fflush(stdout); end
         [source_data, header] = readpvpfile(fname);
         file_type = header.filetype;
         nx = header.nx;
         ny = header.ny;
         nf = header.nf;
         num_frames = size(source_data)(1);
         total_found = total_found + num_frames;
         fprintf("%d frames.\n", num_frames);
         if isOctave, fflush(stdout); end
         source_index = 1;
         per_batch_left = per_batch;
         for f = 1:num_frames
            if rem(f, 5000) == 0
               disp(f);
               if isOctave, fflush(stdout); end
            end
             result{result_index}.values = source_data{source_index}.values;
            result{result_index}.time   = source_data{source_index}.time;
            source_index++;
            per_batch_left--;
            if per_batch_left > 0
               result_index++;
            else
               per_batch_left = per_batch;
               result_index = result_index + num_batch - per_batch + 1;
            end%if
         end%for
         i--;
      end%while

   elseif strcmp(batch_method, "byList") == 1
      i = batch_width - 1;
      while i >= 0
         result_index = total_frames / batch_width * i + 1;
         fname = [directory, "/batchsweep_", num2str(i, "%02d"), "/", layer_name, "_", num2str(i), ".pvp"];
         fprintf("Reading %s... ", fname);
         if isOctave, fflush(stdout); end
         [source_data, header] = readpvpfile(fname);
         file_type = header.filetype;
         nx = header.nx;
         ny = header.ny;
         nf = header.nf;
         num_frames = size(source_data)(1);
         total_found = total_found + num_frames;
         fprintf("%d frames.\n", num_frames);
         if isOctave, fflush(stdout); end
         source_index = 1;
         for f = 1:num_frames
            if rem(f, 5000) == 0
               disp(f);
               if isOctave, fflush(stdout); end
            end
            result{result_index}.values = source_data{source_index}.values;
            result{result_index}.time   = source_data{source_index}.time;
            result_index++;
            source_index++;
        end%for
         i--;
      end%while

   else
      disp("Supported batch methods are byFile or byList. Exiting.");
      return;
   end%if

   if total_found ~= total_frames
      fprintf("Warning: Found %d frames, expected %d.\n", total_found, total_frames);
      if isOctave, fflush(stdout); end
   end%if

   disp("Writing output file...");
   if isOctave, fflush(stdout); end

   if file_type == 4
      writepvpactivityfile([layer_name, ".pvp"], result, true);
   elseif file_type == 6
      writepvpsparsevaluesfile([layer_name, ".pvp"], result, nx, ny, nf, true);
   else
      fprintf("Error: Unsupported filetype %d\n", file_type);
      return;
   end%if

   fprintf("Finished assembling %s\n", [layer_name, ".pvp"]);
   if isOctave, fflush(stdout); end
endfunction
