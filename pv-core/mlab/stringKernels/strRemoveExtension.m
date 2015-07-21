function base_name = strRemoveExtension( file_name )
  %% returns the final folder or file name at the end of the last path separator
  base_ndx = strfind(file_name, ".");
  base_name = file_name(1:base_ndx(end)-1);
endfunction %% strRemoveExtension
