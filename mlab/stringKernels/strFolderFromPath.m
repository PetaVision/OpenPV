function str_folder = strFolderFromPath( str_path )
  %% returns the final folder or file name at the end of the last path separator
  folder_ndx = strfind(str_path, filesep);
  len_ndx = length(folder_ndx);
  last_ndx = folder_ndx(len_ndx);
  len_str = length(str_path);
  if last_ndx == len_str
    last_ndx = folder_ndx(len_ndx-1);
  endif
  str_folder = str_path(last_ndx+1:len_str);
endfunction %% strFolderFromPath
