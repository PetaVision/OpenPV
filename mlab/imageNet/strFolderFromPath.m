function str_folder = strFolderFromPath( str_path )
  %% returns the final folder or file name at the end of the last path separator
  folder_ndx = strfind(str_path, filesep);
  str_folder = str_path(folder_ndx(end)+1:end);
endfunction %% strFolderFromPath
