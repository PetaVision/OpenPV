function str_folder = strFolderFromPath( str_path )
  folder_ndx = strfind(str_path, filesep);
  str_folder = str_path(folder_ndx(end)+1:end);
endfunction %% strFolderFromPath
