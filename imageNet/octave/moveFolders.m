function str_folder = moveFolders( old_name, new_name )
  file_list = glob([old_name, "*.*"]);
  mkdir(new_name);
  movefile(file_list, new_name);
endfunction %% renameFolders
