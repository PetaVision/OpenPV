function str_path = strExtractPath( str_file_pathname )
  %% returns the path, defined as the string up to the last separator
  path_ndx = strfind(str_file_pathname, filesep);
  len_ndx = length(path_ndx);
  last_ndx = path_ndx(len_ndx);
  len_str = length(str_file_pathname);
  if last_ndx == len_str
    last_ndx = path_ndx(len_ndx-1);
  endif
  str_path = str_file_pathname(1:last_ndx);
endfunction %% strExtractPath
