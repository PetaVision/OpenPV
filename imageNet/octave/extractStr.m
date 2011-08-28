  function str_extract = extractStr( str_original, str_ndx)
    len_ndx = length(str_ndx);
    len_str = length(str_original);
    str_extract = str_original(str_ndx(len_ndx)+1:len_str);
  endfunction %% extractStr
