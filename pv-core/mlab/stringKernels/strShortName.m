function str_short = strShortName( str_long )
  short_ndx = strfind(str_long, ",");
  str_short = str_long(1:short_ndx(1)-1);
endfunction %% strShortName
