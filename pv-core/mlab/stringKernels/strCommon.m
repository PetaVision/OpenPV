function str_common = strCommon(str1, str2)
  %% returns the portion of str1 and str2 that are identical
  len_str1 = length(str1);
  len_str2 = length(str2);
  for i_char = 1 : min(len_str1, len_str2)
    if strcmp(str1(i_char),str2(i_char))
      str_common(i_char) = str1(i_char);
    else
      break;
    endif
  endfor
endfunction %% strCommon
