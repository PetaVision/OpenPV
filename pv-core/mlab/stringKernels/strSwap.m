function  [str_new] = strSwap(str_old, pattern_old, pattern_new)
  %% makes a new string by swaping each occurance of the old pattern with the new pattern
  [old_start, old_end] = regexp (str_old, pattern_old, "start", "end", "once");
  num_match = length(old_start);
  str_new = [];
  len_str = length(str_old);
  if old_start(1)>1
    str_new = str_old(1:old_start(1)-1);
  endif
  str_new = [str_new, pattern_new];
  for i_match = 2:num_match
    str_new = [str_new, str_old(old_end(i_match-1)+1:old_start(i_match)-1)];
  endfor
  str_new = [str_new, str_old(old_end(num_match)+1:len_str)];
endfunction %% strSwap