function out = unscramble(in,shuffle_key)
  out = zeros(size(in)(1),size(in)(2));
  for i = 1:numel(in)
    out(shuffle_key(i)) = in(i);
  endfor
endfunction