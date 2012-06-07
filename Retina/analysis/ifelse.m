% --- file: ifelse.m --- %
function RES = ifelse(c,t,f)
  RES = c .* t + (1-c) .* f ;
end
