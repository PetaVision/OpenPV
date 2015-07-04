function out = zero_deflator(infile)
  if ischar(infile)
    in = readpvpfile(infile,100);
  elseif iscell(infile)
    in = infile;
  endif

  for i = 1:length(in)
    non_zero = in{i}.values(:,2)~=0;
    in{i}.values = in{i}.values(non_zero,:);
  endfor
  out = in;
endfunction