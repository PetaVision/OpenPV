function out = zero_inflator(infile)
  if ischar(infile)
    [in,head] = readpvpfile(infile, 100);
  endif
  N = head.nx * head.ny * head.nf;
  for i = 1:length(in)
    current_index = in{1}.values(:,1);
    current_values = in{1}.values(:,2);
    
    tmp = zeros(N,1);
    tmp(current_index + 1) = current_values;
    
    tmp = reshape(tmp, [head.nf, head.nx, head.ny]);

    out{i}.time = in{1}.time;
    out{i}.values = permute(tmp, [3,2,1]);
  endfor
endfunction