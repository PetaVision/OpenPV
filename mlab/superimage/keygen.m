function key = keygen(in)
  key = reshape(randperm(numel(in)), size(in));
endfunction