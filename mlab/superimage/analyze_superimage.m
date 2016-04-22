function analyze_superimage(original_sparse, superimage_recon, decompose_key)
  prescramble = zero_inflator(original_sparse);
  decompose_superimage(superimage_recon, decompose_key);
  postscramble = zero_inflator('sparse1.pvp');
  overlap = min(numel(prescramble), numel(postscramble));
  for i = 1:overlap
    err(i) = mean((prescramble{i}.values - postscramble{i}.values).^2)./mean(prescramble{i}.values);
  endfor
  err'
  plot(err)
endfunction