function decompose_superimage(filename, keyfile)
  load(keyfile);
  superimage = readpvpfile(filename,100);
  depth = length(key.shuffle);

  for i = 1:length(superimage)
    all_flat_acts = zeros(1,numel(superimage{1}.values)/depth);
    # Assign times                                                                                       
    time = superimage{i}.time;
    sparse1{i}.time = time;
    sparse2{i}.time = time;
    for d = 1:depth
      # Turn arbitary superimage matrix into a vector
      unordered_deep_acts = desquarify(superimage{i}.values(:,:,d));
      # Reorder vector at each depth-level, according to shuffle indexes
      ordered_deep_acts = unscramble(unordered_deep_acts, key.shuffle{d});
      # Flatten depth into single vector with averages
      all_flat_acts = all_flat_acts + ordered_deep_acts./depth;
    endfor
    # Assign activations and indexes to values for first outfile
    sparse1{i}.values(:,2) = all_flat_acts(1:key.cutoff);
    sparse1{i}.values(:,1) = (1:length(sparse1{i}.values(:,2)))-1;
    # Assign activations and indexes to values for second outfile
    sparse2{i}.values(:,2) = all_flat_acts(key.cutoff+1:end);
    sparse2{i}.values(:,1) = (1:length(sparse2{i}.values(:,2)))-1;
    disp(["Completed cell " int2str(i) " out of " int2str(length(superimage))]);
    fflush(stdout);
  endfor
  sparse1 = zero_deflator(sparse1);
  sparse1 = zero_deflator(sparse2);
  writepvpsparsevaluesfile("sparse1.pvp", sparse1, key.nx1, key.ny1, key.nf1)
  writepvpsparsevaluesfile("sparse2.pvp", sparse2, key.nx2, key.ny2, key.nf2)
endfunction