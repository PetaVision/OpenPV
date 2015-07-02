function new_new_create_superimage(infile1, infile2)
  depth = 10;
  startime1 = 1;
  startime2 = 1;
  weight1 = 1; 
  weight2 = 1;
  head1 = readpvpheader(fopen(infile1));
  head2 = readpvpheader(fopen(infile2));
  feature_array1 = extract_activity_subset(infile1);
  feature_array2 = extract_activity_subset(infile2);
  
  [r1,c1] = size(feature_array1);
  [r2,c2] = size(feature_array2);

  if size(feature_array1)(1) != size(feature_array2)(1)
    error("Times (rows) do not match");
  endif
  
  key.cutoff = c1;
  key.nx1 = head1.nx;
  key.ny1 = head1.ny;
  key.nf1 = head1.nf;
  key.nx2 = head2.nx;
  key.ny2 = head2.ny;
  key.nf2 = head2.nf;

  for d = 1:depth
    key.shuffle{d} = keygen([feature_array1(1,:), feature_array2(1,:)]); 
  endfor

  for r = 1:r1
    superimage{r}.time = r*head1.time;
    for d = 1:depth
      shuffled  = [feature_array1(r,:)*weight1, feature_array2(r,:)*weight2](key.shuffle{d});
      stratum = squarify(shuffled);
      superimage{r}.values(:,:,d) = stratum;
    endfor
  endfor
  
  #size(superimage)
  #size(superimage{1}.values)
  #superimage{1}.values(1:5,1:5,:)]
  save "superimage_key.mat" key;
  writepvpactivityfile("superimage.pvp", superimage);
 
endfunction