% Takes a pvp file extracts the activity of its features, 
% optionally between some start/stop time or over some 
% threshold. Returns an array, with frames as rows and
% features as columns

function out = extract_activity_subset(infile, stop_time, start_time = 1)

  if !exist("stop_time","var")
    stop_time = readpvpheader(fopen(infile)).nbands;
  endif

  if stop_time <= start_time
    error ("Stop time index must be greater than start time index");
  endif

  [data,header] = readpvpfile(infile,100,stop_time,start_time);

  if (header.filetype != 6)
    error ("Sparse values file required");
  endif


  out = zeros(start_time-stop_time,header.nx*header.ny*header.nf);

  for i = 1:(stop_time - start_time)
    out(i,data{i}.values(:,1)+1) = data{i}.values(:,2);
  endfor

endfunction