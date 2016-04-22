%% function plotstats

function plotstats(statsfile,field,normalize, stats_dir,startline)
  
  if ~exist(stats_dir)  
    mkdir(stats_dir);
  endif
  fname = statsfile(strfind(statsfile,filesep)(end):end-10);
  Stats_fid = fopen(statsfile, "r");
  Stats_line = fgets(Stats_fid);
  ave = [];

  num_N = 1;
  if normalize == 1
    Stats_ndx1 = strfind(Stats_line, "N==");
    Stats_ndx2 = strfind(Stats_line, "Total==");
    Stats_str = Stats_line(Stats_ndx1+3:Stats_ndx2-2);
    num_N = str2num(Stats_str);
  endif
  i = 0;
  for i = 1:startline
    Stats_line = fgets(Stats_fid);
  end
  while (Stats_line ~= -1)
    Stats_ndx1 = strfind(Stats_line,field);
    Stats_ndxs = strfind(Stats_line, " ");
    Stats_ndx2 = Stats_ndxs(find(Stats_ndxs>Stats_ndx1));
    if isempty(Stats_ndx2)
      Stats_ndx2 = length(Stats_line);
    elseif length(Stats_ndx2) > 1
      Stats_ndx2 = Stats_ndx2(1);
    end
    Stats_str = Stats_line(Stats_ndx1+length(field)+2:Stats_ndx2);
    Stats_val = str2num(Stats_str);
    if isempty(ave)
      ave = Stats_val/num_N;
    else
      ave = [ave; Stats_val/num_N];
    endif
    Stats_line = fgets(Stats_fid);
  endwhile
  fclose(Stats_fid);
  h = figure;
  plot(ave); axis tight;
  set(h, "name", ["ave_" fname "_" field]);
  saveas(h, [stats_dir, filesep, fname "_" field "vs_time_"], "png");
  drawnow;
endfunction
