function imageNetDrawBB(original_image, BB_list)

  global NUM_FIGS

  figure
  NUM_FIGS = NUM_FIGS + 1;
  image(original_image)
  hold on

  num_BB = size(BB_list, 1);
  for i_BB = 1 : num_BB

    BB_xmin = BB_list(i_BB, 1);
    BB_xmax = BB_list(i_BB, 2);
    BB_ymin = BB_list(i_BB, 3);
    BB_ymax = BB_list(i_BB, 4);
    
    hold on
    lh = line( [BB_xmin, BB_xmax], [BB_ymin, BB_ymin]);
    hold on
    set(lh, "color", [1 0 0]);
    set(lh, "linewidth", [2]);
    hold on
    lh = line( [BB_xmin, BB_xmax], [BB_ymax, BB_ymax]);
    hold on
    set(lh, "color", [1 0 0]);
    set(lh, "linewidth", [2]);
    hold on
    lh = line( [BB_xmin, BB_xmin], [BB_ymin, BB_ymax]);
    hold on
    set(lh, "color", [1 0 0]);
    set(lh, "linewidth", [2]);
    hold on
    lh = line( [BB_xmax, BB_xmax], [BB_ymin, BB_ymax]);
    hold on
    set(lh, "color", [1 0 0]);
    set(lh, "linewidth", [2]);
    
  endfor

endfunction %% imageNetDrawBB