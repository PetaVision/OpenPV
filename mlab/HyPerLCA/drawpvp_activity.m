%% function drawpvp_activity

function drawpvp_activity(filename, weights, recon_dir)

  fname = filename(strfind(filename,filesep)(end)+1:end-4);
  [astruct, hdr] = readpvpfile(filename, [], [], []);
  num_frames = size(astruct,1);
  h = figure;
  amean = 0;
  astd = 0;
  for i_frame =  1 : num_frames
    atime = astruct{i_frame}.time;
    avals = astruct{i_frame}.values;
    amean = amean + mean(avals(:));
    astd = astd + std(avals(:));
    figure(h);
    set(h, "name", [fname, num2str(atime, "%0d")]);
    imagesc(permute(avals,[2,1,3])); 
    num_colors = size(avals,3);
    if num_colors == 1
      colormap(gray);
    endif
    box off; axis off; axis image;
    saveas(h, [recon_dir, filesep, fname,"_", num2str(atime, "%0d")], "png");

    if ~isempty(weights)
      unwhitened_vals = zeros(size(permute(avals,[2 1 3])));
      for i_color = 1:num_colors
        tmpa = deconvolvemirrorbc(squeeze(avals(:,:,i_color))',weights); %%'
        mean_unwhitened_vals(i_color) = mean(tmpa(:));
        unwhitened_vals(:,:,i_color) = tmpa;
      endfor
      figure(h)
      set(h,"name",[fname,num2str(atime,"%0d"),"_unwhitened"]);
      imagesc(squeeze(unwhitened_vals));
      if num_colors == 1
        colormap(gray);
      endif
      box off; axis off; axis image;
      saveas(h, [recon_dir filesep fname num2str(atime,"%0d") "_unwhitened"], "png");
      drawnow;
    endif
  endfor   %% i_frame

  amean = amean / (num_frames);
  astd = astd / (num_frames);
  disp([fname "_mean = ", num2str(amean), " +/- ", num2str(astd)]);
endfunction