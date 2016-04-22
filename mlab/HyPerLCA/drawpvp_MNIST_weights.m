%% function drawpvp_MNIST_weights

function weights = drawpvp_MNIST_weights(weights,recon_dir)

  [wstruct, hdr] = readpvpfile(weights(1,:),1);
  mw = wstruct{1}.values{1};

  [wstruct, hdr] = readpvpfile(weights(2,:),1);
  lw = wstruct{1}.values{1};

  h = figure;
  set(h,"name","Images and Labels");
  for i = 1:100
      subplot(10, 10, i);
      patch = squeeze(mw(:,:,1,i));
      imagesc(patch_tmp2); colormap(gray);
      [~, label] = max(lw(:,:,:,i),3);
      title(num2str(label)); 
      box off
      axis off
    endfor
  endfor
  saveas(h, [recon_dir, filesep, "ImagesLabels", ".png"]);
  drawnow;
endfunction
