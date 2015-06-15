disp('======= KITTI DevKit Demo =======');
clear all; close all; dbstop error;

addpath('../devkit/matlab/')
addpath('~/workspace/PetaVision/mlab/util')

% error threshold
tau = 3;

%Debug mode, prints out additional images if true
debug = false;

outdirs = {'/nh/compneuro/Data/Depth/geoint/validate/on_clip_128/'; ...
           '/nh/compneuro/Data/Depth/geoint/validate/on_clip_149/'; ...
           '/nh/compneuro/Data/Depth/geoint/validate/on_clip_168/'; ...
          }


%timestamp = [outdir '/timestamps/DepthImage.txt'];
gtPvpFile = ['/nh/compneuro/Data/Depth/LCA/benchmark/validate/recons_run/a10_DepthDownsample.pvp'];
imageDir = '/nh/compneuro/Data/KITTI/stereo_flow/multiview/training/image_2/'

[data_gt, hdr_gt] = readpvpfile(gtPvpFile);

rescaleSize = .08;
framesPeriod = 11;
clips = [128, 149, 168];
%framesIdx = 30; %Frame 129, with the pvp file starting at frame 99
framesOffset = 99;

for(c = 1:length(outdirs))
   outdir = outdirs{c};
   targetClip = clips(c);
   framesIdx = clips(c) - framesOffset; %zero indexed

   outPvpFile = [outdir 'a3_RCorrRecon.pvp'];
   reconPvpFile = [outdir 'a1_LeftRecon.pvp'];
   scoreDir = [outdir 'scores/'];
   mkdir(scoreDir);
   [data_est, hdr_est] = readpvpfile(outPvpFile);
   [data_recon, hdr_recon] = readpvpfile(reconPvpFile);

   for(framesCount = 1:framesPeriod)
      estData = data_est{framesCount}.values' * 256;
      reconData = data_recon{framesCount}.values' * 256;
      gtData = data_gt{framesIdx+1}.values' * 256;
      maxDisp = max(gtData(:));

      handle = figure;
      targetTime = data_est{framesCount}.time;
      targetFrame = framesCount - 1;
      
      imageFilename = sprintf('%s/%06d_%02d.png', imageDir, targetClip, targetFrame)
      im = imread(imageFilename);

      outFilename = sprintf('%s/%06d_%02d.png', scoreDir, targetClip, targetFrame)
      [nx, ny, nf] = size(estData);

      estData(find(gtData == 0)) = 0;

      if(debug)
         g = subplot(4, 1, 1);
      else
         g = subplot(2, 1, 1);
      end

      imagesc(im);
      axis off;
      p = get(g,'position');
      %Left
      p(1) = p(1) - rescaleSize/2;
      %Bottom
      p(2) = p(2) - rescaleSize/2;
      %Width
      p(3) = p(3) + rescaleSize;
      %Height
      p(4) = p(4) + rescaleSize;
      set(g, 'position', p);

      if(debug)
         g = subplot(4, 1, 2);
         imagesc(reconData);
         colormap(gray);
         axis off
         p = get(g,'position');
         %Left
         p(1) = p(1) - rescaleSize/2;
         %Bottom
         p(2) = p(2) - rescaleSize/2;
         %Width
         p(3) = p(3) + rescaleSize;
         %Height
         p(4) = p(4) + rescaleSize;
         set(g, 'position', p);
      end

      if(debug)
         g = subplot(4, 1, 3);
      else
         g = subplot(2, 1, 2);
      end

      imshow(disp_to_color(estData, maxDisp));
      p = get(g,'position');
      %Left
      p(1) = p(1) - rescaleSize/2;
      %Bottom
      p(2) = p(2) - rescaleSize/2;
      %Width
      p(3) = p(3) + rescaleSize;
      %Height
      p(4) = p(4) + rescaleSize;
      set(g, 'position', p);

      if(debug)
         g = subplot(4, 1, 4);
         imshow(disp_to_color(gtData, maxDisp));
         p = get(g,'position');
         %Left
         p(1) = p(1) - rescaleSize/2;
         %Bottom
         p(2) = p(2) - rescaleSize/2;
         %Width
         p(3) = p(3) + rescaleSize;
         %Height
         p(4) = p(4) + rescaleSize;
         set(g, 'position', p);
      end

      saveas(handle, outFilename);
      close(handle);
   end
end
