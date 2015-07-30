disp('======= KITTI DevKit Demo =======');
clear all; close all; dbstop error;

addpath('../devkit/matlab/')
addpath('~/workspace/PetaVision/mlab/util')

% error threshold
tau = 3;

%Debug mode, prints out additional images if true
debug = true;

outdirs = {'/home/ec2-user/mountData/geoint/validate/bias_on_clip_128/'; ...
           '/home/ec2-user/mountData/geoint/validate/bias_on_clip_149/'; ...
           '/home/ec2-user/mountData/geoint/validate/bias_on_clip_168/'; ...
          }
clips = [128, 149, 168];

%outdirs = {'/home/ec2-user/mountData/geoint/validate/bias_on_clip_128/'; ...
%          }
%clips = [128];

%outdirs = {'/home/ec2-user/mountData/geoint/validate/slp_on_clip_128/'; ...
%          }
%clips = [128];



%timestamp = [outdir '/timestamps/DepthImage.txt'];
gtPvpFile = ['/home/ec2-user/mountData/benchmark/validate/aws_rcorr_white_LCA/a3_DepthDownsample.pvp'];
imageDir = 's3://kitti/stereo_flow/multiview/training/image_2'

[data_gt, hdr_gt] = readpvpfile(gtPvpFile);

rescaleSize = .08;
framesPeriod = 11;
%framesIdx = 30; %Frame 129, with the pvp file starting at frame 99
framesOffset = 99;

for(c = 1:length(outdirs))
   outdir = outdirs{c};
   targetClip = clips(c);
   framesIdx = clips(c) - framesOffset; %zero indexed

   slpPvpFile = [outdir 'a4_RCorrRecon.pvp'];
   biasPvpFile = [outdir 'a2_AveDepth.pvp'];
   totalPvpFile = [outdir 'a5_FinalRecon.pvp'];
   reconPvpFile = [outdir 'a1_LeftRecon.pvp'];

   scoreDir = [outdir 'scores/'];
   mkdir(scoreDir);
   [data_slp, hdr_est] = readpvpfile(slpPvpFile);
   [data_bias, hdr_est] = readpvpfile(biasPvpFile);
   [data_est, hdr_est] = readpvpfile(totalPvpFile);
   [data_recon, hdr_recon] = readpvpfile(reconPvpFile);

   for(framesCount = 1:framesPeriod)
      estData = data_est{framesCount}.values' * 256;
      slpData = data_slp{framesCount}.values' * 256;
      biasData = data_bias{framesCount}.values' * 256;
      reconData = data_recon{framesCount}.values' * 256;
      gtData = data_gt{framesIdx+1}.values' * 256;
      maxDisp = max(gtData(:));

      handle = figure;
      targetTime = data_est{framesCount}.time;
      targetFrame = framesCount - 1;
      
      imageFilename = sprintf('%s/%06d_%02d.png', imageDir, targetClip, targetFrame)
      system(['aws s3 cp ', imageFilename, ' tmpImg.png']);
      im = imread('tmpImg.png');

      outFilename = sprintf('%s/%06d_%02d.png', scoreDir, targetClip, targetFrame)
      [nx, ny, nf] = size(estData);

      estData(find(gtData == 0)) = 0;

      if(debug)
         g = subplot(6, 1, 1);
      else
         g = subplot(4, 1, 1);
      end

      imagesc(im);
      axis off;
      %p = get(g,'position');
      %%Left
      %p(1) = p(1) - rescaleSize/2;
      %%Bottom
      %p(2) = p(2) - rescaleSize/2;
      %%Width
      %p(3) = p(3) + rescaleSize;
      %%Height
      %p(4) = p(4) + rescaleSize;
      %set(g, 'position', p);

      if(debug)
         g = subplot(6, 1, 2);
         imagesc(reconData);
         colormap(gray);
         axis off
         %p = get(g,'position');
         %%Left
         %p(1) = p(1) - rescaleSize/2;
         %%Bottom
         %p(2) = p(2) - rescaleSize/2;
         %%Width
         %p(3) = p(3) + rescaleSize;
         %%Height
         %p(4) = p(4) + rescaleSize;
         %set(g, 'position', p);
      end

      if(debug)
         g = subplot(6, 1, 3);
      else
         g = subplot(4, 1, 2);
      end
      imshow(disp_to_color(estData, maxDisp));
      %p = get(g,'position');
      %%Left
      %p(1) = p(1) - rescaleSize/2;
      %%Bottom
      %p(2) = p(2) - rescaleSize/2;
      %%Width
      %p(3) = p(3) + rescaleSize;
      %%Height
      %p(4) = p(4) + rescaleSize;
      %set(g, 'position', p);

      if(debug)
         g = subplot(6, 1, 4);
      else
         g = subplot(4, 1, 3);
      end
      imagesc(slpData);
      %p = get(g,'position');
      %%Left
      %p(1) = p(1) - rescaleSize/2;
      %%Bottom
      %p(2) = p(2) - rescaleSize/2;
      %%Width
      %p(3) = p(3) + rescaleSize;
      %%Height
      %p(4) = p(4) + rescaleSize;
      %set(g, 'position', p);

      if(debug)
         g = subplot(6, 1, 5);
      else
         g = subplot(4, 1, 4);
      end
      imshow(disp_to_color(biasData, maxDisp));
      %p = get(g,'position');
      %%Left
      %p(1) = p(1) - rescaleSize/2;
      %%Bottom
      %p(2) = p(2) - rescaleSize/2;
      %%Width
      %p(3) = p(3) + rescaleSize;
      %%Height
      %p(4) = p(4) + rescaleSize;
      %set(g, 'position', p);

      if(debug)
         g = subplot(6, 1, 6);
         imshow(disp_to_color(gtData, maxDisp));
         %p = get(g,'position');
         %%Left
         %p(1) = p(1) - rescaleSize/2;
         %%Bottom
         %p(2) = p(2) - rescaleSize/2;
         %%Width
         %p(3) = p(3) + rescaleSize;
         %%Height
         %p(4) = p(4) + rescaleSize;
         %set(g, 'position', p);
      end

      saveas(handle, outFilename);
      close(handle);
   end
end
