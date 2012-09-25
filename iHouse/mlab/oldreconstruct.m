%
% Generates STDP params files
%

clear all; close all; more off; clc;
system("clear");
%warning off;
format short g;
global MOVIE_FLAG = 0;
global SWEEP_POS = 0;

addpath('/Users/slundquist/Documents/workspace/HyPerSTDP/analysis/');

%global MOVIE_FLAG = 0;
%global PRINT_FLAG;
%global SWEEP_POS = 0;
%
%setenv("GNUTERM", "x11");
%
%fullOrient_DATASET = "orient_36r";
%DATASET = "orient_36r"; %%orient_36r orient_simple   OlshausenField_raw12x12_tinyAll OlshausenField_whitened32x32_tinyAll1000 blankSquare OlshausenField_raw32x32_tiny
%
%TEMPLATE = "STDPgeneral";
%on_v1_file = "w4_post.pvp";
%off_v1_file = "w5_post.pvp";
%v1_file = "S1.pvp";

numsteps = 500;

%Masquelier params
%pr = 17*0.01/34*0.0085;


%% INDEXES
%STRENGTH_IMAGE2RETINAi = 8;
%
%wMaxInitSTDPi = 9;
%wMinInitSTDPi = 10;
%tauLTPi = 11;
%tauLTDi = 12;
%ampLTPi = 13;
%ampLTDi = 14;
%wMini = 15;
%wMaxi = 16;
%synscalingi = 17;
%synscalingvi = 18;
%movieMarginWidthi = 19;
%
%RUN_FLAG = 1;
%
%MEASURES_FLAG = 1;
%    MEASURES_PLOT_FLAG = 0;
%    MEASURES_OSI_FLAG = 0;
%    MEASURES_GM_FLAG = 1;
%
%global ROTATE_FLAG; ROTATE_FLAG = 0;

v1_cells = 7*7;
img_size = 256;
ign_w = 2;
fid = 0;


%global params;
%params{1} = "false";  %checkpointRead
%params{2} = "true";  %checkpointWrite
%params{3} = DATASET;  %checkpointReadDir
%CHECKPOINTREADi = 0;  %checkpointReadDirIndex
CHECKPOINTSTEPi = 500;  %checkpointWriteStepInterval
%params{6} = "true"; %plasticityFla
DISPLAYPERIODi = 1; %displayPeriod (image display period)
%params{STRENGTH_IMAGE2RETINAi} = 5;
%params{movieMarginWidthi} = 3;
%
%
%
%%STDP params
%%Natural images params
%params{wMaxInitSTDPi} = 0.05;
%params{wMinInitSTDPi} = 0.005;
%params{tauLTPi} = 17;
%params{tauLTDi} = 34;
%params{ampLTPi} = 0.01;
%params{ampLTDi} = 0.0085;
%params{wMini} = 0.001;
%params{wMaxi} = 1;
%params{synscalingi} = 1;
%params{synscalingvi} = 1;
%


disp("------------------------------------------");
disp("------------Generative Measure------------");
disp("------------------------------------------");
    
activityfile = '/Users/slundquist/Documents/workspace/iHouse/output/lif.pvp';
ONpostweightfile = '/Users/slundquist/Documents/workspace/iHouse/output/w5_post.pvp';
OFFpostweightfile = '/Users/slundquist/Documents/workspace/iHouse/output/w6_post.pvp';
ONfilename = 'ON_post.pvp'
OFFfilename = 'OFF_post.pvp'
outputdir = '/Users/slundquist/Documents/workspace/iHouse/output/reconstruct';
sourcefile = '/Users/slundquist/Documents/workspace/iHouse/output/DropInput.txt';
%Read entire dataset
if(fid==0)
    fid = fopen(sourcefile, 'r');
    datasetl = {};
    c=1;
    while(~feof(fid))
        datasetl{c} = fgets(fid);
        c=c+1;
    end
    fclose(fid);
end

post = 1;
hist_per_img = zeros(round(numsteps/CHECKPOINTSTEPi+1), v1_cells, length(datasetl));
diff = ones(round(numsteps/CHECKPOINTSTEPi)+1, length(datasetl)).*Inf;
 
%Generative measure
%1. For each image matrix
%For last image matrix
i=numsteps
%   params{CHECKPOINTREADi} = i;  %checkpointReadDirIndex

PRINT_FLAG = 0;
[data hdr] = readpvpfile(activityfile, outputdir, 'lif');

%Reads the weights Retina_ON > V1 for the time being
PRINT_FLAG = 0;
[d hdrw wm] = readpvpfile(ONpostweightfile, outputdir, ONfilename, post);
[dOff hdrwOff wmOff] = readpvpfile(OFFpostweightfile, outputdir, OFFfilename, post);

wm = cleanWM(wm, v1_cells, hdrw, ign_w, img_size);
wmOff = cleanWM(wmOff, v1_cells, hdrwOff, ign_w, img_size);

if(PARAMSWEEP_FLAG==0)
  figure
  imshow(wm);
end

for p=0:(length(datasetl)-1)
   for f=1:params{DISPLAYPERIODi}
      for v=1:v1_cells
          hist_per_img(round(i/CHECKPOINTSTEPi+1),v,p+1) = hist_per_img(round(i/CHECKPOINTSTEPi+1),v,p+1) + data{p*params{DISPLAYPERIODi}+f}(v);
      end
   end
end

%keyboard

%3. Reconstruct the original image
for p=1:(length(datasetl)) %Loop over images
   img_recons = zeros(img_size);
   for v=1:v1_cells %Loop over cells
      %mean_act = hist_per_img(round(i/params{CHECKPOINTSTEPi})+1,v,p)/params{DISPLAYPERIODi};
      mean_act = hist_per_img(round(i/CHECKPOINTSTEPi)+1,v,p);

      if(mean_act>0) %If cell reconstruct
         [c r] = ind2sub([sqrt(v1_cells) sqrt(v1_cells)], v);
         w=wm((r-1)*img_size+1:r*img_size, (c-1)*img_size+1:c*img_size);
         wOff=wmOff((r-1)*img_size+1:r*img_size, (c-1)*img_size+1:c*img_size);
         img_recons = img_recons .+ (mean_act .* w - mean_act .* wOff);
         if(PARAMSWEEP_FLAG==0)
            [r c mean_act]
         end                            
      end
   end
   img_recons(img_recons<0) = 0;
   if(sum(sum(img_recons))>0)

      img_recons = img_recons./max(max(img_recons));
      img_orig = imread(strtrim(datasetl{p}));

      %Crop img_orig based on the margin width
      %img_orig = img_orig(params{movieMarginWidthi}+1:params{movieMarginWidthi}+img_size, params{movieMarginWidthi}+1:params{movieMarginWidthi}+img_size);
      img_orig = cast(img_orig,'double');
      img_orig = img_orig./max(max(img_orig));
      diff(round(i/CHECKPOINTSTEPi)+1,p) = mean(mean(abs(img_orig-img_recons)));
      if(i==numsteps)%Only plot the last ones
         figure
         subplot(1,2,1);
         imshow(img_orig);
         title('Original');
         subplot(1,2,2);
         imshow(img_recons);
         %sum(sum(img_recons))
         title(['Reconstruction  diff=' num2str(diff(round(i/CHECKPOINTSTEPi)+1,p))]);
      end
   end
end
