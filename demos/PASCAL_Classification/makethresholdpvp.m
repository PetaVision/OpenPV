function makethresholdpvp(inputthresholdsfile, outputpvpfile)
# makethresholdpvp(inputthresholdfile, outputpvpfile)
# inputthresholdfile is a file of thresholds, formatted thus:
# {
#   [1,1] = bottle, 0.27535, 0.03, 0.53119, 0.072304, 0.99007
#   [2,1] = diningtable, 0.11631, 0.07, 0.51693, 0.036686, 0.99718
#   [3,1] = pottedplant, 0.11237, 0.05, 0.50908, 0.020193, 0.99797
#   [4,1] = chair, 0.10691, 0.11, 0.51446, 0.032584, 0.99633
#   [5,1] = person, 0.10435, 0.49, 0.52036, 0.049327, 0.9914
#   [6,1] = horse, 0.045115, 0.09, 0.57957, 0.16892, 0.99023
#   [7,1] = sofa, 0.013709, 0.1, 0.50685, 0.020431, 0.99327
#   [8,1] = background, -0.22364, 1.26, 0.49793, 0.0044646, 0.9914
#   [9,1] = cow, -0.3504, 0.05, 0.50283, 0.0070859, 0.99857
#   [10,1] = bird, -0.36758, 0.1, 0.51441, 0.036693, 0.99213
#   [11,1] = dog, -0.3753, 0.14, 0.5189, 0.045469, 0.99234
#   [12,1] = tvmonitor, -0.40633, 0.06, 0.52262, 0.0497, 0.99555
#   [13,1] = sheep, -0.42595, 0.04, 0.53771, 0.078212, 0.99721
#   [14,1] = cat, -0.49281, 0.12, 0.52863, 0.063391, 0.99387
#   [15,1] = bus, -0.51555, 0.08, 0.53516, 0.078848, 0.99146
#   [16,1] = bicycle, -0.52265, 0.09, 0.59635, 0.20242, 0.99028
#   [17,1] = motorbike, -0.56057, 0.13, 0.56955, 0.14832, 0.99078
#   [18,1] = train, -0.63489, 0.11, 0.52427, 0.054092, 0.99445
#   [19,1] = car, -0.84336, 0.25, 0.51398, 0.035937, 0.99202
#   [20,1] = boat, -0.99052, 0.07, 0.53483, 0.073427, 0.99624
#   [21,1] = aeroplane, -1.3188, 0.09, 0.54851, 0.10482, 0.99221
# }
#
# outputpvpfile is the path to write a pvp weights file.
# It will be a shared weights pvp file, with nxp=1, nyp=1, nfp=21, numKernels=21
# The thresholds are taken from the second number after the category name
# (e.g. in the example above, bottle=0.03, dinigtable=0.07, etc.)
# The categories are defined below.

if isempty(which('readpvpfile'))
   addpath('/home/ec2-user/mountData/workspace/PetaVision/mlab/util/');
end%if

categories = {'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'};

numcategories = numel(categories);

W = cell(1);
W{1}.time = 0;
W{1}.values = cell(1);
W{1}.values{1} = zeros([1 1 numcategories numcategories]);
for k=1:numcategories
   commandstring = sprintf("cat \"%s\" | awk '$3~/^%s,$/ {print $5}'", inputthresholdsfile, categories{k});
   [~,w] = system(commandstring);
   if isempty(w)
      W{1}.values{1}(1,1,k,k) = 0.0;
   else
      threshold = sscanf(w, '%f');
      W{1}.values{1}(1,1,k,k) = 1/threshold;
   end%if
end%for
writepvpsharedweightfile(outputpvpfile, W);
