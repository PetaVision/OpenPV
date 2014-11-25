function rescale()
   numProcs = 16;
   global output_path = "/nh/compneuro/Data/PASCAL/VOC2007_TRAIN/JPEG_SUBSET_US/";
   global target_image_path    = "/nh/compneuro/Data/PASCAL/VOC2007_TRAIN/JPEG_SUBSET/";
   
   %%global output_path = "/Users/wshainin/Pictures/AnimalDB/1_2_Downsample/";
   %%global target_image_path    = "/Users/wshainin/Pictures/AnimalDB/OriginalDB/Targets/";
   %%global distractor_image_path    = "/Users/wshainin/Pictures/AnimalDB/OriginalDB/Distractors/";
   %%list    = dir(fullfile(image_path, '*.jpg'));

   target_list = glob([target_image_path,'*.jpg']);
   %%distractor_list= glob([distractor_image_path,'*.jpg']);
   mkdir(output_path);
   out = parcellfun(numProcs,@downSample, target_list,'UniformOutput',false);
   %%out = parcellfun(numProcs,@downSample, distractor_list,'UniformOutput',false);
end


function [img] = downSample(in)
   global output_path
   
   img = imread(in);
   %%img = mat2gray(rgb2gray(img));
   img = imresize(img, [384 512]);
   
   
   [path, name, ext] = fileparts(in);
   imwrite(img, [output_path, name, ext]);
end

