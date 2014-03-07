%% -Combines the reconstructed images from the three layers of the stack
%% -Combines shuffled images if runType is "stack"
%% -Places all reconstructions from a set in a new folder corresponding to the timestep
%% Recon_S2 prefix: Reconstruction from the stride of 2
%% Recon_S4 prefix: Reconstruction from the stride of 4
%% Recon_S8 prefix: Reconstruction from the stride of 8
%% Recon_XA prefix: Linear combination of all reconstructions
%% Recon_XN prefix: Normalized linear combination

function combineRecon%%(run_type)
  %%if strcmp(run_type, "stack" | nargin == 0)
     recon_path         = '/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_vine/Recon/';
     global shuffle     = 0;
 %% elseif strcmp(run_type, "shuffle")
 %%    recon_path         = '/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_vine_shuffle/Recon/';
 %%    global shuffle     = 1;
 %% end
global inputs_file = '/nh/compneuro/Data/vine/list/2013_01_24_2013_02_01/fileList_2013_01_24_2013_02_01.txt';
global dP = 40;
       nP = 16;

dir_list   = glob([recon_path,'*.png']);
num_images = numel(dir_list);
num_recon  = num_images/3;

for i=1:num_recon
   img_comp{1,1} = dir_list{i};
   img_comp{2,1} = dir_list{i+num_recon};
   img_comp{3,1} = dir_list{i+2*num_recon};
   img_sets{i,1} = img_comp;
end
out = parcellfun(nP, @comb, img_sets,'UniformOutput',false);
%%out = cellfun(@comb, img_sets,'UniformOutput',false);
end

function [img] = comb(in)  %% Input cell contains three full paths to each image in a set
global inputs_file
global dP
global shuffle
im1 = imread(in{1,1});
im2 = imread(in{2,1});
im3 = imread(in{3,1});
img = imlincomb((1/3),im1,(1/3),im2,(1/3),im3);
%%img = (im1*(1/3)+im2*(1/3)+im3*(1/3)); %% For older Octave image packages without imlincomb()

[path,name,ext] = fileparts(in{1});
img_ID          = regexp(name,'_\d+','match'){1}(2:end);
set_path        = [path,filesep,img_ID];
in_img_idx      = ceil(str2double(img_ID)/dP); 
in_img_dir      = textread(inputs_file, '%s', 2, 'headerlines', in_img_idx-2);
%% The recon image names are not consistently reflecting the correct timestep

mkdir(set_path);
copyfile(in_img_dir{1}, set_path); 
copyfile(in_img_dir{2}, set_path); 
movefile(in{1},         set_path);
movefile(in{2},         set_path);
movefile(in{3},         set_path);

imwrite(img, [set_path, filesep, 'Recon_XA_', img_ID, ext]);

img=(img-double(min(img(:))))*(255/((double(max(img(:)))-double(min(img(:))))+(double((max(img(:)))-double(min(img(:))))==0)));

imwrite(img, [set_path, filesep, 'Recon_XN_', img_ID, ext]);
end
