%% -Combines the reconstructed images from the three layers of the stack
%% -Combines shuffled images if runType is "stack"
%% -Places all reconstructions from a set in a new folder corresponding to the timestep
%% Recon_S2 prefix: Reconstruction from the stride of 2
%% Recon_S4 prefix: Reconstruction from the stride of 4
%% Recon_S8 prefix: Reconstruction from the stride of 8
%% Recon_XA prefix: Linear combination of all reconstructions
%% Recon_XN prefix: Normalized linear combination

function combineRecon(run_type, source_orig)
   global inputs_file = '/nh/compneuro/Data/vine/list/2013_01_24_2013_02_01/fileList_2013_01_24_2013_02_01.txt';
   global disp_period = 40;
   num_procs = 16;
   num_layers = 3;
   if nargin < 2
      global s_o = 0;
   else
      global s_o = source_orig;
   end
   if nargin == 0 || strcmp(run_type, "stack")
      recon_path         = '/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_vine/Recon/'
      global shuffle     = 0;
      dir_list   = glob([recon_path,'*.png']);
      num_images = numel(dir_list);
      num_recon  = num_images/num_layers;
   elseif strcmp(run_type, "shuffle")
      recon_path         = '/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_vine_shuffle/Recon/'
      global shuffle     = 1;
      dir_list   = glob([recon_path,'*.png']);
      num_images = numel(dir_list);
      num_recon  = (num_images/num_layers)/2;
   end

   for i=1:num_recon 
      for j=1:num_layers
         img_comp{j,1} = dir_list{i + (j-1)*num_recon};
         if shuffle
            img_comp{j+num_layers,1} = dir_list{i + num_images/2 + (j-1)*num_recon};
         end
      end
      img_sets{i,1} = img_comp;
   end
   
   out = parcellfun(num_procs, @comb, img_sets,'UniformOutput',false);
   %%out = cellfun(@comb, img_sets,'UniformOutput',false);
end

function [img] = comb(in)  %% Input cell contains three full paths to each image in a set
   global inputs_file
   global disp_period
   global shuffle
   global s_o

   %% Probably make the following dynamic, not hard-coded. Lazyass
   im1 = imread(in{1,1});
   im2 = imread(in{2,1});
   im3 = imread(in{3,1});
   %%img = imlincomb((1/3),im1,(1/3),im2,(1/3),im3);
   img = (im1*(1/3)+im2*(1/3)+im3*(1/3)); %% For older Octave image packages without imlincomb()
   
   if shuffle
      shuf_im1 = imread(in{4,1});
      shuf_im2 = imread(in{5,1});
      shuf_im3 = imread(in{6,1});
      %%shuf_img = imlincomb((1/3),shuf_im1,(1/3),shuf_im2,(1/3),shuf_im3);
      shuf_img = (shuf_im1*(1/3)+shuf_im2*(1/3)+shuf_im3*(1/3)); %% For older Octave image packages without imlincomb()
   end
   
   [path,name,ext] = fileparts(in{1});
   img_ID          = regexp(name,'_\d+','match'){1}(2:end);
   set_path        = [path,filesep,img_ID,filesep];
   mkdir(set_path);

   if s_o
      %% TODO: The recon image names are not consistently reflecting the correct timestep. Fix this. Allow for skipped frames
      %%in_img_idx      = ceil(str2double(img_ID)/disp_period); 
      %%in_img_dir      = textread(inputs_file, '%s', 2, 'headerlines', in_img_idx-2);
      %% copyfile(in_img_dir{1}, set_path); 
      %% copyfile(in_img_dir{2}, set_path); 
   end

   movefile(in{1}, set_path);
   movefile(in{2}, set_path);
   movefile(in{3}, set_path);

   %%imwrite(img, [set_path, 'Recon_XA_', img_ID, ext]);
   img=(img-double(min(img(:))))*(255/((double(max(img(:)))-double(min(img(:))))+(double((max(img(:)))-double(min(img(:))))==0)));
   imwrite(img, [set_path, 'Recon_XN_', img_ID, ext]);
   
   if shuffle
      movefile(in{4}, set_path);
      movefile(in{5}, set_path);
      movefile(in{6}, set_path);
      
      %%imwrite(shuf_img, [set_path, 'Shuffle_Recon_XA_', img_ID, ext]);
      shuf_img=(shuf_img-double(min(shuf_img(:))))*(255/((double(max(shuf_img(:)))-double(min(shuf_img(:))))+(double((max(shuf_img(:)))-double(min(shuf_img(:))))==0)));
      imwrite(shuf_img, [set_path, 'Shuffle_Recon_XN_', img_ID, ext]);
   end

end
