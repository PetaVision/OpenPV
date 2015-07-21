%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% Increases the number of weight elements in the feature dimension to NFP in a kernel (weights) pvp file
%% 
%%    Will Shainin 
%%    Mar 20, 2014
%%    modified by Gar Kenyon ~April 3, 2014
%%
%% Input: pvpFile, newNFP,  (wMinInit, wMaxInit, sparseFraction), (outFile)
%%
%%   pvpFile         - Absolute path to input file (Only supports checkpoint weights).
%%   newNFP           - The new number of features in patch. Weights will be added up to newNFP.
%%  (wMinInit        - (wMinInit and wMaxInit determine the range of initialization 
%%   wMaxInit           values (uniform random distribution). sparseFraction defines  
%%   sparseFraction)    the sparsity (% non-zero). Must be passed together, if at all.) 
%%  (outFile)        - (Absolute path to output file. Default appends _NF## to input)
%%
%% Output: pvp kernel file with new weights initialized and appended
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function increaseKernelPatchSize(pvpFile, newNFP, wMinInit, wMaxInit, sparseFraction, outFile)
   %addpath('/nh/home/wshainin/workspace/PetaVision/mlab/util');
   addpath(pwd)

   DEFAULT_wMinInit       = -1.00;
   DEFAULT_wMaxInit       =  1.00;
   DEFAULT_sparseFraction =   .90;

   if nargin < 1 || ~exist(pvpFile,'file') || ~exist('newNFP','var')  || isempty(pvpFile) || isempty(newNFP)
      error('increaseNumFeatInFile:invalidinputarguments',...
         'pvpFile and newNF are required arguments.');
   end%if
   filedata = dir(pvpFile);
   if length(filedata) ~= 1
      error('increaseNumFeatInFile:notonefile',...
         'Path %s should expand to exactly one file; in this case there are %d',...
         pvpFile,length(filedata));
   end%if
   if nargin < 5 || ~exist('wMinInit','var') || ~exist('wMaxInit','var') || ~exist('sparseFraction','var') || isempty(wMinInit) || isempty(wMaxInit) || isempty(sparseFraction)
      warning('increaseNumFeatInFile:initvaluesnotspecified',...
         'Using default values for wMinInit, wMaxInit, and sparseFraction: ');
      wMinInit        = DEFAULT_wMinInit
      wMaxInit        = DEFAULT_wMaxInit
      sparseFraction  = DEFAULT_sparseFraction
   end%if
   if nargin < 6 || ~exist('outFile','var') || isempty(outFile)
	[path,name,ext] = fileparts(pvpFile);
	name_id = name(1:strfind(name, '_W')-1);
	outFile = [path, filesep, name_id, '_NFP', num2str(newNFP), '_W', ext]
	warning('increaseNumFeatInFile:outputfilenotspecified',...
		'Using default naming convention for output file:', ' ', outFile);
   end%if

   [data, hdr] = readpvpfile(pvpFile);
   nf = hdr.nf;

   num_arbors = length(data{1,1}.values);
   for i_arbor = 1 : num_arbors
   for i=1:nf
      for j=hdr.nfp+1:newNFP
         init = unifrnd(wMinInit, wMaxInit, [hdr.nyp, hdr.nxp]);
         rnd  = unifrnd(0, 1, [hdr.nyp, hdr.nxp]);
         rej  = find(rnd < sparseFraction);
         init(rej) = 0.00;
         data{1,1}.values{i_arbor}(:,:,j,i) = init;
      end%for
      %%figure([i]);
      %%imagesc(data{1}.values{1}(:,:,:,i))
   end%for
   end%for
   newSize = size(data{1}.values{1})
   writepvpsharedweightfile(outFile, data);

end%function
