addpath("~/workspace/PetaVision/mlab/imgProc");

workspace_path = "~/workspace";
PASCAL_VOC_path = [workspace_path, filesep, "PASCAL_VOC"];
if ~exist(PASCAL_VOC_path, "dir")
   error(["does not exist PASCAL_VOC_path: ", PASCAL_VOC_path]);
endif
addpath([PASCAL_VOC_path, filesep, "mlab"]);


%% set VOCdevkit paths
VOCdevkit_path = fullfile(PASCAL_VOC_path, "VOCdevkit");
if ~exist(VOCdevkit_path, "dir")
  error("VOCdevkit_path does not exist: ", VOCdevkit_path);
endif
addpath(VOCdevkit_path);
VOCcode_path = fullfile(VOCdevkit_path, "VOCcode");
addpath(VOCcode_path);

