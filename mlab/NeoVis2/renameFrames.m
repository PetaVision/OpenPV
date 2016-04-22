clear all
frame_path = ...
    "/Users/gkenyon/NeoVision2/neovision-programs-petavision/Heli/Challenge/ODD/029/Car3/canny/";
framePathnames = ...
    glob([frame_path, "[0-9]*.png"]);
addpath("~/workspace-indigo/PetaVision/mlab/stringKernels/");
frameNames = cellfun(@strFolderFromPath, framePathnames, "UniformOutput", false);
baseNames = cellfun(@strRemoveExtension, frameNames, "UniformOutput", false);
bareNum = cellfun(@str2num, baseNames);
for i_frame = 1 : length(bareNum)
%%  barePathname = strcat(frame_path, num2str(bareNum(i_frame)), ".png");
  barePathname = strcat(frame_path, num2str(i_frame), ".png");
  copyfile(framePathnames{i_frame}, barePathname);
endfor
