more off;
distPath= "/nh/compneuro/Data/AnimalDB/Original_DB/Distractors/";
targPath= "/nh/compneuro/Data/AnimalDB/Original_DB/Targets/";
inputFile = "/nh/compneuro/Data/AnimalDB/Original_DB_Randomized_List.txt";

%%targPath= "/nh/compneuro/Data/PASCAL/VOC2007_TRAIN/JPEG_SUBSET/";
%%inputFile = "/nh/compneuro/Data/PASCAL/VOC2007_TRAIN/JPEG_SUBSET_List.txt";


distList = glob([distPath, '*.jpg']);
targList = glob([targPath, '*.jpg']);
allList  = [distList; targList];
n        = rand(length(allList),1);
[t, idx] = sort(n);

fileID = fopen(inputFile, "a");

%%for i=1:int32(numel(distList))
%%   imPath = distList{i};
%%   fprintf(fileID, [imPath,'\n']);
%%end

for i=1:int32(numel(allList))
   imPath = allList{idx(i)};
   fprintf(fileID, [imPath,'\n']);
end

fclose(fileID);
