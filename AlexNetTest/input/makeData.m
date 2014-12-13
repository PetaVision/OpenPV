dataName = "train";
numFrames = 1000;

mkdir(dataName);
inputFilename = [dataName, '/input.txt'];
gtFilename = [dataName, '/gt.txt'];

inputFile = fopen(inputFilename, 'w');
gtFile = fopen(gtFilename, 'w');

for(i = 1:numFrames)
   inData = randi([0 3], 1, 1);
   if(inData == 1 || inData == 2);
      outAns = 1;
   else
      outAns = 0;
   end
   fprintf(inputFile, '%d', inData);
   fprintf(gtFile, '%d', outAns);
end

