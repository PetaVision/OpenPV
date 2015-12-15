imgListFilename = '~/mountData/datasets/kitti/list/benchmark_depth_disp_noc.txt';
outFilename = '~/mountData/benchmark/depth_hist.png'

imgList = {};
numLines = 0;
numBins = 256;

fileptr = fopen(imgListFilename, 'r');
fgetl(fileptr); %Throw away first image
while ~feof(fileptr)
   tline = fgetl(fileptr);
   numLines++;
   imgList(numLines) = tline;
end


data = [];
for i = 1 : length(imgList)
   imgList{i}
   depth = imread(imgList{i});
   depth = depth(find(depth != 0));
   data = cat(1, data, depth(:));
end

f = figure;
hist(data, min(data):10:max(data));



title('Histogram of depths', 'FontSize', 28);
xlabel('Near depth          Far depth', 'FontSize', 16);
ylabel('Frequency', 'FontSize', 16);

saveas(f, outFilename);
