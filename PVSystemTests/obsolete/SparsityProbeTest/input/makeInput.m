ysize = 8;
xsize = 8;
%index
numActive = 1;
frameIdx = 0;

for i = 1:(ysize*xsize)-1 %-1 so the entire image isn't the same
   outmat = zeros(ysize, xsize);
   posidx = randperm(xsize*ysize)(1:numActive);
   outmat(posidx) = .5;
   imwrite(outmat, ['input',num2str(frameIdx),'.png']);
   numActive += 1;
   frameIdx += 1;
end
