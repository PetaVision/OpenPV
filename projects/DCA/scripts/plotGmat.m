addpath('~/workspace/pv-core/mlab/util');

InputFiles = ...
   {...
   '~/mountData/DCA/convOnS1/a2_S1.pvp'; '~/mountData/DCA/convOnS1/a3_S2.pvp'; '~/mountData/DCA/convOnS1/a4_S3.pvp';...
   '~/mountData/DCA/convOnS2/a2_S1.pvp'; '~/mountData/DCA/convOnS2/a3_S2.pvp'; '~/mountData/DCA/convOnS2/a4_S3.pvp';...
   '~/mountData/DCA/convOnS3/a2_S1.pvp'; '~/mountData/DCA/convOnS3/a3_S2.pvp'; '~/mountData/DCA/convOnS3/a4_S3.pvp';...
   '~/mountData/MAX/convOnS1/a2_S1.pvp'; '~/mountData/MAX/convOnS1/a4_S2.pvp'; '~/mountData/MAX/convOnS1/a6_S3.pvp';...
   '~/mountData/MAX/convOnS2/a2_S1.pvp'; '~/mountData/MAX/convOnS2/a4_S2.pvp'; '~/mountData/MAX/convOnS2/a6_S3.pvp';...
   '~/mountData/MAX/convOnS3/a2_S1.pvp'; '~/mountData/MAX/convOnS3/a4_S2.pvp'; '~/mountData/MAX/convOnS3/a6_S3.pvp';...
   };

self = [...
   1, 0, 0, ...
   0, 1, 0, ...
   0, 0, 1, ...
   1, 0, 0, ...
   0, 1, 0, ...
   0, 0, 1, ...
];


%Final out matrix should have same number of elements as number of input files
outGVal = zeros(length(InputFiles), 1);
for i = 1:length(InputFiles)
   InputFiles{i}
   [data, hdr] = readpvpfile(InputFiles{i});
   sumVals = 0;
   [nx, ny, nf] = size(data{1}.values);
   numIm = numel(data);
   for j = 1:numIm
      valMat = data{j}.values;
      %Sanity check and set self value to 0
      if(self(i))
         [mval, idx] = max(valMat(:));
         [x, y, f] = ind2sub(size(valMat), idx);
         if(f != j)
            disp(['Check failed, as input feature ', num2str(j), ' max value is located at [', num2str(x), ', ', num2str(y), ', ', num2str(f), ']']);
         end
         %assert(f == j);
         valMat(idx) = 0; %Take out self value
      
         %TODO check if center
      end
      sumVals += sum(valMat(:));
   end
   outGVal(i) = sumVals / (nx*ny*nf*numIm);
end

outGVal


keyboard;

