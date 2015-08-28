s1output = 'deconvS1.pvp';
s2output = 'deconvS2.pvp';
s3output = 'deconvS3.pvp';

%S1 output
s1data = {}
for n = 1:128
   s1data{n}.time = n-1;
   s1data{n}.values = zeros(1, 2);
   s1data{n}.values(1, 1) = n-1;
   s1data{n}.values(1, 2) = 1;
end
writepvpsparsevaluesfile(s1output, s1data, 1, 1, 128);


%S2 output
s2data = {}
for n = 1:256
   s2data{n}.time = n-1;
   s2data{n}.values = zeros(1, 2);
   s2data{n}.values(1, 1) = n-1;
   s2data{n}.values(1, 2) = 1;
end
writepvpsparsevaluesfile(s2output, s2data, 1, 1, 256);


%S3 output
s3data = {}
for n = 1:512
   s3data{n}.time = n-1;
   s3data{n}.values = zeros(1, 2);
   s3data{n}.values(1, 1) = n-1;
   s3data{n}.values(1, 2) = 1;
end
writepvpsparsevaluesfile(s3output, s3data, 1, 1, 512);
