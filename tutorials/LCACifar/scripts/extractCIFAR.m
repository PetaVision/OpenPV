% This script will unpack the MATLAB version of the CIFAR dataset. Assuming
% the dataset was extracted to OpenPV/tutorials/LCACifar/cifar-10-batches-mat,
% running 'octave extractCIFAR.m' will unpack the images into that directory.

cifarPath = cd(cd("../cifar-10-batches-mat/"));
mixed_file_pathname = [cifarPath, '/mixed_cifar.txt'];

if exist(mixed_file_pathname, "file") == 2
    printf("WARNING: %s exists and is being deleted to prevent duplicate entries.\n", mixed_file_pathname);
    unlink(mixed_file_pathname);
end%if

printf("Extracting Images..");
extractImagesOctave([cifarPath, '/data_batch_1.mat'], 1);
extractImagesOctave([cifarPath, '/data_batch_2.mat'], 2);
extractImagesOctave([cifarPath, '/data_batch_3.mat'], 3);
extractImagesOctave([cifarPath, '/data_batch_4.mat'], 4);
extractImagesOctave([cifarPath, '/data_batch_5.mat'], 5);
extractImagesOctave([cifarPath, '/test_batch.mat'], 0, 0);
