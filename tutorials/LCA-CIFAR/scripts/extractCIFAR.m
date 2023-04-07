% This script will unpack the MATLAB version of the CIFAR dataset. Assuming
% the dataset was extracted to OpenPV/tutorials/LCA-CIFAR/cifar-10-batches-mat,
% running 'octave extractCIFAR.m' will unpack the images into that directory.

script_dir = fileparts(mfilename("fullpath"));
% script_dir is the directory containing this script m-file
top_dir = cd(cd([script_dir filesep ".."]));
% top_dir is the directory containing script_dir

input_dir = [top_dir, filesep, "cifar-10-batches-mat"];
output_dir = [top_dir, filesep, "cifar-10-images"];
mixed_file_pathname = [output_dir, filesep, 'mixed_cifar.txt'];

cd(script_dir);

if exist(mixed_file_pathname, "file") == 2
    printf("WARNING: %s exists and is being deleted to prevent duplicate entries.\n", mixed_file_pathname);
    unlink(mixed_file_pathname);
end%if

printf("Extracting Images..\n");
extractImagesOctave([input_dir, filesep, 'data_batch_1.mat'], output_dir, 1);
extractImagesOctave([input_dir, filesep, 'data_batch_2.mat'], output_dir, 2);
extractImagesOctave([input_dir, filesep, 'data_batch_3.mat'], output_dir, 3);
extractImagesOctave([input_dir, filesep, 'data_batch_4.mat'], output_dir, 4);
extractImagesOctave([input_dir, filesep, 'data_batch_5.mat'], output_dir, 5);
extractImagesOctave([input_dir, filesep, 'test_batch.mat'], output_dir, 0, 0);
