all_noise = spatialPattern(DIM, BETA);

for index = 1:DIM(3)
   noise = all_noise(:, :, index);
   if ne(exist(NOISE_DIR), 7)
      mkdir(NOISE_DIR)
   end
   file_path = [NOISE_DIR, '/noise_', int2str(index), '.png'];
   dlmwrite(file_path, noise);
end
