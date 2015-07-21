rm -r /nh/compneuro/Data/ImageNet/PetaVision/CatVsNoCatDog/Train/checkpoints/*
rm -r /nh/compneuro/Data/ImageNet/PetaVision/CatVsNoCatDog/Test/checkpoints/*
rsync --verbose  --progress --stats --compress --rsh=/usr/local/bin/ssh \
      --recursive --times --perms --links  \
      --exclude "*bak" --exclude "*~" \
      /mnt/data/ImageNet/* /nh/compneuro/Data/ImageNet