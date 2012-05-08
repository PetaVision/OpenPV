#!/bin/bash

for i_ver in {001..008}
do
    echo ${i_ver}
    time LD_LIBRARY_PATH=/usr/lib64/openmpi/lib  ~/workspace-indigo/Clique2/Debug/Clique2 -p ~/workspace-indigo/Clique2/input/Heli/Training/Car/canny/Car_bootstrap0/${i_ver}/Heli_Training_Car_Car_bootstrap0_canny_${i_ver}.params 1> /mnt/data/repo/neovision-programs-petavision/Heli/Training/activity/Car/canny/Car_bootstrap0/${i_ver}/Heli_Training_Car_Car_bootstrap0_canny_${i_ver}.log &
done
