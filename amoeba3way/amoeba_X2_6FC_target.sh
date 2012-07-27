#!/bin/bash
mkdir /mnt/data3/repo/neovision-programs-petavision/amoeba2/X2/activity/6FC/target/
mkdir /mnt/data3/repo/neovision-programs-petavision/amoeba/X2/activity/6FC/target/001/
time mpirun -np 1 ~/workspace-indigo/amoeba3way/Debug/amoeba3way -p ~/workspace-indigo/amoeba3way/input/amoeba2/X2/6FC/target/001/amoeba2_X2_6FC_target_001_combine.params 1> /mnt/data3/repo/neovision-programs-petavision/amoeba2/X2/activity/6FC/target/001/amoeba2_X2_6FC_target_001.log
rm /mnt/data3/repo/neovision-programs-petavision/amoeba2/X2/activity/6FC/target/001/w5_last.pvp
rm /mnt/data3/repo/neovision-programs-petavision/amoeba2/X2/activity/6FC/target/001/w6_last.pvp
rm /mnt/data3/repo/neovision-programs-petavision/amoeba2/X2/activity/6FC/target/001/w8_last.pvp
rm /mnt/data3/repo/neovision-programs-petavision/amoeba2/X2/activity/6FC/target/001/w9_last.pvp
