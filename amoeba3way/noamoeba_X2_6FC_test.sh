#!/bin/bash
mkdir /mnt/data/repo/neovision-programs-petavision/noamoeba/X2/activity/6FC/distractor/
mkdir /mnt/data/repo/neovision-programs-petavision/noamoeba/X2/activity/6FC/distractor/001/
time mpirun -np 1 ~/workspace-indigo/amoeba3way/Debug/amoeba3way -p ~/workspace-indigo/amoeba3way/input/noamoeba/X2/6FC/distractor/001/noamoeba_X2_6FC_distractor_001_combine.params 1> /mnt/data/repo/neovision-programs-petavision/noamoeba/X2/activity/6FC/distractor/001/noamoeba_X2_6FC_distractor_001.log
rm /mnt/data/repo/neovision-programs-petavision/noamoeba/X2/activity/6FC/distractor/001/w5_last.pvp
rm /mnt/data/repo/neovision-programs-petavision/noamoeba/X2/activity/6FC/distractor/001/w6_last.pvp
