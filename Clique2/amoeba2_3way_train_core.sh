#! /bin/bash

time ./Debug/Clique2 -p ./input/amoeba2/3way/2FC/t/${1}/amoeba2_3way_2FC2_t_${1}.params 1> /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/2FC/t/${1}/amoeba2_3way_2FC2_t_${1}.log
rm /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/2FC/t/${1}/w4_last.pvp
rm /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/2FC/t/${1}/w5_last.pvp

time ./Debug/Clique2 -p ./input/amoeba2/3way/4FC/t/${1}/amoeba2_3way_4FC2_t_${1}.params 1> /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/4FC/t/${1}/amoeba2_3way_4FC2_t_${1}.log
rm /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/4FC/t/${1}/w4_last.pvp
rm /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/4FC/t/${1}/w5_last.pvp

time ./Debug/Clique2 -p ./input/amoeba2/3way/6FC/t/${1}/amoeba2_3way_6FC2_t_${1}.params 1> /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/6FC/t/${1}/amoeba2_3way_6FC2_t_${1}.log
rm /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/6FC/t/${1}/w4_last.pvp
rm /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/6FC/t/${1}/w5_last.pvp

time ./Debug/Clique2 -p ./input/amoeba2/3way/8FC/t/${1}/amoeba2_3way_8FC2_t_${1}.params 1> /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/8FC/t/${1}/amoeba2_3way_8FC2_t_${1}.log
rm /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/8FC/t/${1}/w4_last.pvp
rm /mnt/data/repo/neovision-programs-petavision/amoeba2/3way/activity/8FC/t/${1}/w5_last.pvp


