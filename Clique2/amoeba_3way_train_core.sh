#! /bin/bash

time ./Debug/Clique2 -p ./input/amoeba/3way/2FC/t/${1}/amoeba_3way_2FC1_t_${1}.params 1> /mnt/data/repo/neovision-programs-petavision/amoeba/3way/activity/2FC/t/${1}/amoeba_3way_2FC1_t_${1}.log

time ./Debug/Clique2 -p ./input/amoeba/3way/4FC/t/${1}/amoeba_3way_4FC1_t_${1}.params 1> /mnt/data/repo/neovision-programs-petavision/amoeba/3way/activity/4FC/t/${1}/amoeba_3way_4FC1_t_${1}.log

time ./Debug/Clique2 -p ./input/amoeba/3way/6FC/t/${1}/amoeba_3way_6FC1_t_${1}.params 1> /mnt/data/repo/neovision-programs-petavision/amoeba/3way/activity/6FC/t/${1}/amoeba_3way_6FC1_t_${1}.log

time ./Debug/Clique2 -p ./input/amoeba/3way/8FC/t/${1}/amoeba_3way_8FC1_t_${1}.params 1> /mnt/data/repo/neovision-programs-petavision/amoeba/3way/activity/8FC/t/${1}/amoeba_3way_8FC1_t_${1}.log

