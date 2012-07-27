#! /bin/bash

time ./Debug/amoeba3way -p ./input/amoeba/X2/2FC/t/${1}/amoeba_X2_2FC1_t_${1}.params 1> /mnt/data3/repo/neovision-programs-petavision/amoeba/X2/activity/2FC/t/${1}/amoeba_X2_2FC1_t_${1}.log

time ./Debug/amoeba3way -p ./input/amoeba/X2/4FC/t/${1}/amoeba_X2_4FC1_t_${1}.params 1> /mnt/data3/repo/neovision-programs-petavision/amoeba/X2/activity/4FC/t/${1}/amoeba_X2_4FC1_t_${1}.log

time ./Debug/amoeba3way -p ./input/amoeba/X2/6FC/t/${1}/amoeba_X2_6FC1_t_${1}.params 1> /mnt/data3/repo/neovision-programs-petavision/amoeba/X2/activity/6FC/t/${1}/amoeba_X2_6FC1_t_${1}.log

time ./Debug/amoeba3way -p ./input/amoeba/X2/8FC/t/${1}/amoeba_X2_8FC1_t_${1}.params 1> /mnt/data3/repo/neovision-programs-petavision/amoeba/X2/activity/8FC/t/${1}/amoeba_X2_8FC1_t_${1}.log

