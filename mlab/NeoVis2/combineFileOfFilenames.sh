#!/bin/bash
set -x
clip_IDs=({001..050})
echo ${clip_IDs[*]}
version_IDs=({001..016})
echo ${version_IDs[*]}
object_ID="distractor"
echo ${object_ID}
petavision_dir="/mnt/data/repo/neovision-programs-petavision/Heli/Training/"
echo ${petavision_dir}
list_dir=${petavision_dir}"list_canny/"
echo ${list_dir}
object_dir=${list_dir}${object_ID}"/"
echo ${object_dir[0]}
if [[ ! -d "${object_dir[0]}" ]]; then
    mkdir ${object_dir[0]}
fi
for version_ID in ${version_IDs[*]}
do
    echo ${version_ID}
    object_file=${object_dir[0]}${object_ID}"_"${version_ID}"_fileOfFilenames.txt"    
    echo ${object_file[0]}
    for clip_ID in ${clip_IDs[*]}
    do
	# echo ${clip_ID}
	clip_list=${list_dir}${clip_ID}"/"${clip_ID}"_"${version_ID}"_fileOfFilenames.txt"
	# echo ${clip_list}
	cat ${clip_list} >> ${object_file}  
    done  #  clip_ID
done  # version_ID