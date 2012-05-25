#!/bin/bash
set -x
anterior_nodes=({101..116})
echo ${anterior_nodes[*]}
anterior_cores=({1..1})
echo ${anterior_cores[*]}

home_path=$HOME
openmpi64_home="/usr/lib64/openmpi"
#openmpi64_lib="/usr/lib64/openmpi/lib"
clique_path=${home_path}"/workspace-indigo/Clique2/"
echo ${clique_path[0]}
exe_path=${clique_path}"Debug/"
echo ${exe_path[0]}
input_path=${clique_path}"input/Heli/Formative/Car_distractor4/canny/mask/"
echo ${input_path[0]}
input_prefix="Heli_Formative_Car_distractor4_canny_mask_"
output_path="/mnt/data/repo/neovision-programs-petavision/Heli/Formative/activity/Car_distractor4/canny/mask/"
output_prefix="Heli_Formative_Car_distractor4_canny_mask"
version_id=0;
version_IDs=({001..016})
echo ${version_IDs[*]}
for i_node in ${anterior_nodes[*]}
do
    echo "i_node=${i_node}"
    for i_core in ${anterior_cores[*]}
    do
	echo "i_core=${i_core}"
	echo "version_id =${version_id}"
	input_params=${input_path}${version_IDs[${version_id}]}"/"${input_prefix}${version_IDs[${version_id}]}".params"
	echo "input_params=$input_params"
	ouput_log=${output_path}${output_prefix}${version_IDs[${version_id}]}".log" 
	echo $output_log
	host_node="10.0.0."${i_node}
	echo "host_node=${host_node}"
	#clush -v -w n${i_node} LD_LIBRARY_PATH=${openmpi64_lib} ${exe_path}Clique2 -p ${input_params} # -B ${output_log}
	# mpirun -np 1 --hostfile ~/.mpi_hosts -bynode --prefix ${openmpi64_home} ${exe_path}Clique2 -p ${input_params} & # 1> ${output_log}
	mpirun -np 8 -H ${host_node} --prefix ${openmpi64_home} ${exe_path}Clique2 -rows 2 -columns 4 -p ${input_params} & # 1> ${output_log}
	version_id=$((${version_id}+1))
    done
done

