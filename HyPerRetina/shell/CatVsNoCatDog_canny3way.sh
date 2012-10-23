#!/bin/bash
set -x
anterior_nodes=({101..116})  # set number of nodes
echo ${anterior_nodes[*]}
anterior_cores=({1..1})  # sets number of jobs per node
echo ${anterior_cores[*]}

home_path=$HOME
openmpi64_home="/usr/lib64/openmpi"
#openmpi64_lib="/usr/lib64/openmpi/lib"
clique_path=${home_path}"/workspace-juno/SynthCog3/"
echo ${clique_path[0]}
exe_path=${clique_path}"Debug/"
echo ${exe_path[0]}
input_path=${clique_path}"input/CatVsNoCatDog/"${2}"/"${1}"/canny3way/"
echo ${input_path[0]}
input_prefix="CatVsNoCatDog_"${2}"_"${1}"_canny3way_"
output_path="/nh/compneuro/Data/ImageNet/PetaVision/CatVsNoCatDog/"${2}"/activity/"${1}"/canny3way/"
output_prefix="CatVsNoCatDog_"${2}"_"${1}"_canny3way_"
version_id=0; #0
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
	output_log=${output_path}${version_IDs[${version_id}]}"/"${output_prefix}${version_IDs[${version_id}]}".log" 
	echo $output_log
	touch $output_log
	host_node="10.0.0."${i_node}
	echo "host_node=${host_node}"
	mpirun -np 4 -H ${host_node} --prefix ${openmpi64_home} ${exe_path}SynthCog3 -rows 2 -columns 2 -p ${input_params} &>${output_log} & 
	version_id=$((${version_id}+1))
    done
done
