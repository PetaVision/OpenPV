#!/bin/bash
set -x
anterior_nodes=({01..16})  # set number of nodes
echo ${anterior_nodes[*]}
for i_node in ${anterior_nodes[*]}
do
    echo "i_node=${i_node}"
    ssh n00${i_node} 
done
