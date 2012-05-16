#!/bin/bash
ant_nodes=({101..116})
for i in ${ant_nodes[*]};  do clush -w 10.0.0.${i} ps -C Clique2 -f; done