#!/bin/bash
ant_nodes=({01..16})
for i in ${ant_nodes[*]};  do clush -w n00${i} ps -C $1 -f; done