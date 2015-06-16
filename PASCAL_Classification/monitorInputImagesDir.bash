#!/usr/bin/env bash
ls inputImages/* | sort > olda
while true
do
    sleep 1
    ls inputImages/* | sort > newa
    diffs="$(diff -Nau olda newa | egrep '^[+][^+]' | cut -c2-)"
    test -n "$diffs" && echo "$diffs"
    mv newa olda
done
