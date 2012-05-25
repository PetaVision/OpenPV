#! /bin/bash
neuro_cores=({001..008})
for i_core in ${neuro_cores[*]}
do
    bash ./amoeba_3way_train_core.sh ${i_core} &
done
