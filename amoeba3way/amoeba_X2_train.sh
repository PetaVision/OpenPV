#! /bin/bash
neuro_cores=({001..008})
for i_core in ${neuro_cores[*]}
do
    bash ./amoeba_X2_train_core.sh ${i_core} &
done
