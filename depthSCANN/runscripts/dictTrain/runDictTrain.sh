lua ~/depthSCANN/input/dictTrain/binoc_learn_512_white.lua > ~/depthSCANN/input/dictTrain/generated/binoc_learn_512_white.params;
#One GPU run
~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/dictTrain/generated/binoc_learn_512_white.params -t 8

#MultiGPU run
#mpirun -np 4 --bind-to none ~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/dictTrain/generated/binoc_learn_512_white.params -t 8;
