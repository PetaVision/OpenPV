lua ~/depthSCANN/input/depthInference/ATA_train_LCA.lua > ~/depthSCANN/input/depthInference/generated/ATA_train_LCA.params;
#One GPU run
~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/depthInference/generated/ATA_train_LCA.params -t 8

#MultiGPU run
#mpirun -np 4 --bind-to none ~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/depthInference/generated/ATA_train_LCA.params -t 8
