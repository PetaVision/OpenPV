lua ~/depthSCANN/input/benchmarkEncoding/encode_LCA.lua > ~/depthSCANN/input/benchmarkEncoding/generated/encode_LCA.params;
#One GPU run
~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/benchmarkEncoding/generated/encode_LCA.params -t 8

#MultiGPU run
#mpirun -np 4 --bind-to none ~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/benchmarkEncoding/generated/encode_LCA.params -t 8;
