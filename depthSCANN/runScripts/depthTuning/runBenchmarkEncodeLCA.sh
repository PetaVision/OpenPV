lua ~/depthSCANN/input/depthTuning/generate_downsample_depth.lua > ~/depthSCANN/input/depthTuning/generated/generate_downsample_depth.params;
#One GPU run
~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/depthTuning/generated/generate_downsample_depth.params -t 8

#MultiGPU run
#mpirun -np 4 --bind-to none ~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/benchmarkEncoding/generated/encode_LCA.params -t 8;
