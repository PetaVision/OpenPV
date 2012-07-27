%
% Generates STDP params files
%

DATASET = "orient_simple"; %%OlshausenField_raw32x32_tiny

RUN_FLAG = 1;

global params;
params{1} = "false";  %checkpointRead
params{2} = "true";  %checkpointWrite
params{3} = DATASET;  %checkpointReadDir
params{4} = 0;  %checkpointReadDirIndex
params{5} = 100;  %checkpointWriteStepInterval

[pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDPParams(DATASET);

if(RUN_FLAG)
    system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file]);
end

addpath([pvp_project_path, "mlab"]);
%Get tunning curves

 %1. For each checkpoint/weight matrix
 V1on = "w2_post.pvp";
 [data hdr wm] = readpvpfile([pvp_output_path, V1on], pvp_output_path, V1on, 1);

 %2. For each orientation (generate 20 different orientations)
 %3. Plot 8x8 tuning curves and time vs gaussian params
 %4. Get selectivity measure: S = mean(x)+-std and mean(sigma)+-std

 %V1off = "w3_post.pvp";
 %[data hdr wm] = readpvpfile([pvp_output_path, V1off], pvp_output_path, V1off, 1);

 


%Generative measure
 %1. For each weight matrix
 %2. For each image
 %3. Get V1 activity and try to reconstruct the original image
 %4. Generative measure: mean(D(I_G,I_O))
