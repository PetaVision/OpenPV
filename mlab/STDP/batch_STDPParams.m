%
% Generates STDP params files
%

DATASET = "orient_simple"; %%OlshausenField_raw32x32_tiny
setenv("GNUTERM", "x11");

RUN_FLAG = 1;

numsteps = 10000;
v1_cells = 8*8;

%%INDEXES
DISPLAYPERIODi=7; 

global params;
params{1} = "false";  %checkpointRead
params{2} = "true";  %checkpointWrite
params{3} = DATASET;  %checkpointReadDir
params{4} = 0;  %checkpointReadDirIndex
params{5} = 100;  %checkpointWriteStepInterval
params{6} = "true"; %plasticityFlag
params{DISPLAYPERIODi} = 20; %displayPeriod (image display period)

[pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDPParams(DATASET, [], [], [], numsteps);

if(RUN_FLAG)
    system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file]);
end

addpath([pvp_project_path, "mlab"]);

%Get tunning curves

disp("---------------------------");
disp("---Tunning curves---");
disp("---------------------------");

%1. For each checkpoint/weight matrix

params{1} = "true";  %checkpointRead
params{2} = "false";  %checkpointWrite
params{3} = DATASET;  %checkpointReadDir
params{5} = 100;  %checkpointWriteStepInterval
params{6} = "false";  %plasticityFlag

global PVP_VERBOSE_FLAG;
PVP_VERBOSE_FLAG = 0;

%Read dataset
fid = fopen([pvp_project_path, "input", filesep, DATASET, '.txt' ], 'r');
datasetl = {};

c=1;
while(~feof(fid))
    datasetl{c} = fgets(fid);
    c=c+1;
end

%Get and plot weights
name = {'w2_post.pvp'} %post point of view
filename = [pvp_output_path, filesep, name{1}];
[data hdr wm]=readpvpfile(filename, [pvp_output_path, filesep], name{1}, 1);
%keyboard

avg_r_per_orient = zeros(numsteps/params{5}+1, v1_cells, length(datasetl));

for i=0:params{5}:numsteps
    params{4} = i;  %checkpointReadDirIndex
    %numsteps = 200;

    %Generates new params file
    [pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDPParams(DATASET, [], [], [], length(datasetl)*params{DISPLAYPERIODi}+i);
    length(datasetl)*params{DISPLAYPERIODi}+i
    %pause
    if(RUN_FLAG)
        system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file]); %Runs new params file

        %Reads V1 activity file (TODO: assumes that writing step for V1 is 1ms)
        [data hdr] = readpvpfile([pvp_output_path, filesep, "S1.pvp"], [pvp_output_path, filesep], "S1.pvp");
        for p=0:(length(datasetl)-1)
            for f=1:params{7}
             for v=1:v1_cells
                 avg_r_per_orient(i/params{5}+1,v,p+1) = avg_r_per_orient(i/params{5}+1,v,p+1) + data{p*params{7}+f}(v);
             end
            end
        end
    end
end
avg_r_per_orient = avg_r_per_orient./params{7};

%Prepare Plot
pict_size = 10;
orient_plot = zeros(size(data{1},1)*size(data{1},2)*pict_size, (numsteps/params{5})*pict_size).*0.1;

%Build image
for i=0:size(data{1},1)*size(data{1},2)-1
    for j=0:numsteps/params{5}-1
        for z=1:size(avg_r_per_orient,3)
            im = imread(strtrim(datasetl{z}));
            orient_plot(i*pict_size+1:i*pict_size+pict_size, j*pict_size+1:j*pict_size+pict_size) += imresize(im, [pict_size pict_size]).*avg_r_per_orient(j+1, i+1, z);
        end
    end
end
figure
imagesc([0  (1/size(data{1},1)*size(data{1},2))*(numsteps/params{5})], [1 size(data{1},1)*size(data{1},2)], orient_plot);
axis image;
colorbar;
%set(gca,'XTick',[],'YTick',[])

% V1on = "w2_post.pvp";
% [data hdr wm] = readpvpfile([pvp_output_path, filesep, V1on], pvp_output_path, V1on, 1);

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
