%
% Generates STDP params files
%

clear all; close all; more off;
format short g;

global MOVIE_FLAG = 0;
global PRINT_FLAG;

addpath('/Users/rcosta/Documents/workspace/HyPerSTDP/analysis/');
setenv("GNUTERM", "x11");

fullOrient_DATASET = "orient_36r";
DATASET = "OlshausenField_raw32x32_tiny"; %%orient_36r orient_simple  OlshausenField_raw32x32_tiny
TEMPLATE = "STDPgeneral";
on_v1_file = "w4_post.pvp";
off_v1_file = "w5_post.pvp";

numsteps = 10000;


%% INDEXES
DISPLAYPERIODi=7;
STRENGTH_IMAGE2RETINAi = 8;

wMaxInitSTDPi = 9;
wMinInitSTDPi = 10;
tauLTPi = 11;
tauLTDi = 12;
ampLTPi = 13;
ampLTDi = 14;

RUN_FLAG = 1;
PARAMSWEEP_FLAG = 0;
    PARAM_SWEEP = ampLTDi;
MEASURES_FLAG = 1;
    MEASURES_PLOT_FLAG = 0;
    MEASURES_OSI_FLAG = 1;
    MEASURES_GM_FLAG = 0;

global ROTATE_FLAG; ROTATE_FLAG = 0;

v1_cells = 8*8;
img_size = 32;


global params;
params{1} = "false";  %checkpointRead
params{2} = "true";  %checkpointWrite
params{3} = DATASET;  %checkpointReadDir
params{4} = 0;  %checkpointReadDirIndex
params{5} = 1000;  %checkpointWriteStepInterval
params{6} = "true"; %plasticityFlag
params{DISPLAYPERIODi} = 20; %displayPeriod (image display period)
params{STRENGTH_IMAGE2RETINAi} = 100;



%STDP params
params{wMaxInitSTDPi} = 0.05;
params{wMinInitSTDPi} = 0.03;
params{tauLTPi} = 20;
params{tauLTDi} = 30;
params{ampLTPi} = 0.05;%0.03;
params{ampLTDi} = 0.0285;%0.0085;


if(PARAMSWEEP_FLAG)
    rg = 0:0.005:0.1; %Range over which 

    disp("---------------------------");
    disp("------Parameter Sweep------");
    disp("---------------------------");

    OSIm = zeros(length(rg),2)
else
    rg = params{PARAM_SWEEP};
    OSIm = zeros(1,2)
end

for ps=1:length(rg)

    params{PARAM_SWEEP} = rg(ps);    
    
    [pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDPParams(DATASET, [], TEMPLATE, [], numsteps);

    if(RUN_FLAG)
        system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file]);
    end


    addpath([pvp_project_path, "mlab"]);

    if(PARAMSWEEP_FLAG==0)
        PRINT_FLAG = 1;
        filename = [pvp_output_path, filesep, on_v1_file];
        readpvpfile(filename, [pvp_output_path, filesep], on_v1_file, 1);
    end

    PRINT_FLAG = 0;


    %Plasticity ratio
    pr = (params{tauLTDi}*params{ampLTDi})/params{tauLTPi}*params{ampLTPi}

 
   

  if(MEASURES_FLAG)

    if(MEASURES_OSI_FLAG)



    disp("---------------------------");
    disp("-------Tunning curves------");
    disp("---------------------------");

    %1. For each checkpoint/weight matrix

    params{1} = "true";  %checkpointRead
    params{2} = "false";  %checkpointWrite
    params{3} = DATASET;  %checkpointReadDir
    %params{5} = 100;  %checkpointWriteStepInterval
    params{6} = "false";  %plasticityFlag

    global PVP_VERBOSE_FLAG;
    PVP_VERBOSE_FLAG = 0;

    %Read Orient dataset
    fid = fopen([pvp_project_path, "input", filesep, fullOrient_DATASET, '.txt' ], 'r');
    datasetl = {};

    c=1;
    while(~feof(fid))
        datasetl{c} = fgets(fid);
        c=c+1;
    end

    %Get and plot weights

    filename = [pvp_output_path, filesep, on_v1_file];
    [data hdr wm]=readpvpfile(filename, [pvp_output_path, filesep], on_v1_file, 1);

    hist_per_orient = zeros(numsteps/params{5}+1, v1_cells, length(datasetl));

    for i=0:params{5}:numsteps
    params{4} = i;  %checkpointReadDirIndex

    %Generates new params file
    [pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDPParams(fullOrient_DATASET, [], TEMPLATE, [], length(datasetl)*params{DISPLAYPERIODi}+i);
    length(datasetl)*params{DISPLAYPERIODi}+i
    %pause
    if(RUN_FLAG)
    system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file]); %Runs new params file

    %Reads V1 activity file (TODO: assumes that writing step for V1 is 1ms)
    [data hdr] = readpvpfile([pvp_output_path, filesep, "S1.pvp"], [pvp_output_path, filesep], "S1.pvp");
    for p=0:(length(datasetl)-1)
        for f=1:params{7}
         for v=1:v1_cells
             hist_per_orient(i/params{5}+1,v,p+1) = hist_per_orient(i/params{5}+1,v,p+1) + data{p*params{DISPLAYPERIODi}+f}(v);
         end
        end
    end
    end
    end
    avg_r_per_orient = (hist_per_orient*1000)./params{7};

    if(MEASURES_PLOT_FLAG)
    %Prepare Plot
    pict_size = 20;
    orient_plot = zeros(size(data{1},1)*size(data{1},2)*pict_size, (numsteps/params{5})*pict_size);

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
    end

    % V1on = "w4_post.pvp";
    % [data hdr wm] = readpvpfile([pvp_output_path, filesep, V1on], pvp_output_path, V1on, 1);

    %V1off = "w5_post.pvp";
    %[data hdr wm] = readpvpfile([pvp_output_path, V1off], pvp_output_path, V1off, 1);








    disp("------------------------------------------");
    disp("-------Orientation Selectivity Index------");
    disp("------------------------------------------");


    %Get Orientation Selectivity Index (OSI)
    osi = zeros(v1_cells,1);
    for x=1:length(osi) %Cell x
      f=fft(sum(hist_per_orient(:,x,:),1));
      %abs(f(2))
    osi(x) = (abs(f(2))/(abs(f(2))+mean(mean(avg_r_per_orient(:,x,:),1))))*100; %A 2nd harmonic/(A 2nd harmonic + delta_firingrate);
    end
    osi(isnan(osi))=0;

    if(MEASURES_PLOT_FLAG)
        figure
        hist(osi,30)
        xlabel('Orientation Selectivity Index');
        ylabel('Count');
        box off;

        plotHistOSI(hist_per_orient, 1, osi(1),36)
    end

    OSIm(ps,:) = [mean(osi) std(osi)];

    %TODO: Plot OSI over time


    end





    if(MEASURES_GM_FLAG)


    disp("------------------------------------------");
    disp("------------Generative Measure------------");
    disp("------------------------------------------");


    %Read Orient dataset
    fid = fopen([pvp_project_path, "input", filesep, DATASET, '.txt' ], 'r');
    datasetl = {};

    c=1;
    while(~feof(fid))
        datasetl{c} = fgets(fid);
        c=c+1;
    end

    post = 1;
    ign_w = 4;
    hist_per_img = zeros(numsteps/params{5}+1, v1_cells, length(datasetl));
    diff = zeros(numsteps/params{5}+1, length(datasetl));

    %Generative measure
     %1. For each image matrix
        for i=numsteps-params{5}:params{5}:numsteps
    %    for i=0:params{5}:numsteps
            params{4} = i;  %checkpointReadDirIndex

            %Generates new params file
            [pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDPParams(DATASET, [], TEMPLATE, [], length(datasetl)*params{DISPLAYPERIODi}+i);
            
            length(datasetl)*params{DISPLAYPERIODi}+i
            %pause

            if(RUN_FLAG)
                system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file]); %Runs new params file

                %2. Get activity and weight matrix
                %Reads V1 activity file (TODO: assumes that writing step for V1 is 1ms)
                [data hdr] = readpvpfile([pvp_output_path, filesep, "S1.pvp"], [pvp_output_path, filesep], "S1.pvp");

                %Reads the weights Retina_ON > V1 for the time being
                [d hdr wm] = readpvpfile([pvp_output_path, filesep, on_v1_file], [pvp_output_path, filesep],on_v1_file, post);
                figure
                imshow(wm);
                for p=0:(length(datasetl)-1)
                    for f=1:params{DISPLAYPERIODi}
                        for v=1:v1_cells
                            hist_per_img(i/params{5}+1,v,p+1) = hist_per_img(i/params{5}+1,v,p+1) + data{p*params{DISPLAYPERIODi}+f}(v);
                        end
                    end
                end

                %keyboard

                %3. Reconstruct the original image
                for p=1:(length(datasetl)) %Loop over images
                    img_recons = zeros(img_size);
                    for v=1:v1_cells %Loop over cells
                        mean_act = mean(hist_per_img(i/params{5}+1,v,p));
                        if(mean_act>0) %If cell reconstruct
                            [r c] = ind2sub([sqrt(v1_cells) sqrt(v1_cells)], v);
                            w=wm((r-1)*hdr.nxp+1:r*hdr.nxp, (c-1)*hdr.nyp+1:c*hdr.nyp);
                            w=w((hdr.nxp-(ign_w*(r-1)-1))-img_size:(hdr.nxp-(ign_w*(r-1))), (hdr.nxp-(ign_w*(c-1)-1))-img_size:(hdr.nxp-(ign_w*(c-1)))); %Get actual weights that are changed
                            img_recons = img_recons .+ (mean_act .* w);
                            [r c mean_act]
                        end
                    end

                    if(sum(sum(img_recons))>0)
                        img_recons = img_recons./max(max(img_recons));
                        img_orig = imread(strtrim(datasetl{p}));
                        if(ROTATE_FLAG==0)
                            img_orig = flipud(rot90(img_orig));
                            %img_orig = rot90(img_orig);
                        end
                        diff(i/params{5}+1,p) = mean(mean(abs(img_orig-img_recons)));
                        %if(i==numsteps)%Only plot the last ones
                            figure
                            subplot(1,2,1);
                            imshow(img_orig);
                            title('Original');
                            subplot(1,2,2);
                            imshow(img_recons);
                            title(['Reconstruction  diff=' num2str(diff(i/params{5}+1,p))]);
                            keyboard
                        %end
                        
                    end
                end    

                %4. Generative measure (use KL divergence): mean(D(I_G,I_O))

            end
        end

        end


    if(PARAMSWEEP_FLAG)
        [rg' OSIm pr]
    else
        [OSIm pr]
    end     
 
    end
end

