clear all;
global numloops = 100;                          % Number of times video frames are looped
global tLCA; tLCA = 50;                         % milliseconds per frame
global columnSizeY = 480/2;                     % Half of the image dimensions for Ganglion cells
global columnSizeX = 480/2;
global numactivity = columnSizeX * columnSizeY;
global PRINT_N; PRINT_N = 1;
global FNUM_ALL; FNUM_ALL = 1; %All frames
global NUM_PROCS; NUM_PROCS = 1;%% nproc();     % Number of processes to run.
global calculate_decaying_spikes = 0;
global numtotalframes;                          % Total number of frames in looped video.
global refractory_period; refractory_period = .004  % In seconds
global vidframes;


function [pois_spikes] = generate_poisson_spikes (spikerate)
    global columnSizeY;
    global columnSizeX;
    global tLCA;
    global numactivity;
    global numtotalframes;
    global numloops;
    global timesteps;
    global refractory_period;
    maxsize = size(spikerate,2);
    for j=1:maxsize

        % Increase spiking rate to account for refractory period
        fasterrate = spikerate{j}./(1-spikerate{j}.*refractory_period);

        % Number of inter-spike intervals
        num_isi = ceil(tLCA/(1000*refractory_period)+1);

        % 3D array of inter-spike intervals for each pixel
        tau_array = repmat(1./fasterrate, [1,1,num_isi]);

        % Gamma Distribution-like 3D array of inter-spike intervals > the refractory period
        isi_arg = rand([size(spikerate{j})(1:2),num_isi]);
        isi_array = (tau_array.* -log(isi_arg) + refractory_period).*1000;

        % Cumulative isi times in milliseconds
        cumtimeofspikes = zeros([size(spikerate{j})(1:2),num_isi]);

        % Rounding.  3rd dimension is the spike number
        cumtimeofspikes(:,:,1:num_isi) = round(cumsum(isi_array,3));

        % This is so the last timestep in each frame doesn't have a higher likelihood of spiking
        cumtimeofspikes(cumtimeofspikes > tLCA+refractory_period*1000) = tLCA+1+refractory_period*1000;

        pois_spikes{j} = zeros(numactivity, tLCA+1+refractory_period*1000);

        % Map the cumulative time of spikes to the correct pixel row and time column in pois_spikes
        for z=1:num_isi
            pois_spikes{j}(sub2ind(size(pois_spikes{j}), [1:numactivity], [cumtimeofspikes(:,:,z)'(:)'])) = 1;
        end

        % Throw away the first few timesteps in each frame, where there is a lower probability of spiking, and
        % the last timestep in each frame, where any cumulative time of spike greater than tLCA would spike in
        % this timestep.
        pois_spikes{j}(:,tLCA+1+refractory_period*1000) = [];
        pois_spikes{j}(:,1:refractory_period*1000)= [];
    end
end



%Calculates the spike rate for all (timesteps/(tLCA*numloops)) chunks
function [m, spikerate] = par_spikerate_calc(cellspikes)
    global columnSizeY;
    global columnSizeX;
    global tLCA;
    global numloops;
    m=0;
    maxsize=size(cellspikes,2);
    for j=1:maxsize
                                                 
        %Spike rate in Hz
        spikerate{j} = mean(cellspikes{j},2).*1000;

        spikerate{j} = reshape(spikerate{j},columnSizeX,columnSizeY);
        spikerate{j} = spikerate{j}';
        q=max(max(spikerate{j}));
        if(q>m)
            m=q;
        end
    end
end
        




rootDir                                    = '/Users/wchavez';
workspaceDir                               = [rootDir, '/Documents/workspace/'];
pvpDir                                     = [rootDir, '/Documents/workspace/TestSquares/'];
outDir                                     = pvpDir;
global outputDir; outputDir                = [outDir, '25000/'];
global outputMovieDir; outputMovieDir      = [outputDir, 'MovieGanglionON/'];
postActivityFile                           = [pvpDir,'GanglionON.pvp'];

if (exist(outDir, 'dir') ~= 7)
	mkdir(outDir);
end
if (exist(outputDir, 'dir') ~= 7)
	mkdir(outputDir);
end
if (exist(outputMovieDir, 'dir') ~= 7)
	mkdir(outputMovieDir);
end


function spikes(activityData)
    global columnSizeX columnSizeY;
	global tLCA;
	global NUM_PROCS;
	global outputDir;
	global outputMovieDir;
	global PRINT_N;
    global calculate_decaying_spikes;
    global numactivity;
    global numloops;
    global numtotalframes;
    global vidframes;


	%Change activitydata into sparse matrix

	activity = activityData{1}.spikeVec;
	time = activityData{1}.frameVec;
	timesteps = activityData{1}.numframes
	sparse_act = sparse(activity, time, 1, numactivity, timesteps);
    numtotalframes = round(timesteps/tLCA)
    vidframes = numtotalframes/numloops
                                                     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Writing spiking jpegs...');
    fflush(1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Just for second loop of video
    for t = 251:500
        spike_mat = zeros(columnSizeY, columnSizeX);
        i = 1:numactivity;
        spike_mat = spike_mat';
        spike_mat(i) = sparse_act(:, t);
        spike_mat = spike_mat';
        print_filename = [outputMovieDir, 'spikes_', num2str(t), '.jpg'];
        imwrite(spike_mat, print_filename);
    end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Calculating spike rates for each cell...');
	fflush(1);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Seperate sparse activity into frames, throwing away first loop of video to get rid of startup artifacts
    for i = vidframes + 1:numtotalframes
        framecell{i-vidframes}=sparse_act(1:numactivity, (i-1)*tLCA+1:i*tLCA);
    end

    %Combine corresponding frames from each loop of video.
    for j=1:numtotalframes/numloops
        combframes{j}=framecell{j};
        for i=1:(numloops-2)
            combframes{j}=[combframes{j},framecell{round(i * numtotalframes/numloops + j)}];
        end
    end

    for j=1:numtotalframes/numloops
        disp('size of combframes{j}:');
        disp(size(combframes{j}));
    end
    disp('numtotalframes:');
    numtotalframes

    %disp(framecell{3}(1:20,10));
    %disp(sparse_act(1:20,30));
    %disp(i);
                        
    %Split combframes for multithreading
    if (mod(numtotalframes/numloops, NUM_PROCS) == 0)
        procSize = floor(numtotalframes / numloops / NUM_PROCS)
        for j=1:NUM_PROCS
            cellspikes{j} = combframes((j-1)*procSize+1:j*procSize);
        end
                        
    else
        procSize = floor(numtotalframes/numloops / (NUM_PROCS - 1))
        lastSize = round(mod(numtotalframes/numloops, NUM_PROCS - 1))
                        
        for j=1:NUM_PROCS-1
            cellspikes{j} = combframes((j-1)*procSize+1:j*procSize);
        end
        cellspikes{NUM_PROCS}=combframes((NUM_PROCS-1)*procSize+1:numtotalframes/numloops);
    end

    for p = 1:NUM_PROCS
    disp('size of cellspikes{p}:');
        size(cellspikes{p})
    end

    %Calculate mean spikerates for each corresponding frame.
    tic;
    if(NUM_PROCS==1)
        [m, spikerate] = cellfun(@par_spikerate_calc, cellspikes, 'UniformOutput', false);
    else
        [m, spikerate] = parcellfun(NUM_PROCS, @par_spikerate_calc, cellspikes, 'UniformOutput', false);
    end
    toc;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Generating Poisson-distributes spikes (with refractory period)');
    fflush(1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    if(NUM_PROCS==1)
        [pois_spikes] = cellfun(@generate_poisson_spikes, spikerate, 'UniformOutput', false);
    else
        [pois_spikes] = parcellfun(NUM_PROCS, @generate_poisson_spikes, spikerate, 'UniformOutput', false);
    end
    toc;
    disp('size of pois_spikes:');
    size(pois_spikes)

    pois_spikes=[pois_spikes{:};];

    disp('size of pois_spikes:');
    size(pois_spikes)

    m=max(cell2mat(m));

    spikerate=[spikerate{:};];
                                                 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Writing spike rate jpegs...');
    fflush(1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for j=1:numtotalframes/numloops
        issparse(spikerate{j})
        spikerate{j} = full(spikerate{j}./m);
        print_filename = [outputMovieDir, 'spikerate_', num2str(j), '.jpg'];
        imwrite(spikerate{j}, print_filename);
    end


                                                 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Writing Poisson Spiking jpegs (with refractory period)...');
    fflush(1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    pois_spikes=[pois_spikes{:}];
    for t = 1:round(timesteps/numloops)
        nOutMat = zeros(columnSizeY, columnSizeX);
        i = 1:numactivity;
        nOutMat = nOutMat';
        nOutMat(i) = pois_spikes(:, t);
        nOutMat = nOutMat';
        print_filename = [outputMovieDir, 'pois_spikes_', num2str(t), '.jpg'];
        imwrite(nOutMat, print_filename);
    end



end






                     
%Script
%Grab activity data
tic;
data = readactivitypvp(postActivityFile);
%Run spikes
spikes(data);
toc;
