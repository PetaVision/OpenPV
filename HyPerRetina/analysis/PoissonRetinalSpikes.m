clear all;
global numloops = 5;
global tLCA; tLCA = 34; %% use  fraction of framerate
global columnSizeY; columnSizeY = 480/2;
global columnSizeX; columnSizeX = 480/2;
global numactivity; numactivity = columnSizeX * columnSizeY;
global PRINT_N; PRINT_N = 1;
global FNUM_ALL; FNUM_ALL = 1; %All frames
global NUM_PROCS; NUM_PROCS = 1;%% nproc();
global calculate_decaying_spikes; calculate_decaying_spikes = 0;
global numspikerates;

%Generates [numactivity x 1] poisson-distributed spikes for each timestep of one/numloops iteration of the video.
function [pois_spikes] = generate_poisson_spikes (spikerate)
    global tLCA;
    global numactivity;
    global numspikerates;
    maxsize=size(spikerate,2);
    for j=1:maxsize
        pois_spikerate{j} = zeros(numactivity,1);
        for t=1:tLCA
            pois_spikes{j}(:,t) = zeros(numactivity,1);
            random_mat = randint(numactivity,1,[0,10000000]);
            pois_spikes{j}(:,t) = (random_mat < (spikerate{j}'(:)*10000000));
            pois_spikerate{j} = pois_spikerate{j} + pois_spikes{j}(:,t);
        end
        pois_spikerate{j} = pois_spikerate{j}./tLCA;
        subtract_mat{j} = spikerate{j}'(:) - pois_spikerate{j};
        sum(sum(subtract_mat{j}));
    end
end

%Calculates the spike rate for each (timesteps/tLCA) chunk
function [m, spikerate] = par_spikerate_calc(cellspikes)
    global tLCA;
    global numactivity;
    global numspikerates;
    global columnSizeY;
    global columnSizeX;
    m=0;
    maxsize=size(cellspikes,2);
    for j=1:maxsize
        spikerate{j}=zeros(numactivity,1);
        for t=1:tLCA
            spikerate{j}=spikerate{j}+cellspikes{j}(:,t);
        end
        spikerate{j}=reshape(spikerate{j},columnSizeX,columnSizeY);
        spikerate{j}=spikerate{j}';
        spikerate{j} = spikerate{j}./tLCA;
    end
end
        




rootDir                                    = '/Users/wchavez';
workspaceDir                               = [rootDir, '/Documents/workspace/'];
pvpDir                                     = [rootDir, '/Documents/workspace/TestSquares/'];
outDir                                     = pvpDir;
global outputDir; outputDir                = [outDir, '850/'];
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
    global numactivity;
                                                  global numloops;

   
	%Change activitydata into sparse matrix
	activity = activityData{1}.spikeVec;
	time = activityData{1}.frameVec;
	timesteps = activityData{1}.numframes
	sparse_act = sparse(activity, time, 1, numactivity, timesteps);
    numspikerates = round(timesteps/tLCA);
                                                     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Writing spiking jpegs...');
    fflush(1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for t = 1:timesteps
        spike_mat = zeros(columnSizeY, columnSizeX);
        i = 1:numactivity;
        spike_mat = spike_mat';
        spike_mat(i) = sparse_act(:, t);
        spike_mat = spike_mat';
        print_filename = [outputMovieDir, 'spikes_', num2str(t), '.jpg'];
        imwrite(spike_mat, print_filename);
    end

                                                 
    if (calculate_decaying_spikes == 1)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	disp('Calculating intSpike...');
	fflush(1);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%Create decay kernel
	tau_kernel = exp(-[0:(5*tLCA)]/tLCA);
	tau_kernel = [zeros(1, (5*tLCA)+1), tau_kernel];

	%Split activity into number of processes
	if (mod(numactivity, NUM_PROCS) == 0)
		procSize = floor(numactivity / NUM_PROCS);
		cellAct = mat2cell(sparse_act, ones(1, NUM_PROCS) .* procSize, timesteps);
	else
		procSize = floor(numactivity / (NUM_PROCS - 1));
		lastSize = mod(numactivity, NUM_PROCS - 1);
		cellAct = mat2cell(sparse_act, [ones(1, NUM_PROCS - 1) .* procSize, lastSize], timesteps);
	end

	%Set rest of variables as cells for parcellfun
	cTau_Kernel{1} = tau_kernel;
	cIdentity{1} = [1];
	cShape{1} = 'same';

	%Create intSpikeCount matrix where it is indexed by (vectorized index, timestep)
	if NUM_PROCS == 1
		cIntSpike = cellfun(@conv2, cellAct, cTau_Kernel, cShape, 'UniformOutput', false);
	else
		cIntSpike = parcellfun(NUM_PROCS, @conv2, cellAct, cTau_Kernel, cShape, 'UniformOutput', false);
	end

	%Recombine from cells, needs to be rotated for collection of cell arrays
	cIntSpike = cellfun(@(x) x', cIntSpike, 'UniformOutput', false);
	intSpike = [cIntSpike{:}]';
    
    end %calculate_decaying_spikes

	

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Calculating spike rates for each cell and generating poisson-equivalent spikes...');
	fflush(1);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
    for i =1:numspikerates
        framecell{i}=sparse_act(1:numactivity, (i-1)*tLCA+1:i*tLCA);
    end
                        
    for j=1:numspikerates/numloops
        combframes{j}=framecell{j};
        for i=1:(numloops-1)
            combframes{j}=[combframes{j},framecell{round(i * numspikerates/numloops + j)}];
        end
    end
                        for j=1:numspikerates/numloops
                        disp(size(combframes{j}));
                        end
                        numspikerates
    %disp(framecell{3}(1:20,10));
    %disp(sparse_act(1:20,30));
    %disp(i);
                        
    %Split framecell for multithreading
    if (mod(numspikerates/numloops, NUM_PROCS) == 0)
        procSize = floor(numspikerates / numloops / NUM_PROCS)
        for j=1:NUM_PROCS
            cellspikes{j} = combframes((j-1)*procSize+1:j*procSize);
        end
                        
    else
        procSize = floor(numspikerates/numloops / (NUM_PROCS - 1))
        lastSize = round(mod(numspikerates/numloops, NUM_PROCS - 1))
                        
        for j=1:NUM_PROCS-1
            cellspikes{j} = combframes((j-1)*procSize+1:j*procSize);
        end
        cellspikes{NUM_PROCS}=combframes((NUM_PROCS-1)*procSize+1:numspikerates/numloops)
    end
    for p = 1:NUM_PROCS
        size(cellspikes{p})
    end
                        tic;
    if(NUM_PROCS==1)
        [m, spikerate] = cellfun(@par_spikerate_calc, cellspikes, 'UniformOutput', false);
    else
        [m, spikerate] = parcellfun(NUM_PROCS, @par_spikerate_calc, cellspikes, 'UniformOutput', false);
    end
                        toc;
                        
    %spikerate=[spikerate{:};];
    m=cell2mat(m);
    m=max(m);
                        
    if(NUM_PROCS==1)
        [pois_spikes] = cellfun(@generate_poisson_spikes, spikerate, 'UniformOutput', false);
    else
        [pois_spikes] = parcellfun(NUM_PROCS, @generate_poisson_spikes, spikerate, 'UniformOutput', false);
    end
    pois_spikes=[pois_spikes{:};];
    pois_spikes=[pois_spikes{:};];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Writing Poisson Spiking jpegs...');
    fflush(1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if(PRINT_N > 0)
        n=timesteps-columns(pois_spikes);
        for t = 1:timesteps-n
            nOutMat = zeros(columnSizeY, columnSizeX);
            i = 1:numactivity;
            nOutMat = nOutMat';
            nOutMat(i) = pois_spikes(:, t);
            nOutMat = nOutMat';
            if (m ~=0)
                nOutMat = nOutMat ./ m;
            end
            print_filename = [outputMovieDir, 'poisson_spikes' , num2str(t), '.jpg'];
            imwrite(nOutMat, print_filename);
        end
    end
end



                     
%Script
%Grab activity data
data = readactivitypvp(postActivityFile);
%Run spikes
spikes(data);
