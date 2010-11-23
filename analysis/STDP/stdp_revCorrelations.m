function [prePatchAverage, nPostSpikes, kPostIndices, timeWeightsSum] = ...
    stdp_revCorrelations(f_file, f_file_pre, W)
% Computes the statistics of STDP dynamics.
% We record the pre-synaptic activity in the receptive field of a post 
% synaptic neuron in a sliding window of size W (in msec).
% 
% REMARKS:
%     - It reads the sparse spike activity from both pre- and 
%       post-synaptic layer.
%     - It reads the pre-synaptic patch indices of post-synaptic neurons
%     ( we choose only post-synaptic neurons not affected by 
%         margins (boundaries in pre-synaptic layer)
%     - if firing_flag is 1, when the post-synaptic neuron spikes, we compute a weighted time 
%      average over the window W for the activity of ALL neurons in the 
%      pre-synaptic layer.
% 
% Author: Marian Anghel



global input_dir output_dir  conn_dir n_time_steps dT


nxpPost = 4; % post-synaptic neuron patch size (receptive field)
nypPost = 4;


nxpPre = 8; % pre-synaptic neuron patch size (projective field)
nypPre = 8;

plot_recfield = 0;
load_data     = 0;

debug1 = 0;  % in initializing pre-synaptic activity matrices
debug2 = 0;
debug3 = 0;  % in reading post-synaptic indexes for pre-synaptic neurons
debug4 = 0;  % in reading pre-synaptic indexes for post-synaptic neurons


%dT = 0.5;       % miliseconds (simulation time step)
%W = 30 / dT;   % size of time window for pre-synaptic spikes
                % NOTE: time step is 0.5
             
             
%% define the vector of weights for time averaging of block activity
STDP_TIME = 20; % miliseconds

timeWeights = zeros(1,W);
%c = exp(-dT/STDP_TIME);
c = 1.0;
timeWeights(1) = 1.0;
timeWeightsSum = timeWeights(1);

for i = 2:W
    timeWeights(i) = c * timeWeights(i-1);
    timeWeightsSum = timeWeightsSum + timeWeights(i);
end
fprintf('the vector of weights for time averaging of block activity:\n');
for i=1:W
    fprintf('time = %f weight = %f\n',i*dT,timeWeights(i));   
end
fprintf('\ntimeWeightsSum = %f\n',timeWeightsSum);
%pause


%% open pre-synaptic layer spiking file
filename = [input_dir, f_file_pre];
fprintf('read spikes from %s\n',filename);

if exist(filename,'file')
    fid_pre = fopen(filename, 'r', 'native');
    [time,numParams,NXpre,NYpre,NFpre] = readHeader(fid_pre);
    fprintf('time = %f numParams = %d NXpre = %d NYpre = %d NFpre = %d \n',...
        time, numParams,NXpre,NYpre,NFpre);
    %pause
else
    disp(['Skipping, could not open ', filename]);
    return
end


%% open post-synaptic layer spiking file
filename = [input_dir, f_file];
fprintf('read spikes from %s\n',filename);

if exist(filename,'file')
    fid_post = fopen(filename, 'r', 'native');
    [time,numParams,NXpost,NYpost,NFpost] = readHeader(fid_post);
    fprintf('time = %f numParams = %d NXpost = %d NYpost = %d NFpost = %d \n',...
        time, numParams,NXpost,NYpost,NFpost);
    %pause
else
    disp(['Skipping, could not open ', filename]);
    return
end


%% read pre-synaptic patch indices of post-synaptic neurons
% NOTE: choose only post-synaptic neurons not affected by 
% margins (boundaries in pre-synaptic layer)

kPostIndices = []; % stores linear indices (restricted space!)
                   % of kPost neurons not affected by margins 
nPostIndices = 0;  % counts number of kPost neurons
                   % not affected by margins
                   
                   
prePatchAverage = {}; % this is a cell of arrays; 
                    % each array is patch size, is cumulative
                    % and stores the sum of time averaged pre-synaptic
                    % neurons activity for each nonmargin post-synaptic
                    % neuron; Needs nPost Spikes to extract the 
                    % time average of the conditioned average patch activity

preActivity = zeros(1,NXpre*NYpre*NFpre);% define pre-synaptc activity vector
    
prePatch        = zeros(W,NXpre*NYpre*NFpre);% define space-time
                    % pre-synaptc activity 

                    % xyPost and kxPost are from 0 to NXpost-1, and NYpost -1
                    % respectively

                    
if ~load_data
    filename = [conn_dir,'PostMarginNeurons.dat'];
    fid_mar = fopen(filename, 'w');
    
    filename = [conn_dir,'PostNonmarginNeurons.dat'];
    fid_non = fopen(filename, 'w');
    
    for kyPost = 0:(NYpost-1)
        for kxPost = 0:(NXpost-1)
            kPost = kyPost * NXpost + kxPost + 1; % shift by 1
            if kyPost >= (nypPost-1) && kyPost <= (NYpost-nypPost) ...
                    && kxPost >= (nxpPost-1) && kxPost <= (NXpost-nxpPost)
                % non-margin neurons
                fprintf(fid_non,'%d %d %d\n',kPost,kxPost,kyPost);
                %fprintf('non: %d %d %d\n',kPost,kxPost,kyPost);
                
                nPostIndices = nPostIndices + 1;
                kPostIndices(nPostIndices) = kPost;
                if debug4
                    fprintf('non: kxPost = %d kyPost = %d kPost = %d\n',...
                        kxPost,kyPost,kPost);
                end
                
                % initialize cell array of pre patch activity matrix
                % this is updated conditioned on post-synaptic neuron
                % firing and does a weighted average of prePatch activity
                % over the time dimension; this is only a space matrix
                prePatchAverage{nPostIndices} = zeros(1,NXpre*NYpre*NFpre);
                %pause
                
            else % margin neurons
                fprintf(fid_mar,'%d %d %d\n',kPost,kxPost,kyPost);
                %fprintf('mar: %d %d %d\n',kPost,kxPost,kyPost);
                if debug4
                    fprintf('mar: kxPost = %d kyPost = %d kPost = %d\n',...
                        kxPost,kyPost,kPost);
                end
            end % condition for margin/nonmargin neurons
            
        end % kxPost loop
        
    end % kyPost loop
    fclose(fid_mar);
    fclose(fid_non);
    
    nPostSpikes  = zeros(1,nPostIndices);% counts the number of spikes for each
    % post-synaptic neuron;
    % cumulative vector; gets appended
    
else % load data
    
    % read indices of non-margin post-synaptic neurons
    % and their spiking numbers
    filename = [output_dir,'nPostSpikes_',num2str(W),'.dat'];
    fid = fopen(filename, 'r');
    data = load(filename); % N x 2 array
    kPostIndices = data(:,1);
    nPostSpikes = data(:,2);
    
    for nPost = 1:numel(kPostIndices)
        kPost = kPostIndices(nPost);
        %fprintf('kPost = %d nPost = %d\n',kPost,nPostSpikes(nPost));
        filename = [output_dir,'STDP_pre_' num2str(W) '_' num2str(kPost) '.dat'];
        data = load(filename);
        prePatchAverage{nPost} = reshape(data',1, NXpre*NYpre);
        
    end % loop over post-synaptic neurons
end


%% advance post-synaptic activity (W+1) steps
% and advance pre-synaptic activity W steps
% and fill the cell of pre-synaptic activity buffers

% read first post record: We preserve causality
% and look at pre-synaptic activity during time window
% [t-w, t-1] conditioned on post-synaptic activity at time t

% advance post-synaptic record once
timePost = fread(fid_post,1,'float64');
fprintf('read first post record: timePost = %f ',timePost);
num_spikes = fread(fid_post, 1, 'int');
fprintf('%d\n',num_spikes);
if num_spikes
   Spost =fread(fid_post, num_spikes, 'int') + 1; % S is a column vector
end

% we do not use Spost here, but we have to read it    
for i_step = 1 : W
    
    timePost = fread(fid_post,1,'float64');
    fprintf('\n\n~~~~~~~~~~~~~~~~~~~~\n\n');
    fprintf('timePost = %f ',timePost);
    num_spikes = fread(fid_post, 1, 'int');
    fprintf('%d ',num_spikes);
    eofstat = feof(fid_post);
    %fprintf('eofstat = %d\n', eofstat);
     if num_spikes     
        Spost =fread(fid_post, num_spikes, 'int') + 1; % S is a column vector
     else
         Spost = [];
     end
     
     preActivity(:) = 0;
     timePre = fread(fid_pre,1,'float64');
     fprintf('timePre = %f ',timePre);
     num_spikes = fread(fid_pre, 1, 'int');
     fprintf('%d\n',num_spikes);
     eofstat = feof(fid_pre);
     %fprintf('Spre:\n');
     % read pre-synaptic indices: ADJUST BY 1
     if num_spikes
        Spre =fread(fid_pre, num_spikes, 'int')+1; % S is a column vector
     else
         Spre = [];
     end
     % update only if there is pre-synaptic activity
     if numel(Spre)
         %fprintf('pre activity:\n');
         preActivity(Spre) = 1;
         %pause
     end

     % update pre patch activity 
     
      prePatch(i_step,:) = preActivity;

end % i_step loop from 1 to W
fprintf('finish advancing W steps in pre layer and (W+1) steps in post layer\n');
pause


%% advance both layers: gather STDP statistics 
% conditioned on post-synaptic
% spiking activity or nonspiking activity depending
% on firing_flag


%% advance sparse activity buffers and write statistics 
%% for non-margin neurons


for i_step = 1 : (n_time_steps-W-1)

    % advance post-synaptic activity
    eofstat = feof(fid_post);
    if eofstat
        fprintf('post eofstat = %d i_step = %d\n', eofstat,i_step);
        break;
    end
    
    timePost = fread(fid_post,1,'float64');
    %fprintf('\n\n~~~~~~~~~~~~~~~~~~~~\n\n');
    %fprintf('timePost = %f ',timePost);
    num_spikes = fread(fid_post, 1, 'int');
    %fprintf('%d ',num_spikes);
    
     %% ADJUST INDICES BY 1
     if num_spikes
        Spost =fread(fid_post, num_spikes, 'int') + 1; % S is a column vector
     else
         Spost = [];
     end
     % advance pre-synaptic activity
     eofstat = feof(fid_pre);
     if eofstat
        fprintf('pre eofstat = %d\n', eofstat);
        break;
     end
     preActivity(:) = 0;
     timePre = fread(fid_pre,1,'float64');
     %fprintf('timePre = %f ',timePre);
     num_spikes = fread(fid_pre, 1, 'int');
     %fprintf('%d\n',num_spikes);
     
     %fprintf('Spre:\n');
     %% ADJUST INDICES BY 1
     if num_spikes
         Spre =fread(fid_pre, num_spikes, 'int')+1; % S is a column vector
     else
         Spre = [];
     end
     % update preActivity vector
     if numel(Spre)
         %fprintf('pre activity:\n');
         preActivity(Spre) = 1;
         %pause
     end
     
     %% copy-shift pre-patch activity block
      prePatch(1:(W-1),:) = prePatch(2:W,:);
      % update pre-patch activity at time W
      prePatch(W,:) = preActivity;
     
      %% compute average weighted activity

     % weighted average
     avgActivity = (timeWeights * prePatch) ./ timeWeightsSum;
                            
     % update pre patch activity for all non-margin post-synaptic neurons
     if numel(Spost)
         for nPost = 1:nPostIndices % loop over nonmargin neurons
             
             kPost = kPostIndices(nPost);
             
             ind = find(Spost == kPost); % check if this neurons fires
             % We can also check if this neuron DOES NOT FIRE
             % and compute pre-synaptic activity for this case!!
             
             
             % set record activity
             % conditioned on post-synaptic neuron activity
             % and firing_flag
             if numel(ind)
                 
                 nPostSpikes(nPost) = nPostSpikes(nPost) + 1;
                 
                 % update prePatch activity for this neuron
                 prePatchAverage{nPost} = prePatchAverage{nPost} + ...
                     avgActivity;
                 
             end % if numel(ind)
             
         end % loop over nonmargin post synaptic neurons
     end % firing condition for post-syn layer
     
     if ~(mod(i_step,20000))
         
         filename = [output_dir,'nPostSpikes_',num2str(W),'.dat'];         
         fid = fopen(filename, 'w');
         for nPost=1:numel(kPostIndices)
             kPost = kPostIndices(nPost);
             fprintf(fid,'%d %d\n',kPost, nPostSpikes(nPost));
         end
         fclose(fid);         
             
         %% open STDP files
         for nPost = 1:numel(kPostIndices)
             kPost = kPostIndices(nPost);
             filename = [output_dir,'STDP_pre_' num2str(W) '_' num2str(kPost) '.dat'];
             
             fid_stat = fopen(filename, 'w');
                          
             prePatchAverage{nPost}=prePatchAverage{nPost}...
                 ./nPostSpikes(nPost);
             
             A = reshape(prePatchAverage{nPost},NXpre,NYpre);
             A = A';
             
             for j=1:NYpre
                 for i=1:NXpre
                     fprintf(fid_stat, '%f ',A(j,i));
                 end
                 fprintf(fid_stat,'\n');
             end
             fclose(fid_stat);
             
             if plot_recfield
                 imagesc(A')
                 pause
             end
         end   
         
         fprintf('\ni_step = %d\n\n',i_step);
         %break;
             
         
     end % write every 20000 time steps

     %pause
end % loop over i_step (time)

% close pre and post spiking files
fprintf('closing fid_pre and fid_post\n');
fclose(fid_pre);
fclose(fid_post);

%% write again at the end of the time step loop
filename = [output_dir,'nPostSpikes_',num2str(W),'.dat'];
fid = fopen(filename, 'w');
for nPost=1:numel(kPostIndices)
    kPost = kPostIndices(nPost);
    fprintf(fid,'%d %d\n',kPost, nPostSpikes(nPost));
end
fclose(fid);
    
for nPost = 1:numel(kPostIndices)
    kPost = kPostIndices(nPost);
    filename = [output_dir,'STDP_pre_' num2str(W) '_' num2str(kPost) '.dat'];
    
    fid_stat = fopen(filename, 'w');
    
    prePatchAverage{nPost}=prePatchAverage{nPost}...
        ./nPostSpikes(nPost);
    
    A = reshape(prePatchAverage{nPost},NXpre,NYpre);
    A = A';
    
    for j=1:NYpre
        for i=1:NXpre
            fprintf(fid_stat, '%f ',A(j,i));
        end
        fprintf(fid_stat,'\n');
    end
    fclose(fid_stat);
end


%% end primary function
    
    
function [time,numParams,NX,NY,NF] = readHeader(fid)
% NOTE: see analysis/python/PVReadWeights.py for reading params
% We call this function first because it rewinds the input file

    head = fread(fid,3,'int');
    if head(3) ~= 2  % PV_INT_TYPE
       disp('incorrect file type')
       return
    end
    numParams = head(2);
    fseek(fid,0,'bof'); % rewind file
    params = fread(fid, numParams-2, 'int'); 
    %pause
    NX         = params(4);
    NY         = params(5);
    NF         = params(6);
    fprintf('numParams = %d ',numParams);
    fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
    %pause
    % read time - last two params
    time = fread(fid,1,'float64');
    fprintf('time = %f\n',time);
    %pause
    
% End subfunction 
%    


