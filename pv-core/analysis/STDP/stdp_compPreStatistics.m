function [prePatchAverage, nPostSpikes, kPostIndices, timeWeightsSum] = ...
    stdp_compPreStatistics(f_file, f_file_pre, W, firing_flag)
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
%     - It reads the post-synaptic neuron indices for pre-synaptic neurons.
%     - if firing_flag is 1, when the post-synaptic neuron spikes, we compute a weighted time 
%      average over the window W for the activity of each pre0-synaptic
%      neuron in the receptive field of the post-synaptic neuron.
%     - if firing_flag is 0, we compute the statistics for the cases when
%      the post-synaptic neuron does not fire.
% 
% Author: Marian Anghel



global input_dir output_dir conn_dir n_time_steps 

debug1 = 0;  % in initializing pre-synaptic activity matrices
debug2 = 0;
debug3 = 0;  % in reading post-synaptic indexes for pre-synaptic neurons
debug4 = 0;  % in reading pre-synaptic indexes for post-synaptic neurons


write_PCP_indexes = 1;
write_CP_indexes = 1;


nxpPost = 4; % post-synaptic neuron patch size (receptive field)
nypPost = 4;


nxpPre = 8; % pre-synaptic neuron patch size (projective field)
nypPre = 8;

patch_size = nxpPost*nypPost;
dT = 0.5;       % miliseconds (simulation time step)
%W = 30 / dT;    % size of time window for pre-synaptic spikes
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
    % define pre-synaptc activity vector
    preActivity = zeros(1,NXpre*NYpre*NFpre);
    
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
                   
nPostSpikes  = zeros(1,NXpost*NYpost*NFpost);
                   % counts the number of spikes for each 
                   % post-synaptic neuron;
                   % cumulative vector; gets appended
                   
nPreSpikes =  zeros(1,NXpost*NYpost*NFpost);  
                   % counts the number of spikes in the receptive field
                   % of each post-synaptic neuron: weighted sum
                   % snap-shot vector; gets updated every time step
                   
prePatchAverage = {}; % this is a cell of arrays; 
                    % each array is patch size, is cumulative
                    % and stores the sum of time averaged pre-synaptic
                    % neurons activity for each nonmargin post-synaptic
                    % neuron; Needs nPost Spikes to extract the 
                    % time average of the conditioned average patch activity
                   
% xyPost and kxPost are from 0 to NXpost-1, and NYpost -1
% respectively

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
            
            filename = [conn_dir 'PCP_' num2str(kxPost) '_' num2str(kyPost) '_0.dat'];
            ind =[];

            if exist(filename,'file')
                fid = fopen(filename, 'r', 'native');
                s=fgetl(fid);             % read first line
                %w=fscanf(fid,'%f',[4,4])  % read weights
                %pause
                % read patch indices
                for jp=1:nypPost
                    for ip=1:nxpPost
                        k = fscanf(fid, '%d',1);
                        i = fscanf(fid, '%d',1);
                        j = fscanf(fid, '%d',1);
                        f = fscanf(fid, '%d',1);
                        %fprintf('%d %d %d %d ',k,i,j,f);
                        ind=[ind;k];
                    end
                    %fprintf('\n');
                end
                fclose(fid);
                ind = ind' + 1; % shift by one
                preIndexes{kPost} = ind;
                if debug4
                    preIndexes{kPost}
                    pause
                end
                % initialize cell array of pre patch activity block
                %  this is a time x space matrix
                prePatch{kPost} = zeros(W,patch_size);
                
                % initialize cell array of pre patch activity matrix
                % this is updated conditioned on post-synaptic neuron
                % firing and does a weighted average of prePatch activity
                % over the time dimension; this is only a space matrix
                prePatchAverage{kPost} = zeros(1,patch_size);
                %pause
            else
                disp(['Skipping, could not open ', filename]);
                return
            end
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

%% write pre-synaptic indexes for post-synaptic neurons
if write_PCP_indexes
    filename = [conn_dir,'PCP_Indexes.dat'];
    fid = fopen(filename, 'w');
    for kPost=1:numel(preIndexes)
        ind =  preIndexes{kPost};
        if numel(ind)
            kxPost=rem(kPost-1,NXpost);
            kyPost=(kPost-1-kxPost)/NXpost;
            fprintf(fid,'%d %d %d ',kPost,kxPost,kyPost);

            for j=1:numel(ind)
                fprintf(fid,'%d ',ind(j));
            end
            fprintf(fid,'\n');
        end
    end
    fclose(fid);
    fprintf('finish writing all post-synaptic neurons patch indexes\n');
end
fprintf('finished reading post-synaptic neurons patch indexes\n');
fprintf('these are indices in the pre-synaptic layer\n');
fprintf('there are %d non-margin neurons\n',nPostIndices);
%pause


%% open STDP files 
for nPost = 1:numel(kPostIndices)
   kPost = kPostIndices(nPost);
   if firing_flag
       filename = [output_dir,'STDP_pre_fire_' num2str(W) '_' num2str(kPost) '.dat'];
   else
       filename = [output_dir,'STDP_pre_notfire_' num2str(W) '_' num2str(kPost) '.dat'];
   end
   fid_stat{nPost} = fopen(filename, 'w');
end



%% read post-synaptic neuron indices for pre-synaptic neurons
% The data format in these CP files is different from the one 
% in PCP files. It only reports the indices of the "real" neurons
% in the post-synaptic layer. That means that we can have nx * ny
% indices where nx <= nxpPre and ny <= nypPre.

for kyPre = 0:(NYpre-1)
    for kxPre = 0:(NXpre-1)
        kPre = kyPre * NXpre + kxPre + 1; % shift by 1
        if debug3
            fprintf('kxPre = %d kyPre = %d kPre = %d\n',...
                kxPre,kyPre,kPre);
        end
        
        filename = [conn_dir 'CP_' num2str(kxPre) '_' num2str(kyPre) '_0.dat'];

        if exist(filename,'file')
            fid = fopen(filename, 'r', 'native');
            s=fgetl(fid);   % read first line
                            % includes kPre in "extended" space!!
            C = textscan(fid,'%d %d %d %d');
            %for i=1:numel(C)
            %    C{i}
            %end
            %pause            
            fclose(fid);
            
            postIndexes{kPre} = C{1}' + 1;
            if debug3
                postIndexes{kPre}
                pause
            end
            
        else
            disp(['Skipping, could not open ', filename]);
            return
        end

    end % kxPre loop

end % kyPre loop

%% write post-synaptic indexes for pre-synaptic neurons
if write_CP_indexes
    filename = [conn_dir,'CP_Indexes.dat'];
    fid = fopen(filename, 'w');
    for kPre=1:numel(postIndexes)
        kxPre=rem(kPre-1,NXpre);
        kyPre=(kPre-1-kxPre)/NXpre;
        fprintf(fid,'%d %d %d: ',kPre,kxPre,kyPre);
        ind =  postIndexes{kPre};
        for j=1:numel(ind)
            fprintf(fid,'%d ',ind(j));
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    fprintf('finish writing all pre-synaptic neurons patch indexes\n');
end
fprintf('finished reading pre-synaptic neurons patch indexes\n');
fprintf('these are indices in the post-synaptic layer\n');
%pause

%% advance post-synaptic activity (W+1) steps
% and advance pre-synaptic activity W steps
% and fill the cell of pre-synaptic activity buffers

% read first post record: We preserve causality
% and look at pre-synaptic activity during time window
% [t-w, t-1] conditioned on post-synaptic activity at time t

timePost = fread(fid_post,1,'float64');
fprintf('read first post record: timePost = %f\n',timePost);
num_spikes = fread(fid_post, 1, 'int');
Spost =fread(fid_post, num_spikes, 'int') + 1 % S is a column vector

% we do not use Spost here, but we have to read it    
for i_step = 1 : W
    
    timePost = fread(fid_post,1,'float64');
    fprintf('\n\n~~~~~~~~~~~~~~~~~~~~\n\n');
    fprintf('timePost = %f\n',timePost);
    num_spikes = fread(fid_post, 1, 'int');
    eofstat = feof(fid_post);
    %fprintf('eofstat = %d\n', eofstat);
          
     Spost =fread(fid_post, num_spikes, 'int') + 1; % S is a column vector
     
     
     preActivity(:) = 0;
     timePre = fread(fid_pre,1,'float64');
     fprintf('timePost = %f timePre = %f\n',timePost,timePre);
     num_spikes = fread(fid_pre, 1, 'int');
     eofstat = feof(fid_pre);
     %fprintf('Spre:\n');
     % read pre-synaptic indices: ADJUST BY 1
     Spre =fread(fid_pre, num_spikes, 'int')+1; % S is a column vector

     % update only if there is pre-synaptic activity
     if numel(Spre)
         %fprintf('pre activity:\n');
         preActivity(Spre) = 1;
         %pause
     end

     % update pre patch activity only for post synaptic
     % non-margin neurons (not afected by boundaries)
     % only if the neuron is in the kPostIndices array!!
     for nPost = 1:numel(kPostIndices)
         kPost = kPostIndices(nPost);

         if debug1
             fprintf('kPost = %d: non-margin post neuron\n',kPost);
             kxPost=rem(kPost-1,NXpost);
             kyPost=(kPost-1-kxPost)/NXpost;
             fprintf('patch indexes for kPost = %d ',kPost);
             fprintf('(kxPost = %d, kyPost = %d):\n',kxPost,kyPost);
             preIndexes{kPost}
             fprintf('patch activity for kPost = %d:\n',kPost)
             preActivity(preIndexes{kPost})
             fprintf('pre patch activity before assignment:\n')
             prePatch{kPost}(i_step,:)
         end

         prePatch{kPost}(i_step,:) = preActivity(preIndexes{kPost});

         if debug1
             fprintf('pre patch activity after assignment:\n')
             prePatch{kPost}(i_step,:)
             pause
         end
     end % loop over post synaptic neurons


end % i_step loop from 1 to W
fprintf('finish advancing W steps in pre layer and (W+1) steps in post layer\n');
%pause


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
    %fprintf('timePost = %f\n',timePost);
    num_spikes = fread(fid_post, 1, 'int');
    
     %% ADJUST INDICES BY 1     
     Spost =fread(fid_post, num_spikes, 'int') + 1; % S is a column vector
          
     % advance pre-synaptic activity
     eofstat = feof(fid_pre);
     if eofstat
        fprintf('pre eofstat = %d\n', eofstat);
        break;
     end
     preActivity(:) = 0;
     timePre = fread(fid_pre,1,'float64');
     fprintf('timePost = %f timePre = %f\n',timePost,timePre);
     num_spikes = fread(fid_pre, 1, 'int');
     
     %fprintf('Spre:\n');
     %% ADJUST INDICES BY 1
     Spre =fread(fid_pre, num_spikes, 'int')+1; % S is a column vector

     % update preActivity vector
     if numel(Spre)
         %fprintf('pre activity:\n');
         preActivity(Spre) = 1;
         %pause
     end
     %update pre patch activity for all non-margin post-synaptic neurons
  
     for k = 1:nPostIndices % loop over nonmargin neurons
         
         record_activity = 0; % flag to be turned on based on 
                              % post-synaptic activity and firing_flag
         
         kPost = kPostIndices(k);
         
         ind = find(Spost == kPost); % check if this neurons fires
                   % We can also check if this neuron DOES NOT FIRE
                   % and compute pre-synaptic activity for this case!!
         
         if numel(ind)
             debug2=0;
         else
             debug2=0;
         end
         if debug2
             fprintf('kPost = %d: non-margin post neuron\n',kPost);
             kxPost=rem(kPost-1,NXpost);
             kyPost=(kPost-1-kxPost)/NXpost;
             fprintf('patch indexes for kPost = %d ',kPost);
             fprintf('(kxPost = %d, kyPost = %d):\n',kxPost,kyPost);
             preIndexes{kPost}
             fprintf('block prepatch activity before updating\n');
             for i=W:-1:1
                 for j=1:patch_size
                     fprintf('%d ',prePatch{kPost}(i,j));
                 end
                 fprintf('\n');
             end
         end
         %% copy-shift pre-patch activity block
         prePatch{kPost}(1:(W-1),:) = prePatch{kPost}(2:W,:);
         
         if debug2
             fprintf('kPost = %d: non-margin post neuron\n',kPost);
             kxPost=rem(kPost-1,NXpost);
             kyPost=(kPost-1-kxPost)/NXpost;
             fprintf('patch indexes for kPost = %d ',kPost);
             fprintf('(kxPost = %d, kyPost = %d):\n',kxPost,kyPost);
             preIndexes{kPost}
             fprintf('patch activity for kPost = %d:\n',kPost)
             preActivity(preIndexes{kPost})
             fprintf('time W: pre patch activity before assignment:\n')
             prePatch{kPost}(W,:)
         end
         
         % update pre-patch activity at time W
         prePatch{kPost}(W,:) = preActivity(preIndexes{kPost});
         
         if debug2
             fprintf('time W: pre patch activity after assignment:\n')
             prePatch{kPost}(W,:)
         end
     
         if debug2     
             fprintf('block prepatch activity after updating\n');
             for i=W:-1:1
                 for j=1:patch_size
                     fprintf('%d ',prePatch{kPost}(i,j));
                 end
                 fprintf('\n');
             end
             pause
         end
         
         % set record activity 
         % conditioned on post-synaptic neuron activity
         % and firing_flag
          if numel(ind)
              
              nPostSpikes(kPost) = nPostSpikes(kPost) + 1;
              
              if debug2
                  fprintf('kPost = %d nSpikes = %d: ',...
                      kPost, nPostSpikes(kPost));
                  fprintf('average block activity:\n');
              end
              
              if firing_flag
                  record_activity = 1;
              end
              
              
          else % neuron does not fire
              
              if ~firing_flag
                  record_activity = 1;
              end
          end % if numel(ind) 
                  
          % update statistics
          if record_activity
              
              % non-weighted average
              %avgActivity = sum(prePatch{kPost});
              % weighted average
              avgActivity = timeWeights * prePatch{kPost};
              nPreSpikes(kPost) = sum(avgActivity);
              
              if debug2
                  for i=1:patch_size
                      fprintf('%f ',avgActivity(i));
                  end

                  fprintf('\nprePatchAverage before updating:\n');
                  for i=1:patch_size
                      fprintf('%f ',prePatchAverage{kPost}(i));
                  end
              end
              
              % update prePatch activity
              prePatchAverage{kPost} = prePatchAverage{kPost} + ...
                  avgActivity;
              
              if debug2
                  fprintf('\nprePatchAverage after updating:\n');
                  for i=1:patch_size
                      fprintf('%f ',prePatchAverage{kPost}(i));
                  end
                  fprintf('\n');
              end
              
              
              fprintf(fid_stat{k},'%f %f ',timePost,nPreSpikes(kPost));
              for i=1:patch_size
                  fprintf(fid_stat{k}, '%f ',avgActivity(i));
              end
              fprintf(fid_stat{k},'\n');
              
              
          end % if record_activity (records pre-synaptic activity
              % based on post-synaptic activity and firing_flag)
     end % loop over nonmargin post synaptic neurons
     %pause
     
     if ~(mod(i_step,20000))
        
         if 1 % we can get this from avgActivity if we write
             % all of it
             if firing_flag
                filename = [output_dir,'prePatchAverage_pre_fire_', num2str(W),'.dat'];
             else
                filename = [output_dir,'prePatchAverage_pre_notfire_', num2str(W),'.dat']; 
             end
             fid = fopen(filename, 'w');
             for i=1:numel(kPostIndices)
                 kPost = kPostIndices(i);
                 patch = prePatchAverage{kPost}; %zeros(1,patch_size);
                 fprintf(fid,'%d ',kPost);
                 for j=1:numel(patch)
                     fprintf(fid,'%f ',patch(j));
                 end
                 fprintf(fid,'\n');
             end
             fclose(fid);
         end
        
         if firing_flag
            filename = [output_dir,'nPostSpikes_fire_',num2str(W),'.dat'];
         else
             filename = [output_dir,'nPostSpikes_notfire_',num2str(W),'.dat'];
         end
        fid = fopen(filename, 'w');
        for i=1:numel(kPostIndices)
            kPost = kPostIndices(i);
            fprintf(fid,'%d %d\n',kPost, nPostSpikes(kPost));
        end
        fclose(fid);
        
        
        fprintf('i_step = %d: break STDP statistics\n',i_step);
        break;
        
     end % write every 5000 time steps

end % loop over i_step (time)

%fclose(fid_stat);
fprintf('closing fid_pre and fid_post\n');
fclose(fid_pre);
fclose(fid_post);

%% close STDP files 
fprintf('closing all fid_stat files\n');
for nPost = 1:numel(kPostIndices)
   fclose(fid_stat{nPost});
end

%
% end primary function
    
    
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


